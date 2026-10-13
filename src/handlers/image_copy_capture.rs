use std::time::Duration;

use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::{Buffer as _, Fourcc, Modifier};
use smithay::backend::drm::DrmNode;
use smithay::backend::renderer::damage::OutputDamageTracker;
use smithay::backend::renderer::gles::GlesRenderer;
use smithay::backend::renderer::sync::SyncPoint;
use smithay::output::{Output, WeakOutput};
use smithay::reexports::calloop::generic::Generic;
use smithay::reexports::calloop::{Interest, LoopHandle, Mode, PostAction};
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::reexports::wayland_server::protocol::wl_pointer::WlPointer;
use smithay::reexports::wayland_server::protocol::wl_shm;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{Client, DisplayHandle, Weak};
use smithay::utils::{
    Buffer as BufferCoords, Logical, Physical, Point, Rectangle, Scale, Size, Transform,
};
use smithay::wayland::dmabuf::get_dmabuf;
use smithay::wayland::image_capture_source::{
    ImageCaptureSource, ImageCaptureSourceHandler, OutputCaptureSourceHandler,
    OutputCaptureSourceState,
};
use smithay::wayland::image_copy_capture::{
    BufferConstraints, CaptureFailureReason, CursorSession, CursorSessionRef, DmabufConstraints,
    Frame, FrameRef, ImageCopyCaptureHandler, ImageCopyCaptureState, Session, SessionRef,
};
use smithay::wayland::shm;
use wayland_backend::server::Credentials;

use crate::backend::Backend;
use crate::cursor::{RenderCursor, XCursor};
use crate::niri::{CastTarget, Niri, State};
use crate::protocols::foreign_toplevel::DynamicCastSource;
use crate::utils::{get_credentials_for_client, CastSessionId, CastStreamId};
use crate::window::mapped::Mapped;

/// Output capture session.
pub struct ImageCopySession {
    pub session: Session,
    pub damage_tracker: OutputDamageTracker,
    /// Frame waiting for output damage.
    pub pending_frame: Option<Frame>,
    /// Cast session id, shared with the cursor session of the same source.
    pub session_id: CastSessionId,
    /// Cast stream id, unique to this session.
    pub stream_id: CastStreamId,
    /// Credentials of the capturing client, if known.
    pub credentials: Option<Credentials>,
    /// For dynamic casts: size of the target we last sent constraints for.
    ///
    /// Used to avoid re-sending the same constraints every refresh, including when the target
    /// isn't picked yet (then it's a 1x1 placeholder).
    pub dynamic_cast_size: Option<(i32, i32)>,
}

/// Cursor capture session of an output.
pub struct ImageCopyCursorSession {
    pub session: CursorSession,
    /// Damage to the cursor image (i.e., not movement).
    pub damage_tracker: OutputDamageTracker,
    /// Frame waiting for cursor image change.
    pub pending_frame: Option<Frame>,
    /// Cast session id, shared with the output session of the same source.
    pub session_id: CastSessionId,
    /// Cast stream id, unique to this session.
    pub stream_id: CastStreamId,
    /// Credentials of the capturing client, if known.
    pub credentials: Option<Credentials>,
}

/// Credentials for the client which created a capture session.
///
/// Returns `None` if the session doesn't have its client (it has already been
/// dropped) or if the credentials cannot be determined (e.g., if peer is
/// credentials_unknown like mutter_service_channel).
fn session_credentials(dh: &DisplayHandle, client: Option<Client>) -> Option<Credentials> {
    get_credentials_for_client(dh, &client?)
}

/// Cast session id of an image capture source.
///
/// Stored on the source so that the output session and the corresponding cursor
/// session are reported as two streams of a single cast session.
fn source_session_id(source: &ImageCaptureSource) -> CastSessionId {
    source.user_data().insert_if_missing(CastSessionId::next);
    *source.user_data().get::<CastSessionId>().unwrap()
}

/// Output captured by a session, or `None` if it's gone (sessions can outlive
/// their output).
pub fn source_output(source: &ImageCaptureSource) -> Option<Output> {
    source.user_data().get::<WeakOutput>()?.upgrade()
}

/// Window captured by a session, and the output it is currently on, if any.
pub fn source_window<'a>(
    niri: &'a Niri,
    source: &ImageCaptureSource,
) -> Option<(&'a Mapped, Option<Output>)> {
    let surface = source
        .user_data()
        .get::<Weak<WlSurface>>()?
        .upgrade()
        .ok()?;
    let (mapped, output) = niri.layout.find_window_and_output(&surface)?;
    Some((mapped, output.cloned()))
}

/// Output of a session's source, following dynamic casts.
pub fn source_session_output(niri: &Niri, source: &ImageCaptureSource) -> Option<Output> {
    if let Some(target) = dynamic_cast_target_or(source, niri) {
        return Some(target.into_output());
    }

    source_output(source).or_else(|| source_window(niri, source).and_then(|(_, o)| o))
}

/// Whether a source follows the dynamic screencast target.
pub fn source_dynamic_cast(source: &ImageCaptureSource) -> bool {
    source.user_data().get::<DynamicCastSource>().is_some()
}

/// What a dynamic cast captures right now.
pub enum DynamicCastTarget<'a> {
    Output(Output),
    Window { mapped: &'a Mapped, output: Output },
}

impl DynamicCastTarget<'_> {
    /// Output that captures of this target must be rendered on.
    pub fn into_output(self) -> Output {
        match self {
            DynamicCastTarget::Output(output) => output,
            DynamicCastTarget::Window { output, .. } => output,
        }
    }
}

/// Resolves the dynamic cast target picked with the `set-dynamic-cast-*` actions.
///
/// Returns `None` when nothing was picked yet, or when the target is gone (closed window,
/// unplugged output); dynamic casts then capture nothing.
pub fn dynamic_cast_target(niri: &Niri) -> Option<DynamicCastTarget<'_>> {
    match &niri.dynamic_cast_target {
        CastTarget::Nothing => None,
        CastTarget::Output { output, .. } => {
            let output = output.upgrade()?;
            // The output may be gone already (the global lingers).
            niri.output_state
                .contains_key(&output)
                .then_some(DynamicCastTarget::Output(output))
        }
        CastTarget::Window { id } => {
            let (_, mapped) = niri
                .layout
                .windows()
                .find(|(_, mapped)| mapped.id().get() == *id)?;
            let output = window_output(niri, mapped)?;
            Some(DynamicCastTarget::Window { mapped, output })
        }
    }
}

/// Dynamic cast target of a source, if it is one.
fn dynamic_cast_target_or<'a>(
    source: &ImageCaptureSource,
    niri: &'a Niri,
) -> Option<DynamicCastTarget<'a>> {
    if !source_dynamic_cast(source) {
        return None;
    }

    dynamic_cast_target(niri)
}

/// Output that a window is on, if any.
fn window_output(niri: &Niri, mapped: &Mapped) -> Option<Output> {
    #[cfg(feature = "xdp-gnome-screencast")]
    if let Some(output) = niri.casting.mapped_cast_output.get(&mapped.window) {
        return Some(output.clone());
    }

    niri.layout
        .outputs()
        .find(|output| {
            niri.layout
                .windows_for_output(output)
                .any(|w| w.id() == mapped.id())
        })
        .cloned()
}

/// Output that a session captures as an output cast right now (following dynamic casts).
pub fn session_output_cast(niri: &Niri, s: &ImageCopySession) -> Option<Output> {
    let source = s.session.source();
    if let Some(target) = dynamic_cast_target_or(&source, niri) {
        // Dynamic casts targeting a window are captured by the window path.
        return match target {
            DynamicCastTarget::Output(output) => Some(output),
            DynamicCastTarget::Window { .. } => None,
        };
    }

    source_output(&source)
}

/// Window captured by a source right now (following dynamic casts).
pub fn source_window_cast<'a>(
    niri: &'a Niri,
    source: &ImageCaptureSource,
) -> Option<(&'a Mapped, Option<Output>)> {
    if let Some(target) = dynamic_cast_target_or(source, niri) {
        return match target {
            DynamicCastTarget::Window { mapped, output } => Some((mapped, Some(output))),
            DynamicCastTarget::Output(_) => None,
        };
    }

    source_window(niri, source)
}

/// Window that a session captures right now (following dynamic casts).
pub fn session_window_cast<'a>(
    niri: &'a Niri,
    s: &ImageCopySession,
) -> Option<(&'a Mapped, Option<Output>)> {
    source_window_cast(niri, &s.session.source())
}

/// Output whose size a dynamic cast uses as its placeholder.
///
/// It's also the output that renders the empty placeholder frames.
pub fn dynamic_cast_placeholder_output(niri: &Niri) -> Option<Output> {
    niri.global_space
        .outputs()
        .find(|output| output.current_mode().is_some())
        .cloned()
}

/// Size that a dynamic cast uses before its target is picked.
fn dynamic_cast_placeholder_size(niri: &Niri) -> Size<i32, BufferCoords> {
    let mode = dynamic_cast_placeholder_output(niri).and_then(|output| output.current_mode());

    match mode {
        Some(mode) => Size::from((mode.size.w, mode.size.h)),
        None => Size::from((1280, 720)),
    }
}

/// Constraints that a dynamic cast starts with before its target is picked.
///
/// Note the dmabuf formats: clients that feed a PipeWire stream (e.g.
/// xdg-desktop-portal-luminous) capture through their dmabuf path, and fail hard when the
/// session doesn't offer one.
fn dynamic_cast_placeholder_constraints(
    renderer: &mut GlesRenderer,
    render_node: Option<DrmNode>,
    size: Size<i32, BufferCoords>,
) -> BufferConstraints {
    BufferConstraints {
        size,
        shm: vec![wl_shm::Format::Xrgb8888, wl_shm::Format::Argb8888],
        dma: capture_dmabuf_constraints(renderer, render_node),
    }
}

/// Same as [`dynamic_cast_placeholder_constraints`], but without dmabuf formats.
///
/// Used when there's no renderer to build dmabuf constraints with (e.g. the headless backend).
fn dynamic_cast_placeholder_constraints_shm_only(
    size: Size<i32, BufferCoords>,
) -> BufferConstraints {
    BufferConstraints {
        size,
        shm: vec![wl_shm::Format::Xrgb8888, wl_shm::Format::Argb8888],
        dma: None,
    }
}

/// Keeps a dynamic cast session's constraints in sync with its target.
///
/// Runs as part of every refresh, before rendering: the client re-negotiates its buffer when we
/// fail its in-flight frame with `BufferConstraints` here. A dynamic cast session is never
/// dropped while there's no target to capture.
pub fn refresh_dynamic_cast_constraints(
    backend: &mut Backend,
    niri: &Niri,
    s: &mut ImageCopySession,
) {
    // Targets that are already gone capture nothing.
    let target = match dynamic_cast_target(niri) {
        Some(DynamicCastTarget::Output(output)) if output.current_mode().is_some() => {
            Some(DynamicCastTarget::Output(output))
        }
        Some(DynamicCastTarget::Window { mapped, output }) => {
            Some(DynamicCastTarget::Window { mapped, output })
        }
        _ => None,
    };

    let size = match &target {
        None => {
            let size = dynamic_cast_placeholder_size(niri);
            (size.w, size.h)
        }
        Some(DynamicCastTarget::Output(output)) => {
            let mode = output.current_mode().unwrap();
            (mode.size.w, mode.size.h)
        }
        Some(DynamicCastTarget::Window { mapped, output }) => {
            let size = window_capture_bbox(mapped, output).size;
            (size.w, size.h)
        }
    };

    // Only react to changes. This also keeps us from retrying forever when the constraints
    // can't be updated (e.g. no renderer): clients will ask again when they need to, and the
    // target changing (including back to "nothing") re-runs this.
    if s.dynamic_cast_size == Some(size) {
        return;
    }
    s.dynamic_cast_size = Some(size);

    if s.session
        .current_constraints()
        .is_some_and(|c| (c.size.w, c.size.h) == size)
    {
        return;
    }

    match &target {
        None => debug!("dynamic cast: no target picked, capturing nothing"),
        Some(DynamicCastTarget::Output(output)) => {
            debug!("dynamic cast: following output {}", output.name())
        }
        Some(DynamicCastTarget::Window { mapped, .. }) => {
            debug!("dynamic cast: following window {}", mapped.id().get())
        }
    }

    let render_node = backend.primary_render_node();
    // We can't drop the session if this fails: it just keeps showing nothing until the
    // constraints can be updated.
    let _ = refresh_session_constraints(backend, s, |renderer| match &target {
        None => Some(dynamic_cast_placeholder_constraints(
            renderer,
            render_node,
            dynamic_cast_placeholder_size(niri),
        )),
        Some(DynamicCastTarget::Output(output)) => {
            output_capture_constraints(renderer, render_node, output)
        }
        Some(DynamicCastTarget::Window { mapped, output }) => Some(window_capture_constraints(
            renderer, render_node, mapped, output,
        )),
    });
}

/// Buffer constraints for capturing an output.
///
/// `render_node` is the primary renderer's DRM render node (see
/// `Backend::primary_render_node()`), and only shm will be supported without
/// one. It is required since querying it from the EGL context fails for the TTY
/// backend: since Mesa 23.3 and until at least 26.1.8, `_eglGetGbmDisplay()`
/// clears the display's EGLDevice once a second EGLDisplay is created for the
/// same GBM device, which the TTY backend does during initialization. See
/// https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/44351.
pub fn output_capture_constraints(
    renderer: &GlesRenderer,
    render_node: Option<DrmNode>,
    output: &Output,
) -> Option<BufferConstraints> {
    let mode = output.current_mode()?;
    let size = Size::<i32, BufferCoords>::from((mode.size.w, mode.size.h));

    Some(BufferConstraints {
        size,
        shm: vec![wl_shm::Format::Xrgb8888],
        dma: capture_dmabuf_constraints(renderer, render_node),
    })
}

/// Buffer constraints for capturing a window.
pub fn window_capture_constraints(
    renderer: &GlesRenderer,
    render_node: Option<DrmNode>,
    mapped: &Mapped,
    output: &Output,
) -> BufferConstraints {
    let size = window_capture_bbox(mapped, output).size;

    BufferConstraints {
        size: Size::from((size.w, size.h)),
        shm: vec![wl_shm::Format::Xrgb8888, wl_shm::Format::Argb8888],
        dma: capture_dmabuf_constraints(renderer, render_node),
    }
}

/// Bounding box of the window's own content, including popups.
pub fn window_capture_bbox(mapped: &Mapped, output: &Output) -> Rectangle<i32, Physical> {
    let scale = Scale::from(output.current_scale().fractional_scale());
    mapped
        .window
        .bbox_with_popups()
        .to_physical_precise_up(scale)
}

/// Dmabuf constraints of the capture source.
fn capture_dmabuf_constraints(
    renderer: &GlesRenderer,
    render_node: Option<DrmNode>,
) -> Option<DmabufConstraints> {
    let node = render_node?;
    let egl = renderer.egl_context();

    // Offer all formats the renderer can draw into to avoid unnecessary conversions, preserving the
    // original order (many clients depend on the order being stable to select the same format when
    // renegotiating).
    let mut formats: Vec<(Fourcc, Vec<Modifier>)> = Vec::new();
    for format in egl.dmabuf_render_formats().iter() {
        if let Some((_, modifiers)) = formats.iter_mut().find(|(code, _)| *code == format.code) {
            modifiers.push(format.modifier)
        } else {
            formats.push((format.code, vec![format.modifier]))
        }
    }
    if formats.is_empty() {
        return None;
    }

    // Put Xrgb8888 and Argb8888 first since some clients always take the first advertised format
    // (e.g. wl-mirror, grim).
    formats.sort_by_key(|(code, _)| match code {
        Fourcc::Xrgb8888 => 0,
        Fourcc::Argb8888 => 1,
        _ => 2,
    });

    Some(DmabufConstraints { node, formats })
}

/// Refresh the session constraints and sends them to the client.
pub fn refresh_session_constraints(
    backend: &mut Backend,
    s: &mut ImageCopySession,
    build: impl FnOnce(&mut GlesRenderer) -> Option<BufferConstraints>,
) -> bool {
    let Some(constraints) = backend.with_primary_renderer(build).flatten() else {
        return false;
    };

    // Cannot capture a frame for outdated constraints, so fail it before sending the new
    // constraints (otherwise clients which re-negotiate on failure may miss the new `done`).
    if let Some(frame) = s.pending_frame.take() {
        frame.fail(CaptureFailureReason::BufferConstraints);
    }
    s.session.update_constraints(constraints);

    true
}

/// Updates the cursor session constraints if the cursor size has changed, returning the current
/// cursor size.
pub fn update_cursor_constraints(
    niri: &Niri,
    s: &mut ImageCopyCursorSession,
    output: &Output,
) -> Size<i32, BufferCoords> {
    let constraints = cursor_capture_constraints(niri, output);
    let size = constraints.size;
    if s.session
        .current_constraints()
        .is_none_or(|c| (c.size.w, c.size.h) != (size.w, size.h))
    {
        // Cannot capture a frame for outdated constraints, so fail it before sending the new
        // constraints (otherwise clients which re-negotiate on failure may miss the new `done`).
        if let Some(frame) = s.pending_frame.take() {
            frame.fail(CaptureFailureReason::BufferConstraints);
        }
        s.session.update_constraints(constraints);
    }
    size
}

/// Buffer constraints for capturing the cursor of an output. Argb8888 since it has alpha.
pub fn cursor_capture_constraints(niri: &Niri, output: &Output) -> BufferConstraints {
    BufferConstraints {
        size: cursor_capture_size(niri, output),
        shm: vec![wl_shm::Format::Argb8888],
        dma: None,
    }
}

/// Size the cursor renders at on this output.
fn cursor_capture_size(niri: &Niri, output: &Output) -> Size<i32, BufferCoords> {
    let int_scale = output.current_scale().integer_scale();
    let scale = Scale::from(output.current_scale().fractional_scale());

    let size: Size<i32, Physical> = match niri.cursor_manager.get_render_cursor(int_scale) {
        RenderCursor::Hidden => Size::from((0, 0)),
        RenderCursor::Surface { surface, .. } => {
            let bbox = smithay::desktop::utils::bbox_from_surface_tree(&surface, (0, 0));
            bbox.to_f64().to_physical_precise_up(scale).size
        }
        RenderCursor::Named {
            scale: buffer_scale,
            cursor,
            ..
        } => {
            // All frames are the same size since CursorManager::load_xcursor()
            // picks one size and rejects frames which differ.
            let (_idx, frame) = cursor.frame(niri.start_time.elapsed().as_millis() as u32);
            // The image is loaded at the integer scale but drawn at the fractional one, so it
            // ends up smaller than its own buffer whenever the two differ.
            let logical = Size::<f64, Logical>::from((
                f64::from(frame.width) / f64::from(buffer_scale),
                f64::from(frame.height) / f64::from(buffer_scale),
            ));
            logical.to_physical_precise_ceil(scale)
        }
    };

    // Fall back to the nominal cursor size when the cursor is currently hidden or has no
    // buffer, so that the session always has valid constraints.
    if size.is_empty() {
        let fallback = i32::from(niri.config.borrow().cursor.xcursor_size) * int_scale;
        return Size::from((fallback, fallback));
    }

    Size::from((size.w, size.h))
}

/// Cursor hotspot in capture buffer coordinates.
pub fn cursor_capture_hotspot(niri: &Niri, output: &Output) -> Point<i32, BufferCoords> {
    let int_scale = output.current_scale().integer_scale();
    let scale = Scale::from(output.current_scale().fractional_scale());

    let hotspot: Point<i32, Physical> = match niri.cursor_manager.get_render_cursor(int_scale) {
        RenderCursor::Hidden => Point::from((0, 0)),
        RenderCursor::Surface { surface, hotspot } => {
            // The tree is shifted to put its bounding box at the origin in
            // render_cursor_for_capture(), so shift the hotspot too.
            let bbox = smithay::desktop::utils::bbox_from_surface_tree(&surface, (0, 0));
            (hotspot - bbox.loc)
                .to_f64()
                .to_physical_precise_round(scale)
        }
        RenderCursor::Named {
            scale: buffer_scale,
            cursor,
            ..
        } => {
            let (_idx, frame) = cursor.frame(niri.start_time.elapsed().as_millis() as u32);
            // Same rescaling as in cursor_capture_size().
            XCursor::hotspot(frame)
                .to_logical(buffer_scale)
                .to_f64()
                .to_physical_precise_round(scale)
        }
    };

    Point::from((hotspot.x, hotspot.y))
}

pub enum CaptureBuffer {
    Dma(Dmabuf),
    Shm,
}

/// Checks a frame's buffer for compatibility with the render helpers.
///
/// Smithay only validates buffers against the protocol constraints, which allow
/// a client to attach a larger buffer than asked for, but the render helpers
/// need an exact match, and checking them here allows mismatches to be reported
/// to the client.
pub fn capture_buffer(
    buffer: &WlBuffer,
    size: Size<i32, BufferCoords>,
    format: wl_shm::Format,
) -> Option<CaptureBuffer> {
    if let Ok(dmabuf) = get_dmabuf(buffer) {
        let size_matches = dmabuf.width() == size.w as u32 && dmabuf.height() == size.h as u32;
        return size_matches.then(|| CaptureBuffer::Dma(dmabuf.clone()));
    }

    // Stride and pool placement are defined by the client, but the format and
    // dimensions need to match.
    let matches = shm::with_buffer_contents(buffer, |_ptr, _len, data| {
        data.format == format && data.width == size.w && data.height == size.h
    })
    .unwrap_or(false);
    matches.then_some(CaptureBuffer::Shm)
}

/// Finishes a frame, waiting for `sync_point` first if the render is still in flight.
///
/// Like `Screencopy::submit_after_sync()`.
pub fn frame_success_after_sync<T>(
    frame: Frame,
    transform: Transform,
    damage: Vec<Rectangle<i32, BufferCoords>>,
    presented: Duration,
    sync_point: Option<SyncPoint>,
    event_loop: &LoopHandle<'static, T>,
) {
    let Some(sync_fd) = sync_point.and_then(|sync| sync.export()) else {
        frame.success(transform, damage, presented);
        return;
    };

    let mut pending = Some((frame, damage));
    let source = Generic::new(sync_fd, Interest::READ, Mode::OneShot);
    let res = event_loop.insert_source(source, move |_, _, _| {
        let (frame, damage) = pending.take().unwrap();
        frame.success(transform, damage, presented);
        Ok(PostAction::Remove)
    });
    if let Err(err) = res {
        // The client will retry (smithay sends an Unknown error when Drop is
        // called on the Frame).
        warn!("error waiting for image copy capture sync point: {err}");
    }
}

impl ImageCaptureSourceHandler for State {}

impl OutputCaptureSourceHandler for State {
    fn output_capture_source_state(&mut self) -> &mut OutputCaptureSourceState {
        &mut self.niri.output_capture_source_state
    }

    fn output_source_created(&mut self, source: ImageCaptureSource, output: &Output) {
        source.user_data().insert_if_missing(|| output.downgrade());
    }
}

impl ImageCopyCaptureHandler for State {
    fn image_copy_capture_state(&mut self) -> &mut ImageCopyCaptureState {
        &mut self.niri.image_copy_capture_state
    }

    fn capture_constraints(&mut self, source: &ImageCaptureSource) -> Option<BufferConstraints> {
        if source_dynamic_cast(source) {
            // Dynamic casts follow the target of the set-dynamic-cast-* actions.
            //
            // Clients need constraints to create a session, but the target may not be picked
            // yet: start with a 1x1 placeholder and let refresh_image_copy_capture() update the
            // constraints once there's something to capture.
            let target = match dynamic_cast_target(&self.niri) {
                Some(DynamicCastTarget::Output(output)) if output.current_mode().is_some() => {
                    Some(DynamicCastTarget::Output(output))
                }
                Some(DynamicCastTarget::Window { mapped, output }) => {
                    Some(DynamicCastTarget::Window { mapped, output })
                }
                _ => None,
            };

            if target.is_none() {
                // Nothing picked yet: hand out the placeholder constraints. Try to include the
                // dmabuf formats: clients that feed a PipeWire stream capture through their
                // dmabuf path.
                let size = dynamic_cast_placeholder_size(&self.niri);
                let render_node = self.backend.primary_render_node();
                return self
                    .backend
                    .with_primary_renderer(|renderer| {
                        dynamic_cast_placeholder_constraints(renderer, render_node, size)
                    })
                    .or_else(|| Some(dynamic_cast_placeholder_constraints_shm_only(size)));
            }

            let render_node = self.backend.primary_render_node();
            return self
                .backend
                .with_primary_renderer(|renderer| match &target {
                    None => Some(dynamic_cast_placeholder_constraints(
                        renderer,
                        render_node,
                        dynamic_cast_placeholder_size(&self.niri),
                    )),
                    Some(DynamicCastTarget::Output(output)) => {
                        output_capture_constraints(renderer, render_node, output)
                    }
                    Some(DynamicCastTarget::Window { mapped, output }) => {
                        Some(window_capture_constraints(renderer, render_node, mapped, output))
                    }
                })
                .flatten()
                // Like with the target not being picked, keep the session alive when we can't
                // build the real constraints (e.g. no renderer).
                .or_else(|| {
                    Some(dynamic_cast_placeholder_constraints_shm_only(
                        dynamic_cast_placeholder_size(&self.niri),
                    ))
                });
        }

        if let Some((mapped, Some(output))) = source_window(&self.niri, source) {
            if !self.niri.output_state.contains_key(&output) {
                return None;
            }

            let render_node = self.backend.primary_render_node();
            return self.backend.with_primary_renderer(|renderer| {
                window_capture_constraints(renderer, render_node, mapped, &output)
            });
        }

        let output = source_output(source)?;
        if !self.niri.output_state.contains_key(&output) {
            return None;
        }

        let render_node = self.backend.primary_render_node();
        self.backend
            .with_primary_renderer(|renderer| {
                output_capture_constraints(renderer, render_node, &output)
            })
            .flatten()
    }

    fn cursor_capture_constraints(
        &mut self,
        source: &ImageCaptureSource,
        _pointer: &WlPointer,
    ) -> Option<BufferConstraints> {
        let output = source_session_output(&self.niri, source)?;
        if !self.niri.output_state.contains_key(&output) {
            return None;
        }

        Some(cursor_capture_constraints(&self.niri, &output))
    }

    fn new_session(&mut self, session: Session) {
        // Will be updated with the output properties before capture.
        let damage_tracker = OutputDamageTracker::new((0, 0), 1.0, Transform::Normal);
        let session_id = source_session_id(&session.source());
        let credentials = session_credentials(&self.niri.display_handle, session.client());
        self.niri.image_copy_sessions.push(ImageCopySession {
            session,
            damage_tracker,
            pending_frame: None,
            session_id,
            stream_id: CastStreamId::next(),
            credentials,
            dynamic_cast_size: None,
        });
    }

    fn new_cursor_session(&mut self, session: CursorSession) {
        // Will be updated with the output properties before capture.
        let damage_tracker = OutputDamageTracker::new((0, 0), 1.0, Transform::Normal);
        let session_id = source_session_id(&session.source());
        let credentials = session_credentials(&self.niri.display_handle, session.client());
        self.niri
            .image_copy_cursor_sessions
            .push(ImageCopyCursorSession {
                session,
                damage_tracker,
                pending_frame: None,
                session_id,
                stream_id: CastStreamId::next(),
                credentials,
            });
        // Send the initial cursor position and hotspot.
        self.niri.refresh_image_copy_cursor_sessions();
    }

    fn frame(&mut self, session: &SessionRef, frame: Frame) {
        // A dynamic cast with nothing picked yet captures an empty frame on the output it uses
        // as its placeholder (see render_for_image_copy_capture()).
        //
        // That output has to be redrawn for the frame to be completed: like any other cast, the
        // frame is captured together with the output's render pass.
        let frame_source = session.source();
        let placeholder_output = if source_dynamic_cast(&frame_source)
            && dynamic_cast_target(&self.niri).is_none()
        {
            dynamic_cast_placeholder_output(&self.niri)
        } else {
            None
        };
        let Some(s) = self
            .niri
            .image_copy_sessions
            .iter_mut()
            .find(|s| s.session == *session)
        else {
            frame.fail(CaptureFailureReason::Unknown);
            return;
        };

        // A session may only have one frame object in flight at a time, a
        // second one is a duplicate_frame protocol error. Smithay doesn't check
        // itself it (create_frame pushes onto active_frames unconditionally)
        // and doesn't expose the session object for us to post_error() on, so
        // fail the frame rather than leak it.
        if s.pending_frame.is_some() {
            warn!("client created a second frame while one was still in flight"); // client bug
            frame.fail(CaptureFailureReason::Unknown);
            return;
        }
        s.pending_frame = Some(frame);

        // The frame is captured on the next redraw with damage.
        let output = if placeholder_output.is_some() {
            placeholder_output
        } else {
            source_session_output(&self.niri, &session.source())
        };
        if let Some(output) = output {
            // The output may be gone already (the global lingers).
            if self.niri.output_exists(&output) {
                self.niri.queue_redraw(&output);
            }
        }
    }

    fn cursor_frame(&mut self, session: &CursorSessionRef, frame: Frame) {
        let Some(s) = self
            .niri
            .image_copy_cursor_sessions
            .iter_mut()
            .find(|s| s.session == *session)
        else {
            frame.fail(CaptureFailureReason::Unknown);
            return;
        };

        // Same as above.
        if s.pending_frame.is_some() {
            warn!("client created a second cursor frame while one was still in flight");
            frame.fail(CaptureFailureReason::Unknown);
            return;
        }
        s.pending_frame = Some(frame);

        // The frame is captured on the next redraw when the cursor image changes.
        if let Some(output) = source_session_output(&self.niri, &session.source()) {
            // The output may be gone already (the global lingers).
            if self.niri.output_exists(&output) {
                self.niri.queue_redraw(&output);
            }
        }
    }

    fn frame_aborted(&mut self, frame: FrameRef) {
        for s in &mut self.niri.image_copy_sessions {
            if s.pending_frame.as_deref() == Some(&frame) {
                s.pending_frame = None;
            }
        }
        for s in &mut self.niri.image_copy_cursor_sessions {
            if s.pending_frame.as_deref() == Some(&frame) {
                s.pending_frame = None;
            }
        }
    }

    fn session_destroyed(&mut self, session: SessionRef) {
        let sessions = &mut self.niri.image_copy_sessions;
        if let Some(idx) = sessions.iter().position(|s| s.session == session) {
            let s = sessions.remove(idx);
            if let Some(frame) = s.pending_frame {
                // Like wlroots.
                frame.fail(CaptureFailureReason::Stopped);
            }
        }
        self.niri.image_copy_capture_state.cleanup();
    }

    fn cursor_session_destroyed(&mut self, session: CursorSessionRef) {
        // FIXME: this doesn't get called by smithay when the underlying
        // ext_image_copy_capture_session_v1 is destroyed, only when the
        // ext_image_copy_capture_cursor_session_v1 is destroyed
        let sessions = &mut self.niri.image_copy_cursor_sessions;
        if let Some(idx) = sessions.iter().position(|s| s.session == session) {
            let s = sessions.remove(idx);
            if let Some(frame) = s.pending_frame {
                // Like wlroots.
                frame.fail(CaptureFailureReason::Stopped);
            }
        }
        self.niri.image_copy_capture_state.cleanup();
    }
}
