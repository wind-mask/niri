use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::renderer::gles::{GlesFrame, GlesRenderer, GlesTexture};
use smithay::backend::renderer::{
    Bind, ExportMem, Frame as _, ImportAll, ImportMem, Offscreen, Renderer, RendererSuper, Texture,
};

use crate::backend::tty::{TtyFrame, TtyRenderer};
use crate::render_helpers::superres::PassSource;

/// Trait with our main renderer requirements to save on the typing.
pub trait NiriRenderer:
    ImportAll
    + ImportMem
    + ExportMem
    + Bind<Dmabuf>
    + Offscreen<GlesTexture>
    + Renderer<TextureId = Self::NiriTextureId, Error = Self::NiriError>
    + AsGlesRenderer
{
    // Associated types to work around the instability of associated type bounds.
    type NiriTextureId: Texture + Clone + Send + 'static + PassSource;
    type NiriError: std::error::Error
        + Send
        + Sync
        + From<<GlesRenderer as RendererSuper>::Error>
        + 'static;
}

impl<R> NiriRenderer for R
where
    R: ImportAll + ImportMem + ExportMem + Bind<Dmabuf> + Offscreen<GlesTexture> + AsGlesRenderer,
    R::TextureId: Texture + Clone + Send + 'static + PassSource,
    R::Error:
        std::error::Error + Send + Sync + From<<GlesRenderer as RendererSuper>::Error> + 'static,
{
    type NiriTextureId = R::TextureId;
    type NiriError = R::Error;
}

/// Trait for getting the underlying `GlesRenderer`.
pub trait AsGlesRenderer {
    fn as_gles_renderer(&mut self) -> &mut GlesRenderer;
}

impl AsGlesRenderer for GlesRenderer {
    fn as_gles_renderer(&mut self) -> &mut GlesRenderer {
        self
    }
}

impl AsGlesRenderer for TtyRenderer<'_> {
    fn as_gles_renderer(&mut self) -> &mut GlesRenderer {
        self.as_mut()
    }
}

/// Trait for getting the underlying `GlesFrame`.
pub trait AsGlesFrame<'frame, 'buffer>
where
    Self: 'frame,
{
    fn as_gles_frame(&mut self) -> &mut GlesFrame<'frame, 'buffer>;
}

impl<'frame, 'buffer> AsGlesFrame<'frame, 'buffer> for GlesFrame<'frame, 'buffer> {
    fn as_gles_frame(&mut self) -> &mut GlesFrame<'frame, 'buffer> {
        self
    }
}

impl<'frame, 'buffer> AsGlesFrame<'frame, 'buffer> for TtyFrame<'_, 'frame, 'buffer> {
    fn as_gles_frame(&mut self) -> &mut GlesFrame<'frame, 'buffer> {
        self.as_mut()
    }
}

/// Opt-in GPU/buffer diagnostics, including in release builds where TRACE is compiled out.
///
/// The default log filter disables this target. Enable it with
/// `RUST_LOG=niri=info,niri::render_helpers::gpu=debug`.
pub(crate) const GPU_LOG_TARGET: &str = "niri::render_helpers::gpu";

/// Whether the GPU/buffer diagnostics are being recorded anywhere.
///
/// The expensive parts of the diagnostics check this instead of a log level: with per-layer
/// filters a level check is only a hint and stays true even when no layer records the target.
static GPU_DIAGNOSTICS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turns the recording of the GPU/buffer diagnostics on or off.
///
/// The compositor sets this when it is told where to write them.
pub fn set_gpu_diagnostics(enabled: bool) {
    GPU_DIAGNOSTICS.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Returns whether the GPU/buffer diagnostics are being recorded.
pub(crate) fn gpu_diagnostics() -> bool {
    GPU_DIAGNOSTICS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Identifies the GLES execution context without querying GL or switching contexts.
///
/// A ContextId is a texture-compatibility group, not a DRM device or a physical memory owner.
/// The device/driver labels come from the capabilities cached during renderer initialization.
pub(crate) struct GpuDebug<'a> {
    context_id: smithay::backend::renderer::ContextId<GlesTexture>,
    capabilities: Option<&'a crate::render_helpers::superres::Capabilities>,
}

impl<'a> GpuDebug<'a> {
    pub(crate) fn renderer(renderer: &'a GlesRenderer) -> Self {
        Self {
            context_id: renderer.context_id(),
            capabilities: renderer.egl_context().user_data().get(),
        }
    }

    pub(crate) fn frame(frame: &'a GlesFrame<'_, '_>) -> Self {
        Self {
            context_id: frame.context_id(),
            capabilities: frame.egl_context().user_data().get(),
        }
    }
}

impl std::fmt::Debug for GpuDebug<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("GlesExecution");
        debug.field("context_id", &self.context_id);
        if let Some(capabilities) = self.capabilities {
            debug
                .field("render_node", &capabilities.render_node)
                .field("vendor", &capabilities.vendor)
                .field("renderer", &capabilities.renderer)
                .field("software", &capabilities.is_software);
        } else {
            debug.field("device", &"unknown (context not initialized)");
        }
        debug.finish()
    }
}

/// A texture name only identifies an object together with its owning context.
/// This describes the GL object, not the driver's physical placement of its storage.
pub(crate) struct TextureDebug<'a>(pub(crate) &'a GlesTexture);

impl std::fmt::Debug for TextureDebug<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GlesTexture")
            .field("texture_id", &self.0.tex_id())
            .field("size", &self.0.size())
            .field("format", &self.0.format())
            .finish()
    }
}

/// Metadata of the original client allocation; never reads its pixel contents.
///
/// fstat identities correlate duplicated plane FDs, unlike the process-local FD number.
/// Neither these identities nor Dmabuf::node() identify the exporter GPU or memory residency.
pub(crate) struct DmabufDebug<'a>(pub(crate) &'a Dmabuf);

impl std::fmt::Debug for DmabufDebug<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::os::fd::AsRawFd as _;

        use smithay::backend::allocator::Buffer as _;
        use smithay::reexports::rustix::fs::fstat;

        let planes: Vec<_> = self
            .0
            .handles()
            .zip(self.0.strides())
            .zip(self.0.offsets())
            .map(|((fd, stride), offset)| {
                let identity = fstat(fd).map(|stat| (stat.st_dev, stat.st_ino));
                (fd.as_raw_fd(), identity, stride, offset)
            })
            .collect();
        f.debug_struct("Dmabuf")
            .field("size", &self.0.size())
            .field("format", &self.0.format())
            .field("flags", &self.0.flags())
            .field(
                "import_node_hint",
                &self.0.node().map(|node| node.to_string()),
            )
            .field("memory_owner", &"unknown")
            .field("planes_fd_dev_ino_stride_offset", &planes)
            .finish()
    }
}

pub(crate) struct ClientBufferDebug<'a>(
    pub(crate) &'a wayland_server::protocol::wl_buffer::WlBuffer,
);

impl std::fmt::Debug for ClientBufferDebug<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use smithay::backend::renderer::buffer_type;
        use smithay::wayland::dmabuf::get_dmabuf;
        use wayland_server::Resource as _;

        let mut debug = f.debug_struct("ClientBuffer");
        debug.field("buffer_id", &self.0.id());
        match get_dmabuf(self.0) {
            Ok(dmabuf) => debug.field("dmabuf", &DmabufDebug(dmabuf)),
            Err(_) => debug.field("kind", &buffer_type(self.0)),
        };
        debug.finish()
    }
}

/// Logs only the currently applied buffer, not a pending synchronized-subsurface attachment.
/// The diagnostic borrows the buffer and does not retain it or change its release/sync lifetime.
pub(crate) fn log_surface_buffer(
    surface: &wayland_server::protocol::wl_surface::WlSurface,
    phase: &'static str,
) {
    use smithay::backend::renderer::utils::RendererSurfaceStateUserData;
    use smithay::wayland::compositor::with_states;
    use wayland_server::Resource as _;

    if !gpu_diagnostics() {
        return;
    }
    with_states(surface, |states| {
        let Some(data) = states.data_map.get::<RendererSurfaceStateUserData>() else {
            return;
        };
        let data = data.lock().unwrap();
        debug!(
            target: GPU_LOG_TARGET,
            phase,
            surface = ?surface.id(),
            commit = ?data.current_commit(),
            buffer = ?data.buffer().map(|buffer| ClientBufferDebug(buffer)),
            "applied client buffer (import hint is not memory ownership)",
        );
    });
}

#[cfg(test)]
mod gpu_diagnostics_tests {
    use std::os::fd::AsFd as _;

    use smithay::backend::allocator::dmabuf::DmabufFlags;
    use smithay::backend::allocator::Fourcc;
    use smithay::backend::renderer::ContextId;
    use smithay::reexports::gbm::Modifier;

    use super::*;

    #[test]
    fn dmabuf_metadata_correlates_duplicate_fds_without_claiming_ownership() {
        let file = std::fs::File::open("/dev/null").unwrap();
        let mut builder = Dmabuf::builder(
            (16, 8),
            Fourcc::Abgr8888,
            Modifier::Linear,
            DmabufFlags::empty(),
        );
        assert!(builder.add_plane(file.as_fd().try_clone_to_owned().unwrap(), 0, 64));
        assert!(builder.add_plane(file.as_fd().try_clone_to_owned().unwrap(), 512, 64));
        let dmabuf = builder.build().unwrap();

        let text = format!("{:?}", DmabufDebug(&dmabuf));
        assert!(text.contains(&format!("{:?}", Fourcc::Abgr8888)), "{text}");
        assert!(text.contains(&format!("{:?}", Modifier::Linear)), "{text}");
        assert!(text.contains("import_node_hint: None"));
        assert!(text.contains("memory_owner: \"unknown\""));
        let identities: Vec<_> = dmabuf
            .handles()
            .map(|fd| {
                let stat = smithay::reexports::rustix::fs::fstat(fd).unwrap();
                (stat.st_dev, stat.st_ino)
            })
            .collect();
        assert_eq!(identities[0], identities[1]);
        assert_eq!(text.matches(&format!("Ok({:?})", identities[0])).count(), 2);
        assert_eq!(dmabuf.node(), None);
        assert_eq!(dmabuf.strides().collect::<Vec<_>>(), vec![64, 64]);
    }

    #[test]
    fn uninitialized_context_is_unknown_not_the_primary_gpu() {
        let gpu = GpuDebug {
            context_id: ContextId::new(),
            capabilities: None,
        };
        let text = format!("{gpu:?}");
        assert!(text.contains("context_id"));
        assert!(text.contains("unknown (context not initialized)"));
        assert!(!text.contains("render_node"));
    }
}
