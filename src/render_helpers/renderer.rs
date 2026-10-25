use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::renderer::gles::{GlesFrame, GlesRenderer, GlesTexture};
#[allow(unused_imports)]
use smithay::backend::renderer::Frame;
use smithay::backend::renderer::{
    Bind, ExportMem, ImportAll, ImportMem, Offscreen, Renderer, Texture,
};

use crate::backend::tty::{TtyFrame, TtyRenderer};
use crate::backend::tty_renderer::TtyOffscreen;
use crate::render_helpers::superres::PassSource;

/// Trait with our main renderer requirements to save on the typing.
pub trait NiriRenderer:
    ImportAll
    + ImportMem
    + ExportMem
    + Bind<Dmabuf>
    + Renderer<TextureId = Self::NiriTextureId, Error = Self::NiriError>
    + AsGlesRenderer
    + AsVulkanRenderer
    + HasOffscreen
    + Offscreen<<Self as HasOffscreen>::Offscreen>
    + Bind<<Self as HasOffscreen>::Offscreen>
    + crate::render_helpers::blend::CaptureBlend
{
    // Associated types to work around the instability of associated type bounds.
    type NiriTextureId: Texture + Clone + Send + 'static + PassSource;
    type NiriError: std::error::Error + Send + Sync + 'static;
}

impl<R> NiriRenderer for R
where
    R: ImportAll
        + ImportMem
        + ExportMem
        + Bind<Dmabuf>
        + AsGlesRenderer
        + AsVulkanRenderer
        + HasOffscreen
        + Offscreen<<R as HasOffscreen>::Offscreen>
        + Bind<<R as HasOffscreen>::Offscreen>
        + crate::render_helpers::blend::CaptureBlend,
    R::TextureId: Texture + Clone + Send + 'static + PassSource,
    R::Error: std::error::Error + Send + Sync + 'static,
{
    type NiriTextureId = R::TextureId;
    type NiriError = R::Error;
}

/// Trait for getting the underlying `GlesRenderer`, if any.
///
/// Returns `None` for the Vulkan renderer; GLES-specific functionality (custom shaders,
/// offscreen effects) degrades gracefully in that case.
pub trait AsGlesRenderer {
    fn as_gles_renderer(&mut self) -> Option<&mut GlesRenderer>;
}

impl AsGlesRenderer for GlesRenderer {
    fn as_gles_renderer(&mut self) -> Option<&mut GlesRenderer> {
        Some(self)
    }
}

impl AsGlesRenderer for TtyRenderer<'_> {
    fn as_gles_renderer(&mut self) -> Option<&mut GlesRenderer> {
        match self {
            TtyRenderer::Gles(renderer) => Some(renderer.as_mut()),
            TtyRenderer::Vulkan(_) => None,
        }
    }
}

impl AsGlesRenderer for smithay::backend::renderer::vulkan::VulkanRenderer {
    fn as_gles_renderer(&mut self) -> Option<&mut GlesRenderer> {
        None
    }
}

/// Trait for getting the underlying `GlesFrame`, if any.
pub trait AsGlesFrame<'frame, 'buffer>
where
    Self: 'frame,
{
    fn as_gles_frame(&mut self) -> Option<&mut GlesFrame<'frame, 'buffer>>;
}

impl<'frame, 'buffer> AsGlesFrame<'frame, 'buffer> for GlesFrame<'frame, 'buffer> {
    fn as_gles_frame(&mut self) -> Option<&mut GlesFrame<'frame, 'buffer>> {
        Some(self)
    }
}

impl<'frame, 'buffer> AsGlesFrame<'frame, 'buffer> for TtyFrame<'_, 'frame, 'buffer> {
    fn as_gles_frame(&mut self) -> Option<&mut GlesFrame<'frame, 'buffer>> {
        match self {
            TtyFrame::Gles(frame) => Some(frame.as_mut()),
            TtyFrame::Vulkan(_) => None,
        }
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
///
/// A window that redraws at the display's rate commits the same buffer state over and over, and
/// this runs for every commit, so only a few states per surface are written at DEBUG; the repeats
/// and everything after them go to TRACE.
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
        let buffer = data.buffer().map(|buffer| &**buffer);
        let first =
            note_buffer_state_logged(surface_key(surface), buffer_state_hash(phase, buffer));

        macro_rules! log_buffer {
            ($level:ident) => {
                $level!(
                    target: GPU_LOG_TARGET,
                    phase,
                    surface = ?surface.id(),
                    commit = ?data.current_commit(),
                    buffer = ?buffer.map(ClientBufferDebug),
                    "applied client buffer (import hint is not memory ownership)",
                );
            };
        }
        if first {
            log_buffer!(debug);
        } else {
            log_buffer!(trace);
        }
    });
}

/// How many buffer states of one surface are worth a DEBUG entry before the rest go to TRACE.
///
/// A window that allocates a fresh buffer for every frame would otherwise write an entry just as
/// often as it commits; the count keeps what one surface can add to the file bounded, and the first
/// states are the ones that say how the client's buffers are being imported.
const LOGGED_BUFFER_STATES_PER_SURFACE: u16 = 4;

/// Returns an identity for a surface that the state table below can keep.
///
/// The handle itself cannot be named here -- Wayland keeps its own type for it -- so it is hashed
/// into one number.
fn surface_key(surface: &wayland_server::protocol::wl_surface::WlSurface) -> u64 {
    use std::hash::{Hash as _, Hasher as _};

    use wayland_server::Resource as _;

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    surface.id().hash(&mut hasher);
    hasher.finish()
}

/// What was already written about one surface's buffers: how many states, and the last one.
type LoggedBufferStates = std::collections::HashMap<u64, (u16, Option<u64>)>;

/// Notes a buffer state of a surface, and returns whether it is worth writing at DEBUG.
///
/// Only a state that differs from the last one written counts, so a window that redraws the same
/// buffer does not repeat the entry. The map has one entry per surface used in a run.
fn note_buffer_state_logged(surface: u64, state: u64) -> bool {
    use std::sync::{Mutex, OnceLock};

    static STATES: OnceLock<Mutex<LoggedBufferStates>> = OnceLock::new();

    let states = STATES.get_or_init(|| Mutex::new(LoggedBufferStates::new()));
    let mut states = states.lock().unwrap();
    let entry = states.entry(surface).or_insert((0, None));

    if entry.1 == Some(state) || entry.0 >= LOGGED_BUFFER_STATES_PER_SURFACE {
        return false;
    }

    entry.0 += 1;
    entry.1 = Some(state);
    true
}

/// Returns what a client's buffer is, in a form that does not change between commits of the same
/// buffer.
///
/// The process-local descriptor numbers are left out: the fstat identity of the planes names the
/// same allocation without changing when a client sends a descriptor again.
fn buffer_state_hash(
    phase: &'static str,
    buffer: Option<&wayland_server::protocol::wl_buffer::WlBuffer>,
) -> u64 {
    use std::hash::{Hash as _, Hasher as _};

    use smithay::backend::allocator::Buffer as _;
    use smithay::reexports::rustix::fs::fstat;
    use smithay::wayland::dmabuf::get_dmabuf;

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    phase.hash(&mut hasher);

    let Some(buffer) = buffer else {
        0u8.hash(&mut hasher);
        return hasher.finish();
    };

    match get_dmabuf(buffer) {
        Ok(dmabuf) => {
            1u8.hash(&mut hasher);
            let size = dmabuf.size();
            (size.w, size.h).hash(&mut hasher);
            format!("{:?}", dmabuf.format()).hash(&mut hasher);
            format!("{:?}", dmabuf.flags()).hash(&mut hasher);
            dmabuf.node().hash(&mut hasher);
            for ((fd, stride), offset) in
                dmabuf.handles().zip(dmabuf.strides()).zip(dmabuf.offsets())
            {
                match fstat(fd) {
                    Ok(stat) => (stat.st_dev, stat.st_ino).hash(&mut hasher),
                    Err(_) => (0u64, 0u64).hash(&mut hasher),
                }
                stride.hash(&mut hasher);
                offset.hash(&mut hasher);
            }
        }
        Err(_) => {
            2u8.hash(&mut hasher);
            format!("{:?}", smithay::backend::renderer::buffer_type(buffer)).hash(&mut hasher);
        }
    }

    hasher.finish()
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

    #[test]
    fn a_surface_writes_a_few_buffer_states_and_then_leaves_the_rest_to_trace() {
        // A key no other test uses, since the table lives for the whole test binary.
        let surface = 0x9f2c_0000_0000_0001u64;

        let logged: Vec<_> = (0..LOGGED_BUFFER_STATES_PER_SURFACE as u64 + 3)
            .map(|state| note_buffer_state_logged(surface, state))
            .collect();
        let expected: Vec<_> = (0..LOGGED_BUFFER_STATES_PER_SURFACE as u64 + 3)
            .map(|state| state < LOGGED_BUFFER_STATES_PER_SURFACE as u64)
            .collect();
        assert_eq!(logged, expected);

        // The same state as the last one written never adds an entry, even early on.
        let other = 0x9f2c_0000_0000_0002u64;
        assert!(note_buffer_state_logged(other, 7));
        assert!(!note_buffer_state_logged(other, 7));
        assert!(note_buffer_state_logged(other, 8));
    }
}

/// Trait for getting the underlying `VulkanRenderer`, if any.
pub trait AsVulkanRenderer {
    fn as_vulkan_renderer(
        &mut self,
    ) -> Option<&mut smithay::backend::renderer::vulkan::VulkanRenderer>;
}

impl AsVulkanRenderer for GlesRenderer {
    fn as_vulkan_renderer(
        &mut self,
    ) -> Option<&mut smithay::backend::renderer::vulkan::VulkanRenderer> {
        None
    }
}

impl AsVulkanRenderer for TtyRenderer<'_> {
    fn as_vulkan_renderer(
        &mut self,
    ) -> Option<&mut smithay::backend::renderer::vulkan::VulkanRenderer> {
        match self {
            TtyRenderer::Gles(_) => None,
            TtyRenderer::Vulkan(renderer) => Some(renderer.as_mut()),
        }
    }
}

impl AsVulkanRenderer for smithay::backend::renderer::vulkan::VulkanRenderer {
    fn as_vulkan_renderer(
        &mut self,
    ) -> Option<&mut smithay::backend::renderer::vulkan::VulkanRenderer> {
        Some(self)
    }
}

/// The offscreen render target texture type of a renderer.
pub trait HasOffscreen: Renderer {
    type Offscreen: Texture + Clone + Send + 'static;

    /// Wraps the renderer's offscreen texture into the renderer-agnostic enum.
    fn wrap_offscreen(texture: Self::Offscreen) -> TtyOffscreen;
    /// Unwraps the renderer-agnostic enum back into this renderer's texture type.
    fn unwrap_offscreen(texture: &mut TtyOffscreen) -> Option<&mut Self::Offscreen>;
    /// Wraps the renderer's surface texture into the renderer-agnostic enum.
    fn wrap_texture(texture: Self::TextureId) -> TtyOffscreen;
}

impl HasOffscreen for GlesRenderer {
    type Offscreen = GlesTexture;

    fn wrap_offscreen(texture: GlesTexture) -> TtyOffscreen {
        TtyOffscreen::Gles(texture)
    }

    fn unwrap_offscreen(texture: &mut TtyOffscreen) -> Option<&mut GlesTexture> {
        match texture {
            TtyOffscreen::Gles(texture) => Some(texture),
            _ => None,
        }
    }

    fn wrap_texture(texture: GlesTexture) -> TtyOffscreen {
        TtyOffscreen::Gles(texture)
    }
}

impl HasOffscreen for TtyRenderer<'_> {
    type Offscreen = crate::backend::tty_renderer::TtyOffscreen;

    fn wrap_offscreen(texture: TtyOffscreen) -> TtyOffscreen {
        texture
    }

    fn unwrap_offscreen(texture: &mut TtyOffscreen) -> Option<&mut TtyOffscreen> {
        Some(texture)
    }

    fn wrap_texture(texture: Self::TextureId) -> TtyOffscreen {
        TtyOffscreen::Multi(texture)
    }
}

impl HasOffscreen for smithay::backend::renderer::vulkan::VulkanRenderer {
    type Offscreen = smithay::backend::renderer::vulkan::VulkanTexture;

    fn wrap_offscreen(texture: Self::Offscreen) -> TtyOffscreen {
        TtyOffscreen::Vulkan(texture)
    }

    fn unwrap_offscreen(texture: &mut TtyOffscreen) -> Option<&mut Self::Offscreen> {
        match texture {
            TtyOffscreen::Vulkan(texture) => Some(texture),
            _ => None,
        }
    }

    fn wrap_texture(texture: Self::TextureId) -> TtyOffscreen {
        TtyOffscreen::Vulkan(texture)
    }
}

/// Renderer bounds needed by the generic capture helpers (screenshots, screencopy,
/// screencasting): offscreen render targets of the renderer's own texture type plus the
/// SDR-capture blend control.
pub trait NiriCaptureRenderer:
    NiriRenderer
    + HasOffscreen
    + Offscreen<<Self as HasOffscreen>::Offscreen>
    + Bind<<Self as HasOffscreen>::Offscreen>
    + crate::render_helpers::blend::CaptureBlend
{
}

impl<R> NiriCaptureRenderer for R where
    R: NiriRenderer
        + HasOffscreen
        + Offscreen<<R as HasOffscreen>::Offscreen>
        + Bind<<R as HasOffscreen>::Offscreen>
        + crate::render_helpers::blend::CaptureBlend
{
}
