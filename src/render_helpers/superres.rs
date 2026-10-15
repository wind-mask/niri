//! Window-level super resolution.
//!
//! This is the first stage of the super resolution work: it probes what the current GLES context
//! can do, and it replaces the sampling that the compositor uses to magnify a client buffer that
//! is smaller than the area it is displayed in.
//!
//! The resampling happens in the final draw instead of in a separate pass, so there is nothing to
//! cache or synchronize: the element that draws the window samples the client buffer with our
//! kernel. Everything else about the window (display geometry, input, frame callbacks, alpha)
//! stays as it is.
//!
//! Super resolution must actually run, so the element that draws the resampled buffer never
//! exposes the client buffer as scanout storage. See [`SuperResElement::underlying_storage`].
//! Enabling the effect for a window therefore rules out direct scanout for it, even when the
//! client buffer happens to match its display area.
//!
//! Everything here is off unless a window rule enables it. If the programs cannot be compiled
//! (unsupported context, compile error) or a surface has nothing to resample, the element falls
//! back to the renderer's own sampling instead of failing the frame.

use std::cell::Cell;
use std::ffi::CStr;
use std::fmt::Write as _;
use std::path::Path;

use niri_config::{Upscaling, UpscalingAlgorithm, UpscalingMode};
use smithay::backend::renderer::element::surface::{
    WaylandSurfaceRenderElement, WaylandSurfaceTexture,
};
use smithay::backend::renderer::element::{Element, Id, Kind, RenderElement, UnderlyingStorage};
use smithay::backend::renderer::gles::{
    ffi, Capability, GlesError, GlesFrame, GlesRenderer, GlesTexProgram, Uniform, UniformName,
    UniformType,
};
use smithay::backend::renderer::utils::{CommitCounter, DamageSet, OpaqueRegions};
use smithay::backend::renderer::{buffer_y_inverted, ImportAll, Renderer, Texture};
use smithay::utils::user_data::UserDataMap;
use smithay::utils::{Buffer, Physical, Rectangle, Scale, Transform};

use crate::backend::tty::{TtyFrame, TtyRenderer, TtyRendererError};
use crate::render_helpers::renderer::AsGlesFrame as _;

/// Resampling kernel for a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleFilter {
    /// Explicit bilinear sampling.
    ///
    /// This reproduces GL_LINEAR, including for the identity case (where the buffer and the
    /// display area are the same size).
    Bilinear,

    /// Catmull-Rom bicubic sampling.
    Bicubic,
}

impl ResampleFilter {
    /// Returns the kernel to use for an algorithm, and whether it fell back to a baseline.
    ///
    /// Algorithms that are not implemented yet fall back to the best baseline kernel rather than
    /// drawing something that we cannot explain.
    fn for_algorithm(algorithm: UpscalingAlgorithm) -> (Self, bool) {
        match algorithm {
            UpscalingAlgorithm::Bilinear => (Self::Bilinear, false),
            UpscalingAlgorithm::Bicubic => (Self::Bicubic, false),
            UpscalingAlgorithm::Fsr1 | UpscalingAlgorithm::Cunny2 | UpscalingAlgorithm::Anime4k => {
                (Self::Bicubic, true)
            }
        }
    }
}

/// GLES capabilities that decide what super resolution can do.
///
/// This is a static probe of the context that niri actually got, which is not necessarily what
/// niri asked for: the TTY backend requests a client version of 2 and the nested backend requests
/// 3.0, and the driver decides what to give us. Everything that is version-gated (ES 3.0
/// instancing, MRT and texture LOD, ES 3.1 compute and gather) must be checked here before it is
/// used, since calling a GL entry point that the context does not have aborts the process.
///
/// Note that this only reports what the driver advertises. Whether a float render target is
/// actually usable still has to be verified by an actual framebuffer completeness check once we
/// allocate one.
#[derive(Debug, Clone)]
pub struct Capabilities {
    pub vendor: String,
    pub renderer: String,
    pub version: String,
    pub glsl_version: String,
    /// Parsed `GL_VERSION`, if it looked like a GLES version string.
    pub es_version: Option<(i32, i32)>,
    /// Parsed `GL_SHADING_LANGUAGE_VERSION`, if it looked like a GLSL ES version string.
    pub glsl_es_version: Option<(i32, i32)>,
    /// DRM render node of the GPU this renderer belongs to, if known.
    pub render_node: Option<String>,
    pub is_software: bool,
    pub max_texture_size: i32,
    pub max_renderbuffer_size: i32,
    pub max_draw_buffers: i32,
    pub max_color_attachments: i32,
    pub max_compute_work_group_count: [i32; 3],
    pub max_compute_work_group_invocations: i32,
    pub extensions: Vec<String>,
    pub instancing: bool,
    pub blit: bool,
    pub texture_gather: bool,
    pub texture_lod: bool,
    pub compute_shaders: bool,
    pub texture_float_linear: bool,
    pub texture_half_float_linear: bool,
    pub color_buffer_float: bool,
    pub color_buffer_half_float: bool,
    pub timer_query: bool,
}

impl Capabilities {
    fn probe(renderer: &mut GlesRenderer, render_node: Option<&Path>) -> Option<Self> {
        let capabilities = renderer.capabilities().to_vec();
        let is_software = renderer.is_software();

        let result = renderer
            .with_context(|gl| unsafe { Self::probe_gl(gl, render_node) })
            .ok()?;

        Some(Self {
            is_software,
            instancing: capabilities.contains(&Capability::Instancing),
            blit: capabilities.contains(&Capability::Blit),
            ..result
        })
    }

    /// # Safety
    ///
    /// Must be called with a current GLES context that is safe to query.
    unsafe fn probe_gl(gl: &ffi::Gles2, render_node: Option<&Path>) -> Self {
        let version = gl_string(gl, ffi::VERSION);
        let es_version = parse_es_version(&version);
        let glsl_version = gl_string(gl, ffi::SHADING_LANGUAGE_VERSION);
        let glsl_es_version = parse_glsl_version(&glsl_version);

        let es3 = es_version.is_some_and(|(major, _)| major >= 3);
        let es31 = es_version.is_some_and(|(major, minor)| major > 3 || (major == 3 && minor >= 1));
        let es32 = es_version.is_some_and(|(major, minor)| major > 3 || (major == 3 && minor >= 2));

        let extensions: Vec<String> = if es3 {
            let mut count = 0;
            gl.GetIntegerv(ffi::NUM_EXTENSIONS, &mut count as *mut _);
            (0..count.max(0))
                .map(|index| gl_string_i(gl, ffi::EXTENSIONS, index as u32))
                .filter(|extension| !extension.is_empty())
                .collect()
        } else {
            gl_string(gl, ffi::EXTENSIONS)
                .split_whitespace()
                .map(String::from)
                .collect()
        };

        let has_extension = |name: &str| extensions.iter().any(|extension| extension == name);

        // ES 3.0 and 3.1 queries. Calling these on an older context gives GL_INVALID_ENUM and
        // leaves the value alone, but the entry points may also be missing entirely, so gate on
        // the version.
        let max_draw_buffers = if es3 {
            gl_int(gl, ffi::MAX_DRAW_BUFFERS)
        } else {
            0
        };
        let max_color_attachments = if es3 {
            gl_int(gl, ffi::MAX_COLOR_ATTACHMENTS)
        } else {
            0
        };

        let max_compute_work_group_count = if es31 {
            let mut value = [0; 3];
            for (index, value) in value.iter_mut().enumerate() {
                gl.GetIntegeri_v(
                    ffi::MAX_COMPUTE_WORK_GROUP_COUNT,
                    index as u32,
                    value as *mut _,
                );
            }
            value
        } else {
            [0; 3]
        };

        let max_compute_work_group_invocations = if es31 {
            gl_int(gl, ffi::MAX_COMPUTE_WORK_GROUP_INVOCATIONS)
        } else {
            0
        };

        Self {
            vendor: gl_string(gl, ffi::VENDOR),
            renderer: gl_string(gl, ffi::RENDERER),
            version,
            glsl_version,
            es_version,
            glsl_es_version,
            render_node: render_node.map(|path| path.display().to_string()),
            // Filled in by the caller from the renderer.
            is_software: false,
            max_texture_size: gl_int(gl, ffi::MAX_TEXTURE_SIZE),
            max_renderbuffer_size: gl_int(gl, ffi::MAX_RENDERBUFFER_SIZE),
            max_draw_buffers,
            max_color_attachments,
            max_compute_work_group_count,
            max_compute_work_group_invocations,
            instancing: false,
            blit: false,
            texture_gather: es31 || has_extension("GL_ARB_texture_gather"),
            texture_lod: es3 || has_extension("GL_EXT_shader_texture_lod"),
            compute_shaders: es31,
            texture_float_linear: es3 || has_extension("GL_OES_texture_float_linear"),
            texture_half_float_linear: es3 || has_extension("GL_OES_texture_half_float_linear"),
            color_buffer_float: es32 || has_extension("GL_EXT_color_buffer_float"),
            color_buffer_half_float: es3 || has_extension("GL_EXT_color_buffer_half_float"),
            timer_query: has_extension("GL_EXT_disjoint_timer_query")
                || has_extension("GL_EXT_disjoint_timer_query_ext"),
            extensions,
        }
    }

    /// Returns whether the context is at least OpenGL ES 3.0.
    pub fn es3(&self) -> bool {
        self.es_version.is_some_and(|(major, _)| major >= 3)
    }

    /// Returns whether compute shaders (OpenGL ES 3.1) are available.
    pub fn compute(&self) -> bool {
        self.compute_shaders
    }

    /// Returns whether at least `count` simultaneous render targets are available.
    pub fn mrt(&self, count: i32) -> bool {
        self.max_draw_buffers >= count && self.max_color_attachments >= count
    }

    /// Returns a single-line summary for logging.
    pub fn summary(&self) -> String {
        let mut s = String::new();

        let es_version = match self.es_version {
            Some((major, minor)) => format!("ES {major}.{minor}"),
            None => "unknown version".to_owned(),
        };

        let glsl = match self.glsl_es_version {
            Some((major, minor)) => format!("ES {major}.{minor:02}"),
            None => self.glsl_version.clone(),
        };

        let _ = write!(
            s,
            "{es_version} ({vendor}, {renderer}), GLSL {glsl}, MRT {draw}/{attach}, \
             max texture {size}, gather {gather}, compute {compute}, float RT {float_rt}, \
             half-float RT {half_rt}, timer query {timer}, software {software}",
            vendor = self.vendor,
            renderer = self.renderer,
            glsl = glsl,
            draw = self.max_draw_buffers,
            attach = self.max_color_attachments,
            size = self.max_texture_size,
            gather = yes_no(self.texture_gather),
            compute = yes_no(self.compute_shaders),
            float_rt = yes_no(self.color_buffer_float),
            half_rt = yes_no(self.color_buffer_half_float),
            timer = yes_no(self.timer_query),
            software = yes_no(self.is_software),
        );

        match &self.render_node {
            Some(node) => {
                let _ = write!(s, ", render node {node}");
            }
            None => {
                let _ = write!(s, ", render node unknown");
            }
        }

        s
    }
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

/// Returns a GL string, or an empty string if the context does not provide one.
fn gl_string(gl: &ffi::Gles2, name: ffi::types::GLenum) -> String {
    let value = unsafe { gl.GetString(name) };
    if value.is_null() {
        return String::new();
    }

    unsafe { CStr::from_ptr(value as *const _) }
        .to_string_lossy()
        .into_owned()
}

/// Returns an indexed GL string, or an empty string if the index is out of range.
fn gl_string_i(gl: &ffi::Gles2, name: ffi::types::GLenum, index: u32) -> String {
    let value = unsafe { gl.GetStringi(name, index) };
    if value.is_null() {
        return String::new();
    }

    unsafe { CStr::from_ptr(value as *const _) }
        .to_string_lossy()
        .into_owned()
}

/// Returns an integer GL parameter, or 0 if the query fails.
fn gl_int(gl: &ffi::Gles2, name: ffi::types::GLenum) -> i32 {
    let mut value = 0;
    unsafe { gl.GetIntegerv(name, &mut value as *mut _) };
    value
}

/// Parses a `GL_VERSION` string, e.g. `OpenGL ES 3.2 Mesa 25.0.0`.
fn parse_es_version(version: &str) -> Option<(i32, i32)> {
    let version = version.trim().strip_prefix("OpenGL ES")?.trim_start();
    // Some drivers report variants like "OpenGL ES-CM 1.1".
    let version = version
        .trim_start_matches('-')
        .trim_start_matches(|c: char| c.is_ascii_alphabetic())
        .trim_start();

    let mut parts = version.split(['.', ' ']);
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

/// Parses a `GL_SHADING_LANGUAGE_VERSION` string, e.g. `OpenGL ES GLSL ES 3.20`.
fn parse_glsl_version(version: &str) -> Option<(i32, i32)> {
    const PREFIX: &str = "GLSL ES";

    let index = version.find(PREFIX)? + PREFIX.len();
    let version = version[index..].trim_start();

    let mut parts = version.split(['.', ' ']);
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

/// The resampling programs, one per kernel.
///
/// These are compiled once per renderer context and stored in the EGL context user data.
#[derive(Debug)]
pub struct ResamplePrograms {
    bilinear: Option<GlesTexProgram>,
    bicubic: Option<GlesTexProgram>,
}

impl ResamplePrograms {
    fn compile(renderer: &mut GlesRenderer) -> Self {
        Self {
            bilinear: Self::compile_filter(renderer, ResampleFilter::Bilinear),
            bicubic: Self::compile_filter(renderer, ResampleFilter::Bicubic),
        }
    }

    fn compile_filter(
        renderer: &mut GlesRenderer,
        filter: ResampleFilter,
    ) -> Option<GlesTexProgram> {
        let uniforms = [
            UniformName::new("niri_tex_size", UniformType::_2f),
            UniformName::new("niri_src_rect", UniformType::_4f),
        ];

        let index = match filter {
            ResampleFilter::Bilinear => 0,
            ResampleFilter::Bicubic => 1,
        };

        let source = include_str!("shaders/superres/resample.frag").replace(
            "// niri:resample-filter",
            &format!("#define NIRI_RESAMPLE_FILTER {index}"),
        );

        match renderer.compile_custom_texture_shader(source, &uniforms) {
            Ok(program) => Some(program),
            Err(err) => {
                warn!("error compiling {filter:?} resampling shader: {err:?}");
                None
            }
        }
    }

    fn get_from_frame<'a>(frame: &'a GlesFrame<'_, '_>) -> Option<&'a Self> {
        frame.egl_context().user_data().get()
    }

    fn program(&self, filter: ResampleFilter) -> Option<GlesTexProgram> {
        match filter {
            ResampleFilter::Bilinear => self.bilinear.clone(),
            ResampleFilter::Bicubic => self.bicubic.clone(),
        }
    }
}

/// Probes super resolution capabilities and compiles the resampling programs.
///
/// Must be called once per renderer context, next to `shaders::init()`.
pub fn init(renderer: &mut GlesRenderer, render_node: Option<&Path>) {
    let Some(capabilities) = Capabilities::probe(renderer, render_node) else {
        warn!("error probing super resolution capabilities");
        return;
    };

    info!("super resolution: {}", capabilities.summary());
    debug!("super resolution capabilities: {capabilities:#?}");

    let programs = ResamplePrograms::compile(renderer);

    let data = renderer.egl_context().user_data();
    if !data.insert_if_missing(|| capabilities) {
        error!("super resolution capabilities were already probed");
    }
    if !data.insert_if_missing(|| programs) {
        error!("super resolution programs were already compiled");
    }
}

/// Returns the capabilities probed for this renderer context, if any.
///
/// This is meant for tests and diagnostics; rendering code uses the programs directly.
pub fn capabilities(renderer: &mut GlesRenderer) -> Option<&Capabilities> {
    renderer.egl_context().user_data().get()
}

/// Renders a client surface with a resampling kernel.
///
/// This wraps the surface element that the compositor would draw anyway and only changes the
/// program that samples the texture, so the surface's viewport, transform, alpha, damage, commit
/// counter and frame callback handling keep working.
#[derive(Debug)]
pub struct SuperResElement<R: Renderer> {
    inner: WaylandSurfaceRenderElement<R>,
    filter: ResampleFilter,
}

impl<R: Renderer> SuperResElement<R> {
    pub fn new(inner: WaylandSurfaceRenderElement<R>, filter: ResampleFilter) -> Self {
        Self { inner, filter }
    }
}

/// Returns the resampling uniforms for a source rect of a texture.
///
/// `src` is in buffer texels and is what the renderer will use to build the texture coordinates,
/// so the shader can use it to clamp its taps the same way.
///
/// `y_inverted` must say whether the renderer flips the texture coordinates for this buffer, so
/// that the shader's texel positions stay consistent with them.
fn resample_uniforms<T: Texture>(
    texture: &T,
    src: Rectangle<f64, Buffer>,
    y_inverted: bool,
) -> Option<[Uniform<'static>; 2]> {
    let tex_size = texture.size();
    if tex_size.w <= 0 || tex_size.h <= 0 {
        return None;
    }

    let src = if y_inverted {
        Rectangle::new(
            (src.loc.x, f64::from(tex_size.h) - src.loc.y - src.size.h).into(),
            src.size,
        )
    } else {
        src
    };

    Some([
        Uniform::new("niri_tex_size", (tex_size.w as f32, tex_size.h as f32)),
        Uniform::new(
            "niri_src_rect",
            [
                src.loc.x as f32,
                src.loc.y as f32,
                src.size.w as f32,
                src.size.h as f32,
            ],
        ),
    ])
}

impl<R: Renderer + ImportAll> SuperResElement<R>
where
    R::TextureId: Texture,
{
    /// Returns the resampling uniforms for a source rect.
    fn resample_uniforms(&self, src: Rectangle<f64, Buffer>) -> Option<[Uniform<'static>; 2]> {
        let WaylandSurfaceTexture::Texture(texture) = self.inner.texture() else {
            return None;
        };

        let y_inverted = buffer_y_inverted(self.inner.buffer()).unwrap_or(false);
        resample_uniforms(texture, src, y_inverted)
    }

    /// Returns the resampling program and its uniforms.
    ///
    /// This returns `None` when there is nothing to resample with, in which case the caller must
    /// fall back to the renderer's own sampling. That keeps the window working when the programs
    /// were not compiled (unsupported context, compile error) or when the surface has no texture
    /// (single-pixel buffers).
    fn resample_program(
        &self,
        frame: &GlesFrame<'_, '_>,
        src: Rectangle<f64, Buffer>,
    ) -> Option<(GlesTexProgram, Vec<Uniform<'static>>)> {
        let program = ResamplePrograms::get_from_frame(frame)?.program(self.filter)?;
        let uniforms = self.resample_uniforms(src)?;
        Some((program, uniforms.to_vec()))
    }
}

impl<R: Renderer + ImportAll> Element for SuperResElement<R> {
    fn id(&self) -> &Id {
        self.inner.id()
    }

    fn current_commit(&self) -> CommitCounter {
        self.inner.current_commit()
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        self.inner.geometry(scale)
    }

    fn transform(&self) -> Transform {
        self.inner.transform()
    }

    fn src(&self) -> Rectangle<f64, Buffer> {
        self.inner.src()
    }

    fn damage_since(
        &self,
        scale: Scale<f64>,
        commit: Option<CommitCounter>,
    ) -> DamageSet<i32, Physical> {
        self.inner.damage_since(scale, commit)
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        self.inner.opaque_regions(scale)
    }

    fn alpha(&self) -> f32 {
        self.inner.alpha()
    }

    fn kind(&self) -> Kind {
        self.inner.kind()
    }
}

impl RenderElement<GlesRenderer> for SuperResElement<GlesRenderer> {
    fn draw(
        &self,
        frame: &mut GlesFrame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        cache: Option<&UserDataMap>,
    ) -> Result<(), GlesError> {
        let Some((program, uniforms)) = self.resample_program(frame, src) else {
            // Fall back to the renderer's own sampling.
            return RenderElement::<GlesRenderer>::draw(
                &self.inner,
                frame,
                src,
                dst,
                damage,
                opaque_regions,
                cache,
            );
        };

        frame.override_default_tex_program(program, uniforms);
        let res = RenderElement::<GlesRenderer>::draw(
            &self.inner,
            frame,
            src,
            dst,
            damage,
            opaque_regions,
            cache,
        );
        frame.clear_tex_program_override();
        res
    }

    fn underlying_storage(&self, _renderer: &mut GlesRenderer) -> Option<UnderlyingStorage<'_>> {
        // The client buffer is resampled, so it is never what ends up on the screen. Handing it
        // out as scanout storage would draw the unresampled buffer behind our back.
        None
    }
}

impl<'render> RenderElement<TtyRenderer<'render>> for SuperResElement<TtyRenderer<'render>> {
    fn draw(
        &self,
        frame: &mut TtyFrame<'render, '_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        cache: Option<&UserDataMap>,
    ) -> Result<(), TtyRendererError<'render>> {
        let program = self.resample_program(frame.as_gles_frame(), src);

        let Some((program, uniforms)) = program else {
            // Fall back to the renderer's own sampling.
            return RenderElement::<TtyRenderer<'render>>::draw(
                &self.inner,
                frame,
                src,
                dst,
                damage,
                opaque_regions,
                cache,
            );
        };

        frame
            .as_gles_frame()
            .override_default_tex_program(program, uniforms);
        let res = RenderElement::<TtyRenderer<'render>>::draw(
            &self.inner,
            frame,
            src,
            dst,
            damage,
            opaque_regions,
            cache,
        );
        frame.as_gles_frame().clear_tex_program_override();
        res
    }

    fn underlying_storage(
        &self,
        _renderer: &mut TtyRenderer<'render>,
    ) -> Option<UnderlyingStorage<'_>> {
        None
    }
}

/// Per-window super resolution state.
///
/// The first stage resamples during the final draw, so there is no cached result texture and no
/// pass graph to keep here yet. What this holds is the resolved configuration, plus whether we
/// already explained a fallback, so that a window with an algorithm or mode that is not
/// implemented logs about it once instead of every frame.
///
/// Later stages (neural network models, multi-pass execution, result caching) attach their state
/// here.
#[derive(Debug)]
pub struct SuperResolutionPipeline {
    /// The mode and algorithm that the current decision was made for.
    resolved: Cell<Option<(UpscalingMode, UpscalingAlgorithm)>>,
    /// Whether we already logged a fallback for the current decision.
    warned_fallback: Cell<bool>,
}

impl SuperResolutionPipeline {
    pub fn new() -> Self {
        Self {
            resolved: Cell::new(None),
            warned_fallback: Cell::new(false),
        }
    }

    /// Returns the resampling kernel to draw this window with, if the effect is enabled.
    ///
    /// This returns `None` when the window rules do not enable super resolution, in which case
    /// the window should be drawn the normal way.
    pub fn filter(&self, rules: &Upscaling) -> Option<ResampleFilter> {
        if !rules.enabled() {
            self.resolved.set(None);
            return None;
        }

        let mode = rules.mode();
        let algorithm = rules.algorithm();
        let (filter, fell_back) = ResampleFilter::for_algorithm(algorithm);

        if self.resolved.get() != Some((mode, algorithm)) {
            self.resolved.set(Some((mode, algorithm)));
            self.warned_fallback.set(false);
        }

        if !self.warned_fallback.get() {
            self.warned_fallback.set(true);

            if mode == UpscalingMode::Magnify {
                warn!(
                    "upscaling mode \"magnify\" is not implemented yet, \
                     treating it as \"filter\""
                );
            }

            if fell_back {
                warn!("upscaling algorithm {algorithm:?} is not implemented yet, using {filter:?}");
            }
        }

        Some(filter)
    }
}

impl Default for SuperResolutionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_es_version_mesa() {
        assert_eq!(parse_es_version("OpenGL ES 3.2 Mesa 25.0.0"), Some((3, 2)));
    }

    #[test]
    fn parse_es_version_es2() {
        assert_eq!(parse_es_version("OpenGL ES 2.0 Mesa 25.0.0"), Some((2, 0)));
    }

    #[test]
    fn parse_es_version_cm() {
        assert_eq!(parse_es_version("OpenGL ES-CM 1.1"), Some((1, 1)));
    }

    #[test]
    fn parse_es_version_desktop_gl() {
        assert_eq!(parse_es_version("4.6 (Core Profile) Mesa 25.0.0"), None);
    }

    #[test]
    fn parse_glsl_version_es3() {
        assert_eq!(parse_glsl_version("OpenGL ES GLSL ES 3.20"), Some((3, 20)));
    }

    #[test]
    fn parse_glsl_version_es2() {
        assert_eq!(parse_glsl_version("OpenGL ES GLSL ES 1.00"), Some((1, 0)));
    }

    #[test]
    fn filter_for_algorithm() {
        assert_eq!(
            ResampleFilter::for_algorithm(UpscalingAlgorithm::Bilinear),
            (ResampleFilter::Bilinear, false)
        );
        assert_eq!(
            ResampleFilter::for_algorithm(UpscalingAlgorithm::Bicubic),
            (ResampleFilter::Bicubic, false)
        );
        assert_eq!(
            ResampleFilter::for_algorithm(UpscalingAlgorithm::Cunny2),
            (ResampleFilter::Bicubic, true)
        );
    }

    #[test]
    fn pipeline_disabled_by_default() {
        let pipeline = SuperResolutionPipeline::new();
        assert_eq!(pipeline.filter(&Upscaling::default()), None);
    }

    #[test]
    fn pipeline_enabled_by_mode() {
        let pipeline = SuperResolutionPipeline::new();
        let rules = Upscaling {
            mode: Some(UpscalingMode::Filter),
            algorithm: None,
        };
        assert_eq!(pipeline.filter(&rules), Some(ResampleFilter::Bilinear));
    }

    #[test]
    fn pipeline_enabled_by_algorithm() {
        let pipeline = SuperResolutionPipeline::new();
        let rules = Upscaling {
            mode: None,
            algorithm: Some(UpscalingAlgorithm::Bicubic),
        };
        assert_eq!(pipeline.filter(&rules), Some(ResampleFilter::Bicubic));
    }

    #[test]
    fn pipeline_disabled_by_mode_off() {
        let pipeline = SuperResolutionPipeline::new();
        let rules = Upscaling {
            mode: Some(UpscalingMode::Off),
            algorithm: Some(UpscalingAlgorithm::Bicubic),
        };
        assert_eq!(pipeline.filter(&rules), None);
    }
    // The kernels are verified on a real GL context below. These tests need an EGL device, so
    // they return early when there is not one (for example on a machine with no GPU access); the
    // parsing and decision tests above do not need one.

    use smithay::backend::allocator::Fourcc;
    use smithay::backend::egl::native::EGLSurfacelessDisplay;
    use smithay::backend::egl::{EGLContext, EGLDisplay};
    use smithay::backend::renderer::gles::GlesTexture;
    use smithay::backend::renderer::{
        Bind as _, Color32F, ExportMem as _, Frame as _, ImportMem as _, Offscreen as _,
    };
    use smithay::utils::Size;

    use crate::render_helpers::copy_framebuffer;

    /// Creates a GLES renderer without a display, like the headless backend does.
    fn surfaceless_renderer() -> Option<GlesRenderer> {
        let display = unsafe { EGLDisplay::new(EGLSurfacelessDisplay).ok()? };
        let context = EGLContext::new(&display).ok()?;
        let renderer = unsafe { GlesRenderer::new(context).ok()? };
        Some(renderer)
    }

    /// A pattern that is distinct along both axes, so that a swap or a flip of the axes shows up.
    fn source_pixels(size: Size<i32, Buffer>) -> Vec<u8> {
        let mut pixels = Vec::new();
        for y in 0..size.h {
            for x in 0..size.w {
                pixels.extend_from_slice(&[
                    (x * 32 + 10) as u8,
                    (y * 32 + 20) as u8,
                    ((x + y) * 16 + 30) as u8,
                    255,
                ]);
            }
        }
        pixels
    }

    /// Reads a texture back in logical, top-down order.
    ///
    /// `copy_framebuffer()` reads GL framebuffer coordinates starting at (0, 0), and the renderer
    /// writes the logical top row into framebuffer row 0, so both an imported texture and a render
    /// target come back with row 0 first.
    fn read_back(
        renderer: &mut GlesRenderer,
        texture: &mut GlesTexture,
        size: Size<i32, Physical>,
    ) -> Vec<u8> {
        let target = renderer.bind(texture).unwrap();
        let mapping = copy_framebuffer(renderer, &target, Fourcc::Abgr8888).unwrap();
        let data = renderer.map_texture(&mapping).unwrap().to_vec();
        drop(target);
        assert_eq!(data.len(), (size.w * size.h * 4) as usize);
        data
    }

    /// Renders `texture` into a texture of `dest_size` with one of our kernels (or with the
    /// renderer's own sampling when `filter` is `None`) and reads the result back in logical,
    /// top-down order.
    fn resample(
        renderer: &mut GlesRenderer,
        texture: &GlesTexture,
        src: Rectangle<f64, Buffer>,
        dest_size: Size<i32, Physical>,
        filter: Option<ResampleFilter>,
    ) -> Vec<u8> {
        let buffer_size = dest_size.to_logical(1).to_buffer(1, Transform::Normal);
        let mut dest = renderer
            .create_buffer(Fourcc::Abgr8888, buffer_size)
            .unwrap();

        {
            let mut target = renderer.bind(&mut dest).unwrap();
            let mut frame = renderer
                .render(&mut target, dest_size, Transform::Normal)
                .unwrap();

            let dest_rect = Rectangle::from_size(dest_size);
            frame.clear(Color32F::TRANSPARENT, &[dest_rect]).unwrap();

            let program = filter.map(|filter| {
                let programs = ResamplePrograms::get_from_frame(&frame)
                    .expect("superres::init() must be called on this renderer");
                let program = programs
                    .program(filter)
                    .expect("the resampling shader must compile");
                let uniforms =
                    resample_uniforms(texture, src, false).expect("texture must have a size");
                (program, uniforms)
            });

            let (program, uniforms) = match &program {
                Some((program, uniforms)) => (Some(program), &uniforms[..]),
                None => (None, &[][..]),
            };

            frame
                .render_texture_from_to(
                    texture,
                    src,
                    dest_rect,
                    &[dest_rect],
                    &[],
                    Transform::Normal,
                    1.0,
                    program,
                    uniforms,
                )
                .unwrap();
            let _ = frame.finish().unwrap();
        }

        read_back(renderer, &mut dest, dest_size)
    }

    /// The Catmull-Rom weights that the shader uses, for the four taps around a fractional offset.
    fn catmull_rom(t: f64) -> [f64; 4] {
        let t2 = t * t;
        let t3 = t2 * t;
        [
            -0.5 * t3 + t2 - 0.5 * t,
            1.5 * t3 - 2.5 * t2 + 1.0,
            -1.5 * t3 + 2.0 * t2 + 0.5 * t,
            0.5 * t3 - 0.5 * t2,
        ]
    }

    /// The expected value of one channel of one destination pixel, mirroring what the shader does.
    ///
    /// `src` is the source in logical, top-down order.
    fn expected_pixel(
        src: &[u8],
        src_size: Size<i32, Buffer>,
        dest_size: Size<i32, Physical>,
        filter: ResampleFilter,
        x: i32,
        y: i32,
        channel: usize,
    ) -> f64 {
        let sample = |tx: i32, ty: i32| -> f64 {
            let tx = tx.clamp(0, src_size.w - 1);
            let ty = ty.clamp(0, src_size.h - 1);
            f64::from(src[((ty * src_size.w + tx) * 4) as usize + channel])
        };

        // The position of the destination pixel center in source texels, in the space where a
        // texel center is at an integer coordinate.
        let ratio = (
            f64::from(src_size.w) / f64::from(dest_size.w),
            f64::from(src_size.h) / f64::from(dest_size.h),
        );
        let pos = (
            (f64::from(x) + 0.5) * ratio.0 - 0.5,
            (f64::from(y) + 0.5) * ratio.1 - 0.5,
        );
        let base = (pos.0.floor(), pos.1.floor());
        let frac = (pos.0 - base.0, pos.1 - base.1);
        let base = (base.0 as i32, base.1 as i32);

        match filter {
            ResampleFilter::Bilinear => {
                let wx = [1.0 - frac.0, frac.0];
                let wy = [1.0 - frac.1, frac.1];
                let mut sum = 0.0;
                for (j, wy) in wy.iter().enumerate() {
                    for (i, wx) in wx.iter().enumerate() {
                        sum += wx * wy * sample(base.0 + i as i32, base.1 + j as i32);
                    }
                }
                sum
            }
            ResampleFilter::Bicubic => {
                let wx = catmull_rom(frac.0);
                let wy = catmull_rom(frac.1);
                let mut sum = 0.0;
                for (j, wy) in wy.iter().enumerate() {
                    for (i, wx) in wx.iter().enumerate() {
                        // The taps are the texel centers base - 1 .. base + 2.
                        sum += wx * wy * sample(base.0 + i as i32 - 1, base.1 + j as i32 - 1);
                    }
                }
                sum
            }
        }
    }

    /// Asserts that a resampled texture matches the kernels that this file claims to implement.
    #[test]
    fn gpu_resample_matches_the_kernels() {
        let Some(mut renderer) = surfaceless_renderer() else {
            return;
        };

        crate::render_helpers::resources::init(&mut renderer);
        crate::render_helpers::shaders::init(&mut renderer);
        init(&mut renderer, None);

        let src_size = Size::from((4, 4));
        let src_physical = Size::<i32, Physical>::from((4, 4));
        let src_pixels = source_pixels(src_size);
        let mut texture = renderer
            .import_memory(&src_pixels, Fourcc::Abgr8888, src_size, false)
            .unwrap();
        let src = read_back(&mut renderer, &mut texture, src_physical);
        assert_eq!(src, src_pixels, "the harness must not change the source");

        let src_rect = Rectangle::from_size(src_size.to_f64());
        let dest_size = Size::from((8, 8));

        for filter in [
            None,
            Some(ResampleFilter::Bilinear),
            Some(ResampleFilter::Bicubic),
        ] {
            // 1:1 must be an exact copy: the taps land on texel centers, where the kernels reduce
            // to sampling the source texel.
            let identity = resample(&mut renderer, &texture, src_rect, src_physical, filter);
            for (i, (got, want)) in identity.iter().zip(src.iter()).enumerate() {
                assert!(
                    i32::from(*got).abs_diff(i32::from(*want)) <= 1,
                    "{filter:?} identity differs at byte {i}: {got} != {want}",
                );
            }
        }

        for filter in [ResampleFilter::Bilinear, ResampleFilter::Bicubic] {
            // 2x is where the kernels differ from the renderer's own sampling, so compare against
            // the kernels themselves.
            let scaled = resample(&mut renderer, &texture, src_rect, dest_size, Some(filter));
            for y in 0..dest_size.h {
                for x in 0..dest_size.w {
                    for channel in 0..4 {
                        let index = ((y * dest_size.w + x) * 4) as usize + channel;
                        let got = f64::from(scaled[index]);
                        let want = expected_pixel(&src, src_size, dest_size, filter, x, y, channel);
                        assert!(
                            (got - want).abs() <= 1.5,
                            "{filter:?} at ({x}, {y}) channel {channel}: {got} != {want}",
                        );
                    }
                }
            }
        }
    }
}
