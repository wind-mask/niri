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

use std::cell::{Cell, RefCell};
use std::ffi::CStr;
use std::fmt::Write as _;
use std::path::Path;

use anyhow::Context as _;
use niri_config::{Upscaling, UpscalingAlgorithm, UpscalingMode};
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::element::surface::{
    WaylandSurfaceRenderElement, WaylandSurfaceTexture,
};
use smithay::backend::renderer::element::{Element, Id, Kind, RenderElement, UnderlyingStorage};
use smithay::backend::renderer::gles::{
    ffi, Capability, GlesError, GlesFrame, GlesRenderer, GlesTexProgram, GlesTexture, Uniform,
    UniformName, UniformType,
};
use smithay::backend::renderer::multigpu::MultiTexture;
use smithay::backend::renderer::utils::{CommitCounter, DamageSet, OpaqueRegions};
use smithay::backend::renderer::{
    buffer_y_inverted, Bind as _, Color32F, ContextId, Frame as _, ImportAll, Offscreen as _,
    Renderer, Texture,
};
use smithay::utils::user_data::UserDataMap;
use smithay::utils::{Buffer, Logical, Physical, Point, Rectangle, Scale, Size, Transform};

use crate::backend::tty::{native_texture, TtyFrame, TtyRenderer, TtyRendererError};
use crate::render_helpers::renderer::{AsGlesFrame as _, NiriRenderer};

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

/// Geometry for the `magnify` upscaling mode, in the output's logical coordinate space.
///
/// The window's own content has size `source` and is fitted (`contain`, centered) into `target`,
/// which is the area the layout gave the window. Everything in `content` is relative to the origin
/// of `target`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MagnifiedGeometry {
    /// The window's own content size.
    pub source: Size<f64, Logical>,
    /// The area that the window is magnified into.
    pub target: Size<f64, Logical>,
    /// Where the scaled content lands inside `target`.
    pub content: Rectangle<f64, Logical>,
    /// Uniform scale factor from `source` to `content.size`.
    pub scale: f64,
}

impl MagnifiedGeometry {
    /// Fits `source` into `target`, preserving aspect ratio and centering the result.
    ///
    /// Returns `None` when `source` or `target` is empty, or when `source` is already at least as
    /// large as `target` in both dimensions (so there is nothing to magnify).
    pub fn new(source: Size<f64, Logical>, target: Size<f64, Logical>) -> Option<Self> {
        if source.w <= 0. || source.h <= 0. || target.w <= 0. || target.h <= 0. {
            return None;
        }

        // Only magnify. A window that is already as big as the target (or bigger) is not
        // this mode's business.
        if source.w >= target.w && source.h >= target.h {
            return None;
        }

        let scale = f64::min(target.w / source.w, target.h / source.h);
        let size = Size::from((source.w * scale, source.h * scale));
        let loc = Point::from(((target.w - size.w) / 2., (target.h - size.h) / 2.));

        Some(Self {
            source,
            target,
            content: Rectangle::new(loc, size),
            scale,
        })
    }

    /// Maps a point in the window's visual area to the content's own coordinate space.
    ///
    /// Returns `None` for points in the letterbox bars, which are not part of the content and must
    /// not receive input.
    pub fn to_source(&self, point: Point<f64, Logical>) -> Option<Point<f64, Logical>> {
        let point = point - self.content.loc;

        if point.x < 0.
            || point.y < 0.
            || point.x >= self.content.size.w
            || point.y >= self.content.size.h
        {
            return None;
        }

        Some(Point::from((point.x / self.scale, point.y / self.scale)))
    }

    /// Maps a rectangle given in the content's coordinate space, relative to the window's visual
    /// origin, into the window's visual area.
    ///
    /// This is what drawing uses, including for pop-ups and subsurfaces, which live in the same
    /// coordinate space as the window's own content.
    pub fn to_target(&self, rect: Rectangle<f64, Logical>) -> Rectangle<f64, Logical> {
        Rectangle::new(
            Point::from((
                self.content.loc.x + rect.loc.x * self.scale,
                self.content.loc.y + rect.loc.y * self.scale,
            )),
            Size::from((rect.size.w * self.scale, rect.size.h * self.scale)),
        )
    }
}

/// Returns whether a buffer of `src` texels has anything for the effect to recover in `dest`
/// pixels.
///
/// A buffer that is one pixel short of its display area is what logical sizes that do not land on
/// whole physical pixels produce with a fractional output scale, and there is nothing in a pixel
/// for a kernel to reconstruct.
fn has_something_to_magnify(src: Size<f64, Buffer>, dest: Size<i32, Physical>) -> bool {
    src.w + 1.0 < f64::from(dest.w) || src.h + 1.0 < f64::from(dest.h)
}

/// How many texels a resampling kernel reads around the one it writes, in source pixels.
///
/// Catmull-Rom and FSR1's EASU both look two texels around the one they write, so anything within
/// that footprint of a change is affected by it too.
const KERNEL_RADIUS: i32 = 2;

/// Maps a rectangle that is relative to an element's origin into the area that element is drawn in
/// when its content is magnified.
///
/// Damage and opaque regions are given relative to the element they belong to, so this is a scale
/// about the origin: the magnified element draws the same content over a larger area. `outward`
/// picks which way to round, which has to over-approximate for damage and under-approximate for
/// opaque regions, and `grow` is how far the resampling kernel reads beyond what changed.
fn magnify_physical_rect(
    rect: Rectangle<i32, Physical>,
    inner: Rectangle<i32, Physical>,
    dest: Rectangle<i32, Physical>,
    outward: bool,
    grow: i32,
) -> Rectangle<i32, Physical> {
    if inner.size.is_empty() {
        // Nothing sensible to scale by; say that the whole area is affected.
        return Rectangle::from_size(dest.size);
    }

    let factor = dest.size.to_f64() / inner.size.to_f64();
    let mapped = rect.to_f64().upscale(factor);

    // Round the edges rather than the location and the size on their own, so that the result
    // covers everything that changed (or, for opaque regions, nothing that did not).
    let mapped = if outward {
        mapped.to_i32_up()
    } else {
        mapped.to_i32_down()
    };

    let grow = Size::from((grow, grow));
    let loc = mapped.loc - grow.to_point();
    let size = mapped.size + grow * 2;

    Rectangle::new(loc, Size::from((size.w.max(0), size.h.max(0))))
}

/// How a window is drawn with super resolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SuperResKernel {
    /// A single-pass resampling filter drawn over the client buffer.
    Filter(ResampleFilter),

    /// FSR1: EASU into an offscreen texture, then RCAS in the final draw.
    Fsr1(Fsr1Params),
}

impl SuperResKernel {
    /// Returns the kernel to use for an algorithm, and whether it fell back to a baseline.
    ///
    /// Algorithms that are not implemented yet fall back to the best baseline kernel rather than
    /// drawing something that we cannot explain.
    fn for_algorithm(algorithm: UpscalingAlgorithm, sharpness: f64) -> (Self, bool) {
        match algorithm {
            UpscalingAlgorithm::Bilinear => (Self::Filter(ResampleFilter::Bilinear), false),
            UpscalingAlgorithm::Bicubic => (Self::Filter(ResampleFilter::Bicubic), false),
            UpscalingAlgorithm::Fsr1 => (
                Self::Fsr1(Fsr1Params {
                    sharpness: sharpness as f32,
                }),
                false,
            ),
            UpscalingAlgorithm::Cunny2 | UpscalingAlgorithm::Anime4k => {
                (Self::Filter(ResampleFilter::Bicubic), true)
            }
        }
    }
}

/// FSR1 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fsr1Params {
    /// Sharpening strength, in RCAS stops of halving.
    pub sharpness: f32,
}

impl Fsr1Params {
    /// Returns the RCAS sharpening scale, `exp2(-sharpness)` of `FsrRcasCon()`.
    fn sharpness_scale(&self) -> f32 {
        (-self.sharpness).exp2()
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

/// The super resolution programs, one per pass.
///
/// These are compiled once per renderer context and stored in the EGL context user data.
#[derive(Debug)]
pub struct SuperResPrograms {
    bilinear: Option<GlesTexProgram>,
    bicubic: Option<GlesTexProgram>,
    easu: Option<GlesTexProgram>,
    rcas: Option<GlesTexProgram>,
}

impl SuperResPrograms {
    fn compile(renderer: &mut GlesRenderer) -> Self {
        Self {
            bilinear: Self::compile_filter(renderer, ResampleFilter::Bilinear),
            bicubic: Self::compile_filter(renderer, ResampleFilter::Bicubic),
            easu: Self::compile_file(
                renderer,
                "easu",
                include_str!("shaders/superres/fsr1_easu.frag"),
                &[
                    UniformName::new("niri_tex_size", UniformType::_2f),
                    UniformName::new("niri_src_rect", UniformType::_4f),
                ],
            ),
            rcas: Self::compile_file(
                renderer,
                "rcas",
                include_str!("shaders/superres/fsr1_rcas.frag"),
                &[
                    UniformName::new("niri_tex_size", UniformType::_2f),
                    UniformName::new("niri_src_rect", UniformType::_4f),
                    UniformName::new("niri_rcas_sharpness", UniformType::_1f),
                ],
            ),
        }
    }

    fn compile_filter(
        renderer: &mut GlesRenderer,
        filter: ResampleFilter,
    ) -> Option<GlesTexProgram> {
        let index = match filter {
            ResampleFilter::Bilinear => 0,
            ResampleFilter::Bicubic => 1,
        };

        let source = include_str!("shaders/superres/resample.frag").replace(
            "// niri:resample-filter",
            &format!("#define NIRI_RESAMPLE_FILTER {index}"),
        );

        Self::compile_file(
            renderer,
            match filter {
                ResampleFilter::Bilinear => "bilinear",
                ResampleFilter::Bicubic => "bicubic",
            },
            &source,
            &[
                UniformName::new("niri_tex_size", UniformType::_2f),
                UniformName::new("niri_src_rect", UniformType::_4f),
            ],
        )
    }

    fn compile_file(
        renderer: &mut GlesRenderer,
        name: &str,
        source: &str,
        uniforms: &[UniformName<'_>],
    ) -> Option<GlesTexProgram> {
        match renderer.compile_custom_texture_shader(source, uniforms) {
            Ok(program) => Some(program),
            Err(err) => {
                warn!("error compiling the {name} shader: {err:?}");
                None
            }
        }
    }

    fn get_from_frame<'a>(frame: &'a GlesFrame<'_, '_>) -> Option<&'a Self> {
        frame.egl_context().user_data().get()
    }

    fn get(renderer: &GlesRenderer) -> Option<&Self> {
        renderer.egl_context().user_data().get()
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

    let programs = SuperResPrograms::compile(renderer);

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
    kernel: ElementKernel,
    /// When set, draw the content into this rectangle (in the output's logical coordinate space)
    /// instead of the element's own geometry. This is the `magnify` mode.
    magnify: Option<Rectangle<f64, Logical>>,
}

/// What an element draws with.
#[derive(Debug, Clone)]
pub enum ElementKernel {
    /// Draw the client buffer with the renderer's own sampling. This is what magnified pop-ups
    /// use, they are still drawn at their element's geometry.
    Plain,

    /// The client buffer, resampled with a single-pass filter.
    Filter(ResampleFilter),

    /// A cached EASU result, sharpened with RCAS.
    Fsr1 {
        /// The EASU result.
        texture: GlesTexture,
        /// The renderer context that the texture belongs to.
        context_id: ContextId<GlesTexture>,
        /// Sharpening scale for RCAS, `exp2(-sharpness)`.
        sharpness_scale: f32,
    },
}

impl<R: Renderer> SuperResElement<R> {
    /// Creates an element that draws the client buffer with a resampling filter.
    pub fn new(inner: WaylandSurfaceRenderElement<R>, filter: ResampleFilter) -> Self {
        Self {
            inner,
            kernel: ElementKernel::Filter(filter),
            magnify: None,
        }
    }

    /// Creates an element that draws the client buffer with the renderer's own sampling.
    pub fn plain(inner: WaylandSurfaceRenderElement<R>) -> Self {
        Self {
            inner,
            kernel: ElementKernel::Plain,
            magnify: None,
        }
    }

    /// Draws the content into `rect` rather than at the element's own geometry.
    pub fn with_magnify(mut self, rect: Rectangle<f64, Logical>) -> Self {
        self.magnify = Some(rect);
        self
    }

    /// Creates an element that draws a cached FSR1 result.
    pub fn fsr1(
        inner: WaylandSurfaceRenderElement<R>,
        texture: GlesTexture,
        context_id: ContextId<GlesTexture>,
        sharpness_scale: f32,
    ) -> Self {
        Self {
            inner,
            kernel: ElementKernel::Fsr1 {
                texture,
                context_id,
                sharpness_scale,
            },
            magnify: None,
        }
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
    /// Returns the resampling uniforms for a source rect of the client buffer.
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
        let ElementKernel::Filter(filter) = self.kernel else {
            return None;
        };

        let program = SuperResPrograms::get_from_frame(frame)?.program(filter)?;
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
        match self.magnify {
            Some(rect) => rect.to_physical_precise_round(scale),
            None => self.inner.geometry(scale),
        }
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
        let damage = self.inner.damage_since(scale, commit);

        let Some(rect) = self.magnify else {
            return damage;
        };

        let inner = self.inner.geometry(scale);
        let dest = rect.to_physical_precise_round(scale);

        // The damage is relative to the element, and this element draws the wrapped one's content
        // over a larger area, so the rects are scaled about the origin; the tracker adds the
        // geometry location back. Growing them by the kernel's footprint keeps the edge of a
        // repainted region from showing what the old frame had there.
        if inner.size.is_empty() {
            return DamageSet::from_slice(&[Rectangle::from_size(dest.size)]);
        }

        let factor = dest.size.to_f64() / inner.size.to_f64();
        let grow = (f64::from(KERNEL_RADIUS) * factor.x.max(factor.y)).ceil() as i32;

        damage
            .into_iter()
            .map(|rect| magnify_physical_rect(rect, inner, dest, true, grow))
            .collect()
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        let regions = self.inner.opaque_regions(scale);

        let Some(rect) = self.magnify else {
            return regions;
        };

        let inner = self.inner.geometry(scale);
        let dest = rect.to_physical_precise_round(scale);

        // These are relative to the element as well, and they round inwards: covering less than we
        // actually do only costs a redraw.
        regions
            .into_iter()
            .map(|rect| magnify_physical_rect(rect, inner, dest, false, 0))
            .collect()
    }

    fn alpha(&self) -> f32 {
        self.inner.alpha()
    }

    fn kind(&self) -> Kind {
        self.inner.kind()
    }
}

/// Draws a surface element with the renderer's own sampling.
#[allow(clippy::too_many_arguments)]
fn draw_inner_gles(
    inner: &WaylandSurfaceRenderElement<GlesRenderer>,
    frame: &mut GlesFrame<'_, '_>,
    src: Rectangle<f64, Buffer>,
    dst: Rectangle<i32, Physical>,
    damage: &[Rectangle<i32, Physical>],
    opaque_regions: &[Rectangle<i32, Physical>],
    cache: Option<&UserDataMap>,
) -> Result<(), GlesError> {
    RenderElement::<GlesRenderer>::draw(inner, frame, src, dst, damage, opaque_regions, cache)
}

/// Draws a cached FSR1 result with the sharpening pass.
///
/// Returns `None` when the result cannot be drawn, in which case the caller must fall back to
/// drawing the client buffer. The result texture belongs to the renderer that produced it, and it
/// is already at the element's size, so this is a straight 1:1 draw.
#[allow(clippy::too_many_arguments)]
fn draw_fsr1(
    frame: &mut GlesFrame<'_, '_>,
    texture: &GlesTexture,
    context_id: ContextId<GlesTexture>,
    sharpness_scale: f32,
    dst: Rectangle<i32, Physical>,
    damage: &[Rectangle<i32, Physical>],
    opaque_regions: &[Rectangle<i32, Physical>],
    alpha: f32,
) -> Option<Result<(), GlesError>> {
    if frame.context_id() != context_id {
        // This happens when the same window is drawn on another GPU, where the cached result is
        // not valid. It is not an error, we just cannot use the cache here.
        debug!("not drawing a super resolution result from another renderer context");
        return None;
    }

    let size = texture.size();
    if size.w <= 0 || size.h <= 0 {
        return None;
    }

    let program = SuperResPrograms::get_from_frame(frame)?.rcas.clone()?;
    let uniforms = [
        Uniform::new("niri_tex_size", (size.w as f32, size.h as f32)),
        Uniform::new("niri_src_rect", [0f32, 0., size.w as f32, size.h as f32]),
        Uniform::new("niri_rcas_sharpness", sharpness_scale),
    ];

    Some(frame.render_texture_from_to(
        texture,
        Rectangle::from_size(size.to_f64()),
        dst,
        damage,
        opaque_regions,
        Transform::Normal,
        alpha,
        Some(&program),
        &uniforms,
    ))
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
        match &self.kernel {
            ElementKernel::Plain => {
                draw_inner_gles(&self.inner, frame, src, dst, damage, opaque_regions, cache)
            }
            ElementKernel::Filter(_) => {
                let Some((program, uniforms)) = self.resample_program(frame, src) else {
                    // Fall back to the renderer's own sampling.
                    return draw_inner_gles(
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
                let res =
                    draw_inner_gles(&self.inner, frame, src, dst, damage, opaque_regions, cache);
                frame.clear_tex_program_override();
                res
            }
            ElementKernel::Fsr1 {
                texture,
                context_id,
                sharpness_scale,
            } => match draw_fsr1(
                frame,
                texture,
                context_id.clone(),
                *sharpness_scale,
                dst,
                damage,
                opaque_regions,
                self.inner.alpha(),
            ) {
                Some(res) => res,
                None => {
                    draw_inner_gles(&self.inner, frame, src, dst, damage, opaque_regions, cache)
                }
            },
        }
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
        match &self.kernel {
            ElementKernel::Plain => RenderElement::<TtyRenderer<'render>>::draw(
                &self.inner,
                frame,
                src,
                dst,
                damage,
                opaque_regions,
                cache,
            ),
            ElementKernel::Filter(_) => {
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
            ElementKernel::Fsr1 {
                texture,
                context_id,
                sharpness_scale,
            } => {
                // Drawing through the underlying GLES frame rather than through the multigpu frame:
                // the cached result is a plain GLES texture that the multigpu frame cannot address.
                // This loses the multigpu frame's cross-GPU damage bookkeeping, which is why this
                // has to be revisited together with mixed-GPU support.
                if let Some(res) = draw_fsr1(
                    frame.as_gles_frame(),
                    texture,
                    context_id.clone(),
                    *sharpness_scale,
                    dst,
                    damage,
                    opaque_regions,
                    self.inner.alpha(),
                ) {
                    return res.map_err(TtyRendererError::Render);
                }

                RenderElement::<TtyRenderer<'render>>::draw(
                    &self.inner,
                    frame,
                    src,
                    dst,
                    damage,
                    opaque_regions,
                    cache,
                )
            }
        }
    }

    fn underlying_storage(
        &self,
        _renderer: &mut TtyRenderer<'render>,
    ) -> Option<UnderlyingStorage<'_>> {
        None
    }
}

/// Gets a renderer's own GLES texture, or the per-GPU texture behind a multigpu texture.
///
/// Offscreen passes run on one GPU's GLES renderer, so they need the texture that belongs to that
/// renderer's context rather than the renderer-independent handle.
pub trait PassSource {
    /// Returns the texture as it exists in the given renderer context.
    fn pass_texture(&self, context_id: &ContextId<GlesTexture>) -> Option<GlesTexture>;
}

impl PassSource for GlesTexture {
    fn pass_texture(&self, _context_id: &ContextId<GlesTexture>) -> Option<GlesTexture> {
        // The element was built with this renderer, so the texture belongs to its context.
        Some(self.clone())
    }
}

impl PassSource for MultiTexture {
    fn pass_texture(&self, context_id: &ContextId<GlesTexture>) -> Option<GlesTexture> {
        native_texture(self, context_id)
    }
}

/// An offscreen result of the EASU pass.
#[derive(Debug)]
struct Fsr1Result {
    texture: GlesTexture,
    context_id: ContextId<GlesTexture>,
    key: Fsr1Key,
}

/// What an EASU result was produced from.
///
/// Resampling a window goes through this pass once per source commit, and the final sharpening
/// pass runs on the result every frame, so this only has to change when the content or the size
/// does. Moving the window, drawing other windows or moving the cursor must not invalidate it.
#[derive(Debug, Clone, PartialEq)]
struct Fsr1Key {
    commit: CommitCounter,
    /// Source rect in buffer texels.
    src: Rectangle<f64, Buffer>,
    /// Destination size in physical pixels.
    size: Size<i32, Physical>,
}

/// Per-window super resolution state.
///
/// This holds the resolved configuration, the cached FSR1 result, and whether we already explained
/// a fallback, so that a window with an algorithm or mode that is not implemented logs about it
/// once instead of every frame.
#[derive(Debug)]
pub struct SuperResolutionPipeline {
    /// The mode and algorithm that the current decision was made for.
    resolved: Cell<Option<(UpscalingMode, UpscalingAlgorithm)>>,
    /// Whether we already logged a fallback for the current decision.
    warned_fallback: Cell<bool>,
    /// Whether we already logged that the offscreen pass failed.
    warned_pass_failure: Cell<bool>,
    /// Cached result of the EASU pass.
    fsr1: RefCell<Option<Fsr1Result>>,
    /// The size that this window's content is magnified into, when it is being magnified.
    magnify_target: Cell<Option<Size<i32, Logical>>>,
}

impl SuperResolutionPipeline {
    pub fn new() -> Self {
        Self {
            resolved: Cell::new(None),
            warned_fallback: Cell::new(false),
            warned_pass_failure: Cell::new(false),
            fsr1: RefCell::new(None),
            magnify_target: Cell::new(None),
        }
    }

    /// Returns the kernel to draw this window with, if the effect is enabled.
    ///
    /// This returns `None` when the window rules do not enable super resolution, in which case
    /// the window should be drawn the normal way.
    pub fn kernel(&self, rules: &Upscaling) -> Option<SuperResKernel> {
        if !rules.enabled() {
            self.resolved.set(None);
            return None;
        }

        let mode = rules.mode();
        let algorithm = rules.algorithm();
        let (kernel, fell_back) = SuperResKernel::for_algorithm(algorithm, rules.sharpness());

        if self.resolved.get() != Some((mode, algorithm)) {
            self.resolved.set(Some((mode, algorithm)));
            self.warned_fallback.set(false);
            self.warned_pass_failure.set(false);
            // The parameters may have changed, so the cached result may no longer match.
            self.fsr1.borrow_mut().take();
        }

        if !self.warned_fallback.get() {
            self.warned_fallback.set(true);

            if fell_back {
                warn!("upscaling algorithm {algorithm:?} is not implemented yet, using {kernel:?}");
            }
        }

        Some(kernel)
    }

    /// Records that this window's content is magnified into `target`, or that it is not anymore.
    pub fn set_magnify_target(&self, target: Option<Size<i32, Logical>>) {
        self.magnify_target.set(target);
    }

    /// Returns the size that this window's content is magnified into, if it is being magnified.
    pub fn magnify_target(&self) -> Option<Size<i32, Logical>> {
        self.magnify_target.get()
    }

    /// Wraps a surface element, running the EASU pass first if the cached result is stale.
    ///
    /// `renderer` must be the GLES renderer that will draw the element, and `scale` is the output
    /// scale that the element is drawn at. The returned element falls back to drawing the client
    /// buffer normally when the pass cannot run or when its result is not valid for the renderer
    /// that ends up drawing it.
    pub fn element<R: NiriRenderer>(
        &self,
        renderer: &mut GlesRenderer,
        elem: WaylandSurfaceRenderElement<R>,
        kernel: SuperResKernel,
        scale: Scale<f64>,
        magnify: Option<Rectangle<f64, Logical>>,
    ) -> SuperResElement<R>
    where
        R::TextureId: PassSource,
    {
        // With magnify, the content is drawn into a rect inside the area the layout gave the
        // window, so that is the size the effect has to produce.
        let dest_size = match magnify {
            Some(rect) => rect.size.to_physical_precise_round(scale),
            None => elem.geometry(scale).size,
        };

        // A buffer that already matches its display area has nothing for the effect to recover: the
        // client is rendering at the physical resolution of the output, which is what Wayland
        // clients that handle fractional scaling do. There is at most a pixel in it, so draw the
        // window the normal way rather than paying for a pass that cannot change anything. The
        // magnify mode is a different thing, since its content is drawn into a bigger area on
        // purpose.
        if magnify.is_none() && !has_something_to_magnify(elem.src().size, dest_size) {
            return SuperResElement::plain(elem);
        }

        let params = match kernel {
            SuperResKernel::Filter(filter) => {
                let mut element = SuperResElement::new(elem, filter);
                element = match magnify {
                    Some(rect) => element.with_magnify(rect),
                    None => element,
                };
                return element;
            }
            SuperResKernel::Fsr1(params) => params,
        };

        let Some(texture) = self.fsr1_result(renderer, &elem, dest_size) else {
            // Fall back to the best single-pass kernel that we have.
            let element = SuperResElement::new(elem, ResampleFilter::Bicubic);
            return match magnify {
                Some(rect) => element.with_magnify(rect),
                None => element,
            };
        };

        let mut element = SuperResElement::fsr1(
            elem,
            texture.texture.clone(),
            texture.context_id.clone(),
            params.sharpness_scale(),
        );
        if let Some(rect) = magnify {
            element = element.with_magnify(rect);
        }
        element
    }

    /// Returns the cached EASU result, running the pass if it is stale.
    fn fsr1_result<R: NiriRenderer>(
        &self,
        renderer: &mut GlesRenderer,
        elem: &WaylandSurfaceRenderElement<R>,
        size: Size<i32, Physical>,
    ) -> Option<Fsr1Result>
    where
        R::TextureId: PassSource,
    {
        if size.w <= 0 || size.h <= 0 {
            return None;
        }

        let src = elem.src();
        let key = Fsr1Key {
            commit: elem.current_commit(),
            src,
            size,
        };

        if let Some(result) = self.fsr1.borrow().as_ref() {
            if result.key == key && result.context_id == renderer.context_id() {
                // The cached result is still what this window would be drawn with.
                return Some(Fsr1Result {
                    texture: result.texture.clone(),
                    context_id: result.context_id.clone(),
                    key,
                });
            }
        }

        let WaylandSurfaceTexture::Texture(texture) = elem.texture() else {
            return None;
        };
        let source = texture.pass_texture(&renderer.context_id())?;

        let y_inverted = buffer_y_inverted(elem.buffer()).unwrap_or(false);
        let uniforms = resample_uniforms(&source, src, y_inverted)?;
        let program = SuperResPrograms::get(renderer)?.easu.clone()?;

        let texture = match run_easu_pass(
            renderer,
            &source,
            src,
            size,
            elem.transform(),
            &program,
            &uniforms,
        ) {
            Ok(texture) => texture,
            Err(err) => {
                if !self.warned_pass_failure.replace(true) {
                    warn!("error running the super resolution pass: {err:?}");
                }
                return None;
            }
        };

        let result = Fsr1Result {
            texture,
            context_id: renderer.context_id(),
            key,
        };

        *self.fsr1.borrow_mut() = Some(Fsr1Result {
            texture: result.texture.clone(),
            context_id: result.context_id.clone(),
            key: result.key.clone(),
        });

        Some(result)
    }
}

/// Runs the EASU pass: renders the client buffer into an offscreen texture at the display size.
#[allow(clippy::too_many_arguments)]
fn run_easu_pass(
    renderer: &mut GlesRenderer,
    source: &GlesTexture,
    src: Rectangle<f64, Buffer>,
    size: Size<i32, Physical>,
    transform: Transform,
    program: &GlesTexProgram,
    uniforms: &[Uniform<'static>],
) -> anyhow::Result<GlesTexture> {
    let buffer_size = size.to_logical(1).to_buffer(1, Transform::Normal);
    let mut texture = renderer
        .create_buffer(Fourcc::Abgr8888, buffer_size)
        .context("error creating texture")?;

    let rect = Rectangle::from_size(size);

    {
        let mut target = renderer
            .bind(&mut texture)
            .context("error binding texture")?;
        let mut frame = renderer
            .render(&mut target, size, Transform::Normal)
            .context("error starting frame")?;

        frame
            .clear(Color32F::TRANSPARENT, &[rect])
            .context("error clearing")?;
        frame
            .render_texture_from_to(
                source,
                src,
                rect,
                &[rect],
                &[rect],
                transform,
                // The element alpha is applied by the sharpening pass.
                1.0,
                Some(program),
                uniforms,
            )
            .context("error in the EASU pass")?;
        let _ = frame.finish().context("error finishing frame")?;
    }

    Ok(texture)
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
    fn kernel_for_algorithm() {
        assert_eq!(
            SuperResKernel::for_algorithm(UpscalingAlgorithm::Bilinear, 0.2),
            (SuperResKernel::Filter(ResampleFilter::Bilinear), false)
        );
        assert_eq!(
            SuperResKernel::for_algorithm(UpscalingAlgorithm::Bicubic, 0.2),
            (SuperResKernel::Filter(ResampleFilter::Bicubic), false)
        );
        assert_eq!(
            SuperResKernel::for_algorithm(UpscalingAlgorithm::Cunny2, 0.2),
            (SuperResKernel::Filter(ResampleFilter::Bicubic), true)
        );

        let (kernel, fell_back) = SuperResKernel::for_algorithm(UpscalingAlgorithm::Fsr1, 0.5);
        assert!(!fell_back);
        assert_eq!(kernel, SuperResKernel::Fsr1(Fsr1Params { sharpness: 0.5 }));
    }

    #[test]
    fn fsr1_sharpness_scale() {
        // `FsrRcasCon()` turns stops of halving into a linear scale.
        assert_eq!(Fsr1Params { sharpness: 0.0 }.sharpness_scale(), 1.0);
        assert_eq!(Fsr1Params { sharpness: 1.0 }.sharpness_scale(), 0.5);
        assert_eq!(Fsr1Params { sharpness: 2.0 }.sharpness_scale(), 0.25);
    }

    #[test]
    fn pipeline_disabled_by_default() {
        let pipeline = SuperResolutionPipeline::new();
        assert_eq!(pipeline.kernel(&Upscaling::default()), None);
    }

    #[test]
    fn pipeline_enabled_by_mode() {
        let pipeline = SuperResolutionPipeline::new();
        let rules = Upscaling {
            mode: Some(UpscalingMode::Filter),
            algorithm: None,
            sharpness: None,
            client_scale: None,
            virtual_size: None,
        };
        assert_eq!(
            pipeline.kernel(&rules),
            Some(SuperResKernel::Filter(ResampleFilter::Bilinear))
        );
    }

    #[test]
    fn pipeline_enabled_by_algorithm() {
        let pipeline = SuperResolutionPipeline::new();
        let rules = Upscaling {
            mode: None,
            algorithm: Some(UpscalingAlgorithm::Bicubic),
            sharpness: None,
            client_scale: None,
            virtual_size: None,
        };
        assert_eq!(
            pipeline.kernel(&rules),
            Some(SuperResKernel::Filter(ResampleFilter::Bicubic))
        );
    }

    #[test]
    fn pipeline_fsr1() {
        let pipeline = SuperResolutionPipeline::new();
        let rules = Upscaling {
            mode: None,
            algorithm: Some(UpscalingAlgorithm::Fsr1),
            sharpness: Some(1.0),
            client_scale: None,
            virtual_size: None,
        };
        assert_eq!(
            pipeline.kernel(&rules),
            Some(SuperResKernel::Fsr1(Fsr1Params { sharpness: 1.0 }))
        );
    }

    #[test]
    fn pipeline_disabled_by_mode_off() {
        let pipeline = SuperResolutionPipeline::new();
        let rules = Upscaling {
            mode: Some(UpscalingMode::Off),
            algorithm: Some(UpscalingAlgorithm::Bicubic),
            sharpness: None,
            client_scale: None,
            virtual_size: None,
        };
        assert_eq!(pipeline.kernel(&rules), None);
    }
    // The kernels are verified on a real GL context below. These tests need an EGL device, so
    // they return early when there is not one (for example on a machine with no GPU access); the
    // parsing and decision tests above do not need one.

    use smithay::backend::allocator::Fourcc;
    use smithay::backend::egl::native::EGLSurfacelessDisplay;
    use smithay::backend::egl::{EGLContext, EGLDisplay};
    use smithay::backend::renderer::gles::GlesTexture;
    use smithay::backend::renderer::{Color32F, ExportMem as _, ImportMem as _};
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
                let programs = SuperResPrograms::get_from_frame(&frame)
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

    #[test]
    fn nothing_to_magnify_when_the_buffer_matches_the_area() {
        let area = Size::<i32, Physical>::from((2560, 1440));

        // A client that renders at the physical resolution of the output, and one that lands a
        // pixel short of it, leave the effect nothing to recover.
        assert!(!has_something_to_magnify(Size::from((2560., 1440.)), area));
        assert!(!has_something_to_magnify(Size::from((2559., 1439.)), area));

        // Two pixels short is real magnification, as is a buffer that is only short in one axis.
        assert!(has_something_to_magnify(Size::from((2558., 1438.)), area));
        assert!(has_something_to_magnify(Size::from((1920., 1440.)), area));
        assert!(has_something_to_magnify(Size::from((2560., 1080.)), area));

        // A big window in a small area is not this mode's business, but it is not a reason to skip
        // the effect either: the window rule enabled it.
        assert!(!has_something_to_magnify(Size::from((3840., 2160.)), area));
    }

    #[test]
    fn magnify_contain_fits_and_centers() {
        let source = Size::from((1280., 720.));

        // A 16:9 content in a 16:9 area fits exactly, no letterbox.
        let m = MagnifiedGeometry::new(source, Size::from((2560., 1440.))).unwrap();
        assert_eq!(m.scale, 2.);
        assert_eq!(m.content.loc, Point::from((0., 0.)));
        assert_eq!(m.content.size, Size::from((2560., 1440.)));

        // A 16:9 content in a 4:3 area gets bars above and below.
        let m = MagnifiedGeometry::new(source, Size::from((1920., 1440.))).unwrap();
        assert_eq!(m.scale, 1.5);
        assert_eq!(m.content.loc, Point::from((0., 180.)));
        assert_eq!(m.content.size, Size::from((1920., 1080.)));

        // A 4:3 content in a 16:9 area gets bars on the sides.
        let m =
            MagnifiedGeometry::new(Size::from((1280., 960.)), Size::from((2560., 1440.))).unwrap();
        assert_eq!(m.scale, 1.5);
        assert_eq!(m.content.loc, Point::from((320., 0.)));
        assert_eq!(m.content.size, Size::from((1920., 1440.)));
    }

    #[test]
    fn magnify_nothing_to_do() {
        // Same size: the window can just be drawn normally.
        assert_eq!(
            MagnifiedGeometry::new(Size::from((100., 100.)), Size::from((100., 100.))),
            None
        );

        // Bigger content is not magnified, it would be a shrink.
        assert_eq!(
            MagnifiedGeometry::new(Size::from((200., 200.)), Size::from((100., 100.))),
            None
        );

        // Degenerate sizes.
        assert_eq!(
            MagnifiedGeometry::new(Size::from((0., 100.)), Size::from((100., 100.))),
            None
        );
        assert_eq!(
            MagnifiedGeometry::new(Size::from((100., 100.)), Size::from((100., 0.))),
            None
        );
    }

    #[test]
    fn magnify_maps_points_both_ways() {
        let m =
            MagnifiedGeometry::new(Size::from((1280., 720.)), Size::from((1920., 1440.))).unwrap();

        // The first content pixel maps to the first pixel of the content area.
        assert_eq!(
            m.to_source(Point::from((0., 180.))),
            Some(Point::from((0., 0.)))
        );
        // And the last one to the last content pixel.
        let last = m.to_source(Point::from((1919.5, 1259.5))).unwrap();
        assert!(
            (last.x - 1279.667).abs() < 1e-3 && (last.y - 719.667).abs() < 1e-3,
            "{last:?}"
        );

        // The letterbox bars are not part of the window.
        assert_eq!(m.to_source(Point::from((100., 0.))), None);
        assert_eq!(m.to_source(Point::from((100., 1439.))), None);
        assert_eq!(m.to_source(Point::from((-1., 500.))), None);
        assert_eq!(m.to_source(Point::from((500., 1440.))), None);

        // Drawing maps the content's own rect onto the whole content area.
        assert_eq!(
            m.to_target(Rectangle::from_size(Size::from((1280., 720.)))),
            m.content
        );

        // A point inside the content maps back to where it came from.
        let point = Point::from((700., 900.));
        let source = m.to_source(point).unwrap();
        let back = m
            .to_target(Rectangle::new(source, Size::from((0., 0.))))
            .loc;
        assert!((back.x - point.x).abs() < 1e-6 && (back.y - point.y).abs() < 1e-6);
    }

    #[test]
    fn magnify_maps_damage() {
        let inner = Rectangle::new(Point::from((10, 20)), Size::from((100, 50)));
        let dest = Rectangle::new(Point::from((0, 0)), Size::from((200, 100)));

        // Doubling, and the rects are relative to the element, so half of it maps onto all of the
        // destination. The inner element's own location does not take part: it cancels out,
        // because both the damage and the result are relative to their element.
        assert_eq!(
            magnify_physical_rect(
                Rectangle::new(Point::from((0, 0)), Size::from((50, 25))),
                inner,
                dest,
                true,
                0,
            ),
            Rectangle::new(Point::from((0, 0)), Size::from((100, 50)))
        );

        // The bottom-right corner maps onto the destination's, without running past it.
        assert_eq!(
            magnify_physical_rect(
                Rectangle::new(Point::from((50, 25)), Size::from((50, 25))),
                inner,
                dest,
                true,
                0,
            ),
            Rectangle::new(Point::from((100, 50)), Size::from((100, 50)))
        );

        // Fractional scaling rounds outwards for damage, so that nothing that changed is left
        // out, and is grown by the kernel's footprint.
        let inner = Rectangle::from_size(Size::from((100, 100)));
        let dest = Rectangle::from_size(Size::from((214, 214)));
        assert_eq!(
            magnify_physical_rect(
                Rectangle::new(Point::from((10, 10)), Size::from((10, 10))),
                inner,
                dest,
                true,
                5,
            ),
            Rectangle::new(Point::from((16, 16)), Size::from((32, 32)))
        );

        // Opaque regions round inwards instead, so that we never claim to cover something we do
        // not actually cover.
        assert_eq!(
            magnify_physical_rect(
                Rectangle::new(Point::from((10, 10)), Size::from((10, 10))),
                inner,
                dest,
                false,
                0,
            ),
            Rectangle::new(Point::from((22, 22)), Size::from((20, 20)))
        );

        // A degenerate inner element has nothing to scale by: the whole area is affected.
        assert_eq!(
            magnify_physical_rect(
                Rectangle::new(Point::from((3, 4)), Size::from((5, 6))),
                Rectangle::from_size(Size::from((0, 0))),
                dest,
                true,
                5,
            ),
            Rectangle::from_size(dest.size)
        );
    }

    /// A grayscale pattern with hard edges and gradients.
    ///
    /// All four channels carry the same luminance, which makes the FSR1 checks below independent
    /// of the channel order that the framebuffer happens to use. The color path is the same code
    /// with per-channel weights, since EASU and RCAS only use the channel values through the
    /// weighted average and the limiter rings.
    fn grayscale_pixels(size: Size<i32, Buffer>) -> Vec<u8> {
        let mut pixels = Vec::new();
        for y in 0..size.h {
            for x in 0..size.w {
                // A diagonal edge plus a gradient, so that both the edge-adaptive weights and the
                // flat parts of the kernel are used.
                let value = if x + y < size.w {
                    40 + x * 8
                } else {
                    200 - y * 4
                };
                let value = value.clamp(0, 255) as u8;
                pixels.extend_from_slice(&[value, value, value, value]);
            }
        }
        pixels
    }

    /// CPU transcriptions of the EASU and RCAS kernels, used to check the shaders.
    ///
    /// These are written from the same reference code as the shaders, so they catch transcription
    /// and coordinate mistakes rather than changes to the algorithm.
    mod reference {
        /// Fetches the alpha of one texel with edge clamping.
        fn texel_alpha(src: &[u8], size: (i32, i32), x: i32, y: i32) -> f64 {
            let x = x.clamp(0, size.0 - 1);
            let y = y.clamp(0, size.1 - 1);
            f64::from(src[((y * size.0 + x) * 4) as usize + 3]) / 255.0
        }

        /// Fetches one texel with edge clamping, like the shaders' tap functions.
        fn texel(src: &[u8], size: (i32, i32), x: i32, y: i32) -> f64 {
            let x = x.clamp(0, size.0 - 1);
            let y = y.clamp(0, size.1 - 1);
            f64::from(src[((y * size.0 + x) * 4) as usize]) / 255.0
        }

        /// Luma times 2, as in the reference.
        fn luma(src: &[u8], size: (i32, i32), x: i32, y: i32) -> f64 {
            let c = texel(src, size, x, y);
            c * 0.5 + (c * 0.5 + c)
        }

        /// Accumulates one weighted tap, with anisotropic windowing and a negative lobe.
        #[allow(clippy::too_many_arguments)]
        fn tap(
            a_c: &mut f64,
            a_w: &mut f64,
            off: (f64, f64),
            dir: (f64, f64),
            len: (f64, f64),
            lob: f64,
            clp: f64,
            c: f64,
        ) {
            let v = (
                off.0 * dir.0 + off.1 * dir.1,
                off.0 * -dir.1 + off.1 * dir.0,
            );
            let v = (v.0 * len.0, v.1 * len.1);
            let d2 = (v.0 * v.0 + v.1 * v.1).min(clp);
            let mut wb = (2.0 / 5.0) * d2 - 1.0;
            let mut wa = lob * d2 - 1.0;
            wb *= wb;
            wa *= wa;
            wb = (25.0 / 16.0) * wb - (25.0 / 16.0 - 1.0);
            let w = wb * wa;
            *a_c += c * w;
            *a_w += w;
        }

        /// Accumulates direction and length for one quadrant.
        fn set(w: f64, l: [f64; 5]) -> (f64, f64, f64) {
            let [la, lb, lc, ld, le] = l;
            let mut dir = (0.0, 0.0);
            let mut len = 0.0;

            let dc = ld - lc;
            let cb = lc - lb;
            let len_x_mag = dc.abs().max(cb.abs());
            let dir_x = ld - lb;
            dir.0 += dir_x * w;
            let len_x = if len_x_mag > 0.0 {
                (dir_x.abs() / len_x_mag).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let len_x = len_x * len_x;
            len += len_x * w;

            let ec = le - lc;
            let ca = lc - la;
            let len_y_mag = ec.abs().max(ca.abs());
            let dir_y = le - la;
            dir.1 += dir_y * w;
            let len_y = if len_y_mag > 0.0 {
                (dir_y.abs() / len_y_mag).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let len_y = len_y * len_y;
            len += len_y * w;

            (dir.0, dir.1, len)
        }

        /// EASU for one output pixel: the luminance value and the resampled alpha.
        pub fn easu(
            src: &[u8],
            src_size: (i32, i32),
            dest_size: (i32, i32),
            x: i32,
            y: i32,
        ) -> (f64, f64) {
            let ratio = (
                f64::from(src_size.0) / f64::from(dest_size.0),
                f64::from(src_size.1) / f64::from(dest_size.1),
            );
            let pos = (
                (f64::from(x) + 0.5) * ratio.0 - 0.5,
                (f64::from(y) + 0.5) * ratio.1 - 0.5,
            );
            let fp = (pos.0.floor(), pos.1.floor());
            let pp = (pos.0 - fp.0, pos.1 - fp.1);
            let fp = (fp.0 as i32, fp.1 as i32);

            let t = |dx: i32, dy: i32| texel(src, src_size, fp.0 + dx, fp.1 + dy);
            let l = |dx: i32, dy: i32| luma(src, src_size, fp.0 + dx, fp.1 + dy);

            let (b, c, e, f, g, h, i, j, k, ll, n, o) = (
                l(0, -1),
                l(1, -1),
                l(-1, 0),
                l(0, 0),
                l(1, 0),
                l(2, 0),
                l(-1, 1),
                l(0, 1),
                l(1, 1),
                l(2, 1),
                l(0, 2),
                l(1, 2),
            );

            let w_s = (1.0 - pp.0) * (1.0 - pp.1);
            let w_t = pp.0 * (1.0 - pp.1);
            let w_u = (1.0 - pp.0) * pp.1;
            let w_v = pp.0 * pp.1;

            let (mut dx, mut dy, mut len) = (0.0, 0.0, 0.0);
            for (w, quad) in [
                (w_s, [b, e, f, g, j]),
                (w_t, [c, f, g, h, k]),
                (w_u, [f, i, j, k, n]),
                (w_v, [g, j, k, ll, o]),
            ] {
                let (x, y, l) = set(w, quad);
                dx += x;
                dy += y;
                len += l;
            }

            let dir2 = dx * dx + dy * dy;
            let zro = dir2 < (1.0 / 32768.0);
            let mut dir_r = 1.0 / dir2.sqrt();
            dir_r = if zro { 1.0 } else { dir_r };
            if zro {
                dx = 1.0;
            }
            dx *= dir_r;
            dy *= dir_r;

            let mut len = len * 0.5;
            len *= len;

            let stretch = (dx * dx + dy * dy) / dx.abs().max(dy.abs());
            let len2 = (1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
            let lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
            let clp = 1.0 / lob;

            let min4 = t(0, 0).min(t(1, 0)).min(t(0, 1)).min(t(1, 1));
            let max4 = t(0, 0).max(t(1, 0)).max(t(0, 1)).max(t(1, 1));

            let mut a_c = 0.0;
            let mut a_w = 0.0;
            for (off, color) in [
                ((0.0, -1.0), t(0, -1)),
                ((1.0, -1.0), t(1, -1)),
                ((-1.0, 1.0), t(-1, 1)),
                ((0.0, 1.0), t(0, 1)),
                ((0.0, 0.0), t(0, 0)),
                ((-1.0, 0.0), t(-1, 0)),
                ((1.0, 1.0), t(1, 1)),
                ((2.0, 1.0), t(2, 1)),
                ((2.0, 0.0), t(2, 0)),
                ((1.0, 0.0), t(1, 0)),
                ((1.0, 2.0), t(1, 2)),
                ((0.0, 2.0), t(0, 2)),
            ] {
                tap(
                    &mut a_c,
                    &mut a_w,
                    (off.0 - pp.0, off.1 - pp.1),
                    (dx, dy),
                    len2,
                    lob,
                    clp,
                    color,
                );
            }

            let color = (a_c / a_w).clamp(min4, max4);

            let alpha = {
                let f = texel(src, src_size, fp.0, fp.1);
                let g = texel(src, src_size, fp.0 + 1, fp.1);
                let j = texel(src, src_size, fp.0, fp.1 + 1);
                let k = texel(src, src_size, fp.0 + 1, fp.1 + 1);
                let lower = f + (g - f) * pp.0;
                let upper = j + (k - j) * pp.0;
                lower + (upper - lower) * pp.1
            };

            (color.min(alpha), alpha)
        }

        /// RCAS for one pixel of the sharpened image.
        ///
        /// The shader also clamps the sharpened color to the resampled alpha, which keeps the
        /// result a valid premultiplied color. That is not part of the reference algorithm, so it
        /// has to be mirrored here.
        pub fn rcas(src: &[u8], size: (i32, i32), sharpness_scale: f64, x: i32, y: i32) -> f64 {
            let t = |dx: i32, dy: i32| texel(src, size, x + dx, y + dy);
            let l = |dx: i32, dy: i32| luma(src, size, x + dx, y + dy);

            let (b, d, e, f, h) = (t(0, -1), t(-1, 0), t(0, 0), t(1, 0), t(0, 1));
            let (bl, dl, el, fl, hl) = (l(0, -1), l(-1, 0), l(0, 0), l(1, 0), l(0, 1));

            let mn4 = b.min(d).min(f).min(h);
            let mx4 = b.max(d).max(f).max(h);

            let peak = (1.0, -4.0);
            let hit_min = mn4.min(e) / (4.0 * mx4);
            let hit_max = (peak.0 - mx4.max(e)) / (4.0 * mn4 + peak.1);
            let lobe = (-(0.25_f64 - 1.0 / 16.0)).max(((-hit_min).max(hit_max)).min(0.0))
                * sharpness_scale;

            let rcp_l = 1.0 / (4.0 * lobe + 1.0);
            let _ = (bl, dl, el, fl, hl);
            let color = (lobe * b + lobe * d + lobe * h + lobe * f + e) * rcp_l;
            color.min(texel_alpha(src, size, x, y))
        }
    }

    /// Renders a texture through the FSR1 pass and checks both stages against the reference.
    #[test]
    fn gpu_fsr1_matches_the_reference() {
        let Some(mut renderer) = surfaceless_renderer() else {
            return;
        };

        crate::render_helpers::resources::init(&mut renderer);
        crate::render_helpers::shaders::init(&mut renderer);
        init(&mut renderer, None);

        let src_size = Size::from((8, 8));
        let src_physical = Size::<i32, Physical>::from((8, 8));
        let pixels = grayscale_pixels(src_size);
        let mut texture = renderer
            .import_memory(&pixels, Fourcc::Abgr8888, src_size, false)
            .unwrap();
        let source = read_back(&mut renderer, &mut texture, src_physical);
        assert_eq!(source, pixels, "the harness must not change the source");

        let src_rect = Rectangle::from_size(src_size.to_f64());
        let dest = Size::<i32, Physical>::from((16, 16));

        // EASU: the client buffer into an offscreen texture at the display size.
        let programs = SuperResPrograms::get(&renderer).unwrap();
        let easu = programs.easu.clone().expect("the EASU shader must compile");
        let rcas = programs.rcas.clone().expect("the RCAS shader must compile");
        let uniforms = resample_uniforms(&texture, src_rect, false).unwrap();

        let mut upsampled = run_easu_pass(
            &mut renderer,
            &texture,
            src_rect,
            dest,
            Transform::Normal,
            &easu,
            &uniforms,
        )
        .unwrap();
        let upsampled_pixels = read_back(&mut renderer, &mut upsampled, dest);

        for y in 0..dest.h {
            for x in 0..dest.w {
                let (want, want_alpha) =
                    reference::easu(&source, (src_size.w, src_size.h), (dest.w, dest.h), x, y);
                for channel in 0..4 {
                    let index = ((y * dest.w + x) * 4) as usize + channel;
                    let got = f64::from(upsampled_pixels[index]) / 255.0;
                    let want = if channel == 3 { want_alpha } else { want };
                    assert!(
                        (got - want).abs() <= 0.01,
                        "EASU at ({x}, {y}) channel {channel}: {got} != {want}",
                    );
                }
            }
        }

        // RCAS: the EASU result, sharpened 1:1.
        let sharpness_scale = 0.5_f32;
        let buffer_size = dest.to_logical(1).to_buffer(1, Transform::Normal);
        let mut sharpened = renderer
            .create_buffer(Fourcc::Abgr8888, buffer_size)
            .unwrap();
        {
            let rect = Rectangle::from_size(dest);
            let mut target = renderer.bind(&mut sharpened).unwrap();
            let mut frame = renderer
                .render(&mut target, dest, Transform::Normal)
                .unwrap();
            frame.clear(Color32F::TRANSPARENT, &[rect]).unwrap();
            frame
                .render_texture_from_to(
                    &upsampled,
                    Rectangle::from_size(buffer_size.to_f64()),
                    rect,
                    &[rect],
                    &[],
                    Transform::Normal,
                    1.0,
                    Some(&rcas),
                    &[
                        Uniform::new("niri_tex_size", (dest.w as f32, dest.h as f32)),
                        Uniform::new("niri_src_rect", [0f32, 0., dest.w as f32, dest.h as f32]),
                        Uniform::new("niri_rcas_sharpness", sharpness_scale),
                    ],
                )
                .unwrap();
            let _ = frame.finish().unwrap();
        }
        let sharpened_pixels = read_back(&mut renderer, &mut sharpened, dest);

        for y in 0..dest.h {
            for x in 0..dest.w {
                let want = reference::rcas(
                    &upsampled_pixels,
                    (dest.w, dest.h),
                    f64::from(sharpness_scale),
                    x,
                    y,
                );
                for channel in 0..3 {
                    let index = ((y * dest.w + x) * 4) as usize + channel;
                    let got = f64::from(sharpened_pixels[index]) / 255.0;
                    assert!(
                        (got - want).abs() <= 0.01,
                        "RCAS at ({x}, {y}) channel {channel}: {got} != {want}",
                    );
                }
            }
        }
    }
}
