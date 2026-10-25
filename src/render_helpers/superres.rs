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
use std::collections::{HashMap, VecDeque};
use std::ffi::CStr;
use std::fmt::{self, Write as _};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
    Renderer, Texture, TextureFilter,
};
use smithay::utils::user_data::UserDataMap;
use smithay::utils::{Buffer, Logical, Physical, Point, Rectangle, Scale, Size, Transform};

use crate::backend::tty::{native_texture, TtyFrame, TtyRenderer, TtyRendererError};
use crate::render_helpers::renderer::{
    gpu_diagnostics, AsGlesFrame as _, ClientBufferDebug, GpuDebug, NiriRenderer, TextureDebug,
    GPU_LOG_TARGET,
};

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

/// The mapping between a magnified window's content and the area it is displayed in.
///
/// This is what the `magnify` mode does: the client keeps its own size, and niri draws its content
/// scaled up to fill the area that the layout asked for. Drawing and input have to agree on where
/// the content ends up, so both go through this.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MagnifiedGeometry {
    /// The client's own logical size, that is, its content as the client thinks of it.
    pub source: Size<f64, Logical>,
    /// The area that the window occupies on the output.
    pub target: Size<f64, Logical>,
    /// Where the content is drawn inside the target, centered.
    pub content: Rectangle<f64, Logical>,
    /// How much the content is scaled up.
    pub scale: f64,
}

impl MagnifiedGeometry {
    /// Returns the geometry for a content of `source` size magnified into `target`.
    ///
    /// Returns `None` when there is nothing to magnify, in which case the window should be drawn
    /// at its own size like any other window.
    pub fn new(source: Size<f64, Logical>, target: Size<f64, Logical>) -> Option<Self> {
        if source.w <= 0. || source.h <= 0. || target.w <= 0. || target.h <= 0. {
            return None;
        }

        // Only magnify, never shrink: a window that is bigger than the area it was given is not
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

    /// A CuNNy network, whose passes run offscreen and whose residuals the final draw applies.
    Cunny(CunnyVariant),
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
            UpscalingAlgorithm::Cunny2 => (Self::Cunny(CunnyVariant::Fast), false),
            UpscalingAlgorithm::Cunny2Large => (Self::Cunny(CunnyVariant::ThreeByTwelve), false),
            UpscalingAlgorithm::Anime4k => (Self::Filter(ResampleFilter::Bicubic), true),
        }
    }
}

impl fmt::Display for SuperResKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Filter(ResampleFilter::Bilinear) => f.write_str("bilinear"),
            Self::Filter(ResampleFilter::Bicubic) => f.write_str("bicubic"),
            Self::Fsr1(params) => write!(f, "fsr1 (sharpness {})", params.sharpness),
            Self::Cunny(CunnyVariant::Fast) => f.write_str("cunny2 (fast-NVL)"),
            Self::Cunny(CunnyVariant::ThreeByTwelve) => f.write_str("cunny2 (3x12-NVL)"),
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
        // Nested/headless contexts may also expose their actual render device through EGL.
        let egl_render_node =
            smithay::backend::egl::EGLDevice::device_for_display(renderer.egl_context().display())
                .ok()
                .and_then(|device| device.render_device_path().ok());
        let render_node = egl_render_node.as_deref().or(render_node);

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
            timer_query: (has_extension("GL_EXT_disjoint_timer_query")
                || has_extension("GL_EXT_disjoint_timer_query_ext"))
                && can_time(gl),
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
    digest: Option<GlesTexProgram>,
    easu: Option<GlesTexProgram>,
    rcas: Option<GlesTexProgram>,
    /// The CuNNy passes and the upscale, compiled the first time they are needed: there are more
    /// than twenty of them, and most sessions never run the effect at all.
    cunny: RefCell<HashMap<u32, GlesTexProgram>>,
    cunny_upscale: Option<GlesTexProgram>,
}

impl SuperResPrograms {
    fn compile(renderer: &mut GlesRenderer) -> Self {
        Self {
            bilinear: Self::compile_filter(renderer, ResampleFilter::Bilinear),
            bicubic: Self::compile_filter(renderer, ResampleFilter::Bicubic),
            digest: Self::compile_digest(renderer),
            easu: Self::compile_file(
                renderer,
                "easu",
                include_str!("shaders/superres/fsr1_easu.frag"),
                &[
                    UniformName::new("niri_tex_size", UniformType::_2f),
                    UniformName::new("niri_src_rect", UniformType::_4f),
                ],
            ),
            cunny: RefCell::new(HashMap::new()),
            cunny_upscale: Some(include_str!("shaders/superres/cunny_upscale.frag")).and_then(
                |source| {
                    Self::compile_file(renderer, "cunny upscale", source, &cunny_upscale_uniforms())
                },
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

    fn compile_digest(renderer: &mut GlesRenderer) -> Option<GlesTexProgram> {
        let source = include_str!("shaders/superres/resample.frag").replace(
            "// niri:resample-filter",
            &format!("#define NIRI_RESAMPLE_FILTER 2\n#define NIRI_DIGEST_SIDE {DIGEST_SIDE}.0"),
        );

        Self::compile_file(
            renderer,
            "digest",
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
                warn!("super resolution: error compiling the {name} shader: {err:?}");
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
        warn!("super resolution: error probing the capabilities");
        return;
    };

    info!("super resolution: {}", capabilities.summary());
    debug!("super resolution capabilities: {capabilities:#?}");

    let programs = SuperResPrograms::compile(renderer);

    let data = renderer.egl_context().user_data();
    if !data.insert_if_missing(|| capabilities) {
        error!("super resolution: the capabilities were already probed");
    }
    if !data.insert_if_missing(|| programs) {
        error!("super resolution: the programs were already compiled");
    }
    debug!(
        target: GPU_LOG_TARGET,
        gpu = ?GpuDebug::renderer(renderer),
        "initialized super-resolution execution context; texture IDs are context-local",
    );
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
    /// Measures how long the passes take, when anything reads the numbers.
    timings: Arc<Timings>,
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

    /// A CuNNy network's residuals, which the final draw applies to the client buffer.
    Cunny {
        /// The residuals, at the source's resolution, one source texel per texel.
        residuals: GlesTexture,
        /// The renderer context that the texture belongs to.
        context_id: ContextId<GlesTexture>,
    },
}

impl ElementKernel {
    fn draw_pass(&self) -> &'static str {
        match self {
            Self::Plain => "renderer-default",
            Self::Filter(ResampleFilter::Bilinear) => "bilinear",
            Self::Filter(ResampleFilter::Bicubic) => "bicubic",
            Self::Fsr1 { .. } => "rcas",
            Self::Cunny { .. } => "cunny-upscale",
        }
    }
}

impl<R: Renderer> SuperResElement<R> {
    /// Creates an element that draws the client buffer with a resampling filter.
    pub fn new(inner: WaylandSurfaceRenderElement<R>, filter: ResampleFilter) -> Self {
        Self {
            inner,
            kernel: ElementKernel::Filter(filter),
            magnify: None,
            timings: Timings::disabled(),
        }
    }

    /// Creates an element that draws the client buffer with the renderer's own sampling.
    pub fn plain(inner: WaylandSurfaceRenderElement<R>) -> Self {
        Self {
            inner,
            kernel: ElementKernel::Plain,
            magnify: None,
            timings: Timings::disabled(),
        }
    }

    /// Draws the content into `rect` rather than at the element's own geometry.
    pub fn with_magnify(mut self, rect: Rectangle<f64, Logical>) -> Self {
        self.magnify = Some(rect);
        self
    }

    /// Measures how long drawing this element takes on the GPU.
    pub fn with_timings(mut self, timings: Arc<Timings>) -> Self {
        self.timings = timings;
        self
    }

    /// Creates an element that draws a CuNNy run's residuals over the client buffer.
    pub fn cunny(
        inner: WaylandSurfaceRenderElement<R>,
        residuals: GlesTexture,
        context_id: ContextId<GlesTexture>,
    ) -> Self {
        Self {
            inner,
            kernel: ElementKernel::Cunny {
                residuals,
                context_id,
            },
            magnify: None,
            timings: Timings::disabled(),
        }
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
            timings: Timings::disabled(),
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
    /// Logs the selected draw path after issuing commands, without querying/waiting for the GPU.
    fn log_draw(
        &self,
        frame: &GlesFrame<'_, '_>,
        executed_pass: &'static str,
        dst: Rectangle<i32, Physical>,
        success: bool,
    ) where
        R::TextureId: PassSource,
    {
        if !gpu_diagnostics() {
            return;
        }
        let native = match self.inner.texture() {
            WaylandSurfaceTexture::Texture(texture) => texture.pass_texture(&frame.context_id()),
            WaylandSurfaceTexture::SolidColor(_) => None,
        };
        let cached = match &self.kernel {
            ElementKernel::Fsr1 {
                texture,
                context_id,
                ..
            } => Some((context_id, TextureDebug(texture))),
            ElementKernel::Cunny {
                residuals,
                context_id,
            } => Some((context_id, TextureDebug(residuals))),
            _ => None,
        };
        trace!(
            target: GPU_LOG_TARGET,
            element = ?self.inner.id(),
            commit = ?self.inner.current_commit(),
            gpu = ?GpuDebug::frame(frame),
            selected_pass = self.kernel.draw_pass(),
            executed_pass,
            draw_fallback = executed_pass != self.kernel.draw_pass(),
            success,
            native_source = ?native.as_ref().map(TextureDebug),
            cached_context_and_texture = ?cached,
            ?dst,
            "final draw commands issued (not a GPU completion notification)",
        );
    }

    /// Returns the program that applies the residuals, with the uniforms and the texture binding
    /// it needs.
    ///
    /// It reads the client buffer as `tex` and the residuals from the texture unit after it, which
    /// is what a texture program has to do to reach more than one texture.
    fn cunny_program(
        &self,
        frame: &mut GlesFrame<'_, '_>,
        residuals: &GlesTexture,
        context_id: ContextId<GlesTexture>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
    ) -> Option<(GlesTexProgram, Vec<Uniform<'static>>)> {
        if frame.context_id() != context_id {
            // This happens when the same window is drawn on another GPU, where the cached result is
            // not valid. It is not an error, we just cannot use the cache here.
            trace!("super resolution: not applying residuals from another renderer context");
            debug!(
                target: GPU_LOG_TARGET,
                gpu = ?GpuDebug::frame(frame),
                cached_context = ?context_id,
                cached_texture = ?TextureDebug(residuals),
                reason = "different texture-compatible context",
                "CuNNy upscale unavailable; renderer-default fallback",
            );
            return None;
        }

        let program = SuperResPrograms::get_from_frame(frame)?
            .cunny_upscale
            .clone()?;
        let mut uniforms = self.resample_uniforms(src)?.to_vec();
        uniforms.push(Uniform::new("niri_tex1", 1i32));
        // The doubled image's size against the destination's, which is what says whether the
        // destination magnifies it (one sample per pixel) or minifies it (average over the pixels
        // each one covers).
        uniforms.push(Uniform::new(
            "niri_footprint",
            [
                2.0 * (src.size.w / f64::from(dst.size.w.max(1))) as f32,
                2.0 * (src.size.h / f64::from(dst.size.h.max(1))) as f32,
            ],
        ));

        let _ = frame.with_context(|gl| unsafe {
            gl.ActiveTexture(ffi::TEXTURE1);
            configure_pass_texture(gl, residuals.tex_id(), ffi::NEAREST);
            gl.ActiveTexture(ffi::TEXTURE0);
        });
        Some((program, uniforms))
    }

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

        let Some(program) =
            SuperResPrograms::get_from_frame(frame).and_then(|programs| programs.program(filter))
        else {
            debug!(target: GPU_LOG_TARGET, gpu = ?GpuDebug::frame(frame), ?filter, reason = "resampling shader unavailable in execution context", "renderer-default fallback");
            return None;
        };
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
        trace!("super resolution: not drawing a result from another renderer context");
        debug!(
            target: GPU_LOG_TARGET,
            gpu = ?GpuDebug::frame(frame),
            cached_context = ?context_id,
            cached_texture = ?TextureDebug(texture),
            reason = "different texture-compatible context",
            "RCAS unavailable; renderer-default fallback",
        );
        return None;
    }

    let size = texture.size();
    if size.w <= 0 || size.h <= 0 {
        return None;
    }

    let Some(program) =
        SuperResPrograms::get_from_frame(frame).and_then(|programs| programs.rcas.clone())
    else {
        debug!(target: GPU_LOG_TARGET, gpu = ?GpuDebug::frame(frame), reason = "RCAS shader unavailable in execution context", "renderer-default fallback");
        return None;
    };
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
        // The element draws with the renderer's own sampling, so there is nothing to measure.
        if matches!(self.kernel, ElementKernel::Plain) {
            let res = draw_inner_gles(&self.inner, frame, src, dst, damage, opaque_regions, cache);
            self.log_draw(frame, "renderer-default", dst, res.is_ok());
            return res;
        }

        self.timings.begin(frame, Pass::Draw);
        let mut executed_pass = "renderer-default";

        let res = match &self.kernel {
            ElementKernel::Plain => unreachable!(),
            ElementKernel::Filter(_) => match self.resample_program(frame, src) {
                // Fall back to the renderer's own sampling.
                None => {
                    draw_inner_gles(&self.inner, frame, src, dst, damage, opaque_regions, cache)
                }
                Some((program, uniforms)) => {
                    executed_pass = self.kernel.draw_pass();
                    frame.override_default_tex_program(program, uniforms);
                    let res = draw_inner_gles(
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
            },
            ElementKernel::Cunny {
                residuals,
                context_id,
            } => match self.cunny_program(frame, residuals, context_id.clone(), src, dst) {
                Some((program, uniforms)) => {
                    executed_pass = self.kernel.draw_pass();
                    frame.override_default_tex_program(program, uniforms);
                    let res = draw_inner_gles(
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
                // Fall back to the renderer's own sampling.
                None => {
                    draw_inner_gles(&self.inner, frame, src, dst, damage, opaque_regions, cache)
                }
            },
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
                Some(res) => {
                    executed_pass = "rcas";
                    res
                }
                None => {
                    draw_inner_gles(&self.inner, frame, src, dst, damage, opaque_regions, cache)
                }
            },
        };

        self.timings.end(frame);
        self.log_draw(frame, executed_pass, dst, res.is_ok());
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
        // Vulkan 渲染器上没有 GLES 帧，超分管线不可用，退回渲染器自带采样。
        if frame.as_gles_frame().is_none() {
            return RenderElement::<TtyRenderer<'render>>::draw(
                &self.inner,
                frame,
                src,
                dst,
                damage,
                opaque_regions,
                cache,
            );
        }

        // The element draws with the renderer's own sampling, so there is nothing to measure.
        if matches!(self.kernel, ElementKernel::Plain) {
            let res = RenderElement::<TtyRenderer<'render>>::draw(
                &self.inner,
                frame,
                src,
                dst,
                damage,
                opaque_regions,
                cache,
            );
            self.log_draw(frame.as_gles_frame().expect("gles frame checked above"), "renderer-default", dst, res.is_ok());
            return res;
        }

        self.timings.begin(frame.as_gles_frame().expect("gles frame checked above"), Pass::Draw);
        let mut executed_pass = "renderer-default";

        let res = match &self.kernel {
            ElementKernel::Plain => unreachable!(),
            ElementKernel::Filter(_) => {
                match self.resample_program(frame.as_gles_frame().expect("gles frame checked above"), src) {
                    // Fall back to the renderer's own sampling.
                    None => RenderElement::<TtyRenderer<'render>>::draw(
                        &self.inner,
                        frame,
                        src,
                        dst,
                        damage,
                        opaque_regions,
                        cache,
                    ),
                    Some((program, uniforms)) => {
                        executed_pass = self.kernel.draw_pass();
                        frame
                            .as_gles_frame()
                            .expect("gles frame checked above")
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
                        frame.as_gles_frame().expect("gles frame checked above").clear_tex_program_override();
                        res
                    }
                }
            }
            ElementKernel::Cunny {
                residuals,
                context_id,
            } => {
                match self.cunny_program(
                    frame.as_gles_frame().expect("gles frame checked above"),
                    residuals,
                    context_id.clone(),
                    src,
                    dst,
                ) {
                    Some((program, uniforms)) => {
                        executed_pass = self.kernel.draw_pass();
                        frame
                            .as_gles_frame()
                            .expect("gles frame checked above")
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
                        frame.as_gles_frame().expect("gles frame checked above").clear_tex_program_override();
                        res
                    }
                    // Fall back to the renderer's own sampling.
                    None => RenderElement::<TtyRenderer<'render>>::draw(
                        &self.inner,
                        frame,
                        src,
                        dst,
                        damage,
                        opaque_regions,
                        cache,
                    ),
                }
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
                match draw_fsr1(
                    frame.as_gles_frame().expect("gles frame checked above"),
                    texture,
                    context_id.clone(),
                    *sharpness_scale,
                    dst,
                    damage,
                    opaque_regions,
                    self.inner.alpha(),
                ) {
                    Some(res) => {
                        executed_pass = "rcas";
                        res.map_err(|err| {
                            TtyRendererError::Gles(
                                smithay::backend::renderer::multigpu::Error::Render(err),
                            )
                        })
                    }
                    None => RenderElement::<TtyRenderer<'render>>::draw(
                        &self.inner,
                        frame,
                        src,
                        dst,
                        damage,
                        opaque_regions,
                        cache,
                    ),
                }
            }
        };

        self.timings.end(frame.as_gles_frame().expect("gles frame checked above"));
        self.log_draw(frame.as_gles_frame().expect("gles frame checked above"), executed_pass, dst, res.is_ok());
        res
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

impl PassSource for smithay::backend::renderer::vulkan::VulkanTexture {
    fn pass_texture(&self, _context_id: &ContextId<GlesTexture>) -> Option<GlesTexture> {
        // 超分管线只在 GLES 上运行，Vulkan 纹理没有对应的 GLES 上下文纹理。
        None
    }
}

impl PassSource for MultiTexture {
    fn pass_texture(&self, context_id: &ContextId<GlesTexture>) -> Option<GlesTexture> {
        native_texture(self, context_id)
    }
}

/// An offscreen result of the EASU pass.
#[derive(Debug, Clone)]
struct Fsr1Result {
    texture: GlesTexture,
    context_id: ContextId<GlesTexture>,
}

/// Geometry and crop that an EASU result was produced from.
///
/// Resampling a window goes through this pass once per distinct source content, and the final
/// sharpening pass runs on the result every frame, so this only has to change when the content or
/// the size does. Moving the window, drawing other windows or moving the cursor must not
/// invalidate it.
#[derive(Debug, Clone, PartialEq)]
struct Fsr1Key {
    /// Source rect in buffer texels.
    src: Rectangle<f64, Buffer>,
    /// Destination size in physical pixels.
    size: Size<i32, Physical>,
}

/// Side length of the offscreen digest texture used to detect unchanged client frames.
const DIGEST_SIDE: i32 = 32;

/// Number of bytes in a [`SourceDigest`] (`32 * 32 * 4` RGBA8 bytes).
const DIGEST_BYTES: usize = (DIGEST_SIDE as usize) * (DIGEST_SIDE as usize) * 4;

/// Minimum offscreen pass work (total output texels written across all offscreen passes) before
/// computing a [`SourceDigest`] is worthwhile.
///
/// Measured on `Mesa Intel(R) Iris(R) Xe Graphics (ADL GT2)` and `NVIDIA GeForce RTX 3050 Ti`:
/// - Computing and reading back a 32×32 digest takes ~220 µs on Iris Xe and ~81 µs on RTX 3050 Ti,
///   dominated by the single offscreen bind + `glReadPixels` synchronization round-trip.
/// - A single-pass EASU run is faster than the digest below 640×360 destination pixels (~212 µs on
///   Iris Xe, ~121 µs on RTX 3050 Ti) and reaches ~368 µs (Iris Xe) / ~108–205 µs (RTX 3050 Ti) at
///   960×540 (518,400 destination texels).
/// - CuNNy-Fast runs 9 offscreen passes and already takes ~534–701 µs on Iris Xe (~512–524 µs on
///   RTX 3050 Ti) at 320×180 source texels (`9 * 57,600 = 518,400` pass texels), scaling up to
///   ~6.17 ms (Iris Xe) / ~1.52 ms (RTX 3050 Ti) at 1280×720.
///
/// Setting the threshold at 500,000 total pass texels (with at least `32 * 32` source texels)
/// skips the digest for small surfaces where running the pass directly is cheaper or comparable,
/// while enabling it for 960×540+ FSR1 outputs and 320×180+ CuNNy inputs.
const DIGEST_MIN_PASS_TEXELS: u64 = 500_000;
const DIGEST_MIN_SOURCE_TEXELS: u64 = (DIGEST_SIDE as u64) * (DIGEST_SIDE as u64);

/// Compact GPU-computed fingerprint of a client buffer's sampled region.
#[derive(Clone, PartialEq, Eq)]
struct SourceDigest(Box<[u8; DIGEST_BYTES]>);

impl fmt::Debug for SourceDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SourceDigest")
            .field(&format_args!("{DIGEST_BYTES}B"))
            .finish()
    }
}

/// Reusable 32×32 offscreen target for computing [`SourceDigest`]s without reallocating.
#[derive(Debug, Default)]
struct DigestTarget {
    texture: Option<(ContextId<GlesTexture>, GlesTexture)>,
}

impl DigestTarget {
    fn compute(
        &mut self,
        renderer: &mut GlesRenderer,
        source: &GlesTexture,
        src: Rectangle<f64, Buffer>,
        y_inverted: bool,
        transform: Transform,
    ) -> Option<SourceDigest> {
        let program = SuperResPrograms::get(renderer)?.digest.clone()?;
        let uniforms = resample_uniforms(source, src, y_inverted)?;
        let context_id = renderer.context_id();

        let digest_phys = Size::<i32, Physical>::from((DIGEST_SIDE, DIGEST_SIDE));
        let digest_buf = digest_phys.to_logical(1).to_buffer(1, Transform::Normal);

        let need_alloc = self
            .texture
            .as_ref()
            .is_none_or(|(id, _)| *id != context_id);
        if need_alloc {
            let tex = renderer.create_buffer(Fourcc::Abgr8888, digest_buf).ok()?;
            self.texture = Some((context_id, tex));
        }
        let (_, texture) = self.texture.as_mut()?;

        let rect = Rectangle::from_size(digest_phys);
        let mut bytes = Box::new([0u8; DIGEST_BYTES]);

        let mut target = renderer.bind(texture).ok()?;
        let mut frame = renderer
            .render(&mut target, digest_phys, Transform::Normal)
            .ok()?;
        frame
            .render_texture_from_to(
                source,
                src,
                rect,
                &[rect],
                &[rect],
                transform,
                1.0,
                Some(&program),
                &uniforms,
            )
            .ok()?;
        frame
            .with_context(|gl| unsafe {
                gl.PixelStorei(ffi::PACK_ALIGNMENT, 1);
                gl.ReadPixels(
                    0,
                    0,
                    DIGEST_SIDE,
                    DIGEST_SIDE,
                    ffi::RGBA,
                    ffi::UNSIGNED_BYTE,
                    bytes.as_mut_ptr().cast(),
                );
            })
            .ok()?;
        let _ = frame.finish().ok()?;

        Some(SourceDigest(bytes))
    }
}

/// Workload of an offscreen super-resolution pipeline, used to decide whether computing a
/// [`SourceDigest`] is cheaper than running the pass.
#[derive(Debug, Clone, Copy)]
struct PassWorkload {
    /// Source region size in buffer texels.
    source_size: Size<i32, Buffer>,
    /// Total output texels written across all offscreen passes of the algorithm.
    pass_texels: u64,
}

impl PassWorkload {
    fn single_pass(source_size: Size<i32, Buffer>, dest_size: Size<i32, Physical>) -> Self {
        let pass_texels = u64::try_from(dest_size.w.max(0)).unwrap_or(0)
            * u64::try_from(dest_size.h.max(0)).unwrap_or(0);
        Self {
            source_size,
            pass_texels,
        }
    }

    fn multi_pass(source_size: Size<i32, Buffer>, pass_count: usize) -> Self {
        let src_texels = u64::try_from(source_size.w.max(0)).unwrap_or(0)
            * u64::try_from(source_size.h.max(0)).unwrap_or(0);
        Self {
            source_size,
            pass_texels: src_texels.saturating_mul(pass_count as u64),
        }
    }

    /// Returns whether computing a [`SourceDigest`] is worthwhile for this workload.
    fn should_use_digest(self) -> bool {
        let src_texels = u64::try_from(self.source_size.w.max(0)).unwrap_or(0)
            * u64::try_from(self.source_size.h.max(0)).unwrap_or(0);
        src_texels >= DIGEST_MIN_SOURCE_TEXELS && self.pass_texels >= DIGEST_MIN_PASS_TEXELS
    }
}

/// Why a cached offscreen pass result could be reused without rerunning the pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheReuse {
    /// Same [`CommitCounter`], geometry and renderer context.
    SameCommit,
    /// New [`CommitCounter`], but Wayland `damage_since` was empty.
    NoDamage { made_for: CommitCounter },
    /// New [`CommitCounter`] with reported damage, but GPU [`SourceDigest`] matched.
    SameDigest { made_for: CommitCounter },
}

/// Cached entry of an offscreen super-resolution pass, shared across all multi-pass algorithms.
#[derive(Debug)]
struct CachedPass<K, V> {
    key: K,
    commit: CommitCounter,
    context_id: ContextId<GlesTexture>,
    digest: Option<SourceDigest>,
    value: V,
}

/// Generic cache for any offscreen super-resolution pass.
///
/// Unifies commit-counter matching, Wayland damage-set checking, and GPU [`SourceDigest`]
/// comparison so every multi-pass upscaler (FSR1, CuNNy variants, and future passes) benefits from
/// unchanged-frame detection through one abstraction.
#[derive(Debug)]
struct PassCache<K, V> {
    entry: Option<CachedPass<K, V>>,
}

impl<K, V> Default for PassCache<K, V> {
    fn default() -> Self {
        Self { entry: None }
    }
}

impl<K: PartialEq, V: Clone> PassCache<K, V> {
    fn clear(&mut self) {
        self.entry = None;
    }

    /// Checks whether the cached result can be reused before looking up the client's GLES texture.
    fn probe_before_import<R: ImportAll>(
        &mut self,
        key: &K,
        commit: CommitCounter,
        context_id: &ContextId<GlesTexture>,
        elem: &WaylandSurfaceRenderElement<R>,
        scale: Scale<f64>,
    ) -> Option<(CacheReuse, V)> {
        let entry = self.entry.as_mut()?;
        if entry.key != *key || entry.context_id != *context_id {
            return None;
        }

        if entry.commit == commit {
            return Some((CacheReuse::SameCommit, entry.value.clone()));
        }

        if damage_since(elem, scale, entry.commit).is_empty() {
            let made_for = entry.commit;
            entry.commit = commit;
            return Some((CacheReuse::NoDamage { made_for }, entry.value.clone()));
        }

        None
    }

    /// Computes a [`SourceDigest`] when `workload` is above the worthwhile threshold and checks
    /// whether the new client buffer carries the exact same pixels as the cached entry.
    ///
    /// Returns `Ok((reuse, value))` on a digest hit, or `Err(computed_digest)` on a miss so the
    /// caller can store `computed_digest` alongside the newly produced pass result without
    /// computing it a second time.
    #[allow(clippy::too_many_arguments)]
    fn probe_digest(
        &mut self,
        key: &K,
        commit: CommitCounter,
        context_id: &ContextId<GlesTexture>,
        workload: PassWorkload,
        digest_target: &mut DigestTarget,
        renderer: &mut GlesRenderer,
        source: &GlesTexture,
        src: Rectangle<f64, Buffer>,
        y_inverted: bool,
        transform: Transform,
    ) -> Result<(CacheReuse, V), Option<SourceDigest>> {
        if !workload.should_use_digest() {
            return Err(None);
        }

        let new_digest = digest_target.compute(renderer, source, src, y_inverted, transform);

        if let Some(entry) = self.entry.as_mut() {
            if entry.key == *key
                && entry.context_id == *context_id
                && new_digest.is_some()
                && entry.digest == new_digest
            {
                let made_for = entry.commit;
                entry.commit = commit;
                return Ok((CacheReuse::SameDigest { made_for }, entry.value.clone()));
            }
        }

        Err(new_digest)
    }

    /// Takes the cached value for intermediate texture reuse if `predicate` matches.
    fn take_reusable(
        &mut self,
        predicate: impl FnOnce(&K, &ContextId<GlesTexture>) -> bool,
    ) -> Option<V> {
        let entry = self.entry.take()?;
        if predicate(&entry.key, &entry.context_id) {
            Some(entry.value)
        } else {
            None
        }
    }

    fn store(
        &mut self,
        key: K,
        commit: CommitCounter,
        context_id: ContextId<GlesTexture>,
        digest: Option<SourceDigest>,
        value: V,
    ) {
        self.entry = Some(CachedPass {
            key,
            commit,
            context_id,
            digest,
            value,
        });
    }
}

/// Messages that should be logged once per resolved configuration rather than every frame.
mod logged {
    /// The configured algorithm is not implemented.
    pub const ALGORITHM_FALLBACK: u8 = 1 << 0;
    /// The offscreen pass failed.
    pub const PASS_FAILURE: u8 = 1 << 1;
    /// The client buffer already matches the display area.
    pub const NOTHING_TO_MAGNIFY: u8 = 1 << 2;
    /// The offscreen pass cannot run for this window.
    pub const NO_PASS: u8 = 1 << 3;
    /// The client buffer is not in an orientation the CuNNy passes can handle.
    pub const CUNNY_ORIENTATION: u8 = 1 << 4;
}

/// Returns the damage the element has accumulated since the given commit.
///
/// A galgame redraws the same scene at the display's rate, so most commits carry the same pixels as
/// the one before, and a pass only has to run again when something actually changed. The renderer
/// answers with the whole buffer when it cannot tell which parts changed, so an empty answer is the
/// only one that proves the content is the same without inspecting the texture.
fn damage_since<R: ImportAll>(
    elem: &WaylandSurfaceRenderElement<R>,
    scale: Scale<f64>,
    commit: CommitCounter,
) -> DamageSet<i32, Physical> {
    elem.damage_since(scale, Some(commit))
}

/// Per-window super resolution state.
///
/// This holds the resolved configuration and the cached offscreen results. It also remembers which
/// messages it already logged for the current configuration, so that a window with, say, an
/// algorithm that is not implemented logs about it once instead of every frame.
///
/// Everything here is logged at `debug!` or below, except for the things the user has to know
/// about (an algorithm that is not implemented, a failing pass), which are logged at `warn!`. The
/// logger target is this module, so the whole feature can be filtered with, for example,
/// `RUST_LOG=niri=info,niri::render_helpers::superres=debug`.
/// Detailed GPU/buffer provenance uses the opt-in [`GPU_LOG_TARGET`] instead; that is where the
/// DEBUG entries are, and the output that repeats for every window and every frame — the raw
/// Smithay cache snapshots, the per-pass completions and the cache hits — is one level below them,
/// at TRACE.
#[derive(Debug)]
pub struct SuperResolutionPipeline {
    /// The mode, algorithm and sharpness that the current decision was made for.
    resolved: Cell<Option<(UpscalingMode, UpscalingAlgorithm, f32)>>,
    /// Which of the once-per-configuration messages were already logged.
    logged: Cell<u8>,
    /// Cached result of the EASU pass.
    fsr1: RefCell<PassCache<Fsr1Key, Fsr1Result>>,
    /// Cached result of a CuNNy run.
    cunny: RefCell<PassCache<CunnyKey, CunnyResult>>,
    /// Reusable 32×32 target for GPU content digests across all upscaling passes.
    digest: RefCell<DigestTarget>,
    /// The size that this window's content is magnified into, when it is being magnified.
    ///
    /// This is here rather than in the window so that everything about the effect, including its
    /// log messages, is under this module's target, which is what the log filter in the wiki
    /// enables.
    magnify_target: Cell<Option<Size<i32, Logical>>>,
    /// Visible output viewport relative to the window's origin, when rendered through a tile.
    ///
    /// Elements outside this rectangle are culled by the damage tracker and skip offscreen
    /// super-resolution passes.
    view_rect: Cell<Option<Rectangle<f64, Logical>>>,
    /// How long this window's passes take on the GPU.
    ///
    /// Shared with the elements that draw the window, which is what measures the final draw.
    timings: Arc<Timings>,
}

/// Which CuNNy network to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CunnyVariant {
    /// `CuNNy-fast-NVL`: four passes.
    Fast,

    /// `CuNNy-3x12-NVL`: the same network with one more 12x12 convolution.
    ThreeByTwelve,
}

impl CunnyVariant {
    /// The passes, in order, with the shaders each of them is made of.
    fn passes(self) -> &'static [CunnyPass] {
        match self {
            CunnyVariant::Fast => FAST_NVL,
            CunnyVariant::ThreeByTwelve => THREE_BY_TWELVE_NVL,
        }
    }

    /// Total number of offscreen draw groups across all passes of this network.
    fn draw_count(self) -> usize {
        self.passes().iter().map(|pass| pass.shaders.len()).sum()
    }

    /// Index for the program cache, so that a program can be found again by what it is made of.
    fn index(self) -> u32 {
        match self {
            CunnyVariant::Fast => 0,
            CunnyVariant::ThreeByTwelve => 1,
        }
    }

    /// How many source-sized textures the passes need, and how many the residuals take.
    fn texture_count(self) -> usize {
        6
    }
}

/// Where a pass reads from or writes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CunnyTexture {
    /// The client buffer.
    Input,
    /// The residuals, which are the network's result.
    Residuals,
    /// One of the source-sized textures that the passes ping-pong between.
    Feature(usize),
}

/// One pass of a network.
#[derive(Debug)]
struct CunnyPass {
    /// The shader for each group of four output channels.
    shaders: &'static [&'static str],
    /// What every one of those shaders reads. The first is drawn as `tex`, the rest are bound to
    /// the texture units that follow it.
    reads: &'static [CunnyTexture],
    /// What each shader writes, in the same order as `shaders`.
    writes: &'static [CunnyTexture],
}

/// The passes of `CuNNy-fast-NVL`.
///
/// The network doubles the image: the passes work at the source's resolution, and the pixel shuffle
/// leaves four sub-pixels of luma residual per source texel, which the final draw applies.
const FAST_NVL: &[CunnyPass] = &[
    CunnyPass {
        shaders: &[
            include_str!("shaders/superres/cunny/generated/fast/pass1_a0.frag"),
            include_str!("shaders/superres/cunny/generated/fast/pass1_a1.frag"),
            include_str!("shaders/superres/cunny/generated/fast/pass1_a2.frag"),
        ],
        reads: &[CunnyTexture::Input],
        writes: &[
            CunnyTexture::Feature(0),
            CunnyTexture::Feature(1),
            CunnyTexture::Feature(2),
        ],
    },
    CunnyPass {
        shaders: &[
            include_str!("shaders/superres/cunny/generated/fast/pass2_a0.frag"),
            include_str!("shaders/superres/cunny/generated/fast/pass2_a1.frag"),
            include_str!("shaders/superres/cunny/generated/fast/pass2_a2.frag"),
        ],
        reads: &[
            CunnyTexture::Feature(0),
            CunnyTexture::Feature(1),
            CunnyTexture::Feature(2),
        ],
        writes: &[
            CunnyTexture::Feature(3),
            CunnyTexture::Feature(4),
            CunnyTexture::Feature(5),
        ],
    },
    CunnyPass {
        shaders: &[
            include_str!("shaders/superres/cunny/generated/fast/pass3_a0.frag"),
            include_str!("shaders/superres/cunny/generated/fast/pass3_a1.frag"),
        ],
        reads: &[
            CunnyTexture::Feature(3),
            CunnyTexture::Feature(4),
            CunnyTexture::Feature(5),
        ],
        writes: &[CunnyTexture::Feature(0), CunnyTexture::Feature(1)],
    },
    CunnyPass {
        shaders: &[include_str!(
            "shaders/superres/cunny/generated/fast/shuffle.frag"
        )],
        reads: &[CunnyTexture::Feature(0), CunnyTexture::Feature(1)],
        writes: &[CunnyTexture::Residuals],
    },
];

/// The passes of `CuNNy-3x12-NVL`, which is the same network with one more 12x12 convolution.
const THREE_BY_TWELVE_NVL: &[CunnyPass] = &[
    CunnyPass {
        shaders: &[
            include_str!("shaders/superres/cunny/generated/3x12/pass1_a0.frag"),
            include_str!("shaders/superres/cunny/generated/3x12/pass1_a1.frag"),
            include_str!("shaders/superres/cunny/generated/3x12/pass1_a2.frag"),
        ],
        reads: &[CunnyTexture::Input],
        writes: &[
            CunnyTexture::Feature(0),
            CunnyTexture::Feature(1),
            CunnyTexture::Feature(2),
        ],
    },
    CunnyPass {
        shaders: &[
            include_str!("shaders/superres/cunny/generated/3x12/pass2_a0.frag"),
            include_str!("shaders/superres/cunny/generated/3x12/pass2_a1.frag"),
            include_str!("shaders/superres/cunny/generated/3x12/pass2_a2.frag"),
        ],
        reads: &[
            CunnyTexture::Feature(0),
            CunnyTexture::Feature(1),
            CunnyTexture::Feature(2),
        ],
        writes: &[
            CunnyTexture::Feature(3),
            CunnyTexture::Feature(4),
            CunnyTexture::Feature(5),
        ],
    },
    CunnyPass {
        shaders: &[
            include_str!("shaders/superres/cunny/generated/3x12/pass3_a0.frag"),
            include_str!("shaders/superres/cunny/generated/3x12/pass3_a1.frag"),
            include_str!("shaders/superres/cunny/generated/3x12/pass3_a2.frag"),
        ],
        reads: &[
            CunnyTexture::Feature(3),
            CunnyTexture::Feature(4),
            CunnyTexture::Feature(5),
        ],
        writes: &[
            CunnyTexture::Feature(0),
            CunnyTexture::Feature(1),
            CunnyTexture::Feature(2),
        ],
    },
    CunnyPass {
        shaders: &[
            include_str!("shaders/superres/cunny/generated/3x12/pass4_a0.frag"),
            include_str!("shaders/superres/cunny/generated/3x12/pass4_a1.frag"),
            include_str!("shaders/superres/cunny/generated/3x12/pass4_a2.frag"),
        ],
        reads: &[
            CunnyTexture::Feature(0),
            CunnyTexture::Feature(1),
            CunnyTexture::Feature(2),
        ],
        writes: &[
            CunnyTexture::Feature(3),
            CunnyTexture::Feature(4),
            CunnyTexture::Feature(5),
        ],
    },
    CunnyPass {
        shaders: &[include_str!(
            "shaders/superres/cunny/generated/3x12/shuffle.frag"
        )],
        reads: &[
            CunnyTexture::Feature(3),
            CunnyTexture::Feature(4),
            CunnyTexture::Feature(5),
        ],
        writes: &[CunnyTexture::Residuals],
    },
];

/// Geometry and variant that a CuNNy result is valid for.
#[derive(Debug, Clone, PartialEq)]
struct CunnyKey {
    /// Which network ran.
    variant: CunnyVariant,
    /// The rect of the buffer that was run.
    src: Rectangle<f64, Buffer>,
    /// Its size, which is the size of the residuals.
    size: Size<i32, Buffer>,
}

/// A network's result: the residuals of the doubled image, one source texel per texel.
#[derive(Debug, Clone)]
struct CunnyResult {
    /// The residuals.
    texture: GlesTexture,
    /// The intermediate feature textures, kept so the next commit at the same size does not
    /// reallocate them.
    features: Vec<GlesTexture>,
    /// The renderer context the texture belongs to.
    context_id: ContextId<GlesTexture>,
}

/// Sets a pass texture to the given filter and edge clamping.
///
/// `GlesRenderer::create_buffer` leaves `GL_TEXTURE_MIN_FILTER` at its initial value of
/// `GL_NEAREST_MIPMAP_LINEAR` with a single mip level, which makes the texture incomplete in
/// OpenGL ES until something sets a non-mipmap filter on it. `GlesFrame::render_texture` only
/// does that for `GL_TEXTURE0`, so any texture bound to `GL_TEXTURE1` or `GL_TEXTURE2` (and
/// the residuals texture, which is never drawn as `GL_TEXTURE0`) would otherwise read back as
/// `(0, 0, 0, 1)` on every sample.
///
/// # Safety
///
/// Must be called with a current GLES context.
unsafe fn configure_pass_texture(gl: &ffi::Gles2, tex_id: u32, filter: ffi::types::GLenum) {
    gl.BindTexture(ffi::TEXTURE_2D, tex_id);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MIN_FILTER, filter as i32);
    gl.TexParameteri(ffi::TEXTURE_2D, ffi::TEXTURE_MAG_FILTER, filter as i32);
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_WRAP_S,
        ffi::CLAMP_TO_EDGE as i32,
    );
    gl.TexParameteri(
        ffi::TEXTURE_2D,
        ffi::TEXTURE_WRAP_T,
        ffi::CLAMP_TO_EDGE as i32,
    );
}

/// Runs a network over the client buffer, returning its feature textures and residuals.
///
/// Every pass is one draw per group of four output channels, since the renderer has no multiple
/// render targets. The feature textures are the network's own and are reused across commits of the
/// same size.
#[allow(clippy::too_many_arguments)]
fn run_cunny(
    renderer: &mut GlesRenderer,
    variant: CunnyVariant,
    source: &GlesTexture,
    src: Rectangle<f64, Buffer>,
    tex_size: Size<i32, Buffer>,
    transform: Transform,
    timings: &Timings,
    reuse: Option<(Vec<GlesTexture>, GlesTexture)>,
) -> anyhow::Result<(Vec<GlesTexture>, GlesTexture)> {
    // The source rect in the client buffer's texels, which is the size the features have.
    let size = src
        .size
        .to_i32_round()
        .to_logical(1, Transform::Normal)
        .to_physical(1);
    anyhow::ensure!(size.w > 0 && size.h > 0, "empty source rect");
    let buffer_size = size.to_logical(1).to_buffer(1, Transform::Normal);

    let (features, residuals) = match reuse
        .filter(|(features, _)| features.len() == variant.texture_count())
    {
        Some(reused) => reused,
        None => {
            let mut features: Vec<GlesTexture> = Vec::with_capacity(variant.texture_count());
            for _ in 0..variant.texture_count() {
                features.push(
                    renderer
                        .create_buffer(Fourcc::Abgr8888, buffer_size)
                        .context("error creating a feature texture")?,
                );
            }

            // The residuals are signed, and the quantization the reference does is on the eight-bit
            // features, not on them, so they go into a half-float texture.
            let residuals: GlesTexture = renderer
                .create_buffer(Fourcc::Abgr16161616f, buffer_size)
                .context("error creating the residuals texture")?;

            renderer
                .with_context(|gl| unsafe {
                    for feature in &features {
                        configure_pass_texture(gl, feature.tex_id(), ffi::NEAREST);
                    }
                    configure_pass_texture(gl, residuals.tex_id(), ffi::NEAREST);
                    gl.BindTexture(ffi::TEXTURE_2D, 0);
                })
                .context("error configuring pass textures")?;

            debug!(
                target: GPU_LOG_TARGET,
                gpu = ?GpuDebug::renderer(renderer),
                ?variant,
                features = ?features.iter().map(TextureDebug).collect::<Vec<_>>(),
                residuals = ?TextureDebug(&residuals),
                "allocated CuNNy intermediates in execution context (not exported as DMA-BUF by these passes)",
            );
            (features, residuals)
        }
    };

    // Every pass samples exact texel centers, and `render_texture` overwrites `GL_TEXTURE0`'s
    // filter with the renderer's filter on every draw. Setting it to `Nearest` for the run keeps
    // both `GL_TEXTURE0` and any feature texture that is later bound to `GL_TEXTURE1`/`GL_TEXTURE2`
    // on point sampling.
    let _ = renderer.downscale_filter(TextureFilter::Nearest);
    let _ = renderer.upscale_filter(TextureFilter::Nearest);

    let run_passes = |renderer: &mut GlesRenderer| -> anyhow::Result<()> {
        for (pass_index, pass) in variant.passes().iter().enumerate() {
            for group in 0..pass.shaders.len() {
                let pass_number = pass_index + 1;
                let group_number = group + 1;
                trace!(
                    target: GPU_LOG_TARGET,
                    ?variant,
                    pass = pass_number,
                    group = group_number,
                    "running a CuNNy pass",
                );
                let program =
                    cunny_program(renderer, variant, pass_index, group).with_context(|| {
                        format!("error compiling pass {pass_number} group {group_number}")
                    })?;

                let uniforms = cunny_uniforms(pass, buffer_size, src, tex_size);

                // A texture handle is cheap to copy, and copying it keeps the pass's reads and its
                // write from borrowing the same list.
                let mut target = match pass.writes[group] {
                    CunnyTexture::Feature(index) => features[index].clone(),
                    CunnyTexture::Residuals => residuals.clone(),
                    CunnyTexture::Input => unreachable!(),
                };

                let mut fb = renderer.bind(&mut target).with_context(|| {
                    format!(
                        "error binding a feature texture (pass {pass_number} group {group_number})"
                    )
                })?;
                let mut frame = renderer
                    .render(&mut fb, size, Transform::Normal)
                    .with_context(|| {
                        format!("error starting pass {pass_number} group {group_number}")
                    })?;

                // A run is measured as a whole: what a user has to weigh is what running the network
                // costs, and the passes are short enough that timing each of them separately would
                // mostly measure the queries. The draw that applies the residuals is measured on its
                // own, since it happens once per composited frame rather than once per client frame.
                let first = pass_index == 0 && group == 0;
                let last =
                    pass_index + 1 == variant.passes().len() && group + 1 == pass.shaders.len();
                if first {
                    timings.begin(&mut frame, Pass::Network);
                }

                // The first input is drawn, so the renderer binds it as `tex`; the others are bound
                // to the texture units after it, which is the only way to reach more than one
                // texture from smithay's texture programs.
                for (i, read) in pass.reads.iter().enumerate().skip(1) {
                    let texture = match read {
                        CunnyTexture::Feature(index) => features[*index].tex_id(),
                        _ => unreachable!(),
                    };
                    frame
                        .with_context(|gl| unsafe {
                            gl.ActiveTexture(ffi::TEXTURE0 + i as u32);
                            configure_pass_texture(gl, texture, ffi::NEAREST);
                            gl.ActiveTexture(ffi::TEXTURE0);
                        })
                        .context("error binding a texture")?;
                }

                let dst = Rectangle::from_size(size);
                let (texture, src_rect, transform) = match pass.reads[0] {
                    CunnyTexture::Feature(index) => (
                        &features[index],
                        Rectangle::from_size(buffer_size).to_f64(),
                        Transform::Normal,
                    ),
                    CunnyTexture::Input => (source, src, transform),
                    CunnyTexture::Residuals => unreachable!(),
                };

                // Mark the whole destination as opaque so the renderer disables `GL_BLEND`: the
                // fourth channel of `gl_FragColor` is a feature channel (or sub-pixel residual),
                // not an alpha value to blend against previous contents of the texture.
                frame
                    .render_texture_from_to(
                        texture,
                        src_rect,
                        dst,
                        &[dst],
                        &[dst],
                        transform,
                        1.0,
                        Some(&program),
                        &uniforms,
                    )
                    .with_context(|| format!("error in pass {pass_number} group {group_number}"))?;
                if last {
                    timings.end(&mut frame);
                }
                let sync = frame.finish().with_context(|| {
                    format!("error finishing pass {pass_number} group {group_number}")
                })?;
                drop(fb);
                trace!(
                    target: GPU_LOG_TARGET,
                    gpu = ?GpuDebug::renderer(renderer),
                    pass = pass_index + 1,
                    group,
                    reads = ?pass.reads,
                    writes = ?pass.writes[group],
                    target = ?TextureDebug(&target),
                    completion_fence = sync.contains_fence(),
                    "CuNNy finish returned; diagnostics add no waits (renderer may synchronize internally)",
                );
            }
        }
        Ok(())
    };

    let res = run_passes(renderer);
    let _ = renderer.downscale_filter(TextureFilter::Linear);
    let _ = renderer.upscale_filter(TextureFilter::Linear);
    res?;

    Ok((features, residuals))
}

/// The uniforms a CuNNy pass is drawn with.
///
/// Which of them a pass uses depends on what it reads, but the renderer has to be told about a
/// uniform when the program is compiled rather than when it is drawn with: passing one that was
/// not declared fails the whole draw, and the pass then falls back to the resampling kernels.
/// The uniforms the upscale program is drawn with.
fn cunny_upscale_uniforms() -> Vec<UniformName<'static>> {
    vec![
        UniformName::new("niri_tex_size", UniformType::_2f),
        UniformName::new("niri_src_rect", UniformType::_4f),
        UniformName::new("niri_footprint", UniformType::_2f),
        UniformName::new("niri_tex1", UniformType::_1i),
    ]
}

fn cunny_pass_uniforms() -> Vec<UniformName<'static>> {
    vec![
        UniformName::new("niri_tex_size", UniformType::_2f),
        UniformName::new("niri_src_rect", UniformType::_4f),
        UniformName::new("niri_size", UniformType::_2f),
        UniformName::new("niri_src_size", UniformType::_2f),
        UniformName::new("niri_tex1", UniformType::_1i),
        UniformName::new("niri_tex2", UniformType::_1i),
    ]
}

/// Returns the program of one group of a pass, compiling it the first time it is asked for.
fn cunny_program(
    renderer: &mut GlesRenderer,
    variant: CunnyVariant,
    pass: usize,
    group: usize,
) -> Option<GlesTexProgram> {
    let key = variant.index() * 1000 + pass as u32 * 10 + group as u32;

    {
        let cached = SuperResPrograms::get(renderer)
            .and_then(|programs| programs.cunny.borrow().get(&key).cloned());
        if let Some(program) = cached {
            return Some(program);
        }
    }

    let source = variant.passes()[pass].shaders[group];
    let program =
        SuperResPrograms::compile_file(renderer, "cunny", source, &cunny_pass_uniforms())?;

    if let Some(programs) = SuperResPrograms::get(renderer) {
        programs.cunny.borrow_mut().insert(key, program.clone());
    }

    Some(program)
}

/// The uniforms of a pass: the size of what it reads, and the extra texture units.
fn cunny_uniforms(
    pass: &CunnyPass,
    feature_size: Size<i32, Buffer>,
    src: Rectangle<f64, Buffer>,
    tex_size: Size<i32, Buffer>,
) -> Vec<Uniform<'static>> {
    let mut rv = Vec::new();

    match pass.reads[0] {
        // The first pass reads the client buffer, whose size is the window's rather than the source
        // rect's, and needs the rect of it that is being resampled.
        CunnyTexture::Input => {
            rv.push(Uniform::new(
                "niri_tex_size",
                (tex_size.w as f32, tex_size.h as f32),
            ));
            rv.push(Uniform::new(
                "niri_src_rect",
                [
                    src.loc.x as f32,
                    src.loc.y as f32,
                    src.size.w as f32,
                    src.size.h as f32,
                ],
            ));
        }
        CunnyTexture::Feature(_) => {
            // The pixel shuffle names its size after what it reads, the convolutions after the
            // features they write.
            let name = if matches!(pass.writes, [CunnyTexture::Residuals]) {
                "niri_src_size"
            } else {
                "niri_size"
            };
            rv.push(Uniform::new(
                name,
                (feature_size.w as f32, feature_size.h as f32),
            ));
        }
        CunnyTexture::Residuals => (),
    }

    // A sampler uniform is a texture unit, so these say which unit the texture was bound to.
    for (i, read) in pass.reads.iter().enumerate().skip(1) {
        if let CunnyTexture::Feature(_) = read {
            let name = match i {
                1 => "niri_tex1",
                2 => "niri_tex2",
                _ => continue,
            };
            rv.push(Uniform::new(name, i as i32));
        }
    }

    rv
}

/// Which pass is being measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    /// The EASU pass, which runs offscreen whenever the window's buffer changes.
    Easu,
    /// The final draw: the resampling filter, or RCAS over the cached EASU result.
    Draw,
    /// A run of a neural network, all of its passes together.
    Network,
}

impl Pass {
    /// Every pass, in the order they are logged.
    const ALL: [Pass; 3] = [Pass::Easu, Pass::Draw, Pass::Network];

    fn name(self) -> &'static str {
        match self {
            Pass::Easu => "easu",
            Pass::Draw => "draw",
            Pass::Network => "network",
        }
    }

    fn index(self) -> usize {
        match self {
            Pass::Easu => 0,
            Pass::Draw => 1,
            Pass::Network => 2,
        }
    }
}

/// What one pass took over the last interval.
#[derive(Debug, Clone, Copy)]
pub struct PassReport {
    /// Which pass this is.
    pub pass: Pass,
    /// Average GPU time of a single pass.
    pub average: Duration,
    /// How many passes were measured.
    pub count: u64,
}

/// What the passes took over the last interval.
#[derive(Debug, Clone)]
pub struct Report {
    /// How long the interval was.
    pub interval: Duration,
    /// One entry per pass that ran, in the order they are logged.
    pub passes: Vec<PassReport>,
}

impl Report {
    /// How much GPU time the passes took in total over the interval.
    pub fn total(&self) -> Duration {
        self.passes
            .iter()
            .map(|pass| pass.average * u32::try_from(pass.count).unwrap_or(u32::MAX))
            .sum()
    }
}

/// Formats what the passes took, for the log.
fn format_timings(report: &Report) -> String {
    let mut rv = String::new();

    for (index, pass) in report.passes.iter().enumerate() {
        if index > 0 {
            rv.push_str(" + ");
        }

        let _ = write!(
            rv,
            "{name} {average:.2} ms in {count} pass{plural}",
            name = pass.pass.name(),
            average = pass.average.as_secs_f64() * 1000.,
            count = pass.count,
            plural = if pass.count == 1 { "" } else { "es" },
        );
    }

    let _ = write!(
        rv,
        ", {total:.0} ms of GPU time in {interval} ms",
        total = report.total().as_secs_f64() * 1000.,
        interval = report.interval.as_millis(),
    );

    rv
}

/// How long to accumulate results before logging them.
const TIMING_INTERVAL: Duration = Duration::from_secs(1);

/// How many results to keep waiting for before giving up on the driver handing them back.
const MAX_PENDING: usize = 64;

/// Measures how long a pass takes on the GPU.
///
/// Timer queries are asynchronous: a result is only there a few frames after the pass it measured,
/// so results are read when the driver has them and skipped otherwise, which is what keeps this
/// from stalling the frame. What has been measured is logged once per interval, so the numbers are
/// averages over that interval.
#[derive(Debug)]
pub struct Timings {
    /// Whether this window measures at all.
    enabled: bool,
    inner: Mutex<Inner>,
}

impl Timings {
    /// Returns timings that measure nothing, for elements that are built without a pipeline.
    fn disabled() -> Arc<Self> {
        Arc::new(Self {
            enabled: false,
            inner: Mutex::new(Inner::default()),
        })
    }

    /// Returns timings for a window, measured whenever the driver can and anything reads them.
    fn for_window() -> Arc<Self> {
        Arc::new(Self {
            enabled: true,
            inner: Mutex::new(Inner::default()),
        })
    }

    /// Notes the GL context the passes run in, giving up on what was measured in a different one.
    ///
    /// The context is identified by the address of its `EGLContext`, which lives inside the
    /// renderer for as long as the context does. Rendering to several GPUs at once means the
    /// passes run in more than one context, and then this keeps resetting, which is why such a
    /// setup does not report timings.
    fn set_context(&self, context: usize) {
        let mut inner = self.inner.lock().unwrap();
        if inner.context != Some(context) {
            // GL objects do not outlive their context, so the query names do not have to be
            // deleted here.
            *inner = Inner {
                context: Some(context),
                ..Default::default()
            };
        }
    }

    /// Starts measuring a pass. Every call has to be followed by [`Timings::end`].
    fn begin(&self, frame: &mut GlesFrame<'_, '_>, pass: Pass) {
        // Measuring costs GL calls and, on some drivers, flushes work. Only do it when the numbers
        // are going to be logged anyway, and when the driver can measure at all.
        if !self.enabled
            || !(tracing::enabled!(tracing::Level::DEBUG)
                || tracing::enabled!(target: GPU_LOG_TARGET, tracing::Level::DEBUG))
        {
            return;
        }
        if !matches!(frame.with_context(can_time), Ok(true)) {
            return;
        }

        self.set_context(context_address(frame.egl_context()));

        let mut inner = self.inner.lock().unwrap();

        // Read what the driver has ready: a result shows up a few frames later, and reading them
        // here means that no pass ever waits for its own result.
        let _ = frame.with_context(|gl| inner.collect(gl));

        // If a query is somehow still running, close it here: leaving it active would make every
        // later measurement fail.
        if let Some((pass, query)) = inner.started.take() {
            let _ = frame.with_context(|gl| unsafe { gl.EndQueryEXT(ffi::TIME_ELAPSED_EXT) });
            inner.pending.push_back((pass, query));
        }

        let mut query = 0;
        let _ = frame.with_context(|gl| unsafe { gl.GenQueriesEXT(1, &mut query) });
        if query == 0 {
            return;
        }

        let _ = frame.with_context(|gl| unsafe { gl.BeginQueryEXT(ffi::TIME_ELAPSED_EXT, query) });
        inner.started = Some((pass, query));
    }

    /// Finishes measuring a pass. Does nothing when nothing was started.
    fn end(&self, frame: &mut GlesFrame<'_, '_>) {
        let mut inner = self.inner.lock().unwrap();
        let Some((pass, query)) = inner.started.take() else {
            return;
        };

        let _ = frame.with_context(|gl| unsafe { gl.EndQueryEXT(ffi::TIME_ELAPSED_EXT) });
        inner.pending.push_back((pass, query));
    }

    /// Returns what to log for the interval that just ended, if anything was measured.
    fn take_report(&self) -> Option<Report> {
        let mut inner = self.inner.lock().unwrap();
        inner.report(Instant::now(), TIMING_INTERVAL)
    }
}

/// Returns whether the driver can measure passes with timer queries.
///
/// A GL entry point that is not there cannot be called at all, so nothing may use the query
/// functions before this says that they are all loaded.
fn can_time(gl: &ffi::Gles2) -> bool {
    gl.GenQueriesEXT.is_loaded()
        && gl.BeginQueryEXT.is_loaded()
        && gl.EndQueryEXT.is_loaded()
        && gl.DeleteQueriesEXT.is_loaded()
        && gl.GetQueryObjectuivEXT.is_loaded()
        && gl.GetQueryObjectui64vEXT.is_loaded()
}

/// The address of a GL context, to tell contexts apart.
fn context_address(context: &smithay::backend::egl::EGLContext) -> usize {
    std::ptr::from_ref(context) as usize
}

#[derive(Debug, Default)]
struct Inner {
    /// The GL context that the queries belong to.
    context: Option<usize>,
    /// The query that is being filled in right now, if a pass is being measured.
    started: Option<(Pass, u32)>,
    /// Queries that are waiting for their result, oldest first.
    pending: VecDeque<(Pass, u32)>,
    /// Accumulated results of the current interval.
    totals_ns: [u64; Pass::ALL.len()],
    /// How many results went into each total.
    counts: [u64; Pass::ALL.len()],
    /// When the first result of the interval arrived.
    interval_start: Option<Instant>,
    /// Results that had to be dropped, which means the numbers are incomplete.
    dropped: u64,
}

impl Inner {
    /// Reads the results that the driver has ready.
    ///
    /// Results come back in the order the passes were submitted, so this stops at the first query
    /// that is not ready yet.
    fn collect(&mut self, gl: &ffi::Gles2) {
        // If the GPU hiccuped, the results that are in flight do not mean anything.
        let mut disjoint = 0;
        unsafe { gl.GetIntegerv(ffi::GPU_DISJOINT_EXT, &mut disjoint) };
        if disjoint != 0 {
            for (_, query) in self.pending.drain(..) {
                unsafe { gl.DeleteQueriesEXT(1, &query) };
            }
            self.dropped += 1;
            return;
        }

        while let Some(&(pass, query)) = self.pending.front() {
            let mut available = 0;
            unsafe {
                gl.GetQueryObjectuivEXT(query, ffi::QUERY_RESULT_AVAILABLE_EXT, &mut available)
            };
            if available == 0 {
                break;
            }

            let mut ns = 0;
            unsafe { gl.GetQueryObjectui64vEXT(query, ffi::QUERY_RESULT_EXT, &mut ns) };
            unsafe { gl.DeleteQueriesEXT(1, &query) };

            self.pending.pop_front();
            self.record(pass, ns, Instant::now());
        }

        // Rather than piling up results that the driver never hands back, give up on the oldest.
        while self.pending.len() > MAX_PENDING {
            let (_, query) = self.pending.pop_front().unwrap();
            unsafe { gl.DeleteQueriesEXT(1, &query) };
            self.dropped += 1;
        }
    }

    /// Adds one measured pass to the interval.
    fn record(&mut self, pass: Pass, ns: u64, now: Instant) {
        if self.interval_start.is_none() {
            self.interval_start = Some(now);
        }

        let index = pass.index();
        self.totals_ns[index] += ns;
        self.counts[index] += 1;
    }

    /// Returns what to log for the interval that just ended, and starts a new interval.
    ///
    /// `min_interval` is how long to accumulate before logging at all; tests pass zero to get what
    /// has been measured right away.
    fn report(&mut self, now: Instant, min_interval: Duration) -> Option<Report> {
        let start = self.interval_start?;
        let interval = now.duration_since(start);
        if interval < min_interval {
            return None;
        }

        let mut passes = Vec::new();
        for pass in Pass::ALL {
            let index = pass.index();
            let count = self.counts[index];
            if count == 0 {
                continue;
            }

            passes.push(PassReport {
                pass,
                average: Duration::from_nanos(self.totals_ns[index] / count),
                count,
            });
        }

        self.totals_ns = Default::default();
        self.counts = Default::default();
        self.interval_start = Some(now);

        if passes.is_empty() {
            return None;
        }

        Some(Report { interval, passes })
    }
}

impl SuperResolutionPipeline {
    pub fn new() -> Self {
        Self {
            resolved: Cell::new(None),
            logged: Cell::new(0),
            fsr1: RefCell::new(PassCache::default()),
            cunny: RefCell::new(PassCache::default()),
            digest: RefCell::new(DigestTarget::default()),
            magnify_target: Cell::new(None),
            view_rect: Cell::new(None),
            timings: Timings::for_window(),
        }
    }

    /// Sets the visible output viewport relative to the window's origin for the current render.
    pub fn set_view_rect(&self, view_rect: Option<Rectangle<f64, Logical>>) {
        self.view_rect.set(view_rect);
    }

    /// Returns the visible output viewport relative to the window's origin, if set.
    pub fn view_rect(&self) -> Option<Rectangle<f64, Logical>> {
        self.view_rect.get()
    }

    /// Returns the kernel to draw this window with, if the effect is enabled.
    ///
    /// This returns `None` when the window rules do not enable super resolution, in which case
    /// the window should be drawn the normal way.
    pub fn kernel(&self, rules: &Upscaling, id: u64) -> Option<SuperResKernel> {
        if !rules.enabled() {
            if self.resolved.take().is_some() {
                self.fsr1.borrow_mut().clear();
                self.cunny.borrow_mut().clear();
                debug!("super resolution: window {id} back to the renderer's own sampling");
            }
            return None;
        }

        let mode = rules.mode();
        let algorithm = rules.algorithm();
        let sharpness = rules.sharpness();
        let (kernel, fell_back) = SuperResKernel::for_algorithm(algorithm, sharpness);

        if self.resolved.get() != Some((mode, algorithm, sharpness as f32)) {
            self.resolved.set(Some((mode, algorithm, sharpness as f32)));
            self.logged.set(0);
            // The parameters may have changed, so the cached result may no longer match.
            self.fsr1.borrow_mut().clear();
            self.cunny.borrow_mut().clear();

            debug!("super resolution: window {id} uses {kernel}");
        }

        if self.log_once(logged::ALGORITHM_FALLBACK) && fell_back {
            warn!(
                "super resolution: algorithm {algorithm:?} is not implemented yet, using {kernel}"
            );
        }

        Some(kernel)
    }

    /// Returns whether the message behind `flag` should be logged now, and remembers it.
    fn log_once(&self, flag: u8) -> bool {
        let logged = self.logged.get();
        if logged & flag != 0 {
            return false;
        }

        self.logged.set(logged | flag);
        true
    }

    /// Records that this window's content is magnified into `target`, or that it is not anymore.
    ///
    /// `content` is the client's own size, which it keeps in this mode. Everything is logged here
    /// so that the whole feature can be followed with one filter.
    pub fn set_magnify_target(
        &self,
        id: u64,
        content: Size<i32, Logical>,
        target: Option<Size<i32, Logical>>,
    ) {
        if self.magnify_target.get() == target {
            return;
        }

        self.magnify_target.set(target);

        match target {
            Some(target) => debug!(
                "super resolution: magnifying window {id}: its content stays {w}×{h} \
                 and is drawn into {tw}×{th}",
                w = content.w,
                h = content.h,
                tw = target.w,
                th = target.h,
            ),
            None => debug!("super resolution: window {id} is not magnified anymore"),
        }
    }

    /// Returns the size that this window's content is magnified into, if it is being magnified.
    pub fn magnify_target(&self) -> Option<Size<i32, Logical>> {
        self.magnify_target.get()
    }

    /// Returns a network's residuals for this window, running it if the cached result is stale.
    fn cunny_result<R: NiriRenderer>(
        &self,
        renderer: &mut GlesRenderer,
        variant: CunnyVariant,
        elem: &WaylandSurfaceRenderElement<R>,
        scale: Scale<f64>,
        id: u64,
    ) -> Option<CunnyResult>
    where
        R::TextureId: PassSource,
    {
        let commit = elem.current_commit();
        let _span = debug_span!(
            target: GPU_LOG_TARGET,
            "cunny",
            window = id,
            element = ?elem.id(),
            ?commit,
            ?variant,
        )
        .entered();
        // The passes read and write axis-aligned texel grids, and the residuals only line up with
        // the client buffer when it is not flipped, so other buffers keep the plain kernels.
        if elem.transform() != Transform::Normal
            || buffer_y_inverted(elem.buffer()).unwrap_or(false)
        {
            trace!(
                target: GPU_LOG_TARGET,
                gpu = ?GpuDebug::renderer(renderer),
                buffer_transform = ?elem.transform(),
                y_inverted = buffer_y_inverted(elem.buffer()).unwrap_or(false),
                reason = "unsupported CuNNy buffer orientation",
                "selecting resampling fallback",
            );
            if self.log_once(logged::CUNNY_ORIENTATION) {
                debug!(
                    "super resolution: window {id} has a buffer the {variant:?} network cannot \
                     read, drawing it with the resampling kernels"
                );
            }
            return None;
        }

        let src = elem.src();
        let size = src.size.to_i32_round();
        if size.w <= 0 || size.h <= 0 {
            return None;
        }

        let key = CunnyKey { variant, src, size };
        let context_id = renderer.context_id();

        if let Some((reuse, result)) =
            self.cunny
                .borrow_mut()
                .probe_before_import(&key, commit, &context_id, elem, scale)
        {
            match reuse {
                CacheReuse::SameCommit => {
                    trace!("super resolution: reusing the {variant:?} result for window {id}");
                    trace!(
                        target: GPU_LOG_TARGET,
                        cache = "hit",
                        gpu = ?GpuDebug::renderer(renderer),
                        result = ?TextureDebug(&result.texture),
                        "reusing CuNNy residuals; network not rerun",
                    );
                }
                CacheReuse::NoDamage { made_for } | CacheReuse::SameDigest { made_for } => {
                    trace!(
                        target: GPU_LOG_TARGET,
                        cache = "unchanged",
                        window = id,
                        ?commit,
                        ?made_for,
                        gpu = ?GpuDebug::renderer(renderer),
                        "the window has no damage since the network ran; reusing its residuals",
                    );
                }
            }
            return Some(result);
        }

        let WaylandSurfaceTexture::Texture(texture) = elem.texture() else {
            return None;
        };
        let Some(source) = texture.pass_texture(&context_id) else {
            debug!(
                target: GPU_LOG_TARGET,
                gpu = ?GpuDebug::renderer(renderer),
                reason = "no native texture for execution context",
                "offscreen pass unavailable; falling back",
            );
            return None;
        };
        let tex_size = source.size();
        let workload = PassWorkload::multi_pass(size, variant.draw_count());

        let digest = match self.cunny.borrow_mut().probe_digest(
            &key,
            commit,
            &context_id,
            workload,
            &mut self.digest.borrow_mut(),
            renderer,
            &source,
            src,
            false,
            elem.transform(),
        ) {
            Ok((CacheReuse::SameDigest { made_for }, result)) => {
                trace!(
                    target: GPU_LOG_TARGET,
                    cache = "unchanged",
                    reason = "source digest matched",
                    window = id,
                    ?commit,
                    ?made_for,
                    gpu = ?GpuDebug::renderer(renderer),
                    result = ?TextureDebug(&result.texture),
                    "the client buffer pixels did not change; reusing CuNNy residuals",
                );
                return Some(result);
            }
            Ok((_, result)) => return Some(result),
            Err(digest) => digest,
        };

        trace!(
            "super resolution: running the {variant:?} network for window {id}, {w}×{h}",
            w = size.w,
            h = size.h,
        );

        debug!(
            target: GPU_LOG_TARGET,
            cache = "miss",
            gpu = ?GpuDebug::renderer(renderer),
            source = ?TextureDebug(&source),
            "submitting CuNNy network",
        );
        let reuse = self
            .cunny
            .borrow_mut()
            .take_reusable(|cached, cached_ctx| {
                cached.size == key.size && *cached_ctx == context_id
            })
            .map(|result| (result.features, result.texture));

        let (features, texture) = match run_cunny(
            renderer,
            variant,
            &source,
            src,
            tex_size,
            elem.transform(),
            &self.timings,
            reuse,
        ) {
            Ok(out) => out,
            Err(err) => {
                // Logged every time rather than once: a pass that fails every frame is exactly what
                // the Hybrid work has to be able to see, and the file this goes to keeps everything.
                debug!(
                    target: GPU_LOG_TARGET,
                    window = id,
                    ?variant,
                    source = ?TextureDebug(&source),
                    error = ?err,
                    "CuNNy run failed; drawing this window with the fallback",
                );
                if self.log_once(logged::PASS_FAILURE) {
                    warn!("super resolution: error running the pass for window {id}: {err:?}");
                }
                return None;
            }
        };

        let result = CunnyResult {
            texture,
            features,
            context_id: context_id.clone(),
        };
        self.cunny
            .borrow_mut()
            .store(key, commit, context_id, digest, result.clone());
        Some(result)
    }

    /// Wraps a surface element, running the EASU or CuNNy pass first if the cached result is stale.
    ///
    /// `renderer` must be the GLES renderer that will draw the element, and `scale` is the output
    /// scale that the element is drawn at. When `on_screen` is `false`, the element lies outside
    /// the visible viewport and skips offscreen passes entirely. The returned element falls back
    /// to drawing the client buffer normally when the pass cannot run or when its result is not
    /// valid for the renderer that ends up drawing it.
    #[allow(clippy::too_many_arguments)]
    pub fn element<R: NiriRenderer>(
        &self,
        renderer: &mut GlesRenderer,
        elem: WaylandSurfaceRenderElement<R>,
        kernel: SuperResKernel,
        scale: Scale<f64>,
        id: u64,
        magnify: Option<Rectangle<f64, Logical>>,
        on_screen: bool,
    ) -> SuperResElement<R>
    where
        R::TextureId: PassSource,
    {
        let _span = debug_span!(
            target: GPU_LOG_TARGET,
            "superres",
            window = id,
            element = ?elem.id(),
            commit = ?elem.current_commit(),
        )
        .entered();
        // With magnify, the content is drawn into a rect inside the area the layout gave the
        // window, so that is the size the effect has to produce.
        let dest_size = match magnify {
            Some(rect) => rect.size.to_physical_precise_round(scale),
            None => elem.geometry(scale).size,
        };

        // The passes run in this renderer's context, which is where their timings belong.
        self.timings
            .set_context(context_address(renderer.egl_context()));
        self.log_timings(id, kernel, dest_size, renderer);

        if !on_screen {
            trace!(
                target: GPU_LOG_TARGET,
                gpu = ?GpuDebug::renderer(renderer),
                ?dest_size,
                "element is offscreen; skipping super-resolution pass",
            );
            let mut element = SuperResElement::plain(elem);
            if let Some(rect) = magnify {
                element = element.with_magnify(rect);
            }
            return element;
        }

        if gpu_diagnostics() {
            let native = match elem.texture() {
                WaylandSurfaceTexture::Texture(texture) => {
                    texture.pass_texture(&renderer.context_id())
                }
                WaylandSurfaceTexture::SolidColor(_) => None,
            };
            // A client texture with no native texture in this context is the window that goes
            // through Smithay's import/copy path, which is what the Hybrid work has to see. The
            // healthy case is the same entry for every window of every frame, so it stays out of
            // the default file.
            let native_is_missing =
                native.is_none() && matches!(elem.texture(), WaylandSurfaceTexture::Texture(_));
            macro_rules! log_input {
                ($level:ident) => {
                    $level!(
                        target: GPU_LOG_TARGET,
                        gpu = ?GpuDebug::renderer(renderer),
                        kernel = %kernel,
                        client_buffer = ?ClientBufferDebug(elem.buffer()),
                        native_texture = ?native.as_ref().map(TextureDebug),
                        src = ?elem.src(),
                        copy_path = "unknown (Smithay internal; native availability is not zero-copy)",
                        sync_state = "not inspected; native lookup does not wait",
                        "super-resolution input in execution context",
                    );
                };
            }
            if native_is_missing {
                log_input!(debug);
            } else {
                log_input!(trace);
            }
            if let WaylandSurfaceTexture::Texture(texture) = elem.texture() {
                trace!(
                    target: GPU_LOG_TARGET,
                    smithay_texture = ?texture,
                    "raw import/cache state (not a stable path classifier)",
                );
            }
        }

        // A buffer that already matches its display area has nothing for the effect to recover: the
        // client is rendering at the physical resolution of the output, which is what Wayland
        // clients that handle fractional scaling do. There is at most a pixel in it, so draw the
        // window the normal way rather than paying for a pass that cannot change anything. The
        // magnify mode is a different thing, since its content is drawn into a bigger area on
        // purpose.
        if magnify.is_none() && !has_something_to_magnify(elem.src().size, dest_size) {
            trace!(
                target: GPU_LOG_TARGET,
                gpu = ?GpuDebug::renderer(renderer),
                src = ?elem.src(),
                ?dest_size,
                "nothing to magnify; selecting renderer-default sampling",
            );
            let src = elem.src().size;
            if self.log_once(logged::NOTHING_TO_MAGNIFY) {
                debug!(
                    "super resolution: window {id} has nothing to magnify: \
                     the buffer is {src_w}×{src_h} and the display area is {dst_w}×{dst_h}, \
                     drawing it the normal way",
                    src_w = src.w as i32,
                    src_h = src.h as i32,
                    dst_w = dest_size.w,
                    dst_h = dest_size.h,
                );
            }

            return SuperResElement::plain(elem);
        }

        let params = match kernel {
            SuperResKernel::Cunny(variant) => {
                let Some(result) = self.cunny_result(renderer, variant, &elem, scale, id) else {
                    trace!(
                        target: GPU_LOG_TARGET,
                        requested_kernel = %kernel,
                        fallback_kernel = "bicubic",
                        gpu = ?GpuDebug::renderer(renderer),
                        "CuNNy result unavailable; selecting single-pass fallback",
                    );
                    if self.log_once(logged::NO_PASS) {
                        debug!(
                            "super resolution: cannot run {kernel} for window {id}, \
                             drawing it with {kernel_alt}",
                            kernel_alt = SuperResKernel::Filter(ResampleFilter::Bicubic),
                        );
                    }

                    // Fall back to the best single-pass kernel that we have, still magnifying.
                    let mut element = SuperResElement::new(elem, ResampleFilter::Bicubic);
                    if let Some(rect) = magnify {
                        element = element.with_magnify(rect);
                    }
                    return element.with_timings(self.timings.clone());
                };

                let mut element = SuperResElement::cunny(elem, result.texture, result.context_id);
                if let Some(rect) = magnify {
                    element = element.with_magnify(rect);
                }
                return element.with_timings(self.timings.clone());
            }
            SuperResKernel::Filter(filter) => {
                let mut element = SuperResElement::new(elem, filter);
                element = match magnify {
                    Some(rect) => element.with_magnify(rect),
                    None => element,
                };
                return element.with_timings(self.timings.clone());
            }
            SuperResKernel::Fsr1(params) => params,
        };

        let Some(texture) = self.fsr1_result(renderer, &elem, dest_size, scale, id) else {
            trace!(
                target: GPU_LOG_TARGET,
                requested_kernel = %kernel,
                fallback_kernel = "bicubic",
                gpu = ?GpuDebug::renderer(renderer),
                "EASU result unavailable; selecting single-pass fallback",
            );
            if self.log_once(logged::NO_PASS) {
                debug!(
                    "super resolution: cannot run {kernel} for window {id}, \
                     drawing it with {kernel_alt}",
                    kernel_alt = SuperResKernel::Filter(ResampleFilter::Bicubic),
                );
            }

            // Fall back to the best single-pass kernel that we have, still magnifying.
            let mut element = SuperResElement::new(elem, ResampleFilter::Bicubic);
            if let Some(rect) = magnify {
                element = element.with_magnify(rect);
            }
            return element.with_timings(self.timings.clone());
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
        element.with_timings(self.timings.clone())
    }

    /// Logs how long this window's passes took, once per interval.
    fn log_timings(
        &self,
        id: u64,
        kernel: SuperResKernel,
        size: Size<i32, Physical>,
        renderer: &GlesRenderer,
    ) {
        let Some(report) = self.timings.take_report() else {
            return;
        };

        debug!(
            gpu = ?GpuDebug::renderer(renderer),
            "super resolution: window {id} {kernel}, {w}×{h}: {timings}",
            w = size.w,
            h = size.h,
            timings = format_timings(&report),
        );
        debug!(
            target: GPU_LOG_TARGET,
            window = id,
            gpu = ?GpuDebug::renderer(renderer),
            kernel = %kernel,
            ?size,
            timings = format_timings(&report),
            coverage = "EASU + final draw; excludes CuNNy network/import/copy/output waits",
            "GPU timer-query report (not end-to-end latency or GPU utilization)",
        );
    }

    /// Returns the cached EASU result, running the pass if it is stale.
    fn fsr1_result<R: NiriRenderer>(
        &self,
        renderer: &mut GlesRenderer,
        elem: &WaylandSurfaceRenderElement<R>,
        size: Size<i32, Physical>,
        scale: Scale<f64>,
        id: u64,
    ) -> Option<Fsr1Result>
    where
        R::TextureId: PassSource,
    {
        let commit = elem.current_commit();
        let _span = debug_span!(
            target: GPU_LOG_TARGET,
            "easu",
            window = id,
            element = ?elem.id(),
            ?commit,
        )
        .entered();
        if size.w <= 0 || size.h <= 0 {
            return None;
        }

        let src = elem.src();
        let src_size = src.size.to_i32_round();
        if src_size.w <= 0 || src_size.h <= 0 {
            return None;
        }

        let key = Fsr1Key { src, size };
        let context_id = renderer.context_id();

        if let Some((reuse, result)) =
            self.fsr1
                .borrow_mut()
                .probe_before_import(&key, commit, &context_id, elem, scale)
        {
            match reuse {
                CacheReuse::SameCommit => {
                    trace!(
                        target: GPU_LOG_TARGET,
                        cache = "hit",
                        gpu = ?GpuDebug::renderer(renderer),
                        result = ?TextureDebug(&result.texture),
                        "reusing EASU result; EASU not rerun",
                    );
                    trace!(
                        "super resolution: reusing the result for window {id}, \
                         {w}×{h}",
                        w = size.w,
                        h = size.h,
                    );
                }
                CacheReuse::NoDamage { made_for } | CacheReuse::SameDigest { made_for } => {
                    trace!(
                        target: GPU_LOG_TARGET,
                        cache = "unchanged",
                        window = id,
                        ?commit,
                        ?made_for,
                        gpu = ?GpuDebug::renderer(renderer),
                        "the window has no damage since the pass ran; reusing its result",
                    );
                }
            }
            return Some(result);
        }

        let WaylandSurfaceTexture::Texture(texture) = elem.texture() else {
            return None;
        };
        let Some(source) = texture.pass_texture(&context_id) else {
            debug!(
                target: GPU_LOG_TARGET,
                gpu = ?GpuDebug::renderer(renderer),
                reason = "no native texture for execution context",
                "offscreen pass unavailable; falling back",
            );
            return None;
        };

        let y_inverted = buffer_y_inverted(elem.buffer()).unwrap_or(false);
        let workload = PassWorkload::single_pass(src_size, size);

        let digest = match self.fsr1.borrow_mut().probe_digest(
            &key,
            commit,
            &context_id,
            workload,
            &mut self.digest.borrow_mut(),
            renderer,
            &source,
            src,
            y_inverted,
            elem.transform(),
        ) {
            Ok((CacheReuse::SameDigest { made_for }, result)) => {
                trace!(
                    target: GPU_LOG_TARGET,
                    cache = "unchanged",
                    reason = "source digest matched",
                    window = id,
                    ?commit,
                    ?made_for,
                    gpu = ?GpuDebug::renderer(renderer),
                    result = ?TextureDebug(&result.texture),
                    "the client buffer pixels did not change; reusing EASU result",
                );
                return Some(result);
            }
            Ok((_, result)) => return Some(result),
            Err(digest) => digest,
        };

        let reason = self.cache_miss_reason(&key, commit, renderer);
        trace!(
            "super resolution: resampling window {id}, {src_w}×{src_h} -> {dst_w}×{dst_h} \
             (×{ratio:.3}, {reason})",
            src_w = src.size.w as i32,
            src_h = src.size.h as i32,
            dst_w = size.w,
            dst_h = size.h,
            ratio = f64::from(size.w) / src.size.w,
        );

        let uniforms = resample_uniforms(&source, src, y_inverted)?;
        let Some(program) =
            SuperResPrograms::get(renderer).and_then(|programs| programs.easu.clone())
        else {
            debug!(target: GPU_LOG_TARGET, gpu = ?GpuDebug::renderer(renderer), reason = "EASU shader unavailable in execution context", "offscreen pass unavailable; falling back");
            return None;
        };

        debug!(
            target: GPU_LOG_TARGET,
            cache = "miss",
            reason,
            gpu = ?GpuDebug::renderer(renderer),
            source = ?TextureDebug(&source),
            "submitting EASU",
        );
        let texture = match run_easu_pass(
            renderer,
            &source,
            src,
            size,
            elem.transform(),
            &program,
            &uniforms,
            &self.timings,
        ) {
            Ok(texture) => texture,
            Err(err) => {
                debug!(
                    target: GPU_LOG_TARGET,
                    window = id,
                    ?size,
                    error = ?err,
                    "EASU run failed; drawing this window with the fallback",
                );
                if self.log_once(logged::PASS_FAILURE) {
                    warn!("super resolution: error running the pass for window {id}: {err:?}");
                }
                return None;
            }
        };

        let result = Fsr1Result {
            texture,
            context_id: context_id.clone(),
        };
        self.fsr1
            .borrow_mut()
            .store(key, commit, context_id, digest, result.clone());

        Some(result)
    }
}

impl SuperResolutionPipeline {
    /// Explains why the cached EASU result cannot be reused.
    fn cache_miss_reason(
        &self,
        key: &Fsr1Key,
        commit: CommitCounter,
        renderer: &GlesRenderer,
    ) -> &'static str {
        let cache = self.fsr1.borrow();
        let Some(entry) = cache.entry.as_ref() else {
            return "no cached result";
        };

        if entry.context_id != renderer.context_id() {
            "different renderer context"
        } else if entry.key.src != key.src {
            "different source rect"
        } else if entry.key.size != key.size {
            "different display size"
        } else if entry.commit != commit {
            "new source commit"
        } else {
            "unknown"
        }
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
    timings: &Timings,
) -> anyhow::Result<GlesTexture> {
    let buffer_size = size.to_logical(1).to_buffer(1, Transform::Normal);
    let mut texture = renderer
        .create_buffer(Fourcc::Abgr8888, buffer_size)
        .context("error creating texture")?;
    debug!(
        target: GPU_LOG_TARGET,
        pass = "easu",
        gpu = ?GpuDebug::renderer(renderer),
        source = ?TextureDebug(source),
        intermediate = ?TextureDebug(&texture),
        "allocated offscreen result in execution context (not exported as DMA-BUF by these passes)",
    );

    let rect = Rectangle::from_size(size);

    {
        let mut target = renderer
            .bind(&mut texture)
            .context("error binding texture")?;
        let mut frame = renderer
            .render(&mut target, size, Transform::Normal)
            .context("error starting frame")?;

        // The whole offscreen pass is measured, including the clear: it is the cost of running
        // EASU for one client frame.
        timings.begin(&mut frame, Pass::Easu);

        let mut result = frame
            .clear(Color32F::TRANSPARENT, &[rect])
            .context("error clearing");
        if result.is_ok() {
            result = frame
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
                .context("error in the EASU pass");
        }

        timings.end(&mut frame);
        result?;

        let sync = frame.finish().context("error finishing frame")?;
        drop(target);
        trace!(
            target: GPU_LOG_TARGET,
            pass = "easu",
            gpu = ?GpuDebug::renderer(renderer),
            result = ?TextureDebug(&texture),
            completion_fence = sync.contains_fence(),
            "EASU finish returned; diagnostics add no waits (renderer may synchronize internally)",
        );
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
    fn diagnostics_names_the_selected_single_pass() {
        assert_eq!(ElementKernel::Plain.draw_pass(), "renderer-default");
        assert_eq!(
            ElementKernel::Filter(ResampleFilter::Bilinear).draw_pass(),
            "bilinear"
        );
        assert_eq!(
            ElementKernel::Filter(ResampleFilter::Bicubic).draw_pass(),
            "bicubic"
        );
    }

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
            (SuperResKernel::Cunny(CunnyVariant::Fast), false)
        );
        assert_eq!(
            SuperResKernel::for_algorithm(UpscalingAlgorithm::Cunny2Large, 0.2),
            (SuperResKernel::Cunny(CunnyVariant::ThreeByTwelve), false)
        );

        // The larger network is implemented but not selectable yet, so that the algorithm the
        // window rules can name keeps being the one that has been measured.
        assert_eq!(
            SuperResKernel::for_algorithm(UpscalingAlgorithm::Anime4k, 0.2),
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
        assert_eq!(pipeline.kernel(&Upscaling::default(), 1), None);
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
            pipeline.kernel(&rules, 1),
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
            pipeline.kernel(&rules, 1),
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
            pipeline.kernel(&rules, 1),
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
        assert_eq!(pipeline.kernel(&rules, 1), None);
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
    fn timing_report_aggregates_and_resets() {
        let start = Instant::now();
        let mut inner = Inner::default();

        // Results come back a few frames after the pass, so there is nothing to log yet.
        assert!(inner
            .report(start + Duration::from_millis(500), TIMING_INTERVAL)
            .is_none());

        inner.record(Pass::Easu, 1_000_000, start);
        inner.record(Pass::Easu, 5_000_000, start);
        inner.record(Pass::Draw, 2_000_000, start);

        let report = inner
            .report(start + Duration::from_millis(1500), TIMING_INTERVAL)
            .expect("expected a report");
        assert_eq!(report.passes.len(), 2);
        assert_eq!(report.passes[0].pass, Pass::Easu);
        assert_eq!(report.passes[0].count, 2);
        assert_eq!(report.passes[0].average, Duration::from_millis(3));
        assert_eq!(report.passes[1].pass, Pass::Draw);
        assert_eq!(report.passes[1].average, Duration::from_millis(2));
        assert_eq!(report.total(), Duration::from_millis(8));
        assert_eq!(report.interval, Duration::from_millis(1500));
        assert_eq!(
            format_timings(&report),
            "easu 3.00 ms in 2 passes + draw 2.00 ms in 1 pass, 8 ms of GPU time in 1500 ms"
        );

        // What has been logged is not logged again, and the next interval starts over.
        assert!(inner
            .report(start + Duration::from_millis(1600), TIMING_INTERVAL)
            .is_none());
        assert_eq!(inner.counts, [0, 0, 0]);

        // The interval that was just logged starts a new one.
        inner.record(Pass::Easu, 1_000_000, start + Duration::from_millis(2000));
        let report = inner
            .report(start + Duration::from_millis(2600), TIMING_INTERVAL)
            .expect("expected a report");
        assert_eq!(report.passes[0].count, 1);
        assert_eq!(report.passes[0].average, Duration::from_millis(1));
        assert_eq!(report.interval, Duration::from_millis(1100));
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

    /// Compiles every generated CuNNy pass, and says what the driver thinks of it.
    ///
    /// The passes are compiled lazily, so a driver that refuses one of them only shows up as a
    /// window that quietly falls back to the resampling kernels. This is also the only way to see
    /// why: the renderer's error type does not carry the compiler's message.
    #[test]
    fn gpu_cunny_shaders_compile() {
        let Some(mut renderer) = surfaceless_renderer() else {
            return;
        };

        crate::render_helpers::resources::init(&mut renderer);
        crate::render_helpers::shaders::init(&mut renderer);
        init(&mut renderer, None);

        // The upscale is the one shader of the effect that is written by hand rather than
        // generated, so it is the one nothing else checks.
        let upscale = include_str!("shaders/superres/cunny_upscale.frag");
        let mut failures = Vec::new();
        if SuperResPrograms::compile_file(
            &mut renderer,
            "cunny upscale",
            upscale,
            &cunny_upscale_uniforms(),
        )
        .is_none()
        {
            let log = renderer
                .with_context(|gl| unsafe { compile_log(gl, upscale) })
                .unwrap_or_else(|err| format!("<{err:?}>"));
            failures.push(format!("cunny upscale: {log}"));
        }

        // The programs the renderer keeps are the ones a session draws with, so check the one that
        // matters rather than only that the source compiles.
        assert!(
            SuperResPrograms::get(&renderer)
                .expect("the init above installs the programs")
                .cunny_upscale
                .is_some(),
            "the upscale program did not survive init"
        );
        for (variant_name, variant) in [
            ("fast", CunnyVariant::Fast),
            ("3x12", CunnyVariant::ThreeByTwelve),
        ] {
            for (pass_index, pass) in variant.passes().iter().enumerate() {
                for (group, source) in pass.shaders.iter().enumerate() {
                    if SuperResPrograms::compile_file(
                        &mut renderer,
                        "cunny",
                        source,
                        &cunny_pass_uniforms(),
                    )
                    .is_some()
                    {
                        continue;
                    }

                    let log = renderer
                        .with_context(|gl| unsafe { compile_log(gl, source) })
                        .unwrap_or_else(|err| format!("<{err:?}>"));
                    failures.push(format!(
                        "{variant_name} pass{} group{group}: {log}",
                        pass_index + 1
                    ));
                }
            }
        }

        assert!(
            failures.is_empty(),
            "the driver refused {} of the passes:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    /// Runs a network over a small source, which is what the uniform declarations are for.
    ///
    /// Compiling says nothing about this: a program that was compiled without the uniforms it is
    /// drawn with compiles fine and then fails every draw, which is how the network came to run on
    /// every frame and be thrown away on every frame.
    #[test]
    fn gpu_cunny_network_runs() {
        let Some(mut renderer) = surfaceless_renderer() else {
            return;
        };

        let size = Size::<i32, Buffer>::from((64, 36));

        // The residuals are half-float, and a context that cannot render to one cannot run the
        // network at all.
        let half_float_target: Result<GlesTexture, _> =
            renderer.create_buffer(Fourcc::Abgr16161616f, size);
        if half_float_target.is_err() {
            return;
        }
        crate::render_helpers::resources::init(&mut renderer);
        crate::render_helpers::shaders::init(&mut renderer);
        init(&mut renderer, None);

        let timings = Timings::disabled();
        let src_rect = Rectangle::from_size(size.to_f64());
        let dest = Size::<i32, Physical>::from((size.w * 2, size.h * 2));
        let dest_buf = dest.to_logical(1).to_buffer(1, Transform::Normal);
        let upscale = SuperResPrograms::get(&renderer)
            .unwrap()
            .cunny_upscale
            .clone()
            .expect("the CuNNy upscale shader must compile");

        let render_upscaled = |renderer: &mut GlesRenderer,
                               source: &GlesTexture,
                               residuals: &GlesTexture,
                               out_size: Size<i32, Physical>|
         -> Vec<u8> {
            let out_buf = out_size.to_logical(1).to_buffer(1, Transform::Normal);
            let mut out = renderer.create_buffer(Fourcc::Abgr8888, out_buf).unwrap();
            {
                let rect = Rectangle::from_size(out_size);
                let mut target = renderer.bind(&mut out).unwrap();
                let mut frame = renderer
                    .render(&mut target, out_size, Transform::Normal)
                    .unwrap();
                frame.clear(Color32F::TRANSPARENT, &[rect]).unwrap();
                frame
                    .with_context(|gl| unsafe {
                        gl.ActiveTexture(ffi::TEXTURE1);
                        configure_pass_texture(gl, residuals.tex_id(), ffi::NEAREST);
                        gl.ActiveTexture(ffi::TEXTURE0);
                    })
                    .unwrap();
                frame
                    .render_texture_from_to(
                        source,
                        src_rect,
                        rect,
                        &[rect],
                        &[rect],
                        Transform::Normal,
                        1.0,
                        Some(&upscale),
                        &[
                            Uniform::new("niri_tex_size", (size.w as f32, size.h as f32)),
                            Uniform::new("niri_src_rect", [0.0, 0.0, size.w as f32, size.h as f32]),
                            Uniform::new(
                                "niri_footprint",
                                [
                                    2.0 * (size.w as f32 / out_size.w as f32),
                                    2.0 * (size.h as f32 / out_size.h as f32),
                                ],
                            ),
                            Uniform::new("niri_tex1", 1i32),
                        ],
                    )
                    .unwrap();
                let _ = frame.finish().unwrap();
            }
            read_back(renderer, &mut out, out_size)
        };

        // 1. Flat dark input: an incomplete texture (sampling as (0, 0, 0, 1)) or enabled alpha
        // blending on the fourth channel turns every (1, 1) sub-pixel white (255) instead of ~50.
        let flat_pixels = vec![50u8, 50, 50, 255].repeat((size.w * size.h) as usize);
        let flat_source = renderer
            .import_memory(&flat_pixels, Fourcc::Abgr8888, size, false)
            .expect("error importing flat source");
        let (features, flat_residuals) = run_cunny(
            &mut renderer,
            CunnyVariant::Fast,
            &flat_source,
            src_rect,
            size,
            Transform::Normal,
            &timings,
            None,
        )
        .expect("error running Fast network on flat input");
        let flat_out = render_upscaled(&mut renderer, &flat_source, &flat_residuals, dest);
        // Check interior 2x2 blocks away from the border clamp.
        for y in 8..(dest.h - 8) {
            for x in 8..(dest.w - 8) {
                let idx = ((y * dest.w + x) * 4) as usize;
                for c in 0..3 {
                    let got = flat_out[idx + c];
                    assert!(
                        got.abs_diff(50) <= 3,
                        "flat input produced checkerboard artifact at ({x}, {y}) ch {c}: got {got}, expected ~50"
                    );
                }
            }
        }

        // 2. Edge/gradient input: reuse the intermediate textures, verify that the network's
        // residuals actually take effect (differing from plain bilinear on edges) and that both
        // 2x and non-integer (<2x) footprints produce bounded output without checkerboard spikes.
        let pixels = grayscale_pixels(size);
        let source = renderer
            .import_memory(&pixels, Fourcc::Abgr8888, size, false)
            .expect("error importing the source");
        let (_, residuals) = run_cunny(
            &mut renderer,
            CunnyVariant::Fast,
            &source,
            src_rect,
            size,
            Transform::Normal,
            &timings,
            Some((features, flat_residuals)),
        )
        .expect("error running Fast network on edge input");

        let cunny_2x = render_upscaled(&mut renderer, &source, &residuals, dest);
        let bilinear_2x = resample(&mut renderer, &source, src_rect, dest, None);
        let _ = dest_buf;
        let max_diff = cunny_2x
            .iter()
            .zip(bilinear_2x.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(
            max_diff > 0 && max_diff < 120,
            "expected non-zero bounded residual reconstruction, got max_diff={max_diff}"
        );

        // Non-integer scale (<2x, so `niri_footprint > 1.0` averages sub-pixels).
        let dest_1_5x = Size::<i32, Physical>::from((size.w * 3 / 2, size.h * 3 / 2));
        let cunny_1_5x = render_upscaled(&mut renderer, &source, &residuals, dest_1_5x);
        let bilinear_1_5x = resample(&mut renderer, &source, src_rect, dest_1_5x, None);
        let max_diff_1_5x = cunny_1_5x
            .iter()
            .zip(bilinear_1_5x.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(
            max_diff_1_5x > 0 && max_diff_1_5x < 120,
            "expected bounded 1.5x reconstruction, got max_diff={max_diff_1_5x}"
        );

        // 3. 3x12-NVL variant: verify that the M3x4 input pass and 12x12 shuffle weights produce
        // non-zero bounded residuals too.
        let (_, residuals_3x12) = run_cunny(
            &mut renderer,
            CunnyVariant::ThreeByTwelve,
            &source,
            src_rect,
            size,
            Transform::Normal,
            &timings,
            None,
        )
        .expect("error running 3x12 network on edge input");
        let cunny_3x12_2x = render_upscaled(&mut renderer, &source, &residuals_3x12, dest);
        let max_diff_3x12 = cunny_3x12_2x
            .iter()
            .zip(bilinear_2x.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        assert!(
            max_diff_3x12 > 0 && max_diff_3x12 < 120,
            "expected non-zero bounded 3x12 reconstruction, got max_diff={max_diff_3x12}"
        );
    }

    /// Compiles one fragment shader and returns the driver's log for it.
    ///
    /// # Safety
    ///
    /// Must be called with a current GLES context.
    unsafe fn compile_log(gl: &ffi::Gles2, source: &str) -> String {
        let shader = gl.CreateShader(ffi::FRAGMENT_SHADER);
        let ptr = source.as_ptr().cast();
        gl.ShaderSource(shader, 1, &ptr, std::ptr::null());
        gl.CompileShader(shader);

        let mut status = 0;
        gl.GetShaderiv(shader, ffi::COMPILE_STATUS, &mut status);
        if status != 0 {
            gl.DeleteShader(shader);
            return String::from("(compiled on its own?!)");
        }

        let mut log = [0u8; 4096];
        let mut len = 0;
        gl.GetShaderInfoLog(shader, log.len() as i32, &mut len, log.as_mut_ptr().cast());
        gl.DeleteShader(shader);

        let len = (len.max(0) as usize).min(log.len());
        String::from_utf8_lossy(&log[..len]).to_string()
    }

    /// Measures a real pass with timer queries, the way a session does.
    #[test]
    fn gpu_timings_measure_a_pass() {
        let Some(mut renderer) = surfaceless_renderer() else {
            return;
        };

        crate::render_helpers::resources::init(&mut renderer);
        crate::render_helpers::shaders::init(&mut renderer);
        init(&mut renderer, None);

        // Drivers that cannot measure have nothing to report, and that is not a failure here.
        if !renderer.with_context(can_time).unwrap_or(false) {
            return;
        }

        let src_size = Size::from((8, 8));
        let pixels = grayscale_pixels(src_size);
        let texture = renderer
            .import_memory(&pixels, Fourcc::Abgr8888, src_size, false)
            .unwrap();

        let src_rect = Rectangle::from_size(src_size.to_f64());
        let dest = Size::<i32, Physical>::from((16, 16));

        let programs = SuperResPrograms::get(&renderer).unwrap();
        let easu = programs.easu.clone().expect("the EASU shader must compile");
        let uniforms = resample_uniforms(&texture, src_rect, false).unwrap();

        // The passes are only measured when their log line would be visible, so pretend that it is.
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .finish();

        let report = tracing::subscriber::with_default(subscriber, || {
            let timings = Arc::new(Timings {
                enabled: true,
                inner: Mutex::new(Inner::default()),
            });

            // A result is read when the driver hands it back, which is what a later pass does, so
            // run a few passes. The test cannot wait for frames, so make the GPU finish instead.
            for _ in 0..4 {
                run_easu_pass(
                    &mut renderer,
                    &texture,
                    src_rect,
                    dest,
                    Transform::Normal,
                    &easu,
                    &uniforms,
                    &timings,
                )
                .unwrap();

                let _ = renderer.with_context(|gl| unsafe { gl.Finish() });
            }

            let mut inner = timings.inner.lock().unwrap();
            inner.report(Instant::now(), Duration::ZERO)
        });

        let report = report.expect("expected the passes to be measured");
        assert_eq!(report.passes.len(), 1);
        assert_eq!(report.passes[0].pass, Pass::Easu);
        assert!(report.passes[0].count >= 1, "{report:?}");
        assert!(report.passes[0].average > Duration::ZERO, "{report:?}");
        assert!(
            report.passes[0].average < Duration::from_secs(1),
            "{report:?}"
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

    /// A writer that collects the log output of a test.
    #[derive(Clone, Default)]
    struct LogBuffer(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl LogBuffer {
        fn contents(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }

    impl std::io::Write for LogBuffer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogBuffer {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Checks that the effect explains itself once per configuration change, and not per frame.
    #[test]
    fn pipeline_logs_once_per_change() {
        let log = LogBuffer::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(log.clone())
            .with_ansi(false)
            .without_time()
            .with_max_level(tracing::Level::DEBUG)
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            let pipeline = SuperResolutionPipeline::new();
            let fsr1 = Upscaling {
                mode: None,
                algorithm: Some(UpscalingAlgorithm::Fsr1),
                sharpness: Some(0.4),
                client_scale: None,
                virtual_size: None,
            };

            // Called for every frame, so it must only log when the result changes.
            for _ in 0..3 {
                assert_eq!(
                    pipeline.kernel(&fsr1, 42),
                    Some(SuperResKernel::Fsr1(Fsr1Params { sharpness: 0.4 }))
                );
            }
            let logged = log.contents();
            assert_eq!(
                logged
                    .lines()
                    .filter(|line| line.contains("window 42 uses"))
                    .count(),
                1,
                "{logged}"
            );
            assert!(logged.contains("fsr1 (sharpness 0.4)"), "{logged}");
            assert!(logged.contains("window 42"), "{logged}");

            // Changing the sharpness is a different configuration.
            let sharper = Upscaling {
                sharpness: Some(0.1),
                ..fsr1
            };
            let _ = pipeline.kernel(&sharper, 42);

            // Turning the effect off says so.
            let off = Upscaling {
                mode: Some(UpscalingMode::Off),
                ..fsr1
            };
            assert_eq!(pipeline.kernel(&off, 42), None);

            let logged = log.contents();
            assert_eq!(
                logged
                    .lines()
                    .filter(|line| line.contains("window 42 uses"))
                    .count(),
                2,
                "{logged}"
            );
            assert!(
                logged.contains("back to the renderer's own sampling"),
                "{logged}"
            );
        });
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

        let timings = Timings::disabled();
        let mut upsampled = run_easu_pass(
            &mut renderer,
            &texture,
            src_rect,
            dest,
            Transform::Normal,
            &easu,
            &uniforms,
            &timings,
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

    #[test]
    fn pass_workload_digest_threshold() {
        // Below the 32x32 source floor: never run the digest.
        assert!(!PassWorkload::multi_pass(Size::from((16, 16)), 12).should_use_digest());

        // Single-pass EASU below 500k destination pixels (e.g. 640x360 = 230,400): cheaper to
        // run EASU directly than to synchronize a digest readback.
        assert!(
            !PassWorkload::single_pass(Size::from((320, 180)), Size::from((640, 360)))
                .should_use_digest()
        );

        // Single-pass EASU at 960x540 (518,400 destination texels): above threshold.
        assert!(
            PassWorkload::single_pass(Size::from((480, 270)), Size::from((960, 540)))
                .should_use_digest()
        );

        // CuNNy-Fast (9 passes) at 240x135 (9 * 32,400 = 291,600 pass texels): below threshold.
        assert!(
            !PassWorkload::multi_pass(Size::from((240, 135)), CunnyVariant::Fast.draw_count())
                .should_use_digest()
        );

        // CuNNy-Fast (9 passes) at 320x180 (9 * 57,600 = 518,400 pass texels): above threshold.
        assert!(
            PassWorkload::multi_pass(Size::from((320, 180)), CunnyVariant::Fast.draw_count())
                .should_use_digest()
        );
    }

    #[test]
    fn gpu_source_digest_detects_unchanged_and_changed_frames() {
        let Some(mut renderer) = surfaceless_renderer() else {
            return;
        };
        crate::render_helpers::resources::init(&mut renderer);
        crate::render_helpers::shaders::init(&mut renderer);
        init(&mut renderer, None);

        let size = Size::<i32, Buffer>::from((320, 180));
        let src_rect = Rectangle::from_size(size.to_f64());
        let pixels_a = grayscale_pixels(size);
        let tex_a = renderer
            .import_memory(&pixels_a, Fourcc::Abgr8888, size, false)
            .unwrap();
        // Separate swapchain texture carrying the exact same pixels.
        let tex_b = renderer
            .import_memory(&pixels_a, Fourcc::Abgr8888, size, false)
            .unwrap();
        assert_ne!(tex_a.tex_id(), tex_b.tex_id());

        let mut target = DigestTarget::default();
        let digest_a = target
            .compute(&mut renderer, &tex_a, src_rect, false, Transform::Normal)
            .expect("digest must succeed");
        let digest_b = target
            .compute(&mut renderer, &tex_b, src_rect, false, Transform::Normal)
            .expect("digest must succeed");
        assert_eq!(
            digest_a, digest_b,
            "identical pixels in different swapchain textures must produce the same digest"
        );

        // Mutate a small 4x4 character-sized glyph patch in the dialog area.
        let mut pixels_c = pixels_a.clone();
        for dy in 120..124 {
            for dx in 160..164 {
                let idx = ((dy * size.w + dx) * 4) as usize;
                pixels_c[idx] = pixels_c[idx].wrapping_add(40);
                pixels_c[idx + 1] = pixels_c[idx + 1].wrapping_add(40);
                pixels_c[idx + 2] = pixels_c[idx + 2].wrapping_add(40);
            }
        }
        let tex_c = renderer
            .import_memory(&pixels_c, Fourcc::Abgr8888, size, false)
            .unwrap();
        let digest_c = target
            .compute(&mut renderer, &tex_c, src_rect, false, Transform::Normal)
            .expect("digest must succeed");
        assert_ne!(
            digest_a, digest_c,
            "a small glyph change must change the source digest"
        );
    }

    #[test]
    fn gpu_pass_cache_reuses_unchanged_and_respects_threshold() {
        let Some(mut renderer) = surfaceless_renderer() else {
            return;
        };
        crate::render_helpers::resources::init(&mut renderer);
        crate::render_helpers::shaders::init(&mut renderer);
        init(&mut renderer, None);

        let size = Size::<i32, Buffer>::from((320, 180));
        let src_rect = Rectangle::from_size(size.to_f64());
        let pixels_a = grayscale_pixels(size);
        let tex_a = renderer
            .import_memory(&pixels_a, Fourcc::Abgr8888, size, false)
            .unwrap();
        let tex_b = renderer
            .import_memory(&pixels_a, Fourcc::Abgr8888, size, false)
            .unwrap();
        let mut pixels_c = pixels_a.clone();
        pixels_c[0] = pixels_c[0].wrapping_add(80);
        let tex_c = renderer
            .import_memory(&pixels_c, Fourcc::Abgr8888, size, false)
            .unwrap();

        let key = CunnyKey {
            variant: CunnyVariant::Fast,
            src: src_rect,
            size,
        };
        let context_id = renderer.context_id();
        let workload = PassWorkload::multi_pass(size, CunnyVariant::Fast.draw_count());
        let mut digest_target = DigestTarget::default();
        let mut cache: PassCache<CunnyKey, u32> = PassCache::default();

        let mut c1 = CommitCounter::default();
        c1.increment();
        let mut c2 = c1;
        c2.increment();
        let mut c3 = c2;
        c3.increment();

        // First commit: cache is empty, computes digest and returns Err(Some(digest)).
        let digest_1 = cache
            .probe_digest(
                &key,
                c1,
                &context_id,
                workload,
                &mut digest_target,
                &mut renderer,
                &tex_a,
                src_rect,
                false,
                Transform::Normal,
            )
            .expect_err("first commit must miss");
        assert!(digest_1.is_some());
        cache.store(key.clone(), c1, context_id.clone(), digest_1, 77);

        // Second commit with a different texture handle carrying identical pixels: digest hit!
        let (reuse, val) = cache
            .probe_digest(
                &key,
                c2,
                &context_id,
                workload,
                &mut digest_target,
                &mut renderer,
                &tex_b,
                src_rect,
                false,
                Transform::Normal,
            )
            .expect("second commit with identical pixels must hit via digest");
        assert_eq!(reuse, CacheReuse::SameDigest { made_for: c1 });
        assert_eq!(val, 77);
        assert_eq!(
            cache.entry.as_ref().unwrap().commit,
            c2,
            "digest hit must advance the cached CommitCounter"
        );

        // Third commit with modified pixels: digest mismatch -> cache miss.
        let digest_3 = cache
            .probe_digest(
                &key,
                c3,
                &context_id,
                workload,
                &mut digest_target,
                &mut renderer,
                &tex_c,
                src_rect,
                false,
                Transform::Normal,
            )
            .expect_err("modified pixels must miss");
        assert!(digest_3.is_some());

        // Below-threshold workload skips digest computation completely.
        let small_workload = PassWorkload::single_pass(size, Size::from((640, 360)));
        let below = cache
            .probe_digest(
                &key,
                c3,
                &context_id,
                small_workload,
                &mut digest_target,
                &mut renderer,
                &tex_b,
                src_rect,
                false,
                Transform::Normal,
            )
            .expect_err("below-threshold workload must not hit via digest");
        assert!(below.is_none());
    }
}
