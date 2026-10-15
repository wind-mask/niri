use knuffel::errors::DecodeError;
use niri_ipc::ColumnDisplay;

use crate::appearance::{
    BackgroundEffect, BackgroundEffectRule, BlockOutFrom, BorderRule, CornerRadius, ShadowRule,
    TabIndicatorRule,
};
use crate::layout::DefaultPresetSize;
use crate::utils::{MergeWith, RegexEq};
use crate::FloatOrInt;

/// Size that a window's client is told to render at, in logical pixels.
///
/// This is what the `virtual-size` upscaling rule takes: the width and the height, like
/// `virtual-size 1280 720`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowSize {
    pub width: i32,
    pub height: i32,
}

impl<S> knuffel::Decode<S> for WindowSize
where
    S: knuffel::traits::ErrorSpan,
{
    fn decode_node(
        node: &knuffel::ast::SpannedNode<S>,
        ctx: &mut knuffel::decode::Context<S>,
    ) -> Result<Self, DecodeError<S>> {
        // Check for unexpected type name.
        if let Some(type_name) = &node.type_name {
            ctx.emit_error(DecodeError::unexpected(
                type_name,
                "type name",
                "no type name expected for this node",
            ));
        }

        let decode_dimension = |ctx: &mut knuffel::decode::Context<S>,
                                val: &knuffel::ast::Value<S>| {
            // Check for unexpected type name.
            if let Some(typ) = &val.type_name {
                ctx.emit_error(DecodeError::TypeName {
                    span: typ.span().clone(),
                    found: Some((**typ).clone()),
                    expected: knuffel::errors::ExpectedType::no_type(),
                    rust_type: "int",
                });
            }

            let dimension: i32 = match *val.literal {
                knuffel::ast::Literal::Int(ref x) => match x.try_into() {
                    Ok(x) => x,
                    Err(err) => {
                        ctx.emit_error(DecodeError::conversion(&val.literal, err));
                        0
                    }
                },
                _ => {
                    ctx.emit_error(DecodeError::scalar_kind(
                        knuffel::decode::Kind::Int,
                        &val.literal,
                    ));
                    0
                }
            };

            if dimension <= 0 {
                ctx.emit_error(DecodeError::conversion(&val.literal, "size must be > 0"));
            }

            dimension
        };

        let mut args = node.arguments.iter();
        let width = args
            .next()
            .ok_or_else(|| DecodeError::missing(node, "width is required"))?;
        let width = decode_dimension(ctx, width);
        let height = args
            .next()
            .ok_or_else(|| DecodeError::missing(node, "height is required"))?;
        let height = decode_dimension(ctx, height);

        for arg in args {
            ctx.emit_error(DecodeError::unexpected(
                &arg.literal,
                "argument",
                "unexpected argument",
            ));
        }

        Ok(Self { width, height })
    }
}

/// Super resolution rules for a window.
///
/// Super resolution replaces the sampling that the compositor uses to draw a client buffer that
/// is smaller than the area it is displayed in. It is off unless a rule enables it: either
/// `mode` is set to something other than `"off"`, or `algorithm` is set, which enables `filter`
/// mode with that algorithm.
///
/// As a rule of thumb, only enable it for windows that keep rendering at a lower resolution than
/// their display area, since that is the only case where it can do anything.
#[derive(knuffel::Decode, Debug, Default, Clone, Copy, PartialEq)]
pub struct UpscalingRule {
    /// What to do with the window's content.
    ///
    /// - `"off"`: don't apply super resolution. This is the default.
    /// - `"filter"`: keep the window's display geometry, but replace the sampling used to magnify
    ///   its buffer.
    /// - `"magnify"`: magnify the window's content to a larger display area. Put the window in
    ///   fullscreen and niri fills the output with its content instead, without resizing the
    ///   client, keeping the aspect ratio and mapping input back into the client's coordinates.
    ///   With `virtual-size`, this happens wherever the window is, and the client always renders
    ///   the configured size.
    #[knuffel(child, unwrap(argument))]
    pub mode: Option<UpscalingMode>,

    /// Resampling algorithm to use.
    ///
    /// `"fsr1"` runs FSR1 (EASU then RCAS) and uses `sharpness`, and `"cunny2"` runs a CuNNy
    /// network, which reconstructs the image instead of resampling it and is meant for
    /// illustrations and text. `"anime4k"` is not implemented yet and currently falls back to
    /// `"bicubic"`.
    #[knuffel(child, unwrap(argument))]
    pub algorithm: Option<UpscalingAlgorithm>,

    /// Sharpening strength for algorithms that sharpen.
    ///
    /// This is FSR1's RCAS sharpness, in stops of halving: 0 is maximum sharpening, 1 is half of
    /// it, and 2 is a quarter. Defaults to 0.2, which is what AMD's FSR1 sample uses.
    #[knuffel(child, unwrap(argument))]
    pub sharpness: Option<FloatOrInt<0, 2>>,

    /// Scale to advertise to this window's client.
    ///
    /// By default, niri tells every client the scale of the output it is on, and clients that
    /// support fractional scaling render that many pixels per logical pixel. Setting this
    /// overrides the value for this window's client, so it renders at a different pixel density: a
    /// value below the output's scale makes the client render fewer pixels than the area it is
    /// displayed in has, which is exactly what super resolution magnifies.
    ///
    /// For example, with `client-scale 1.0` on an output with `scale 1.5`, the client renders one
    /// pixel per logical pixel, and niri magnifies it 1.5× while drawing it (with super
    /// resolution, if it is enabled for this window). This is meant for games and similar
    /// windows that would otherwise render at the output's full resolution.
    ///
    /// Note that this is a lie the client cannot tell apart from a smaller output scale, so
    /// everything the client draws itself, including its decorations and text, is sized for it
    /// too.
    ///
    /// Zero means "use the output's scale", which is the default.
    #[knuffel(child, unwrap(argument))]
    pub client_scale: Option<FloatOrInt<0, 10>>,

    /// Size to tell this window's client to render at, in logical pixels.
    ///
    /// Two arguments, the width and the height, like `virtual-size 1280 720`. The client is
    /// configured with this size instead of the area the layout gives the window, so it renders at
    /// a resolution of your choosing while niri draws its content into that area. This is the
    /// other half of "a game that renders at a lower resolution than the screen":
    /// `client-scale` decides how many pixels the client renders per logical pixel, and this
    /// decides the logical size it renders at.
    ///
    /// Setting it turns on the magnify mode, since that is the only mode where it does anything.
    ///
    /// The size is only used when it fits in the area the layout gives the window. A virtual size
    /// that is bigger than the area in both dimensions would only make the window overflow it, so
    /// such a window is left alone, and its client is given the area's size like any other window.
    #[knuffel(child)]
    pub virtual_size: Option<WindowSize>,
}

/// Resolved super resolution rules.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Upscaling {
    /// What to do with the window's content. `None` means unset.
    pub mode: Option<UpscalingMode>,

    /// Resampling algorithm. `None` means unset.
    pub algorithm: Option<UpscalingAlgorithm>,

    /// Sharpening strength, in FSR1 RCAS stops. `None` means unset.
    pub sharpness: Option<f64>,

    /// Scale to advertise to the window's client. `None` means unset.
    pub client_scale: Option<f64>,

    /// Size to tell the window's client to render at. `None` means unset.
    pub virtual_size: Option<WindowSize>,
}

impl Upscaling {
    /// Returns whether super resolution should run for this window.
    pub fn enabled(&self) -> bool {
        match self.mode {
            Some(UpscalingMode::Off) => false,
            Some(UpscalingMode::Filter | UpscalingMode::Magnify) => true,
            None => self.algorithm.is_some() || self.virtual_size.is_some(),
        }
    }

    /// Returns the mode to use, defaulting to `filter`.
    pub fn mode(&self) -> UpscalingMode {
        match self.mode {
            Some(mode) => mode,
            // A virtual size does nothing outside of the magnify mode, so setting it turns that on,
            // the same way setting an algorithm turns on the filter mode.
            None if self.virtual_size.is_some() => UpscalingMode::Magnify,
            // Otherwise the mode defaults to `filter`, which is what setting only an algorithm
            // gets, and what `UpscalingMode::default()` is.
            None => UpscalingMode::default(),
        }
    }

    /// Returns the algorithm to use, defaulting to bilinear.
    pub fn algorithm(&self) -> UpscalingAlgorithm {
        self.algorithm.unwrap_or_default()
    }

    /// Returns the sharpness to use, defaulting to what AMD's FSR1 sample uses.
    pub fn sharpness(&self) -> f64 {
        self.sharpness.unwrap_or(0.2)
    }

    /// Returns the scale to advertise to this window's client, if a rule overrides it.
    ///
    /// Zero (and unset) mean that the client should use the scale of the output it is on.
    pub fn client_scale(&self) -> Option<f64> {
        self.client_scale.filter(|scale| *scale > 0.)
    }

    /// Returns the size to tell this window's client to render at, if a rule sets one.
    pub fn virtual_size(&self) -> Option<(i32, i32)> {
        self.virtual_size.map(|size| (size.width, size.height))
    }
}

impl MergeWith<UpscalingRule> for Upscaling {
    fn merge_with(&mut self, part: &UpscalingRule) {
        if let Some(x) = part.mode {
            self.mode = Some(x);
        }

        if let Some(x) = part.algorithm {
            self.algorithm = Some(x);
        }

        if let Some(x) = part.sharpness {
            self.sharpness = Some(x.0);
        }

        if let Some(x) = part.client_scale {
            self.client_scale = Some(x.0);
        }

        if let Some(x) = part.virtual_size {
            self.virtual_size = Some(x);
        }
    }
}

/// What to do with a window's content.
#[derive(knuffel::DecodeScalar, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum UpscalingMode {
    /// Don't apply super resolution.
    Off,

    /// Replace the sampling used to magnify the window's buffer, without changing where the
    /// window is displayed.
    #[default]
    Filter,

    /// Magnify the window's content to a larger display area.
    Magnify,
}

/// Resampling algorithm for super resolution.
#[derive(knuffel::DecodeScalar, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum UpscalingAlgorithm {
    /// Explicit bilinear resampling.
    ///
    /// Reproduces what the renderer does anyway, so it is mainly useful as a reference and as a
    /// way to verify the execution path.
    #[default]
    Bilinear,

    /// Catmull-Rom bicubic resampling.
    Bicubic,

    /// AMD FidelityFX Super Resolution 1.
    ///
    /// Runs EASU into an offscreen texture and then RCAS on the result, using `sharpness`.
    Fsr1,

    /// CuNNy2 neural network.
    ///
    /// A network trained on illustrations, which reconstructs the image instead of resampling it.
    /// It runs at the client buffer's resolution, so its cost follows the client's resolution
    /// rather than the output's, and it does not sharpen.
    Cunny2,

    /// Anime4K neural network.
    ///
    /// Not implemented yet.
    Anime4k,
}

#[derive(knuffel::Decode, Debug, Default, Clone, PartialEq)]
pub struct WindowRule {
    #[knuffel(children(name = "match"))]
    pub matches: Vec<Match>,
    #[knuffel(children(name = "exclude"))]
    pub excludes: Vec<Match>,

    // Rules applied at initial configure.
    #[knuffel(child)]
    pub default_column_width: Option<DefaultPresetSize>,
    #[knuffel(child)]
    pub default_window_height: Option<DefaultPresetSize>,
    #[knuffel(child, unwrap(argument))]
    pub open_on_output: Option<String>,
    #[knuffel(child, unwrap(argument))]
    pub open_on_workspace: Option<String>,
    #[knuffel(child, unwrap(argument))]
    pub open_maximized: Option<bool>,
    #[knuffel(child, unwrap(argument))]
    pub open_maximized_to_edges: Option<bool>,
    #[knuffel(child, unwrap(argument))]
    pub open_fullscreen: Option<bool>,
    #[knuffel(child, unwrap(argument))]
    pub open_floating: Option<bool>,
    #[knuffel(child, unwrap(argument))]
    pub open_focused: Option<bool>,
    #[knuffel(child, unwrap(argument))]
    pub on_xdg_activate: Option<OnXdgActivate>,

    // Rules applied dynamically.
    #[knuffel(child, unwrap(argument))]
    pub min_width: Option<u16>,
    #[knuffel(child, unwrap(argument))]
    pub min_height: Option<u16>,
    #[knuffel(child, unwrap(argument))]
    pub max_width: Option<u16>,
    #[knuffel(child, unwrap(argument))]
    pub max_height: Option<u16>,

    #[knuffel(child, default)]
    pub focus_ring: BorderRule,
    #[knuffel(child, default)]
    pub border: BorderRule,
    #[knuffel(child, default)]
    pub shadow: ShadowRule,
    #[knuffel(child, default)]
    pub tab_indicator: TabIndicatorRule,
    #[knuffel(child, unwrap(argument))]
    pub draw_border_with_background: Option<bool>,
    #[knuffel(child, unwrap(argument))]
    pub opacity: Option<f32>,
    #[knuffel(child)]
    pub geometry_corner_radius: Option<CornerRadius>,
    #[knuffel(child, unwrap(argument))]
    pub clip_to_geometry: Option<bool>,
    #[knuffel(child, unwrap(argument))]
    pub baba_is_float: Option<bool>,
    #[knuffel(child, unwrap(argument))]
    pub block_out_from: Option<BlockOutFrom>,
    #[knuffel(child, unwrap(argument))]
    pub variable_refresh_rate: Option<VrrRule>,
    #[knuffel(child, unwrap(argument, str))]
    pub default_column_display: Option<ColumnDisplay>,
    #[knuffel(child)]
    pub default_floating_position: Option<FloatingPosition>,
    #[knuffel(child, unwrap(argument))]
    pub scroll_factor: Option<FloatOrInt<0, 100>>,
    #[knuffel(child, unwrap(argument))]
    pub pinch_sensitivity: Option<FloatOrInt<0, 100>>,
    #[knuffel(child, unwrap(argument))]
    pub tiled_state: Option<bool>,
    #[knuffel(child, default)]
    pub background_effect: BackgroundEffectRule,
    #[knuffel(child, default)]
    pub popups: PopupsRule,

    #[knuffel(child, default)]
    pub upscaling: UpscalingRule,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VrrRule {
    /// No effect on VRR.
    Off,
    /// Enable VRR when the window is visible.
    On,
    /// Enable VRR when the window is the current fullscreen window.
    FullscreenOnly,
}

/// Rules for popup surfaces.
#[derive(knuffel::Decode, Debug, Default, Clone, PartialEq)]
pub struct PopupsRule {
    #[knuffel(child, unwrap(argument))]
    pub opacity: Option<f32>,
    #[knuffel(child)]
    pub geometry_corner_radius: Option<CornerRadius>,
    #[knuffel(child, default)]
    pub background_effect: BackgroundEffectRule,
}

/// Resolved popup-specific rules.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct ResolvedPopupsRules {
    /// Extra opacity to draw popups with.
    pub opacity: Option<f32>,

    /// Corner radius to assume the popups have.
    pub geometry_corner_radius: Option<CornerRadius>,

    /// Background effect configuration for popups.
    pub background_effect: BackgroundEffect,
}

impl MergeWith<PopupsRule> for ResolvedPopupsRules {
    fn merge_with(&mut self, part: &PopupsRule) {
        if let Some(x) = part.opacity {
            self.opacity = Some(x);
        }
        if let Some(x) = part.geometry_corner_radius {
            self.geometry_corner_radius = Some(x);
        }
        self.background_effect.merge_with(&part.background_effect);
    }
}

#[derive(knuffel::Decode, Debug, Default, Clone, PartialEq)]
pub struct Match {
    #[knuffel(property, str)]
    pub app_id: Option<RegexEq>,
    #[knuffel(property, str)]
    pub title: Option<RegexEq>,
    #[knuffel(property)]
    pub is_active: Option<bool>,
    #[knuffel(property)]
    pub is_focused: Option<bool>,
    #[knuffel(property)]
    pub is_active_in_column: Option<bool>,
    #[knuffel(property)]
    pub is_floating: Option<bool>,
    #[knuffel(property)]
    pub is_window_cast_target: Option<bool>,
    #[knuffel(property)]
    pub is_urgent: Option<bool>,
    #[knuffel(property)]
    pub at_startup: Option<bool>,
}

#[derive(knuffel::Decode, Debug, Clone, Copy, PartialEq)]
pub struct FloatingPosition {
    #[knuffel(property)]
    pub x: FloatOrInt<-65535, 65535>,
    #[knuffel(property)]
    pub y: FloatOrInt<-65535, 65535>,
    #[knuffel(property, default)]
    pub relative_to: RelativeTo,
}

#[derive(knuffel::DecodeScalar, Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnXdgActivate {
    Ignore,
    SetUrgent,
    Focus,
}

#[derive(knuffel::DecodeScalar, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum RelativeTo {
    #[default]
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Top,
    Bottom,
    Left,
    Right,
}

impl<S: knuffel::traits::ErrorSpan> knuffel::DecodeScalar<S> for VrrRule {
    fn raw_decode(
        val: &knuffel::span::Spanned<knuffel::ast::Literal, S>,
        _ctx: &mut knuffel::decode::Context<S>,
    ) -> Result<VrrRule, DecodeError<S>> {
        const ERROR: &str = "expected `true`, `\"fullscreen-only\"`, or `false`";
        match &**val {
            knuffel::ast::Literal::Bool(b) => Ok(if *b { VrrRule::On } else { VrrRule::Off }),
            knuffel::ast::Literal::String(s) => match &s[..] {
                // Also accept bool strings to reduce confusion.
                "true" => Ok(VrrRule::On),
                "false" => Ok(VrrRule::Off),
                "fullscreen-only" => Ok(VrrRule::FullscreenOnly),
                _ => Err(DecodeError::conversion(val, ERROR)),
            },
            _ => Err(DecodeError::conversion(val, ERROR)),
        }
    }

    fn type_check(
        type_name: &Option<knuffel::span::Spanned<knuffel::ast::TypeName, S>>,
        ctx: &mut knuffel::decode::Context<S>,
    ) {
        if let Some(typ) = type_name {
            ctx.emit_error(DecodeError::TypeName {
                span: typ.span().clone(),
                found: Some((**typ).clone()),
                expected: knuffel::errors::ExpectedType::no_type(),
                rust_type: stringify!(VrrRule),
            });
        }
    }
}
