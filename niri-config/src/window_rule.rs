use niri_ipc::ColumnDisplay;

use crate::appearance::{
    BackgroundEffect, BackgroundEffectRule, BlockOutFrom, BorderRule, CornerRadius, ShadowRule,
    TabIndicatorRule,
};
use crate::layout::DefaultPresetSize;
use crate::utils::{MergeWith, RegexEq};
use crate::FloatOrInt;

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
    /// - `"magnify"`: magnify the window's content to a larger display area. Not implemented yet;
    ///   currently behaves like `"filter"`.
    #[knuffel(child, unwrap(argument))]
    pub mode: Option<UpscalingMode>,

    /// Resampling algorithm to use.
    ///
    /// `"fsr1"` runs FSR1 (EASU then RCAS) and uses `sharpness`. Algorithms other than
    /// `"bilinear"`, `"bicubic"` and `"fsr1"` are not implemented yet and currently fall back to
    /// `"bicubic"`.
    #[knuffel(child, unwrap(argument))]
    pub algorithm: Option<UpscalingAlgorithm>,

    /// Sharpening strength for algorithms that sharpen.
    ///
    /// This is FSR1's RCAS sharpness, in stops of halving: 0 is maximum sharpening, 1 is half of
    /// it, and 2 is a quarter. Defaults to 0.2, which is what AMD's FSR1 sample uses.
    #[knuffel(child, unwrap(argument))]
    pub sharpness: Option<FloatOrInt<0, 2>>,
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
}

impl Upscaling {
    /// Returns whether super resolution should run for this window.
    pub fn enabled(&self) -> bool {
        match self.mode {
            Some(UpscalingMode::Off) => false,
            Some(UpscalingMode::Filter | UpscalingMode::Magnify) => true,
            None => self.algorithm.is_some(),
        }
    }

    /// Returns the mode to use, defaulting to `Off`.
    pub fn mode(&self) -> UpscalingMode {
        self.mode.unwrap_or_default()
    }

    /// Returns the algorithm to use, defaulting to bilinear.
    pub fn algorithm(&self) -> UpscalingAlgorithm {
        self.algorithm.unwrap_or_default()
    }

    /// Returns the sharpness to use, defaulting to what AMD's FSR1 sample uses.
    pub fn sharpness(&self) -> f64 {
        self.sharpness.unwrap_or(0.2)
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
    /// Not implemented yet.
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
    pub variable_refresh_rate: Option<bool>,
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
