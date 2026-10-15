#version 100

//_DEFINES_

// niri:resample-filter

// Resampling kernel for window-level super resolution.
//
// This shader is compiled through smithay's texture program interface, so it is used with
// smithay's implicit texture vertex shader. That means:
//
// - `v_coords` is the normalized texture coordinate of the current fragment, already
//   including the source crop, the buffer transform and the y-inversion of the texture.
// - `tex` is the client buffer texture, `alpha` is the element alpha and blending, texture
//   filtering and the texture transform are set up by the renderer already.
//
// The kernel is selected by NIRI_RESAMPLE_FILTER (substituted by superres.rs):
//
//   0 - explicit bilinear sampling, which reproduces GL_LINEAR for the identity case
//   1 - Catmull-Rom bicubic sampling
//
// We sample at explicit texel positions instead of relying on the renderer's filter, so that
// the kernel does not depend on whatever filtering the driver happens to have set. Taps are
// clamped to the source rect instead of relying on the texture wrap mode, which the renderer
// does not define for us.

#if defined(EXTERNAL)
#extension GL_OES_EGL_image_external : require
#endif

precision highp float;

#if defined(EXTERNAL)
uniform samplerExternalOES tex;
#else
uniform sampler2D tex;
#endif

uniform float alpha;
varying vec2 v_coords;

#if defined(DEBUG_FLAGS)
uniform float tint;
#endif

// Full size of the client buffer texture in texels.
uniform vec2 niri_tex_size;
// Source rect in texels that is being resampled: (x, y, w, h).
uniform vec4 niri_src_rect;

#ifndef NIRI_RESAMPLE_FILTER
#define NIRI_RESAMPLE_FILTER 0
#endif

const float NIRI_HALF_TEXEL = 0.5;

// Samples the texture at an absolute texel position, where texel (i, j) spans
// [i, i + 1] x [j, j + 1] and its center is at (i + 0.5, j + 0.5).
//
// Out-of-bounds taps clamp to the source rect, matching GL_CLAMP_TO_EDGE for the common case
// of a source rect that covers the whole texture.
vec4 niri_texel(vec2 pos) {
    vec2 lo = niri_src_rect.xy + vec2(NIRI_HALF_TEXEL);
    vec2 hi = niri_src_rect.xy + niri_src_rect.zw - vec2(NIRI_HALF_TEXEL);
    pos = clamp(pos, lo, hi);
    return texture2D(tex, pos / niri_tex_size);
}

#if NIRI_RESAMPLE_FILTER == 1
// Catmull-Rom weights for the four taps around a fractional offset `t` in [0, 1).
vec4 niri_catmull_rom(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return vec4(
        -0.5 * t3 + 1.0 * t2 - 0.5 * t,
         1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
         0.5 * t3 - 0.5 * t2
    );
}
#endif

void main() {
    // Position in texel space; subtract half a texel to get to the space where integer
    // coordinates are texel centers.
    vec2 pos = v_coords * niri_tex_size - NIRI_HALF_TEXEL;
    vec2 base = floor(pos);

#if NIRI_RESAMPLE_FILTER == 1
    vec2 fract_pos = pos - base;
    vec4 wx = niri_catmull_rom(fract_pos.x);
    vec4 wy = niri_catmull_rom(fract_pos.y);

    // Bicubic taps the texel centers base - 1 .. base + 2, which are at base - 0.5 .. base + 2.5
    // in texel space.
    vec4 row0 =
          niri_texel(vec2(base.x - 0.5, base.y - 0.5)) * wx.x
        + niri_texel(vec2(base.x + 0.5, base.y - 0.5)) * wx.y
        + niri_texel(vec2(base.x + 1.5, base.y - 0.5)) * wx.z
        + niri_texel(vec2(base.x + 2.5, base.y - 0.5)) * wx.w;
    vec4 row1 =
          niri_texel(vec2(base.x - 0.5, base.y + 0.5)) * wx.x
        + niri_texel(vec2(base.x + 0.5, base.y + 0.5)) * wx.y
        + niri_texel(vec2(base.x + 1.5, base.y + 0.5)) * wx.z
        + niri_texel(vec2(base.x + 2.5, base.y + 0.5)) * wx.w;
    vec4 row2 =
          niri_texel(vec2(base.x - 0.5, base.y + 1.5)) * wx.x
        + niri_texel(vec2(base.x + 0.5, base.y + 1.5)) * wx.y
        + niri_texel(vec2(base.x + 1.5, base.y + 1.5)) * wx.z
        + niri_texel(vec2(base.x + 2.5, base.y + 1.5)) * wx.w;
    vec4 row3 =
          niri_texel(vec2(base.x - 0.5, base.y + 2.5)) * wx.x
        + niri_texel(vec2(base.x + 0.5, base.y + 2.5)) * wx.y
        + niri_texel(vec2(base.x + 1.5, base.y + 2.5)) * wx.z
        + niri_texel(vec2(base.x + 2.5, base.y + 2.5)) * wx.w;

    vec4 color = row0 * wy.x + row1 * wy.y + row2 * wy.z + row3 * wy.w;
#else
    vec2 fract_pos = pos - base;

    // Bilinear taps the texel centers base and base + 1, which are at base + 0.5 and base + 1.5
    // in texel space.
    vec4 color = mix(
        mix(
            niri_texel(vec2(base.x + 0.5, base.y + 0.5)),
            niri_texel(vec2(base.x + 1.5, base.y + 0.5)),
            fract_pos.x
        ),
        mix(
            niri_texel(vec2(base.x + 0.5, base.y + 1.5)),
            niri_texel(vec2(base.x + 1.5, base.y + 1.5)),
            fract_pos.x
        ),
        fract_pos.y
    );
#endif

#if defined(NO_ALPHA)
    color = vec4(color.rgb, 1.0);
#endif

    // Apply final alpha and tint, same as the renderer's implicit texture shader.
    color = color * alpha;

#if defined(DEBUG_FLAGS)
    if (tint == 1.0)
        color = vec4(0.0, 0.2, 0.0, 0.2) + color * 0.8;
#endif

    gl_FragColor = color;
}
