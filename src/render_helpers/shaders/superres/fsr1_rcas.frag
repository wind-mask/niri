#version 100

//_DEFINES_

// FSR1 RCAS (Robust Contrast Adaptive Sharpening) pass for window-level super resolution.
//
// This is a port of AMD's `FsrRcasF` from `ffx_fsr1.h`:
//   https://github.com/GPUOpen-Effects/FidelityFX-FSR/blob/a21ffb8f6c13233ba336352bdff293894c706575/ffx-fsr/ffx_fsr1.h
//   (FidelityFX FSR v1.0.2, MIT licensed, Copyright (c) 2021 Advanced Micro Devices, Inc.)
//   sha256 93c3922362ea7fc99cbcc698ca30c98de4f8c246d1fbb0b09e015ddef38ce3a5
// The MIT permission notice from that file applies to this port as well.
//
// This pass runs at the output resolution on the upsampled result, which is what FSR1 does with
// EASU's output. Changes from the reference:
//
// - `APrxMedRcpF1` is replaced by an exact reciprocal, since the reference implements it with a
//   float bit trick and a Newton step that GLSL ES 1.00 does not have. See
//   `gpu_fsr1_matches_the_reference` in superres.rs for how far that moves the result.
// - The reference leaves alpha to the caller (`FSR_RCAS_PASSTHROUGH_ALPHA`). Window buffers are
//   premultiplied, so alpha is taken from the center texel and the sharpened color is clamped to
//   it, which keeps the result a valid premultiplied color. For an opaque window this clamp never
//   does anything.

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

// Full size of the input texture in texels.
uniform vec2 niri_tex_size;
// Source rect in texels: (x, y, w, h).
uniform vec4 niri_src_rect;
// Sharpening scale, `exp2(-sharpness)` of the configured sharpness, as in `FsrRcasCon()`.
uniform float niri_rcas_sharpness;

// This is set at the limit of providing unnatural results for sharpening.
const float NIRI_RCAS_LIMIT = 0.25 - (1.0 / 16.0);

// Fetches one texel by index, clamping to the source rect. The reference loads exact texels.
vec3 niri_rcas_load(vec2 index) {
    vec2 lo = niri_src_rect.xy;
    vec2 hi = niri_src_rect.xy + niri_src_rect.zw - vec2(1.0);
    index = clamp(index, lo, hi);
    return texture2D(tex, (index + vec2(0.5)) / niri_tex_size).rgb;
}

vec4 niri_rcas_load4(vec2 index) {
    vec2 lo = niri_src_rect.xy;
    vec2 hi = niri_src_rect.xy + niri_src_rect.zw - vec2(1.0);
    index = clamp(index, lo, hi);
    return texture2D(tex, (index + vec2(0.5)) / niri_tex_size);
}

void main() {
    // Integer pixel position in this pass's input.
    vec2 pos = v_coords * niri_tex_size;
    vec2 ip = floor(pos);

    // Algorithm uses minimal 3x3 pixel neighborhood.
    //    b
    //  d e f
    //    h
    vec3 b = niri_rcas_load(ip + vec2(0.0, -1.0));
    vec3 d = niri_rcas_load(ip + vec2(-1.0, 0.0));
    vec4 ee = niri_rcas_load4(ip);
    vec3 e = ee.rgb;
    vec3 f = niri_rcas_load(ip + vec2(1.0, 0.0));
    vec3 h = niri_rcas_load(ip + vec2(0.0, 1.0));

    // Min and max of ring.
    float mn4R = min(min(b.r, min(d.r, f.r)), h.r);
    float mn4G = min(min(b.g, min(d.g, f.g)), h.g);
    float mn4B = min(min(b.b, min(d.b, f.b)), h.b);
    float mx4R = max(max(b.r, max(d.r, f.r)), h.r);
    float mx4G = max(max(b.g, max(d.g, f.g)), h.g);
    float mx4B = max(max(b.b, max(d.b, f.b)), h.b);

    // Immediate constants for peak range.
    vec2 peakC = vec2(1.0, -4.0);

    // Limiters.
    float hitMinR = min(mn4R, e.r) / (4.0 * mx4R);
    float hitMinG = min(mn4G, e.g) / (4.0 * mx4G);
    float hitMinB = min(mn4B, e.b) / (4.0 * mx4B);
    float hitMaxR = (peakC.x - max(mx4R, e.r)) / (4.0 * mn4R + peakC.y);
    float hitMaxG = (peakC.x - max(mx4G, e.g)) / (4.0 * mn4G + peakC.y);
    float hitMaxB = (peakC.x - max(mx4B, e.b)) / (4.0 * mn4B + peakC.y);

    float lobeR = max(-hitMinR, hitMaxR);
    float lobeG = max(-hitMinG, hitMaxG);
    float lobeB = max(-hitMinB, hitMaxB);
    float lobe = max(-NIRI_RCAS_LIMIT, min(max(max(lobeR, lobeG), lobeB), 0.0))
        * niri_rcas_sharpness;

    // The reference also computes a noise term, but only uses it with `FSR_RCAS_DENOISE`, which is
    // deliberately not enabled here, matching the reference default.

    // Resolve.
    float rcpL = 1.0 / (4.0 * lobe + 1.0);
    vec3 color = vec3(
        (lobe * b.r + lobe * d.r + lobe * h.r + lobe * f.r + e.r) * rcpL,
        (lobe * b.g + lobe * d.g + lobe * h.g + lobe * f.g + e.g) * rcpL,
        (lobe * b.b + lobe * d.b + lobe * h.b + lobe * f.b + e.b) * rcpL
    );

    float outAlpha = ee.a;

    // Keep the result a valid premultiplied color.
    color = min(color, vec3(outAlpha));

#if defined(NO_ALPHA)
    outAlpha = 1.0;
    color = color;
#endif

    // Apply the element alpha.
    vec4 result = vec4(color, outAlpha) * alpha;

#if defined(DEBUG_FLAGS)
    if (tint == 1.0)
        result = vec4(0.0, 0.2, 0.0, 0.2) + result * 0.8;
#endif

    gl_FragColor = result;
}
