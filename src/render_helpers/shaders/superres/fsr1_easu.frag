#version 100

//_DEFINES_

// FSR1 EASU (Edge Adaptive Spatial Upsampling) pass for window-level super resolution.
//
// This is a port of AMD's `FsrEasuF` from `ffx_fsr1.h`:
//   https://github.com/GPUOpen-Effects/FidelityFX-FSR/blob/a21ffb8f6c13233ba336352bdff293894c706575/ffx-fsr/ffx_fsr1.h
//   (FidelityFX FSR v1.0.2, MIT licensed, Copyright (c) 2021 Advanced Micro Devices, Inc.)
//   sha256 93c3922362ea7fc99cbcc698ca30c98de4f8c246d1fbb0b09e015ddef38ce3a5
// The MIT permission notice from that file applies to this port as well.
//
// Changes from the reference, all of which are deliberate:
//
// - The reference fetches its taps with `textureGather`, and gets the output pixel position through
//   the `FsrEasuCon()` constants. Here the taps are 12 individual texel fetches at the positions
//   that the reference's gather calls resolve to (the b/c/e/f/g/h/i/j/k/l/n/o layout from the
//   reference comments), and the position comes from `v_coords`, which smithay's vertex shader
//   already interpolates for us. This removes the need for the `con0`..`con3` constants, which
//   only exist to turn an output pixel index into a position in the input viewport.
// - `ARcpF1`, `APrxLoRcpF1` and `APrxLoRsqF1` are replaced by their exact counterparts, since the
//   reference implements them with float bit tricks that GLSL ES 1.00 does not have. See
//   `gpu_fsr1_matches_the_reference` in superres.rs for how far that moves the result.
// - Alpha is not part of EASU (the reference leaves it to the caller). Window buffers are
//   premultiplied, so instead of copying or sharpening alpha we resample it bilinearly from the
//   same taps that EASU already needs.
//
// The kernel is only used to magnify, and it is fed the client buffer as it is; it does not
// degenerate to a copy at 1:1 the way the plain resampling kernels do.

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
// Source rect in texels: (x, y, w, h).
uniform vec4 niri_src_rect;

// Fetches one texel by index, clamping to the source rect like the reference's gathers do at the
// edges. The reference uses gather, i.e. exact texels and no filtering.
vec4 niri_easu_texel(vec2 index) {
    vec2 lo = niri_src_rect.xy;
    vec2 hi = niri_src_rect.xy + niri_src_rect.zw - vec2(1.0);
    index = clamp(index, lo, hi);
    return texture2D(tex, (index + vec2(0.5)) / niri_tex_size);
}

// Luma times 2, as in the reference.
float niri_easu_luma(vec3 c) {
    return c.b * 0.5 + (c.r * 0.5 + c.g);
}

// Accumulates direction and length for one of the four quadrants of the bilinear mix.
//
// `w` is the bilinear weight of this quadrant. The five lumas are the 'a/b/c/d/e' of the
// reference's diagram:
//     a
//   b c d
//     e
vec3 niri_easu_set(float w, float lA, float lB, float lC, float lD, float lE) {
    vec2 dir = vec2(0.0);
    float len = 0.0;

    // Direction is the '+' diff, then takes magnitude from abs average of both sides of 'c'.
    // Length converts gradient reversal to 0, smoothly to non-reversal at 1, shaped, then adding
    // horz and vert terms.
    // The reference takes the reciprocal of the gradient magnitude with an approximation that
    // returns a large finite value for zero, so a flat neighborhood contributes nothing here. An
    // exact reciprocal would be an infinity there, and then zero times infinity is a NaN, so this
    // has to be explicit.
    float dc = lD - lC;
    float cb = lC - lB;
    float lenX = max(abs(dc), abs(cb));
    float dirX = lD - lB;
    dir.x += dirX * w;
    lenX = lenX > 0.0 ? clamp(abs(dirX) / lenX, 0.0, 1.0) : 0.0;
    lenX *= lenX;
    len += lenX * w;

    float ec = lE - lC;
    float ca = lC - lA;
    float lenY = max(abs(ec), abs(ca));
    float dirY = lE - lA;
    dir.y += dirY * w;
    lenY = lenY > 0.0 ? clamp(abs(dirY) / lenY, 0.0, 1.0) : 0.0;
    lenY *= lenY;
    len += lenY * w;

    return vec3(dir, len);
}

// Accumulates one weighted tap, with anisotropic windowing and a negative lobe.
void niri_easu_tap(
    inout vec3 aC,
    inout float aW,
    vec2 off,
    vec2 dir,
    vec2 len,
    float lob,
    float clp,
    vec3 c
) {
    // Rotate offset by direction.
    vec2 v = vec2(off.x * dir.x + off.y * dir.y, off.x * -dir.y + off.y * dir.x);
    // Anisotropy.
    v *= len;
    // Compute distance^2, limited to the window, since at the corners two taps can easily be
    // outside.
    float d2 = v.x * v.x + v.y * v.y;
    d2 = min(d2, clp);
    // Approximation of lanczos2 without sin(), rcp() or sqrt() to get x:
    //  (25/16 * (2/5 * x^2 - 1)^2 - (25/16 - 1)) * (1/4 * x^2 - 1)^2
    float wB = (2.0 / 5.0) * d2 - 1.0;
    float wA = lob * d2 - 1.0;
    wB *= wB;
    wA *= wA;
    wB = (25.0 / 16.0) * wB - (25.0 / 16.0 - 1.0);
    float w = wB * wA;
    // Do weighted average.
    aC += c * w;
    aW += w;
}

void main() {
    // Position of this output pixel in the input viewport, in the space where texel centers are at
    // integer coordinates. The reference computes the same value as `ip * con0.xy + con0.zw`.
    vec2 pos = v_coords * niri_tex_size - niri_src_rect.xy - vec2(0.5);
    vec2 fp = floor(pos);
    vec2 pp = pos - fp;

    // 12-tap kernel:
    //    b c
    //  e f g h
    //  i j k l
    //    n o
    vec4 b = niri_easu_texel(fp + vec2(0.0, -1.0));
    vec4 c = niri_easu_texel(fp + vec2(1.0, -1.0));
    vec4 e = niri_easu_texel(fp + vec2(-1.0, 0.0));
    vec4 f = niri_easu_texel(fp + vec2(0.0, 0.0));
    vec4 g = niri_easu_texel(fp + vec2(1.0, 0.0));
    vec4 h = niri_easu_texel(fp + vec2(2.0, 0.0));
    vec4 i = niri_easu_texel(fp + vec2(-1.0, 1.0));
    vec4 j = niri_easu_texel(fp + vec2(0.0, 1.0));
    vec4 k = niri_easu_texel(fp + vec2(1.0, 1.0));
    vec4 l = niri_easu_texel(fp + vec2(2.0, 1.0));
    vec4 n = niri_easu_texel(fp + vec2(0.0, 2.0));
    vec4 o = niri_easu_texel(fp + vec2(1.0, 2.0));

    float bL = niri_easu_luma(b.rgb);
    float cL = niri_easu_luma(c.rgb);
    float eL = niri_easu_luma(e.rgb);
    float fL = niri_easu_luma(f.rgb);
    float gL = niri_easu_luma(g.rgb);
    float hL = niri_easu_luma(h.rgb);
    float iL = niri_easu_luma(i.rgb);
    float jL = niri_easu_luma(j.rgb);
    float kL = niri_easu_luma(k.rgb);
    float lL = niri_easu_luma(l.rgb);
    float nL = niri_easu_luma(n.rgb);
    float oL = niri_easu_luma(o.rgb);

    // Accumulate for bilinear interpolation.
    float wS = (1.0 - pp.x) * (1.0 - pp.y);
    float wT = pp.x * (1.0 - pp.y);
    float wU = (1.0 - pp.x) * pp.y;
    float wV = pp.x * pp.y;

    vec3 dirLen = niri_easu_set(wS, bL, eL, fL, gL, jL);
    dirLen += niri_easu_set(wT, cL, fL, gL, hL, kL);
    dirLen += niri_easu_set(wU, fL, iL, jL, kL, nL);
    dirLen += niri_easu_set(wV, gL, jL, kL, lL, oL);

    vec2 dir = dirLen.xy;
    float len = dirLen.z;

    // Normalize with approximation, and cleanup close to zero.
    vec2 dir2 = dir * dir;
    float dirR = dir2.x + dir2.y;
    bool zro = dirR < (1.0 / 32768.0);
    dirR = inversesqrt(dirR);
    dirR = zro ? 1.0 : dirR;
    dir.x = zro ? 1.0 : dir.x;
    dir *= dirR;

    // Transform from {0 to 2} to {0 to 1} range, and shape with square.
    len = len * 0.5;
    len *= len;

    // Stretch kernel {1.0 vert|horz, to sqrt(2.0) on diagonal}.
    float stretch = (dir.x * dir.x + dir.y * dir.y) / max(abs(dir.x), abs(dir.y));
    // Anisotropic length after rotation,
    //  x := 1.0 lerp to 'stretch' on edges
    //  y := 1.0 lerp to 2x on edges
    vec2 len2 = vec2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
    // Based on the amount of 'edge', the window shifts from +/-{sqrt(2.0) to slightly beyond 2.0}.
    float lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
    // Set distance^2 clipping point to the end of the adjustable window.
    float clp = 1.0 / lob;

    // Min/max of the four nearest, for deringing.
    vec3 min4 = min(min(f.rgb, g.rgb), min(j.rgb, k.rgb));
    vec3 max4 = max(max(f.rgb, g.rgb), max(j.rgb, k.rgb));

    // Accumulation.
    vec3 aC = vec3(0.0);
    float aW = 0.0;
    niri_easu_tap(aC, aW, vec2(0.0, -1.0) - pp, dir, len2, lob, clp, b.rgb); // b
    niri_easu_tap(aC, aW, vec2(1.0, -1.0) - pp, dir, len2, lob, clp, c.rgb); // c
    niri_easu_tap(aC, aW, vec2(-1.0, 1.0) - pp, dir, len2, lob, clp, i.rgb); // i
    niri_easu_tap(aC, aW, vec2(0.0, 1.0) - pp, dir, len2, lob, clp, j.rgb); // j
    niri_easu_tap(aC, aW, vec2(0.0, 0.0) - pp, dir, len2, lob, clp, f.rgb); // f
    niri_easu_tap(aC, aW, vec2(-1.0, 0.0) - pp, dir, len2, lob, clp, e.rgb); // e
    niri_easu_tap(aC, aW, vec2(1.0, 1.0) - pp, dir, len2, lob, clp, k.rgb); // k
    niri_easu_tap(aC, aW, vec2(2.0, 1.0) - pp, dir, len2, lob, clp, l.rgb); // l
    niri_easu_tap(aC, aW, vec2(2.0, 0.0) - pp, dir, len2, lob, clp, h.rgb); // h
    niri_easu_tap(aC, aW, vec2(1.0, 0.0) - pp, dir, len2, lob, clp, g.rgb); // g
    niri_easu_tap(aC, aW, vec2(1.0, 2.0) - pp, dir, len2, lob, clp, o.rgb); // o
    niri_easu_tap(aC, aW, vec2(0.0, 2.0) - pp, dir, len2, lob, clp, n.rgb); // n

    // Normalize and dering.
    vec3 color = min(max4, max(min4, aC / aW));

    // Window buffers are premultiplied, so keep the result premultiplied by the resampled alpha
    // that goes with it.
    float outAlpha = mix(mix(f.a, g.a, pp.x), mix(j.a, k.a, pp.x), pp.y);
    color = min(color, vec3(outAlpha));

#if defined(NO_ALPHA)
    outAlpha = 1.0;
#endif

    // Apply final alpha and tint, same as the renderer's implicit texture shader. The pass is
    // always run with alpha 1.0; the element alpha is applied by the sharpening pass.
    vec4 result = vec4(color, outAlpha) * alpha;

#if defined(DEBUG_FLAGS)
    if (tint == 1.0)
        result = vec4(0.0, 0.2, 0.0, 0.2) + result * 0.8;
#endif

    gl_FragColor = result;
}
