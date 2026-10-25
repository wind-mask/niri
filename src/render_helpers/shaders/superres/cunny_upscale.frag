#version 100

//_DEFINES_

// CuNNy upscale for window-level super resolution.
//
// This is the last step of the CuNNy port: it takes the network's residuals, which the pixel
// shuffle pass wrote into the four channels of one source-sized texture, and applies them to the
// client buffer, magnified to the destination.
//
// It is written by hand rather than generated, because it holds no weights: everything that came
// out of the reference's tables is in the passes that `cunny/generate.py` writes. What it does
// follows the reference's own output stage, whose structure the port had to change:
//
// - The reference writes a doubled image, one output pixel per sub-pixel of it, and each of those
//   samples the client buffer with linear filtering and adds its own residual to the luma. Here the
//   residuals sit in the four channels of a source-sized texture, so this shader reconstructs the
//   same doubled image by interpolating between those four channels: a destination pixel falls
//   between up to four doubled pixels, and each of them is a channel of the source texel it
//   belongs to, selected by its parity. The selection is a dot product with a mask built from that
//   parity, since shading-language versions without dynamic indexing cannot index a vector either.
// - The reference writes an opaque result. A compositor has to keep the alpha, so the buffer is
//   unpremultiplied around the luma work and premultiplied again at the end, exactly as the passes
//   that read the buffer do.
// - The reference's luma work is in Rec.601, with the two matrices it inlines; they are dot
//   products here, and the same ones.
//
// The destination can be any size, not just twice the source: bilinear interpolation of the
// doubled image is what the reference gets when a player scales its output to a non-integer factor.
// There is deliberately no sharpening after the network.
//
// TODO: the client buffer can be y-inverted, which this does not handle yet: `niri_src_rect` is
// passed the way the resampling shaders get it, while a run of the network is in the orientation it
// was displayed in. Until that is worked out, the pipeline that calls this only runs the network on
// buffers it is sure about, and draws the others with the plain resampling kernels.

#if defined(EXTERNAL)
#extension GL_OES_EGL_image_external : require
#endif

precision highp float;

#if defined(EXTERNAL)
uniform samplerExternalOES tex;
#else
uniform sampler2D tex;
#endif

// The residuals: its size is the source rect's size in texels.
uniform sampler2D niri_tex1;

uniform float alpha;
varying vec2 v_coords;

// Full size of the client buffer texture in texels.
uniform vec2 niri_tex_size;
// Source rect in texels that is being upscaled: (x, y, w, h).
uniform vec4 niri_src_rect;
// How much of the doubled image one destination pixel covers, which is twice the source rect
// divided by the destination's size. Above one, the doubled image is minified and its detail has to
// be averaged over that footprint rather than sampled.
uniform vec2 niri_footprint;

// One residual texel: its four channels are the sub-pixels of one source texel, in the order the
// shuffling pass wrote them: (0,0), (1,0), (0,1), (1,1).
vec4 niri_residual_texel(vec2 idx) {
    vec2 texel = floor(idx * 0.5);
    vec2 uv = (clamp(texel, vec2(0.0), niri_src_rect.zw - vec2(1.0)) + vec2(0.5))
        / niri_src_rect.zw;
    return texture2D(niri_tex1, uv);
}

// The residual of one pixel of the doubled image, at `q` in its texel coordinates.
//
// The four doubled-image pixels around a destination pixel are each a sub-pixel of a source texel,
// so each of them contributes a single channel of one residual texel. Which channel that is comes
// from the pixel's parity, and folding that selection into the interpolation weight turns the sum
// into a dot product, with one fetch per pixel instead of one per channel.
float niri_subpixel(vec2 q) {
    q = floor(q);
    vec2 parity = q - 2.0 * floor(q * 0.5);
    vec2 k = vec2(1.0) - parity;
    vec2 j = vec2(1.0) - k;
    return dot(niri_residual_texel(q), vec4(k.x * k.y, j.x * k.y, k.x * j.y, j.x * j.y));
}

// The residual of the doubled image at `p`, in its texel coordinates.
//
// A destination pixel need not land on a pixel of the doubled image, so this interpolates
// bilinearly between the four around it. Where the destination minifies the doubled image, one
// sample per pixel would alias whatever detail the network put there, so the samples are averaged
// over the footprint instead.
float niri_residual(vec2 p) {
    if (niri_footprint.x <= 1.0 && niri_footprint.y <= 1.0) {
        vec2 i0 = floor(p);
        vec2 f = p - i0;
        return mix(mix(niri_subpixel(i0), niri_subpixel(i0 + vec2(1.0, 0.0)), f.x),
                   mix(niri_subpixel(i0 + vec2(0.0, 1.0)), niri_subpixel(i0 + vec2(1.0, 1.0)), f.x),
                   f.y);
    }

    vec2 h = niri_footprint * 0.25;
    return 0.25 * (niri_subpixel(p + vec2(-h.x, -h.y)) + niri_subpixel(p + vec2(h.x, -h.y))
                   + niri_subpixel(p + vec2(-h.x, h.y)) + niri_subpixel(p + vec2(h.x, h.y)));
}

void main() {
#ifdef NIRI_UV_TRANSFORM
    // The Vulkan renderer draws this as a custom quad, where `v_coords` spans the destination
    // rectangle rather than the source rect in the client texture. Map it back before anything
    // else reads it.
    vec2 uv = niri_uv_offset + niri_v_coords * niri_uv_scale;
#else
#define uv v_coords
#endif

    // Where this destination pixel lands in the client buffer, in its texel space, and then
    // relative to the source rect, which is where the doubled image starts.
    vec2 in_rect = uv * niri_tex_size - niri_src_rect.xy;

    // Where it lands in the doubled image, in texel coordinates.
    vec2 p = clamp(in_rect * 2.0 - vec2(0.5), vec2(0.0), niri_src_rect.zw * 2.0 - vec2(1.0));

    float residual = niri_residual(p);

    // The client buffer, sampled the way the renderer's own shader would: one bilinear fetch, on
    // the coordinates the vertex shader already put the crop and the transform into.
    //
    // The taps are clamped into the source rect like the resampling kernels do, instead of relying
    // on the sampler's wrap mode: the Vulkan renderer's custom textures clamp to a transparent
    // border, which would fade the edge pixels out against nothing.
    vec2 tex_lo = niri_src_rect.xy + vec2(0.5);
    vec2 tex_hi = niri_src_rect.xy + niri_src_rect.zw - vec2(0.5);
    vec2 base_texel = clamp(uv * niri_tex_size, tex_lo, tex_hi);
    vec4 base = texture2D(tex, base_texel / niri_tex_size);

    // The network sees straight color: a window buffer is premultiplied, but a transparent window
    // is not what this effect is for, so the division is guarded.
    float a = base.a;
    vec3 rgb = a > 0.0 ? base.rgb / a : base.rgb;

    vec3 yuv = vec3(dot(rgb, vec3(0.299, 0.587, 0.114)),
                    dot(rgb, vec3(-0.169, -0.331, 0.5)),
                    dot(rgb, vec3(0.5, -0.419, -0.081)));
    yuv.x = clamp(yuv.x + residual, 0.0, 1.0);
    rgb = vec3(yuv.x + 1.401687 * yuv.z - 0.00093 * yuv.y,
               yuv.x - 0.3437 * yuv.y - 0.71417 * yuv.z,
               yuv.x + 1.77216 * yuv.y + 0.00099 * yuv.z);

    gl_FragColor = vec4(rgb * a, a) * alpha;
}
