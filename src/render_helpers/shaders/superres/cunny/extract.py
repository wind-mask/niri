#!/usr/bin/env python3
"""Reads a CuNNy network out of the Magpie effect that ships it.

The reference is `magpie/normal/CuNNy-{fast,3x12}-NVL.hlsl` from
https://github.com/Blinue/CuNNy, commit dc3f51aa411b73d692aae16294d252199e4d1e93, which is
GPL-3.0, the same license as niri. The weights are trained by the CuNNy project
(https://github.com/funnyplanter/CuNNy) and are part of that GPL-covered work.

The effects are written for MagpieFX, a compute-shader dialect: every pass gathers a 3x3
neighbourhood of each input texture and accumulates it into a few vectors of output channels. A
pass walks its body in order and reuses the same sample names for the next input texture to keep
the register count down, so the statements are read in order rather than matched all at once. The
weights are written inline, either as a V4 for a single input channel or as an M4 for four of them,
which is what this reads back. The port itself uses that same M4 grouping, because a `mat4` times a
`vec4` is exactly the four dot products it does.

What comes out is a pass per line: which texture each pass reads, how the channels of those
textures map onto the network's channels, the 3x3 weights and the biases, where the ReLU is, and
which texture each accumulator ends up in. The last pass is deliberately not handled here beyond
its weights: it is the pixel shuffle, whose structure the port spells out itself.

Run it with `python3 extract.py <effect.hlsl>` to print what it found. This file grows into the
generator that writes the GLSL shaders and the weight tables that the port loads, at which point
running it is how those are updated.
"""

import re
import sys


def parse(path):
    text = open(path).read()
    blocks = re.split(r"^//!PASS ", text, flags=re.M)[1:]

    passes = []
    for block in blocks:
        lines = block.splitlines()
        num = int(lines[0].split()[0])
        desc = ""
        inputs = []
        outputs = []
        for line in lines[1:]:
            if line.startswith("//!DESC"):
                desc = line[len("//!DESC") :].strip()
            elif line.startswith("//!IN"):
                inputs = [t.strip() for t in line[len("//!IN") :].split(",")]
            elif line.startswith("//!OUT"):
                outputs = [t.strip() for t in line[len("//!OUT") :].split(",")]
            elif line.startswith("//!"):
                continue
            else:
                break

        passes.append(parse_pass(num, desc, inputs, outputs, "\n".join(lines)))

    return passes


def parse_pass(num, desc, inputs, outputs, body):
    # Which texture each `L<i>` reads.
    tables = {
        int(m.group(1)): m.group(2)
        for m in re.finditer(r"#define L(\d)\(x, y\) V[34]\(O\((\w+), x, y\)", body)
    }
    luma = [
        float(v)
        for m in re.finditer(r"#define L(\d)\(x, y\) min16float\(dot\(float3\(([^)]*)\)", body)
        for v in m.group(2).split(",")
    ]

    offsets = {}
    source = {}
    weights = []
    biases = {}
    relu = set()
    out_ch = {}

    for statement in [s for line in body.splitlines() for s in line.split(";")]:
        statement = statement.strip()
        if not statement:
            continue

        m = re.match(r"s(\d)_(\d)_(\d) = L(\d)\((-?[\d.]+), (-?[\d.]+)\)$", statement)
        if m:
            i, j, k = int(m.group(1)), int(m.group(2)), int(m.group(3))
            offsets[(i, j, k)] = (int(float(m.group(5))), int(float(m.group(6))))
            source[(i, j, k)] = int(m.group(4))
            continue

        m = re.match(r"r(\d) \+= V4\(([^)]*)\) \* s(\d)_(\d)_(\d)$", statement)
        if m:
            out = int(m.group(1))
            values = [float(v) for v in m.group(2).split(",")]
            index = (int(m.group(3)), int(m.group(4)), int(m.group(5)))
            for lane, value in enumerate(values):
                # A single luma channel spread over four output channels.
                weights.append((out, source[index], offsets[index], lane, 0, value))
            continue

        m = re.match(r"r(\d) \+= mul\(s(\d)_(\d)_(\d), M(3x4|4)\(([^)]*)\)\)$", statement)
        if m:
            out = int(m.group(1))
            index = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
            rows = 3 if m.group(5) == "3x4" else 4
            values = [float(v) for v in m.group(6).split(",")]
            assert len(values) == rows * 4, statement
            for row in range(rows):
                for column in range(4):
                    # mul(vector, matrix): out[column] += sum(row) vector[row] * matrix[row][column].
                    weights.append(
                        (
                            out,
                            source[index],
                            offsets[index],
                            column,
                            row,
                            values[row * 4 + column],
                        )
                    )
            continue

        m = re.match(r"r(\d) \+= V4\(([^)]*)\)$", statement)
        if m:
            biases[int(m.group(1))] = [float(v) for v in m.group(2).split(",")]
            continue

        m = re.match(r"r(\d) = max\(r\d, 0\.0\)$", statement)
        if m:
            relu.add(int(m.group(1)))
            continue

        m = re.match(r"T(\d)\[gxy\] = r(\d)$", statement)
        if m:
            out_ch[int(m.group(2))] = int(m.group(1))
            continue

    # Flatten: a channel is its texture times four plus its lane within the vector, and a tap is
    # its y offset plus one times three plus its x offset plus one.
    table = {}
    for out, tex, (dx, dy), out_lane, in_lane, value in weights:
        channel = tex * 4 + in_lane
        tap = (dy + 1) * 3 + (dx + 1)
        table.setdefault((out, out_lane), {})[(channel, tap)] = value

    return {
        "num": num,
        "desc": desc,
        "inputs": inputs,
        "outputs": outputs,
        "luma": luma,
        "out_texture": out_ch,
        "relu": sorted(relu),
        "weights": table,
        "biases": biases,
    }


def main(path):
    for p in parse(path):
        print(f"pass {p['num']} ({p['desc']}): in {p['inputs']} out {p['outputs']}")
        print(f"  relu {p['relu']} out textures {p['out_texture']} luma {p['luma']}")
        for (acc, lane), v in sorted(p["weights"].items()):
            channels = sorted({c for c, _ in v})
            taps = sorted({t for _, t in v})
            missing = [
                (c, t)
                for c in range(max(channels) + 1)
                for t in range(9)
                if (c, t) not in v
            ]
            print(
                f"  r{acc}.{'xyzw'[lane]}: {len(v)} weights, channels {channels}, taps {taps}, missing {missing[:3]}"
            )
        print(f"  biases {p['biases']}")


if __name__ == "__main__":
    main(sys.argv[1])
