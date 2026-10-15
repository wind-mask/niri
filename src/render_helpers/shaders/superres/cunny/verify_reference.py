#!/usr/bin/env python3
"""Diffs the ported CuNNy passes against the reference effect they came from.

The reference is parsed with extract.py, which turns each pass into weights keyed
by output channel and by (input texture, channel, tap). The generated shaders
carry the same tables as `niri_w{input}_{tap}_{lane}` constants. Comparing them
finds a pass that reads or weights its inputs differently from the reference,
which is what a wrong port looks like: the network still runs, and its residuals
are wrong everywhere at once.

Run: python3 verify_reference.py
"""

import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/tmp/pi-github-repos/runtime-CIg4wo/1aa09303404ae9db4754f6e2639d4c806e5147c00938a7c7984f725af88fa2bc"
    "/magpie/normal/CuNNy-fast-NVL.hlsl"
)


def load_extract():
    spec = importlib.util.spec_from_file_location("extract", HERE / "extract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_generated(variant, shuffle_pass):
    """Returns (pass, input, tap, output channel) -> [4 input channel weights]."""
    constants = {}
    files = sorted((HERE / "generated" / variant).glob("pass*_a*.frag"))
    for path in files:
        match = re.search(r"pass(\d+)_a(\d+)", path.name)
        pass_number, group = int(match.group(1)), int(match.group(2))
        text = path.read_text()
        for m in re.finditer(r"const vec4 niri_w(\d+)_(\d+)_(\d+) = vec4\(([^)]*)\)", text):
            input_index, tap, lane = (int(g) for g in m.groups()[:3])
            values = [float(v) for v in m.group(4).split(",")]
            constants[(pass_number, input_index, tap, group * 4 + lane)] = values

    shuffle = HERE / "generated" / variant / "shuffle.frag"
    if shuffle.exists():
        files.append(shuffle)
        text = shuffle.read_text()
        for m in re.finditer(r"const vec4 niri_s(\d+)_(\d+)_(\d+) = vec4\(([^)]*)\)", text):
            lane, input_index, tap = (int(g) for g in m.groups()[:3])
            values = [float(v) for v in m.group(4).split(",")]
            constants[(shuffle_pass, input_index, tap, lane)] = values

    return constants, [p.name for p in files]


def main():
    extract = load_extract()
    passes = extract.parse(str(REFERENCE))
    print(f"reference: {REFERENCE.name}, {len(passes)} passes")
    for p in passes:
        print(f"  pass {p['num']} ({p['desc']}): in {p['inputs']} out {p['outputs']}"
              f" relu {p['relu']} luma {p['luma']}")

    variant = "3x12" if "3x12" in REFERENCE.name else "fast"
    ours, files = parse_generated(variant, passes[-1]["num"])
    print(f"generated ({variant}): {len(files)} shaders, {len(ours)} weight vectors")

    luma_rgb = (0.299, 0.587, 0.114)
    missing, mismatched, checked = [], [], 0
    for p in passes:
        is_rgb_shuffle = p is passes[-1] and (1, 0) in p["weights"]
        if is_rgb_shuffle:
            for lane in range(4):
                for (input_channel, tap) in p["weights"][(0, lane)]:
                    want = sum(
                        luma_rgb[ch] * p["weights"][(ch, lane)].get((input_channel, tap), 0.0)
                        for ch in range(3)
                    )
                    input_index, component = divmod(input_channel, 4)
                    key = (p["num"], input_index, tap, lane)
                    got = ours.get(key)
                    checked += 1
                    if got is None:
                        missing.append(key)
                    elif abs(got[component] - want) > 1e-6 + 1e-4 * abs(want):
                        mismatched.append((key, component, want, got[component]))
            continue

        for (acc, lane), table in p["weights"].items():
            output_channel = acc * 4 + lane
            for (input_channel, tap), want in table.items():
                input_index, component = divmod(input_channel, 4)
                key = (p["num"], input_index, tap, output_channel)
                got = ours.get(key)
                checked += 1
                if got is None:
                    missing.append(key)
                    continue
                if abs(got[component] - want) > 1e-6 + 1e-4 * abs(want):
                    mismatched.append((key, component, want, got[component]))
    print(f"\nchecked {checked} weights: {len(missing)} missing, {len(mismatched)} mismatched")
    for key in missing[:8]:
        print(f"  missing {key}")
    for key, component, want, got in mismatched[:12]:
        print(f"  pass {key[0]} input {key[1]} tap {key[2]} output channel {key[3]}"
              f" channel {component}: reference {want:.6g} ours {got:.6g}")

    extra = set(ours) - {
        (p["num"], in_ch // 4, tap, acc * 4 + lane)
        for p in passes
        for (acc, lane), table in p["weights"].items()
        for (in_ch, tap) in table
    }
    if extra:
        print(f"\n{len(extra)} generated weights the reference does not have, first:"
              f" {sorted(extra)[:6]}")


if __name__ == "__main__":
    main()
