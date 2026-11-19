"""Round-2 TEST noise bands (task of Fri 13 Nov 2026, test-run preparation).

check  Does configs/noise_round2.yaml hold WNS, packed-LUT and dynamic-power bands for every TEST design? Reports present or
       missing per design and metric (no power value is printed). Writes data/test_r2_2026-11-13/noise_check.json.
write  Writes the NEW file configs/noise_round2_test.yaml (configs/noise_round2.yaml is pinned in the template v2.1 and bandit
       freeze lists and is never edited): the entries of noise_round2.yaml copied verbatim where they exist, plus ONLY the entries
       that already exist in two recorded round-1 files, each with its source and the file's SHA-256. Nothing is computed, rounded,
       interpolated or invented; a metric with no recorded value stays missing and the writer refuses.
         WNS and packed LUTs  data/exp_evening_2026-11-04/test_b2/overnight_baseline_2026-11-04/noise_bands_2026-11-04.json
         dynamic power        data/scratch_saturday_2026-11-07/test_power_noise_bands.json  (dynamic_power_w_band, unit W)
usage: tr_noise.py check | write
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[2]
NOISE_R2 = PROJECT / "configs/noise_round2.yaml"
OUT_YAML = PROJECT / "configs/noise_round2_test.yaml"
OUT_JSON = PROJECT / "data/test_r2_2026-11-13/noise_check.json"
SRC_WL = PROJECT / "data/exp_evening_2026-11-04/test_b2/overnight_baseline_2026-11-04/noise_bands_2026-11-04.json"
SRC_PW = PROJECT / "data/scratch_saturday_2026-11-07/test_power_noise_bands.json"
TEST = ("rosetta_digit-recognition", "corescore_500_mod", "vexriscv_re-place_v2", "fir_symmetric_systolic", "fir_systolic_transposed")
METRICS = ("wns_ns", "packed_luts", "dynamic_power_w")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def present() -> dict[str, dict[str, bool]]:
    bands = yaml.safe_load(NOISE_R2.read_text(encoding="utf-8"))["bands"]
    return {d: {m: m in bands.get(d, {}) for m in METRICS} for d in TEST}


def recorded() -> dict[str, dict[str, bool]]:
    wl = json.loads(SRC_WL.read_text(encoding="utf-8"))["bands"]
    pw = json.loads(SRC_PW.read_text(encoding="utf-8"))["bands"]
    return {d: {"wns_ns": "wns_ns" in wl.get(d, {}), "packed_luts": "packed_luts" in wl.get(d, {}),
                "dynamic_power_w": "dynamic_power_w_band" in pw.get(d, {})} for d in TEST}


def check() -> dict:
    have, rec = present(), recorded()
    out = {"noise_round2_sha256": sha(NOISE_R2), "designs": {}, "missing_in_noise_round2": [], "missing_everywhere": []}
    for d in TEST:
        out["designs"][d] = {m: {"in_noise_round2": have[d][m], "recorded_in_round1_test_files": rec[d][m]} for m in METRICS}
        for m in METRICS:
            if not have[d][m]:
                out["missing_in_noise_round2"].append(f"{d}:{m}")
                if not rec[d][m]:
                    out["missing_everywhere"].append(f"{d}:{m}")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out


def block_lines(text_lines: list[str], design: str) -> list[str]:
    """The verbatim child lines of one design in noise_round2.yaml (4-space indented)."""

    start = text_lines.index(f"  {design}:")
    out = []
    for line in text_lines[start + 1:]:
        if line.startswith("    "):
            out.append(line)
        else:
            break
    return out


def write() -> int:
    if OUT_YAML.exists():
        print("refused: configs/noise_round2_test.yaml exists (written once)")
        return 2
    chk = check()
    if chk["missing_everywhere"]:
        print("refused: no recorded value for", chk["missing_everywhere"])
        return 3
    wl_doc = json.loads(SRC_WL.read_text(encoding="utf-8"))
    wl, pw_doc = wl_doc["bands"], json.loads(SRC_PW.read_text(encoding="utf-8"))
    if not str(pw_doc.get("unit", "")).startswith("W (absolute)"):
        print("refused: power band unit is", pw_doc.get("unit"))
        return 4
    pw = pw_doc["bands"]
    r2_lines = NOISE_R2.read_text(encoding="utf-8").splitlines()
    have = present()
    lines = [
        "# Round-2 TEST noise bands, written 2026-11-13 (test-run preparation). NEW file; configs/noise_round2.yaml is pinned and untouched.",
        f"# Base: configs/noise_round2.yaml sha256 {sha(NOISE_R2)} (its entries for the TEST designs are copied verbatim).",
        "# Supplement: only values that already exist in recorded round-1 files, each with source and file SHA-256; none computed or invented.",
        f"#   WNS and packed LUTs: {SRC_WL.relative_to(PROJECT)} sha256 {sha(SRC_WL)} (method {wl_doc.get('method')})",
        f"#   dynamic power: {SRC_PW.relative_to(PROJECT)} sha256 {sha(SRC_PW)} (dynamic_power_w_band, unit W)",
        "# Same schema as noise_round2.yaml (orch.config.load_noise).",
        "bands:",
    ]
    controls = wl_doc.get("source_result_sha256", {})
    for d in TEST:
        lines.append(f"  {d}:")
        copied = block_lines(r2_lines, d) if d in yaml.safe_load(NOISE_R2.read_text())["bands"] else []
        lines.extend(copied)
        if not have[d]["wns_ns"]:
            n = max(1, len(controls.get(d, {})))
            src = f"{SRC_WL.relative_to(PROJECT)} bands.{d}.wns_ns (sha256 {sha(SRC_WL)}; method {wl_doc.get('method')})"
            lines.append(f'    wns_ns: {{value: {wl[d]["wns_ns"]}, basis: ns, relative: false, samples: {n}, source: "{src}"}}')
        if not have[d]["packed_luts"]:
            n = max(1, len(controls.get(d, {})))
            src = f"{SRC_WL.relative_to(PROJECT)} bands.{d}.packed_luts (sha256 {sha(SRC_WL)}; method {wl_doc.get('method')})"
            lines.append(f'    packed_luts: {{value: {wl[d]["packed_luts"]}, basis: packed_luts, relative: false, samples: {n}, source: "{src}"}}')
        if not have[d]["dynamic_power_w"]:
            n = max(1, int(pw[d].get("n_controls") or 1))
            src = f"{SRC_PW.relative_to(PROJECT)} bands.{d}.dynamic_power_w_band (sha256 {sha(SRC_PW)})"
            lines.append(f'    dynamic_power_w: {{value: {pw[d]["dynamic_power_w_band"]}, basis: W, relative: false, samples: {n}, source: "{src}"}}')
    OUT_YAML.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # load check: every TEST design has the three metrics
    sys.path.insert(0, str(PROJECT))
    from orch.config import load_noise

    cfg = load_noise(OUT_YAML)
    for d in TEST:
        got = sorted(k.value for k in cfg.bands[d])
        if got != sorted(METRICS):
            print("load check failed", d, got)
            return 5
    print(json.dumps({"written": str(OUT_YAML.relative_to(PROJECT)), "sha256": sha(OUT_YAML), "designs": len(TEST)}))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        print(json.dumps(check(), indent=1, sort_keys=True))
        raise SystemExit(0)
    raise SystemExit(write() if cmd == "write" else 1)
