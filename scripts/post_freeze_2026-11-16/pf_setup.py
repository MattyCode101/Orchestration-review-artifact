"""Post-freeze setup (Mon 16 Nov 2026): the run folder, the prices file, the queue configuration and the seeded telemetry cache.
Idempotent: nothing that exists is overwritten. Run with TR_QUEUE=pf (set here), and once more with TR_DRY=1 for the VALIDATION smoke folder.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["TR_QUEUE"] = "pf"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_common as tc  # noqa: E402

PRICES = {
    "gateway:gpt-5.6-terra": {"input_per_million": 2.0, "output_per_million": 12.0},
    "gateway:Qwen/Qwen3.5-397B-A17B-FP8": {"input_per_million": 0.0, "output_per_million": 0.0},
}
PROVENANCE = """# Prices of the post-freeze rows (Mon 16 Nov 2026)

- gemini-3.1-flash-lite: US$0.25 input and US$1.50 output per million tokens (PRICE_GEMINI in tr_common.py, unchanged since 13 Nov).
- gateway:gpt-5.6-terra: the placeholder of 14 Nov, US$2 and US$12 per million tokens (OpenAI list prices; the gateway publishes no per-token price; UNCONFIRMED).
- gateway:Qwen/Qwen3.5-397B-A17B-FP8: the gateway lists no price for this model (the /models entry carries no pricing field; checked 16 Nov), so the price is
  n/a. The entry is 0 and 0 only so that the driver can run; it means NOT APPLICABLE, not free. Qwen is limited by its own token ledger
  (1.5 million tokens, smoke test included), never by US$.
"""


def main() -> int:
    tc.ROOT.mkdir(parents=True, exist_ok=True)
    base = tc.PRICES.parent
    base.mkdir(parents=True, exist_ok=True)
    if not tc.PRICES.exists():
        tc.jwrite(tc.PRICES, PRICES)
        (base / "prices_provenance.md").write_text(PROVENANCE, encoding="utf-8")
        print("wrote", tc.PRICES.name)
    if not tc.CONFIG.exists():
        tc.jwrite(tc.CONFIG, {"max_active": 24, "override_max_active": 8, "load_guard": 160.0, "stall_override_min": 15, "admit_per_loop": 4})
        print("wrote", tc.CONFIG.name)
    print(json.dumps({"root": str(tc.ROOT), "dry": tc.DRY, "telemetry_seed": tc.seed_telemetry_cache(), "plan_sha": tc.plan_sha()[:12]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
