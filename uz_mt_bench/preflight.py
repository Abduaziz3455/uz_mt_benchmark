"""Phase 0b — preflight: prove every system is downloaded and runnable.

Runs ONE tiny translation through each selected system. That single call
exercises the whole path end-to-end — env vars / API keys, endpoint reachability
and auth, model download or Ollama tag resolution, weight load, and actual
generation — so a green row here means Step 5 won't die on that system hours in.

Catches the known footguns before you spend money on the full run:
  - missing GEMINI_API_KEY / OLLAMA_API_KEY
  - Ollama server down, or a tag that doesn't resolve (`ollama pull` / fix SPECS)
  - HF model gated / not `huggingface-cli login`'d, or not yet downloaded
  - torchvision mismatch breaking the transformers import (see runbook Step 1)

Heavy local models (NLLB, neuronai-uzbek) download + load real weights here; use
--skip-heavy to check only the API / Ollama systems fast. VRAM is released
between local models so they don't accumulate.

Usage:
    python -m uz_mt_bench.preflight            # all 9
    python -m uz_mt_bench.preflight --systems gemini-3.5-flash,gemma4-12b
    python -m uz_mt_bench.preflight --skip-heavy
"""

from __future__ import annotations

import argparse
import gc
import time
import traceback

from .systems import SPEC_BY_KEY, build_system, resolve_keys

# Short, entity-bearing probe: verifies translation AND that a digit ID survives.
PROBE = "Hello, your order 12345 is ready. How can I help you today?"
_HEAVY = {"nllb", "hf_local"}          # kinds that load local weights into VRAM


def _release_vram() -> None:
    gc.collect()
    try:  # best-effort; torch may not be imported if no heavy system ran
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _check(key: str, text: str) -> dict:
    spec = SPEC_BY_KEY[key]
    t0 = time.time()
    try:
        system = build_system(key)
        out = system.translate(text)
        dt = int((time.time() - t0) * 1000)
        del system
        ok = bool(out and out.strip())
        return {
            "key": key, "kind": spec.kind, "ok": ok, "ms": dt,
            "out": (out or "").strip(),
            "err": "" if ok else "empty output",
        }
    except Exception as exc:  # noqa: BLE001 - report every failure mode uniformly
        dt = int((time.time() - t0) * 1000)
        return {
            "key": key, "kind": spec.kind, "ok": False, "ms": dt,
            "out": "", "err": f"{type(exc).__name__}: {exc}".splitlines()[0][:200],
            "trace": traceback.format_exc(),
        }
    finally:
        if spec.kind in _HEAVY:
            _release_vram()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--systems", default="all",
                    help="'all' (9) or a comma-separated list of system keys")
    ap.add_argument("--skip-heavy", action="store_true",
                    help="skip local-weight systems (NLLB, HF) — check APIs/Ollama only")
    ap.add_argument("--text", default=PROBE, help="probe sentence to translate")
    ap.add_argument("--verbose", action="store_true", help="print full traceback on failure")
    args = ap.parse_args()

    keys = resolve_keys(args.systems)
    if args.skip_heavy:
        keys = [k for k in keys if SPEC_BY_KEY[k].kind not in _HEAVY]

    print(f"preflight: {len(keys)} system(s) · probe: {args.text!r}\n")
    results = []
    for key in keys:
        print(f"  … {key:22s} ", end="", flush=True)
        r = _check(key, args.text)
        results.append(r)
        status = "ok  " if r["ok"] else "FAIL"
        detail = r["out"][:70] if r["ok"] else r["err"]
        print(f"{status} {r['ms']:>7d}ms  {detail}")
        if not r["ok"] and args.verbose and r.get("trace"):
            print("\n".join("      " + ln for ln in r["trace"].splitlines()))

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    print(f"\n{'=' * 60}")
    print(f"READY: {len(ok)}/{len(results)}")
    if bad:
        print("NOT READY:")
        for r in bad:
            print(f"  - {r['key']:22s} ({r['kind']}): {r['err']}")
        print("\nFix the above, then re-run preflight. Full run is safe once all green.")
        raise SystemExit(1)
    print("All systems downloaded and runnable — safe to run Step 5.")


if __name__ == "__main__":
    main()
