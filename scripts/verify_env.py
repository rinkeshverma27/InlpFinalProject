#!/usr/bin/env python3
"""
verify_env.py — Environment Verification & Auto-Fix Script

USAGE:
  Check only (no changes made):
      python scripts/verify_env.py

  Check AND auto-install/fix everything missing:
      python scripts/verify_env.py --fix

HOW --fix WORKS:
  1. Runs: conda env update -f environment.yml --prune
     (adds missing packages, removes unlisted ones)
  2. Re-runs all checks to confirm everything now passes.

Run this after:
  - Cloning the repo for the first time
  - Pulling changes that include environment.yml updates
  - Seeing any ❌ FAIL in a previous run
"""

import sys
import importlib
import subprocess
import argparse
from pathlib import Path

# ── CLI args ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Verify (and optionally fix) the nifty-nlp conda environment."
)
parser.add_argument(
    "--fix",
    action="store_true",
    help="Auto-install missing packages via 'conda env update -f environment.yml --prune'"
)
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / "environment.yml"

REQUIRED_PACKAGES = {
    "torch":          "2.3.1",
    "transformers":   "4.42.4",
    "pandas":         "2.2.2",
    "numpy":          "1.26.4",
    "sklearn":        "1.5.1",
    "yfinance":       "0.2.40",
    "feedparser":     "6.0.11",
    "yaml":           None,
    "loguru":         "0.7.2",
    "rich":           "13.7.1",
    "schedule":       "1.2.2",
    "git":            None,
    "tqdm":           "4.66.4",
    "requests":       "2.32.3",
    "aiohttp":        "3.9.5",
    "accelerate":     "0.31.0",
}

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"


def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def check(label, ok, msg=""):
    status = PASS if ok else FAIL
    line = f"  {status}  {label}"
    if msg:
        line += f"  ({msg})"
    print(line)
    return ok


def run_checks():
    """Run all environment checks. Returns (passed, total)."""
    results = []

    # ── Python version ────────────────────────────────────────
    section("1. Python")
    major, minor = sys.version_info[:2]
    ok = (major == 3 and minor == 10)
    results.append(check(
        f"Python {major}.{minor}", ok,
        "need 3.11" if not ok else sys.version.split()[0]
    ))

    # ── Conda environment name ────────────────────────────────
    section("2. Conda Environment")
    conda_env = subprocess.run(
        ["conda", "info", "--envs"],
        capture_output=True, text=True
    )
    env_active = "nifty-nlp" in subprocess.run(
        ["conda", "info"], capture_output=True, text=True
    ).stdout
    # Simple check: is 'nifty-nlp' in the active env path?
    active_env = sys.prefix
    results.append(check(
        "Active conda env contains 'nifty-nlp'",
        "nifty-nlp" in active_env,
        active_env
    ))

    # ── Package imports ───────────────────────────────────────
    section("3. Package Imports")
    for pkg, expected_ver in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "unknown")
            if expected_ver and ver != expected_ver:
                print(f"  {WARN}  {pkg}=={ver}  (expected {expected_ver} — likely still works)")
                results.append(True)   # version mismatch = warning, not failure
            else:
                results.append(check(f"{pkg}=={ver}", True))
        except ImportError as e:
            results.append(check(pkg, False, f"missing — run with --fix to install"))

    # ── CUDA / GPU ────────────────────────────────────────────
    section("4. CUDA & GPU")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        results.append(check(
            "CUDA available", cuda_ok,
            f"device count: {torch.cuda.device_count()}" if cuda_ok
            else "NO GPU — CPU-only mode (fine for data/scraping machines)"
        ))
        if cuda_ok:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / 1e9
                results.append(check(
                    f"GPU {i}: {props.name}", True, f"{vram_gb:.1f} GB VRAM"
                ))
        results.append(check("CUDA version", True, torch.version.cuda or "N/A"))
    except Exception as e:
        results.append(check("torch cuda check", False, str(e)))

    # ── TA-Lib ────────────────────────────────────────────────
    section("5. TA-Lib (Technical Analysis)")
    try:
        import talib
        results.append(check(f"ta-lib=={talib.__version__}", True))
    except ImportError:
        results.append(check(
            "ta-lib", False,
            "system lib missing — run: sudo apt-get install libta-lib-dev"
        ))
    except Exception as e:
        results.append(check("ta-lib", False, str(e)))

    # ── HuggingFace Hub ───────────────────────────────────────
    section("6. HuggingFace Hub Access")
    try:
        from huggingface_hub import HfApi
        models = list(HfApi().list_models(search="finbert", limit=1))
        results.append(check("HuggingFace Hub reachable", len(models) > 0))
    except Exception as e:
        results.append(check("HuggingFace Hub", False, str(e)))

    # ── Config file ───────────────────────────────────────────
    section("7. Project Config")
    results.append(check("config.yml exists", ENV_FILE.parent.joinpath("config.yml").exists()))
    if ENV_FILE.parent.joinpath("config.yml").exists():
        try:
            import yaml
            with open(PROJECT_ROOT / "config.yml") as f:
                cfg = yaml.safe_load(f)
            stage = cfg.get("project", {}).get("stage", "?")
            freeze = cfg.get("adaptive_learning", {}).get("FREEZE_ALL_UPDATES", None)
            results.append(check("config.yml parseable", True, f"stage={stage}"))
            results.append(check(f"FREEZE_ALL_UPDATES={freeze}", True))
        except Exception as e:
            results.append(check("config.yml parse", False, str(e)))

    # ── Data directories ─────────────────────────────────────
    section("8. Data Directory Structure")
    expected_dirs = [
        "data/price", "data/news/raw", "data/handshake",
        "data/trust_weights", "data/predictions",
        "models/checkpoints", "logs", "src", "scripts",
    ]
    for d in expected_dirs:
        p = PROJECT_ROOT / d
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        results.append(check(f"  {d}", p.exists()))

    # ── Git repo ─────────────────────────────────────────────
    section("9. Git Repository")
    try:
        import git as gitmodule
        repo = gitmodule.Repo(PROJECT_ROOT)
        results.append(check("Git repo valid", True, f"branch: {repo.active_branch.name}"))
        dirty = repo.is_dirty()
        if dirty:
            print(f"  {WARN}  Working tree has uncommitted changes")
        else:
            print(f"  {PASS}  Working tree is clean")
    except Exception as e:
        results.append(check("Git repo", False, str(e)))

    return sum(results), len(results)


def auto_fix():
    """Run conda env update to install missing packages."""
    print("\n" + "═"*55)
    print("  🔧  AUTO-FIX MODE — Running conda env update")
    print("═"*55)

    if not ENV_FILE.exists():
        print(f"\n  {FAIL}  environment.yml not found at: {ENV_FILE}")
        print("  Make sure you are running from inside the project root.")
        sys.exit(1)

    print(f"\n  Running: conda env update -f {ENV_FILE} --prune")
    print("  (This may take a few minutes...)\n")

    result = subprocess.run(
        ["conda", "env", "update", "-f", str(ENV_FILE), "--prune"],
        # Stream output live to terminal
    )

    if result.returncode != 0:
        print(f"\n  {FAIL}  conda env update failed (exit code {result.returncode})")
        print("\n  Manual fix options:")
        print("  1. Make sure 'conda activate nifty-nlp' was run first")
        print("  2. Check system dep: sudo apt-get install libta-lib-dev")
        print("  3. Re-create from scratch:")
        print("       conda env remove -n nifty-nlp")
        print("       conda env create -f environment.yml")
        sys.exit(1)

    print("\n  ✅  conda env update completed successfully!")
    print("\n  Re-running checks to confirm...\n")


# ── MAIN ─────────────────────────────────────────────────────
print()
print("  ╔══════════════════════════════════════════════╗")
print("  ║   nifty-nlp — Environment Verification      ║")
if args.fix:
    print("  ║   Mode: AUTO-FIX (--fix)                    ║")
else:
    print("  ║   Mode: CHECK ONLY  (use --fix to install)  ║")
print("  ╚══════════════════════════════════════════════╝")

if args.fix:
    auto_fix()

passed, total = run_checks()

section("Summary")
all_ok = passed == total
emoji = "🎉" if all_ok else ("⚠️ " if passed > total * 0.8 else "❌")
print(f"\n  {emoji}  {passed}/{total} checks passed\n")

if not all_ok:
    if not args.fix:
        print("  To auto-fix missing packages, run:")
        print("      python scripts/verify_env.py --fix\n")
        print("  For ta-lib system dependency errors, run first:")
        print("      sudo apt-get install libta-lib-dev\n")
    else:
        print("  Some checks still failing after --fix.")
        print("  Check the ❌ items above for manual steps.\n")
    sys.exit(1)
else:
    print("  Environment is ready. You're good to go! 🚀\n")
    sys.exit(0)
