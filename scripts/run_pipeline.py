from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print(f"\n=== RUN: {' '.join(cmd)} ===")
    p = subprocess.run(cmd, cwd=str(ROOT))
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main() -> None:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"Pipeline start (UTC): {ts}")

    # 1) Data
    _run([sys.executable, "scripts/fetch_data.py"])
    _run([sys.executable, "scripts/run_qa.py"])

    # 2) Indicators / Positions
    _run([sys.executable, "scripts/compute_indicators.py"])
    _run([sys.executable, "scripts/compute_positions_macd.py"])
    _run([sys.executable, "scripts/compute_positions_ema_ratio.py"])
    _run([sys.executable, "scripts/compute_positions_reversion_vwap.py"])

    # 3) Backtests
    _run([sys.executable, "scripts/run_backtest_macd.py"])
    _run([sys.executable, "scripts/run_backtest_ema_ratio.py"])
    _run([sys.executable, "scripts/run_backtest_reversion_vwap.py"])

    # 4) Sweeps + compare table
    _run([sys.executable, "scripts/sweep_costs_macd.py"])
    _run([sys.executable, "scripts/sweep_costs_ema_ratio.py"])
    _run([sys.executable, "scripts/compare_strategies.py"])

    # 5) Optional ML
    ml_script = ROOT / "scripts" / "train_classifier_and_backtest.py"
    if ml_script.exists():
        _run([sys.executable, str(ml_script)])

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
    