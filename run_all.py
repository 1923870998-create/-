#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all — 批量运行所有预测脚本，生成全部 CSV 和 PNG
======================================================
按依赖顺序执行，捕获错误不中断，最后输出汇总。
"""

import subprocess, sys, os, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SCRIPTS = [
    # ---- 数据准备 ----
    ("drone_actual.py",            "数据展示 (生成 dronetotal.csv)"),

    # ---- 统计时序 (无外生) ----
    ("drone_arima_bic.py",        "ARIMA BIC"),
    ("drone_arima_noci.py",       "ARIMA NoCI"),
    ("drone_arima_gpr.py",        "ARIMA + GPR"),
    ("drone_pure_gpr.py",         "Pure GPR"),

    # ---- 统计时序 (含外生) ----
    ("drone_arimax.py",            "ARIMAX"),
    ("drone_arimax_gpr.py",        "ARIMAX+GPR"),
    ("drone_pure_gpr_2d.py",       "Pure GPR 2D"),

    # ---- ML ----
    ("drone_prophet.py",           "Prophet"),
    ("drone_prophet_x.py",         "Prophet-X"),

    # ---- DL ----
    ("drone_lstm_tcn_xgb.py",      "LSTM-TCN+XGBoost"),

    # ---- 整合 & 比较 ----
    ("drone_integrate.py",         "整合所有 CSV"),
    ("drone_compare.py",           "多模型比较"),
]

PYTHON = sys.executable
results = []

print("=" * 70)
print("BATCH RUNNER — Drone Prediction Pipeline")
print("=" * 70)
print(f"Python: {PYTHON}")
print(f"Scripts to run: {len(SCRIPTS)}")
print()

for i, (script, desc) in enumerate(SCRIPTS, 1):
    label = f"[{i}/{len(SCRIPTS)}] {script}"
    print(f"{label}")
    print(f"  {desc}")
    if not os.path.exists(script):
        print(f"  SKIP: file not found")
        results.append((script, "SKIP", "not found"))
        continue

    t0 = time.time()
    try:
        r = subprocess.run([PYTHON, script], capture_output=True, text=True, timeout=600)
        elapsed = time.time() - t0
        if r.returncode == 0:
            print(f"  OK  ({elapsed:.0f}s)")
            results.append((script, "OK", f"{elapsed:.0f}s"))
        else:
            err = r.stderr.strip().split('\n')[-3:] if r.stderr else ["unknown"]
            print(f"  FAIL ({elapsed:.0f}s)")
            for line in err:
                print(f"    {line[:120]}")
            results.append((script, "FAIL", err[-1][:80] if err else ""))
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (>600s)")
        results.append((script, "TIMEOUT", ">600s"))
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append((script, "ERROR", str(e)[:80]))
    print()

# ---- Summary ----
print("=" * 70)
print("SUMMARY")
print("=" * 70)
ok = sum(1 for _, s, _ in results if s == "OK")
fail = sum(1 for _, s, _ in results if s not in ("OK", "SKIP"))
skip = sum(1 for _, s, _ in results if s == "SKIP")
for script, status, info in results:
    mark = "✓" if status == "OK" else "✗" if status == "FAIL" else "-" if status == "SKIP" else "?"
    print(f"  {mark} {script:<40s} {status:<8s} {info}")
print(f"\n  Total:{len(results)}  OK:{ok}  Fail:{fail}  Skip:{skip}")
print("Done.")
