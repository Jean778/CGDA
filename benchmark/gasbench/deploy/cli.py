"""``python -m gasbench deploy`` — deployment profiling CLI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .cost_model import (
    model_from_linear,
    profile_report,
    PLATFORMS,
)
from .optimize import quantize_profile, prune_profile, optimization_curve
from .wireless import system_latency_energy, transmit, LINKS
from .scale import fleet_energy, duty_cycle_matrix
from .fidelity import calibrated_estimates, report as fidelity_report


def _csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df, dict):
        df = [df]
    pd.DataFrame(df).to_csv(path, index=False, float_format="%.6g")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="gasbench deploy",
        description="Simulation-first deployment profiling of the CGDA models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--arch", default="mlp", choices=["mlp"])
    ap.add_argument("--dim", type=int, default=128, help="input feature dim")
    ap.add_argument("--classes", type=int, default=6, help="number of classes")
    ap.add_argument("--platforms", default="stm32h7,rp4b,edge_tpu",
                    help="comma list of platform keys")
    ap.add_argument("--quantize", action="store_true", help="include INT8 columns")
    ap.add_argument("--prune", type=float, default=0.0,
                    help="structured prune rate (0..0.9) for the summary")
    ap.add_argument("--prune-curve", action="store_true",
                    help="emit latency/energy vs prune-rate curve per platform")
    ap.add_argument("--wireless", default="ble",
                    help="comma list of links for system-level view")
    ap.add_argument("--cloud", action="store_true",
                    help="cloud mode: transmit raw 128-feat (int8) instead of a label")
    ap.add_argument("--nodes", type=int, default=10000, help="fleet size")
    ap.add_argument("--freq", type=float, default=6.0,
                    help="inferences per node per hour")
    ap.add_argument("--baseline-mw", type=float, default=400.0,
                    help="system baseline per node (mW): gas-sensor array + MCU + "
                         "radio standby of the inspection-robot sensing subsystem, "
                         "excluding locomotion drive")
    ap.add_argument("--host-ms", type=float, default=0.1321,
                    help="measured x86 end-to-end latency for calibration")
    ap.add_argument("--out", default="results/deploy", help="output dir")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    platforms = [x.strip() for x in args.platforms.split(",") if x.strip()]
    links = [x.strip() for x in args.wireless.split(",") if x.strip()]

    prof = model_from_linear(None, dim=args.dim, nclasses=args.classes, dtype="fp32")
    print("=" * 72)
    print(f"CGDA MLP  [{args.dim} -> {' -> '.join(map(str, prof.widths))} -> {args.classes}]")
    print(f"  MACs={prof.total_macs:,}  params={prof.total_parameters:,}  "
          f"FP32 weights={prof.weight_bytes/1024:.1f} KB  "
          f"INT8 weights={prof.weight_bytes/4/1024:.1f} KB")
    print("=" * 72)

    # 1. Per-platform latency/energy
    rows = profile_report(prof, platforms=platforms, quantize=args.quantize,
                          host_ms=args.host_ms)
    _csv(rows, out / "platform_profile.csv")
    print("\n[1] Per-platform latency/energy (analytical estimates)")
    print(pd.DataFrame(rows).to_string(index=False))

    # 2. Quantization + pruning summary on the primary platform
    if args.quantize and args.prune > 0:
        q = quantize_profile(prune_profile(prof, args.prune))
        rows2 = profile_report(q, platforms=platforms[:1], quantize=True, host_ms=args.host_ms)
        best = next((r for r in rows2 if r["quantized"] == "int8"), rows2[0])
        print(f"\n[2] INT8 + prune {args.prune:.0%}  ->  "
              f"{best['latency_ms']} ms / {best['energy_mj']} mJ on {platforms[0]} "
              f"(fp32+prune {args.prune:.0%}: {rows2[0]['latency_ms']} ms)")
        _csv(rows2, out / "quantized_pruned_summary.csv")

    # 3. Prune curve on the primary platform
    if args.prune_curve:
        curve = []
        for plat in platforms[:1]:
            curve += optimization_curve(prof, plat, quantize=args.quantize,
                                        host_ms=args.host_ms)
        _csv(curve, out / "prune_curve.csv")
        print(f"\n[3] Latency/energy vs prune rate ({platforms[0]}, "
              f"{'int8' if args.quantize else 'fp32'}):")
        print(pd.DataFrame(curve)[["prune_rate", "latency_ms", "energy_mj"]].to_string(index=False))

    # 4. Wireless system-level view (uses the first platform's numbers)
    if links:
        plat = platforms[0]
        base_rows = profile_report(prof, platforms=(plat,), quantize=args.quantize,
                                   host_ms=args.host_ms)
        base = next((r for r in base_rows if r["quantized"] == "int8"), base_rows[0])
        payload = 1 if not args.cloud else (128 if args.quantize else 512)
        sysrows = []
        for ln in links:
            s = system_latency_energy(base["latency_ms"], base["energy_mj"], payload, ln)
            s["link"] = ln
            sysrows.append(s)
        _csv(sysrows, out / "system_level.csv")
        print(f"\n[4] System-level per inference ({plat}, "
              f"payload={payload}B, {'cloud' if args.cloud else 'edge'} mode)")
        print(pd.DataFrame(sysrows)[
            ["link", "system_latency_ms", "system_energy_mj", "wireless_s", "wireless_mj"]
        ].to_string(index=False))

    # 5. Fleet-scale annual energy (primary link)
    if links:
        ln = links[0]
        t = transmit(1 if not args.cloud else payload, ln)
        fe = fleet_energy(base["energy_mj"], t["energy_mj"], args.freq, args.nodes,
                          baseline_mw=args.baseline_mw)
        print(f"\n[5] Fleet energy via {ln}: {fe['annual_kwh']:.2f} kWh/year "
              f"for {args.nodes:,} nodes @ {args.freq}/h "
              f"({fe['annual_kwh_per_node']:.5f} kWh/node/year; "
              f"baseline {args.baseline_mw:.0f} mW => {fe['baseline_kwh']:.2f} kWh)")
        grid = duty_cycle_matrix(base["energy_mj"], t["energy_mj"], args.nodes,
                                 baseline_mw=args.baseline_mw)
        print("    duty-cycle grid (kWh/year; rows=freq/h 6/60/600, cols=active 0.3/0.6/1.0):")
        print("    " + pd.DataFrame(grid, index=[6, 60, 600],
                                    columns=[0.3, 0.6, 1.0]).round(1).to_string().replace("\n", "\n    "))
        _csv({"link": ln, "nodes": args.nodes, "freq_per_hour": args.freq,
              "baseline_mw": args.baseline_mw, "annual_kwh": fe["annual_kwh"],
              "inference_kwh": fe["inference_kwh"], "baseline_kwh": fe["baseline_kwh"]},
             out / "fleet_energy.csv")

    # 6. Fidelity anchoring
    fid = calibrated_estimates(prof, platforms=platforms, host_ms=args.host_ms)
    _csv(fid, out / "fidelity.csv")
    print("\n[6] " + fidelity_report(fid).replace("\n", "\n    "))

    print(f"\nArtifacts written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
