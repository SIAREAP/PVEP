from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from heat import get_alpha_per_c, initial_inner_diameter_for_target_temp_mm
from sensor_test3 import SleeveDesign, TrueDynamicsParams, sample_true_required_setpoint_temp_c


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _size_groups_from_csv(s: str) -> List[float]:
    xs = [x.strip() for x in (s or "").split(",") if x.strip()]
    out = [float(x) for x in xs]
    if not out:
        raise ValueError("size groups cannot be empty")
    return out


def _materials_from_csv(s: str) -> List[str]:
    xs = [x.strip() for x in (s or "").split(",") if x.strip()]
    if not xs:
        raise ValueError("materials cannot be empty")
    return xs


def _pick_theory_temp_c(*, rng: random.Random, material: str, target_mm: float) -> float:
    """
    人工设计一个“理论温度”作为反推初始内径的锚点。
    你要三组（尺寸不同且波动不同），这里简单做：尺寸越大，理论温度范围稍微更高/更宽一点。
    """
    _ = material
    # 三组：180/200/220 的默认想法
    if target_mm <= 185:
        lo, hi = 250.0, 320.0
    elif target_mm <= 205:
        lo, hi = 260.0, 340.0
    else:
        lo, hi = 270.0, 360.0
    return float(rng.uniform(lo, hi))


def _delta_d_mm(*, d0_mm: float, alpha_per_c: float, room_temp_c: float, temp_c: float) -> float:
    """线膨胀：ΔD = D0*alpha*(T-T0)。"""
    return float(d0_mm) * float(alpha_per_c) * float(float(temp_c) - float(room_temp_c))


def generate_tasks(
    *,
    seed: int,
    room_temp_c: float,
    materials: List[str],
    size_groups_mm: List[float],
    n_per_material: int,
    params: TrueDynamicsParams,
) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed))
    tasks: List[Dict[str, Any]] = []

    for mat in materials:
        # 均匀覆盖每个尺寸组；若不能整除，前面的组多拿 1 个
        base = int(n_per_material) // int(len(size_groups_mm))
        rem = int(n_per_material) % int(len(size_groups_mm))
        counts = [base + (1 if i < rem else 0) for i in range(len(size_groups_mm))]

        task_idx_in_mat = 0
        for gi, target_mm in enumerate(size_groups_mm):
            for _k in range(int(counts[gi])):
                t_theory = _pick_theory_temp_c(rng=rng, material=str(mat), target_mm=float(target_mm))
                d0 = initial_inner_diameter_for_target_temp_mm(
                    target_shaft_diameter_mm=float(target_mm),
                    material=str(mat),
                    room_temperature_c=float(room_temp_c),
                    target_heating_temperature_c=float(t_theory),
                )
                # 工程上保留两位小数
                d0 = round(float(d0), 2)

                design = SleeveDesign(
                    name=f"{mat}-{int(target_mm)}-{task_idx_in_mat:03d}",
                    material=str(mat),
                    target_shaft_mm=float(target_mm),
                    initial_inner_mm=float(d0),
                    room_temp_c=float(room_temp_c),
                )

                # 每个任务固定 world_seed，便于复现实验
                world_seed = int(seed) * 1_000_000 + (hash(mat) % 100_000) * 10_000 + int(task_idx_in_mat) * 10 + int(gi)
                world_rng = random.Random(int(world_seed))
                t_needed, meta = sample_true_required_setpoint_temp_c(design, rng=world_rng, params=params)

                alpha_nom = get_alpha_per_c(str(mat))
                delta_d = _delta_d_mm(
                    d0_mm=float(d0),
                    alpha_per_c=float(alpha_nom),
                    room_temp_c=float(room_temp_c),
                    temp_c=float(t_needed),
                )

                tasks.append(
                    {
                        "task_id": f"{mat}-{int(target_mm)}-{task_idx_in_mat:03d}",
                        "material": str(mat),
                        "size_group_target_mm": float(target_mm),
                        "room_temp_c": float(room_temp_c),
                        "initial_inner_mm": float(d0),
                        "target_shaft_mm": float(target_mm),
                        "t_theory_c": float(t_theory),
                        "t_needed_c": float(t_needed),
                        "delta_d_mm": float(delta_d),
                        # meta（可选但很有用）
                        **{f"meta_{k}": float(v) for k, v in dict(meta).items()},
                        "world_seed": int(world_seed),
                    }
                )
                task_idx_in_mat += 1

    return tasks


def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    _ensure_dir(out_csv.parent)
    # 统一列顺序：先放常用字段，再放 meta_*
    base_cols = [
        "task_id",
        "material",
        "size_group_target_mm",
        "room_temp_c",
        "initial_inner_mm",
        "target_shaft_mm",
        "t_theory_c",
        "t_needed_c",
        "delta_d_mm",
        "world_seed",
    ]
    meta_cols = sorted({k for r in rows for k in r.keys() if k.startswith("meta_")})
    cols = base_cols + meta_cols

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def write_simple_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    """
    简化版 CSV：只保留“定义任务/复现任务”必需的列。
    不包含绘图相关字段（delta_d_mm 等）和生成过程元数据（meta_*）。
    """
    _ensure_dir(out_csv.parent)
    cols = [
        "task_id",
        "material",
        "size_group_target_mm",
        "initial_inner_mm",
        "target_shaft_mm",
        "t_needed_c",
        "world_seed",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def plot_by_material(
    *,
    rows: List[Dict[str, Any]],
    out_dir: Path,
    room_temp_c: float,
) -> List[Path]:
    import matplotlib.pyplot as plt  # lazy import (no-plot runs don't need matplotlib)

    _ensure_dir(out_dir)
    mats = sorted({str(r["material"]) for r in rows})
    out_paths: List[Path] = []

    for mat in mats:
        xs = [r for r in rows if str(r["material"]) == mat]
        if not xs:
            continue

        plt.figure(figsize=(8.2, 5.2))
        for r in xs:
            t0 = float(room_temp_c)
            t1 = float(r["t_needed_c"])
            y1 = float(r["delta_d_mm"])
            # 每个任务画一条线段： (room_temp, 0) -> (t_needed, ΔD)
            plt.plot([t0, t1], [0.0, y1], alpha=0.35, linewidth=1.0)

        plt.title(f"{mat} | each task as a line segment (T: room→needed, ΔD)")
        plt.xlabel("Temperature (°C)")
        plt.ylabel("ΔD (mm)  (nominal alpha, from room temp)")
        plt.grid(True, alpha=0.25)
        out_png = out_dir / f"tasks_{mat}_segments.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()
        out_paths.append(out_png)

    return out_paths


def main() -> None:
    p = argparse.ArgumentParser(description="Generate sleeve tasks CSV (10 per material) and per-material segment plots.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--room-temp-c", type=float, default=16.0)
    p.add_argument("--materials", type=str, default="碳素钢,铜合金,铝合金")
    p.add_argument("--size-groups-mm", type=str, default="180,200,220")
    p.add_argument("--n-per-material", type=int, default=10)
    p.add_argument("--out-csv", type=str, default=str(Path(__file__).resolve().parent / "data" / "sleeve_tasks_10x3.csv"))
    p.add_argument(
        "--out-simple-csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "sleeve_tasks_simple_10x3.csv"),
        help="输出简化版 CSV（不含 meta_* / delta_d_mm / room_temp_c / t_theory_c 等）",
    )
    p.add_argument("--plot", action="store_true", help="是否生成每个材料的线段图（默认不画）")
    p.add_argument("--out-plot-dir", type=str, default=str(Path(__file__).resolve().parent / "plot" / "tasks"))

    args = p.parse_args()

    params = TrueDynamicsParams()
    rows = generate_tasks(
        seed=int(args.seed),
        room_temp_c=float(args.room_temp_c),
        materials=_materials_from_csv(str(args.materials)),
        size_groups_mm=_size_groups_from_csv(str(args.size_groups_mm)),
        n_per_material=int(args.n_per_material),
        params=params,
    )
    out_csv = Path(str(args.out_csv))
    write_csv(rows, out_csv)

    out_simple_csv = Path(str(args.out_simple_csv))
    write_simple_csv(rows, out_simple_csv)

    print(f"[ok] wrote CSV: {out_csv} (rows={len(rows)})")
    print(f"[ok] wrote SIMPLE CSV: {out_simple_csv} (rows={len(rows)})")

    if bool(args.plot):
        out_pngs = plot_by_material(rows=rows, out_dir=Path(str(args.out_plot_dir)), room_temp_c=float(args.room_temp_c))
        for pth in out_pngs:
            print(f"[ok] wrote plot: {pth}")


if __name__ == "__main__":
    main()

