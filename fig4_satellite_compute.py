"""复现论文图 4：max λ_i 随"每颗卫星的计算能力 f_max"的变化。

论文原图
--------
Fig. 4: The maximum λ_i for ground users versus different computing capability
        of each satellite.
横轴：f_max ∈ {3, 5, 7, 9, 11, 13} Gcycles/s
纵轴：The maximum λ_i
论文结论：
  - CL 全部本地计算，与 f_max 无关，是一条水平线（默认参数下恒为 1/6 ≈ 0.167）；
  - f_max < 8 时 CL 优于 JBHCO-PG，f_max > 9 后 JBHCO-PG 反超；
  - JBHCO-MB 始终低于 JBHCO-PB、CL 和 JBHCO-PG。

运行：
    python fig4_satellite_compute.py
    python fig4_satellite_compute.py --runs 10 --rerun
"""

from __future__ import annotations

import argparse

from experiment_utils import add_common_arguments, plot_curves, sweep_parameter

# 横坐标：单颗卫星的计算能力 f_max，单位 Gcycle/s
SATELLITE_COMPUTE_VALUES = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="论文图 4：不同卫星算力")
    add_common_arguments(parser)
    arguments = parser.parse_args()

    curves = sweep_parameter(
        parameter_name="sat_compute_capacity_GHz",     # SimConfig 中对应 f_max 的字段
        parameter_values=SATELLITE_COMPUTE_VALUES,
        cache_name="fig4_satellite_compute",
        num_runs=arguments.runs,
        base_overrides={"max_swap_iterations": arguments.swaps},
        force_rerun=arguments.rerun,
    )

    plot_curves(
        x_values=SATELLITE_COMPUTE_VALUES,
        curves=curves,
        x_label="The computing capability of each satellite (Gcycles/s)",
        output_name="fig4_satellite_compute.png",
        title="Fig. 4  max $\\lambda_i$ vs. satellite computing capability",
    )


if __name__ == "__main__":
    main()
