"""复现论文图 5：max λ_i 随"处理 1 bit 所需 CPU 周期数"的变化。

论文原图
--------
Fig. 5: The maximum λ_i for ground users versus different number of CPU cycles
        for processing one bit.
横轴：每 bit 所需 CPU 周期 ∈ {100, 150, 200, 250, 300, 350} cycle
纵轴：The maximum λ_i
论文结论：
  - 每 bit 周期数越大，计算负载越重，所有算法的 λ_i 都上升；
  - JBHCO-PB 与 CL 表现接近；
  - JBHCO-PG 始终最差，周期数为 350 时高达 0.928，而其他算法都低于 0.6；
  - JBHCO-MB 因为同时具备"本地/卸载二选一"和更好的 BHTP，表现最好。

运行：
    python fig5_cpu_cycles.py
    python fig5_cpu_cycles.py --runs 10 --rerun
"""

from __future__ import annotations

import argparse

from experiment_utils import add_common_arguments, plot_curves, sweep_parameter

# 横坐标：处理 1 bit 数据所需的 CPU 周期数，单位 cycle/bit
CPU_CYCLE_VALUES = [100, 150, 200, 250, 300, 350]


def main() -> None:
    parser = argparse.ArgumentParser(description="论文图 5：不同每 bit CPU 周期数")
    add_common_arguments(parser)
    arguments = parser.parse_args()

    curves = sweep_parameter(
        parameter_name="cpu_cycles_per_bit",           # 决定 O_i = M_i × 该值
        parameter_values=CPU_CYCLE_VALUES,
        cache_name="fig5_cpu_cycles",
        num_runs=arguments.runs,
        base_overrides={"max_swap_iterations": arguments.swaps},
        force_rerun=arguments.rerun,
    )

    plot_curves(
        x_values=CPU_CYCLE_VALUES,
        curves=curves,
        x_label="Number of CPU cycles for processing one bit (cycle)",
        output_name="fig5_cpu_cycles.png",
        title="Fig. 5  max $\\lambda_i$ vs. CPU cycles per bit",
    )


if __name__ == "__main__":
    main()
