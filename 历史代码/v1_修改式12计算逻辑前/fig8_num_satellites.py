"""复现论文图 8：max λ_i 随"卫星数量 N_S"的变化。

论文原图
--------
Fig. 8: The maximum λ_i for ground users versus different number of satellites.
横轴：N_S ∈ {2, 3, 4, 5, 6, 7}
纵轴：The maximum λ_i
论文结论：
  - N_S 增大意味着系统能提供更多容量和计算资源，JBHCO-MB 的 λ_i 随之下降；
  - CL 不与卫星协作，曲线保持水平；
  - JBHCO-MB 与 JBHCO-PB 表现接近，但前者始终更低 —— 论文把这归因于
    swap 操作能找到更好的 BHTP，从而给地面用户带来更多容量；
  - 平均而言 JBHCO-MB 比 JBHCO-PB 低 3.68%，比 CL 和 JBHCO-PG 分别低 15.27% 和 60.93%。

注意
----
N_S=7 时 N_S·N_K = 28 > N_I = 25，同一时隙内无法让所有波束对准互不相同的小区，
BHTP 会出现冗余照明，这是 C1 约束下的必然结果（见 solvers 中
_fill_slot_to_beam_capacity 的说明）。

运行：
    python fig8_num_satellites.py
    python fig8_num_satellites.py --runs 10 --rerun
"""

from __future__ import annotations

import argparse

from experiment_utils import add_common_arguments, plot_curves, sweep_parameter

# 横坐标：卫星数量 N_S
SATELLITE_COUNTS = [2, 3, 4, 5, 6, 7]


def main() -> None:
    parser = argparse.ArgumentParser(description="论文图 8：不同卫星数量")
    add_common_arguments(parser)
    arguments = parser.parse_args()

    curves = sweep_parameter(
        parameter_name="num_satellites",               # SimConfig 中对应 N_S 的字段
        parameter_values=SATELLITE_COUNTS,
        cache_name="fig8_num_satellites",
        num_runs=arguments.runs,
        base_overrides={"max_swap_iterations": arguments.swaps},
        force_rerun=arguments.rerun,
    )

    plot_curves(
        x_values=SATELLITE_COUNTS,
        curves=curves,
        x_label="The number of satellites",
        output_name="fig8_num_satellites.png",
        title="Fig. 8  max $\\lambda_i$ vs. number of satellites",
    )


if __name__ == "__main__":
    main()
