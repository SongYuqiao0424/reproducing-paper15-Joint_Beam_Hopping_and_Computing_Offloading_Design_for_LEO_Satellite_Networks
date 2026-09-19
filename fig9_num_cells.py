"""复现论文图 9：max λ_i 随"地面小区数量 N_I"的变化。

论文原图
--------
Fig. 9: The maximum λ_i for ground users versus different number of ground cells.
横轴：N_I ∈ {21, 23, 25, 27, 29, 31}
纵轴：The maximum λ_i
论文结论：
  - 小区（用户）越多，卫星上可用的计算资源被摊得越薄，卸载类算法的 λ_i 上升；
  - JBHCO-PG 最差；JBHCO-MB 和 JBHCO-PB 因为同时利用了 MEC 和本地算力，
    优于 CL 和 JBHCO-PG；
  - 小区数超过 25 之后 JBHCO-MB 的优势更明显，N_I=31 时比 JBHCO-PB 低 13.45%；
  - 平均而言比 JBHCO-PB 低 6.37%，比 CL 和 JBHCO-PG 分别低 17.11% 和 49.06%。

运行：
    python fig9_num_cells.py
    python fig9_num_cells.py --runs 10 --rerun
"""

from __future__ import annotations

import argparse

from experiment_utils import add_common_arguments, plot_curves, sweep_parameter

# 横坐标：地面小区数量 N_I（本模型中也等于用户数）
CELL_COUNTS = [21, 23, 25, 27, 29, 31]


def main() -> None:
    parser = argparse.ArgumentParser(description="论文图 9：不同地面小区数量")
    add_common_arguments(parser)
    arguments = parser.parse_args()

    curves = sweep_parameter(
        parameter_name="num_ground_cells",             # SimConfig 中对应 N_I 的字段
        parameter_values=CELL_COUNTS,
        cache_name="fig9_num_cells",
        num_runs=arguments.runs,
        base_overrides={"max_swap_iterations": arguments.swaps},
        force_rerun=arguments.rerun,
    )

    plot_curves(
        x_values=CELL_COUNTS,
        curves=curves,
        x_label="The number of ground cells",
        output_name="fig9_num_cells.png",
        title="Fig. 9  max $\\lambda_i$ vs. number of ground cells",
    )


if __name__ == "__main__":
    main()
