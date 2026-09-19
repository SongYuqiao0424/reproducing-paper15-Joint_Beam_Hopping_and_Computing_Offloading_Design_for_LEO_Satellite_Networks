"""复现论文图 6：max λ_i 随"每个用户任务数据量 M_i"的变化。

论文原图
--------
Fig. 6: The maximum λ_i for ground users versus different data size of computing
        task for each ground user.
横轴：M_i 的取值区间 ∈ {0~5, 5~10, 10~15, 15~20, 20~25, 25~30} Mbit
纵轴：The maximum λ_i
论文结论：
  - 任务数据量越大，总计算时延越大，所有算法的 λ_i 单调上升；
  - JBHCO-PG 始终最差；
  - JBHCO-MB 与 JBHCO-PB 接近，但 JBHCO-MB 始终更优，平均比 JBHCO-PB 和 CL
    分别降低 5.93% 和 16.55%，比 JBHCO-PG 降低 50.69%。

实现说明
--------
横坐标是"区间"而不是单个数值，所以用 value_to_overrides 把区间翻译成
SimConfig 的 task_data_size_range_Mbit 字段，并用 0,1,2,... 做等距刻度、
再换成 "0~5" 这样的文字标签。

运行：
    python fig6_task_size.py
    python fig6_task_size.py --runs 10 --rerun
"""

from __future__ import annotations

import argparse

from experiment_utils import add_common_arguments, plot_curves, sweep_parameter

# 横坐标：任务数据量 M_i 的均匀分布区间，单位 Mbit
TASK_SIZE_RANGES = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0),
                    (15.0, 20.0), (20.0, 25.0), (25.0, 30.0)]


def main() -> None:
    parser = argparse.ArgumentParser(description="论文图 6：不同任务数据量")
    add_common_arguments(parser)
    arguments = parser.parse_args()

    curves = sweep_parameter(
        parameter_name="task_data_size_range_Mbit",    # M_i 的抽样区间
        parameter_values=TASK_SIZE_RANGES,
        cache_name="fig6_task_size",
        num_runs=arguments.runs,
        base_overrides={"max_swap_iterations": arguments.swaps},
        # 把 (low, high) 元组翻译成配置覆盖项
        value_to_overrides=lambda size_range: {"task_data_size_range_Mbit": size_range},
        force_rerun=arguments.rerun,
    )

    plot_curves(
        x_values=list(range(len(TASK_SIZE_RANGES))),
        curves=curves,
        x_label="The data size of computing task for each ground user (Mbit)",
        output_name="fig6_task_size.png",
        title="Fig. 6  max $\\lambda_i$ vs. task data size",
        x_tick_labels=[f"{int(low)}~{int(high)}" for low, high in TASK_SIZE_RANGES],
    )


if __name__ == "__main__":
    main()
