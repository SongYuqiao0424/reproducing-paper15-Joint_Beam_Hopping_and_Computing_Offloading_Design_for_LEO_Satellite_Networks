"""复现论文图 7：max λ_i 随"系统带宽 B"的变化。

论文原图
--------
Fig. 7: The maximum λ_i for ground users versus different system bandwidth.
横轴：B ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6} GHz
纵轴：The maximum λ_i
论文结论：
  - CL 全部本地计算，与带宽无关，是一条水平线；
  - 带宽越大上行容量越高、上行传输时延越小，因此卸载类算法的 λ_i 随 B 下降；
  - 带宽小于 0.4 GHz 时 JBHCO-MB 的优势更明显；B=0.1 GHz 时相比 JBHCO-PB 和 CL
    分别降低 12.48% 和 15.73%，相比 JBHCO-PG 降低 56.28%。

注意
----
默认参数下上行传输时延只占总时延的百分之几（计算时延占主导），
所以这条曲线的整体变化幅度不大，与论文图 7 的纵轴跨度一致。

运行：
    python fig7_bandwidth.py
    python fig7_bandwidth.py --runs 10 --rerun
"""

from __future__ import annotations

import argparse

from experiment_utils import add_common_arguments, plot_curves, sweep_parameter

# 横坐标：系统带宽 B，单位 GHz
BANDWIDTH_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def main() -> None:
    parser = argparse.ArgumentParser(description="论文图 7：不同系统带宽")
    add_common_arguments(parser)
    arguments = parser.parse_args()

    curves = sweep_parameter(
        parameter_name="system_bandwidth_GHz",         # SimConfig 中对应 B 的字段
        parameter_values=BANDWIDTH_VALUES,
        cache_name="fig7_bandwidth",
        num_runs=arguments.runs,
        base_overrides={"max_swap_iterations": arguments.swaps},
        force_rerun=arguments.rerun,
    )

    plot_curves(
        x_values=BANDWIDTH_VALUES,
        curves=curves,
        x_label="The system bandwidth (GHz)",
        output_name="fig7_bandwidth.png",
        title="Fig. 7  max $\\lambda_i$ vs. system bandwidth",
    )


if __name__ == "__main__":
    main()
