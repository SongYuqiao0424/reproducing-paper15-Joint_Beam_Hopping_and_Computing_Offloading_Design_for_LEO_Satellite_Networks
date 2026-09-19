"""复现论文图 3：max λ_i 随"每个地面用户的计算能力 f^l_i"的变化。

论文原图
--------
Fig. 3: The maximum λ_i for ground users versus different computing capability
        of each ground user.
横轴：f^l_i ∈ {0.2, 0.4, 0.6, 0.8, 1.0, 1.2} Gcycles/s
纵轴：The maximum λ_i
论文结论：
  - CL 曲线随 f^l_i 增大迅速下降（本地时延 = O_i/f^l_i，与 f^l_i 成反比）；
  - JBHCO-PG 把所有任务都卸载上星，所以对 f^l_i 完全不敏感，是一条水平线；
  - f^l_i < 0.5 时 CL 比 JBHCO-PG 差，f^l_i > 0.7 后反超；
  - JBHCO-MB 始终最好，f^l_i=0.2 时相比 CL 降低 78.36%。

运行：
    python fig3_user_compute.py
    python fig3_user_compute.py --runs 10 --rerun
"""

from __future__ import annotations

import argparse

from experiment_utils import add_common_arguments, plot_curves, sweep_parameter

# 横坐标：用户本地计算能力 f^l_i，单位 Gcycle/s
LOCAL_COMPUTE_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]


def main() -> None:
    parser = argparse.ArgumentParser(description="论文图 3：不同用户本地算力")
    add_common_arguments(parser)
    arguments = parser.parse_args()

    curves = sweep_parameter(
        parameter_name="user_local_compute_GHz",      # SimConfig 中对应 f^l_i 的字段
        parameter_values=LOCAL_COMPUTE_VALUES,
        cache_name="fig3_user_compute",
        num_runs=arguments.runs,
        base_overrides={"max_swap_iterations": arguments.swaps},
        force_rerun=arguments.rerun,
    )

    plot_curves(
        x_values=LOCAL_COMPUTE_VALUES,
        curves=curves,
        x_label="The computing capability of each ground user (Gcycles/s)",
        output_name="fig3_user_compute.png",
        title="Fig. 3  max $\\lambda_i$ vs. user computing capability",
    )


if __name__ == "__main__":
    main()
