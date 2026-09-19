"""复现论文图 2(b)：JBHCO-MB 随 swap 次数 N_max 的收敛曲线。

论文原图
--------
Fig. 2b: Convergence curves of JBHCO-MB
横轴：N_max（swap 次数），论文取 20 / 50 / 100 / 600 / 4000 / 9000 六个刻度
纵轴：The maximum λ_i
四条曲线对应四种网络规模：N_S=5/N_I=25、N_S=5/N_I=29、N_S=6/N_I=25、N_S=7/N_I=25
论文结论："the curves of JBHCO-MB almost all converge to a local optimum after 100 swaps"
        因此后续所有仿真都把 swap 次数定为 100。

本脚本做法
----------
不必为每个 N_max 各跑一次。只要跑一次 max(N_max) 次 swap 的完整 JBHCO-MB，
并让 proposed_jbhco_mb 记录每次 swap 之后的 G(Ω)，然后在这条历史曲线上按
需要的 N_max 取样即可，速度快一个数量级。

运行：
    python fig2b_jbhcomb_convergence.py                # 默认最多 1000 次 swap
    python fig2b_jbhcomb_convergence.py --max-swaps 9000 --runs 3 --rerun
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from config import build_config
from experiment_utils import RESULT_DIR, add_common_arguments, plot_generic_series
from proposed_algo import initialize_simulation, proposed_jbhco_mb

# 论文图 2 使用的四种网络规模场景：(卫星数 N_S, 地面小区数 N_I)
SCENARIOS = [(5, 25), (5, 29), (6, 25), (7, 25)]

# 论文图 2b 的横坐标刻度（等距排列的非线性刻度）
PAPER_TICKS = [20, 50, 100, 600, 4000, 9000]


def collect_swap_curve(
    num_satellites: int,
    num_cells: int,
    num_runs: int,
    max_swaps: int,
    sample_points: List[int],
) -> List[float]:
    """跑若干组随机任务，返回各采样点 N_max 处平均的 max λ_i。"""
    accumulated = [0.0] * len(sample_points)
    for run_index in range(num_runs):
        config = build_config(
            num_satellites=num_satellites,
            num_ground_cells=num_cells,
            solver_backend="fast",
            random_seed=7 + run_index,
        )
        network, channel_model = initialize_simulation(config)
        # record_history=True 会返回长度 = max_swaps+1 的历史，history[n] 就是
        # "做完 n 次 swap 之后的 G(Ω)"，n=0 即初始匹配结果
        result = proposed_jbhco_mb(
            config, network, channel_model, record_history=True, max_swaps=max_swaps
        )
        history = result["history"]
        for index, n_max in enumerate(sample_points):
            position = min(n_max, len(history) - 1)
            accumulated[index] += history[position] / num_runs
    return accumulated


def main() -> None:
    parser = argparse.ArgumentParser(description="论文图 2(b)：JBHCO-MB 收敛曲线")
    add_common_arguments(parser)
    parser.add_argument("--max-swaps", type=int, default=1000,
                        help="最大 swap 次数；论文画到 9000，默认 1000 已足以看出 100 次后收敛")
    arguments = parser.parse_args()

    # 只保留不超过 max_swaps 的刻度
    sample_points = [tick for tick in PAPER_TICKS if tick <= arguments.max_swaps]
    if not sample_points:
        sample_points = [arguments.max_swaps]

    RESULT_DIR.mkdir(exist_ok=True)
    cache_path = RESULT_DIR / "fig2b_jbhcomb_convergence.json"
    if cache_path.exists() and not arguments.rerun:
        with cache_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        series, sample_points = payload["series"], payload["sample_points"]
        print(f"[缓存] 直接读取 {cache_path.name}")
    else:
        series: Dict[str, Dict[str, List[float]]] = {}
        for num_satellites, num_cells in SCENARIOS:
            label = f"$N_S$={num_satellites}, $N_I$={num_cells}"
            print(f"[计算] 场景 {label}（最多 {arguments.max_swaps} 次 swap）")
            curve = collect_swap_curve(
                num_satellites, num_cells, arguments.runs, arguments.max_swaps, sample_points
            )
            # 横坐标用 0,1,2,... 等距排列，再换成论文的刻度文字，形状才和论文一致
            series[label] = {"x": list(range(len(sample_points))), "y": curve}
            print(f"       各 N_max 处的 max λ_i：{[round(v, 4) for v in curve]}")
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump({"series": series, "sample_points": sample_points},
                      handle, ensure_ascii=False, indent=2)
        print(f"[保存] 数值结果写入 {cache_path}")

    path = plot_generic_series(
        series,
        x_label=r"$N_{\max}$",
        y_label=r"The maximum $\lambda_i$",
        output_name="fig2b_jbhcomb_convergence.png",
        title="Fig. 2(b) Convergence curves of JBHCO-MB",
    )
    # 把等距横坐标替换成论文的非线性刻度文字
    import matplotlib.pyplot as plt
    from experiment_utils import FIGURE_DIR
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    palette = ["#000000", "#E02020", "#2050E0", "#12A050"]
    markers = ["s", "o", "^", "v"]
    for index, (name, data) in enumerate(series.items()):
        axis.plot(data["x"], data["y"], color=palette[index % 4], marker=markers[index % 4],
                  label=name, linewidth=1.6, markersize=5)
    axis.set_xticks(list(range(len(sample_points))))
    axis.set_xticklabels([str(tick) for tick in sample_points])
    axis.set_xlabel(r"$N_{\max}$", fontsize=11)
    axis.set_ylabel(r"The maximum $\lambda_i$", fontsize=11)
    axis.set_title("Fig. 2(b) Convergence curves of JBHCO-MB", fontsize=11)
    axis.grid(True, linestyle=":", alpha=0.45)
    axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "fig2b_jbhcomb_convergence.png", dpi=300)
    plt.close(figure)
    print(f"[出图] {FIGURE_DIR / 'fig2b_jbhcomb_convergence.png'}（已替换为论文刻度）")


if __name__ == "__main__":
    main()
