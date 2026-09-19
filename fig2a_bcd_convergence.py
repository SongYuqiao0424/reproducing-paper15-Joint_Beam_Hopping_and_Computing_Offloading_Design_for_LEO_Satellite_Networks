"""复现论文图 2(a)：BCD 算法的收敛曲线。

论文原图
--------
Fig. 2a: Convergence curves of BCD
横轴：迭代次数（论文标注为 N_max，但这张图画的是 Algorithm 3 内部 BCD 的迭代序号）
纵轴：The maximum λ_i
四条曲线对应四种网络规模：N_S=5/N_I=25、N_S=5/N_I=29、N_S=6/N_I=25、N_S=7/N_I=25
论文结论："curves ... could converge within at most eight BCD iterations"

本脚本做法
----------
对每种网络规模，先用匹配理论生成初始 BHTP，再调用 Algorithm 3，
把它内部记录的 bcd_history（每轮迭代后的 max_i λ_i）画出来。
为了让曲线平滑，对多组随机任务取平均（--runs 控制次数）。

运行：
    python fig2a_bcd_convergence.py
    python fig2a_bcd_convergence.py --runs 10 --rerun
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

from config import build_config
from experiment_utils import RESULT_DIR, add_common_arguments, plot_generic_series
from proposed_algo import initialize_simulation
from solvers import design_beam_hopping_matching, solve_joint_offloading_and_allocation

# 论文图 2 使用的四种网络规模场景：(卫星数 N_S, 地面小区数 N_I)
SCENARIOS = [(5, 25), (5, 29), (6, 25), (7, 25)]

MAX_ITERATIONS = 8      # 横坐标最多画到第 8 次迭代，与论文结论一致


def collect_bcd_curve(num_satellites: int, num_cells: int, num_runs: int) -> List[float]:
    """跑若干组随机任务，返回平均后的 BCD 收敛曲线（长度 = MAX_ITERATIONS）。"""
    accumulated = [0.0] * MAX_ITERATIONS
    for run_index in range(num_runs):
        config = build_config(
            num_satellites=num_satellites,
            num_ground_cells=num_cells,
            solver_backend="fast",
            max_bcd_iterations=MAX_ITERATIONS,
            random_seed=7 + run_index,
        )
        network, channel_model = initialize_simulation(config)
        # 先构造初始 BHTP（Algorithm 4 第 2-8 行），再看 Algorithm 3 内部的 BCD 收敛情况
        plan = design_beam_hopping_matching(config, network, channel_model)
        result = solve_joint_offloading_and_allocation(config, network, channel_model, plan)
        history = list(result["bcd_history"])
        # BCD 提前收敛时历史比 8 短，用最后一个值补齐，画出来就是一条水平尾巴
        while len(history) < MAX_ITERATIONS:
            history.append(history[-1] if history else 0.0)
        for index in range(MAX_ITERATIONS):
            accumulated[index] += history[index] / num_runs
    return accumulated


def main() -> None:
    parser = argparse.ArgumentParser(description="论文图 2(a)：BCD 收敛曲线")
    add_common_arguments(parser)
    arguments = parser.parse_args()

    RESULT_DIR.mkdir(exist_ok=True)
    cache_path = RESULT_DIR / "fig2a_bcd_convergence.json"
    if cache_path.exists() and not arguments.rerun:
        with cache_path.open(encoding="utf-8") as handle:
            series = json.load(handle)
        print(f"[缓存] 直接读取 {cache_path.name}")
    else:
        series: Dict[str, Dict[str, List[float]]] = {}
        for num_satellites, num_cells in SCENARIOS:
            label = f"$N_S$={num_satellites}, $N_I$={num_cells}"
            print(f"[计算] 场景 {label}")
            curve = collect_bcd_curve(num_satellites, num_cells, arguments.runs)
            series[label] = {"x": list(range(1, MAX_ITERATIONS + 1)), "y": curve}
            print(f"       收敛曲线：{[round(v, 4) for v in curve]}")
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(series, handle, ensure_ascii=False, indent=2)
        print(f"[保存] 数值结果写入 {cache_path}")

    plot_generic_series(
        series,
        x_label="Number of BCD iterations",
        y_label=r"The maximum $\lambda_i$",
        output_name="fig2a_bcd_convergence.png",
        title="Fig. 2(a) Convergence curves of BCD",
    )


if __name__ == "__main__":
    main()
