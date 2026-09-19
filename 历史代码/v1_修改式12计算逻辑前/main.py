"""单次仿真主入口：在论文表 II 的默认参数下跑一遍四种算法并打印/保存结果。

运行方式
--------
    python main.py                  # 默认用 fast 后端（秒级）
    python main.py --backend cvxpy  # 严格按论文用 CVXPY 解 P̃2/P̃3（慢很多，用于核对建模）
    python main.py --seed 11        # 换一组随机任务

流程
----
    Step 1  读取仿真参数（config.SimConfig，默认值 = 论文表 II）
    Step 2  生成卫星网络拓扑与信道模型
    Step 3  打印链路预算，确认信道量级合理
    Step 4  运行 JBHCO-MB（论文 Algorithm 4）
    Step 5  运行三个对照算法 JBHCO-PB / CL / JBHCO-PG
    Step 6  打印对比结果并写入 simulation_result.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from baselines import run_computing_locally, run_jbhco_pb, run_jbhco_pg
from config import SimConfig, build_config, describe_config
from proposed_algo import initialize_simulation, proposed_jbhco_mb
from solvers import illuminating_satellites


def parse_arguments() -> argparse.Namespace:
    """定义命令行参数。"""
    # 创建命令行参数解析器，并设置程序说明文字
    parser = argparse.ArgumentParser(description="论文 15 联合波束跳跃与计算卸载仿真")
    # 添加命令行参数，使用方式如：python main.py --backend cvxpy --seed 11 --swaps 20
    parser.add_argument("--backend", choices=["fast", "cvxpy"], default="fast",
                        help="凸子问题求解后端；cvxpy 严格对应论文的 CVX，但慢得多")
    parser.add_argument("--seed", type=int, default=7, help="随机种子（决定任务大小抽样）")
    parser.add_argument("--swaps", type=int, default=None, help="覆盖 N_max（swap 次数）")
    return parser.parse_args()


def print_link_budget(config: SimConfig, network, channel_model) -> None:
    """打印一条典型链路的预算，用于核对信道建模是否合理。

    量级参考（表 II 默认参数，星下点正下方的小区）：
        FSPL ≈ 183.6 dB，接收 SINR ≈ 19 dB，单波束瞬时容量 ≈ 2.5 Gbit/s
    """
    sat_id, cell_id = 0, 0
    distance_km = network.sat_to_cell_distance[(sat_id, cell_id)]
    path_loss_db = channel_model.path_loss_db[(sat_id, cell_id)]
    sinr = channel_model.get_sinr(sat_id, cell_id)
    capacity = channel_model.get_capacity(sat_id, cell_id)
    print("[INFO] 链路预算抽样（卫星 0 → 小区 0，无干扰参考值）：")
    print(f"  - 距离 d_i,s        ：{distance_km:.1f} km")
    print(f"  - 路径损耗 PL       ：{path_loss_db:.2f} dB")
    print(f"  - 信噪比 SNR        ：{10.0 * math.log10(max(sinr, 1e-30)):.2f} dB")
    print(f"  - 瞬时容量 C        ：{capacity / 1e9:.3f} Gbit/s")


def summarize(name: str, result: dict) -> str:
    """把一个算法的结果压成一行文本。"""
    return (
        f"  {name:<10s} max λ_i = {result['max_lambda']:.4f}   "
        f"最大时延 = {result['maximum_latency_s']:.3f} s   "
        f"本地 {len(result['local_users']):2d} / 卸载 {len(result['offloaded_users']):2d}   "
        f"C4 满足: {'是' if result['constraint_C4_ok'] else '否'}"
    )


def main() -> None:
    """仿真主函数。"""
    arguments = parse_arguments()
    print("=" * 78)
    print("论文 15：Joint Beam Hopping and Computing Offloading for LEO Satellite Networks")
    print("=" * 78)
    start_time = time.time()

    # ---------- Step 1：读取仿真参数 ----------
    # 根据命令行参数构建仿真配置对象 SimConfig
    config = build_config(solver_backend=arguments.backend, random_seed=arguments.seed)
    if arguments.swaps is not None:
        config.max_swap_iterations = arguments.swaps
    print("[INFO] 仿真配置（默认值取自论文表 II）：")
    print(f"  - 卫星数 N_S            ：{config.num_satellites}")
    print(f"  - 地面小区数 N_I        ：{config.num_ground_cells}")
    print(f"  - 每星波束数 N_K        ：{config.num_beams_per_satellite}")
    print(f"  - BHP 时隙数 N_w        ：{config.num_slots_per_BHP}（Δt={config.slot_duration_s*1e3:.0f} ms，"
          f"T_BH={config.beam_hopping_period_s*1e3:.0f} ms）")
    print(f"  - 系统带宽 B            ：{config.system_bandwidth_GHz} GHz")
    print(f"  - 卫星算力 f_max        ：{config.sat_compute_capacity_GHz} Gcycle/s")
    print(f"  - 本地算力 f^l_i        ：{config.user_local_compute_GHz} Gcycle/s")
    print(f"  - 任务量 M_i            ：{config.task_data_size_range_Mbit} Mbit")
    print(f"  - swap 次数 N_max       ：{config.max_swap_iterations}")
    print(f"  - 凸子问题求解后端      ：{config.solver_backend}")

    # ---------- Step 2：初始化网络与信道 ----------
    network, channel_model = initialize_simulation(config)
    print("[INFO] 已构建卫星网络拓扑与信道模型：")
    print(f"  - 卫星节点数：{len(network.satellites)}")
    print(f"  - 地面小区数：{len(network.ground_cells)}")
    print(f"  - 用户任务数：{len(network.user_tasks)}")

    # ---------- Step 3：链路预算体检 ----------
    print_link_budget(config, network, channel_model)

    # ---------- Step 4：运行 JBHCO-MB ----------
    print("[INFO] 正在运行 JBHCO-MB（论文 Algorithm 4）……")
    proposed = proposed_jbhco_mb(config, network, channel_model)
    proposed["algorithm"] = "JBHCO-MB"
    print(f"  - 初始匹配目标 G(Ω)  ：{proposed['history'][0]:.4f}")
    print(f"  - swap 后目标 G(Ω)   ：{proposed['objective_value']:.4f}")
    print(f"  - 有效 swap / 尝试   ：{proposed['accepted_swaps']} / {proposed['attempted_swaps']}")
    print(f"  - BCD 收敛历史       ：{[round(v, 4) for v in proposed['bcd_history']]}")
    print(f"  - 用时               ：{proposed['elapsed_seconds']:.2f} s")

    # ---------- Step 5：运行三个对照算法 ----------
    print("[INFO] 正在运行对照算法 JBHCO-PB / CL / JBHCO-PG……")
    polling_bcd = run_jbhco_pb(config, network, channel_model)
    local_only = run_computing_locally(config, network, channel_model)
    polling_greedy = run_jbhco_pg(config, network, channel_model)

    # ---------- Step 6：结果对比 ----------
    print("-" * 78)
    print("[结果] 四种算法的最大时延满意度 max λ_i（越小越好）：")
    print(summarize("JBHCO-MB", proposed))
    print(summarize("JBHCO-PB", polling_bcd))
    print(summarize("CL", local_only))
    print(summarize("JBHCO-PG", polling_greedy))
    baseline = proposed["max_lambda"]
    for name, result in (("JBHCO-PB", polling_bcd), ("CL", local_only), ("JBHCO-PG", polling_greedy)):
        reduction = (result["max_lambda"] - baseline) / result["max_lambda"] * 100.0
        print(f"  JBHCO-MB 相比 {name:<9s} 降低 {reduction:6.2f}%")
    print("-" * 78)

    # 校验约束 C1：每个时隙每颗卫星都必须正好点亮 N_K 个小区
    beam_counts = {value for row in proposed["served_cells_per_satellite_slot"] for value in row}
    print(f"[检查] 约束 C1（每星每槽波束数）实际取值集合：{beam_counts}，应为 "
          f"{{{config.num_beams_per_satellite}}}")
    coverage = [
        len(illuminating_satellites(proposed["beam_hopping_plan"], task.cell_id))
        for task in network.user_tasks
    ]
    print(f"[检查] 每个小区可接入的卫星数：最小 {min(coverage)}，最大 {max(coverage)}")

    # ---------- 保存结构化结果 ----------
    summary = {
        "config": describe_config(config),
        "results": {
            name: {
                "max_lambda": result["max_lambda"],
                "maximum_latency_s": result["maximum_latency_s"],
                "num_local_users": len(result["local_users"]),
                "num_offloaded_users": len(result["offloaded_users"]),
                "constraint_C4_ok": result["constraint_C4_ok"],
                "normalized_latencies": result["normalized_latencies"],
            }
            for name, result in (
                ("JBHCO-MB", proposed),
                ("JBHCO-PB", polling_bcd),
                ("CL", local_only),
                ("JBHCO-PG", polling_greedy),
            )
        },
        "jbhco_mb_detail": {
            "swap_history": proposed["history"],
            "bcd_history": proposed["bcd_history"],
            "accepted_swaps": proposed["accepted_swaps"],
            "served_cells_per_satellite_slot": proposed["served_cells_per_satellite_slot"],
        },
        "elapsed_seconds": time.time() - start_time,
    }
    result_path = Path(__file__).resolve().parent / "simulation_result.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"[INFO] 结果已保存到 {result_path}")
    print(f"[INFO] 总耗时：{summary['elapsed_seconds']:.2f} s")


if __name__ == "__main__":
    main()
