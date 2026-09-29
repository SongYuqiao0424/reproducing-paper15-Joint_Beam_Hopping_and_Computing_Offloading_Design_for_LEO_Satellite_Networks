"""论文 15 的主算法：JBHCO-MB（Algorithm 4）。

JBHCO-MB = 匹配理论(Matching) + 块坐标下降(BCD) 的联合波束跳跃与计算卸载算法。

论文 Algorithm 4 的伪代码与本文件函数的对应关系：
    第 1 行      初始化场景参数、迭代计数 n、最大 swap 次数 N_max
    第 2-8 行    构造偏好列表并完成初始 BHTP 匹配
                 → solvers.design_beam_hopping_matching()
    第 9 行      对当前匹配 Ω 用 Algorithm 3 求最优 A 和 f^S
                 → solvers.solve_joint_offloading_and_allocation()
    第 10 行     计算 G(Ω) = min max λ_i（式 (36)）
                 → solvers.evaluate_objective()
    第 11-15 行  随机 swap 主循环：抽取 Ω^{t,t'}_{i,i'}，重解 Algorithm 3，
                 若构成 swap-blocking pair（定义 3）就接受
                 → solvers.propose_swap() + 本文件的 proposed_jbhco_mb()
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Tuple

from channel_model import ChannelModel
from config import SimConfig
from satellite_network import SatelliteNetwork
from solvers import (
    design_beam_hopping_matching,
    evaluate_objective,
    propose_swap,
    solve_joint_offloading_and_allocation,
)


def initialize_simulation(config: SimConfig) -> Tuple[SatelliteNetwork, ChannelModel]:
    """构建仿真所需的网络拓扑和信道模型。

    这是所有算法（JBHCO-MB 和三个对照算法）共用的初始化入口，
    保证四种方案面对完全相同的拓扑、信道和任务集合，比较才公平。
    """
    network = SatelliteNetwork(config)
    channel_model = ChannelModel(config, network)
    # 刷新一次"无干扰参考容量"，方便 main.py 打印链路预算做 sanity check。
    channel_model.update_channel_matrix()
    return network, channel_model


def proposed_jbhco_mb(
    config: SimConfig,
    network: SatelliteNetwork,
    channel_model: ChannelModel,
    record_history: bool = True,
    max_swaps: Optional[int] = None,
) -> Dict[str, object]:
    """论文 Algorithm 4：JBHCO-MB 主算法。

    参数
    ----
    record_history : 是否逐次记录 swap 后的 G(Ω)。论文图 2b 需要这条曲线。
    max_swaps      : 覆盖 config.max_swap_iterations（N_max），供图 2b 扫描不同 N_max 时复用。

    返回
    ----
    字典，含最终 BHTP、关联矩阵 A、资源分配 f^S、目标值 G(Ω)、
    收敛历史 history（第 0 项是初始匹配的结果，之后每次 swap 记一项）等。
    """
    swap_budget = config.max_swap_iterations if max_swaps is None else max_swaps
    start_time = time.time()

    # ---- 第 2-8 行：匹配理论构造初始 BHTP ----
    beam_hopping_plan = design_beam_hopping_matching(config, network, channel_model)

    # ---- 第 9-10 行：对初始匹配求 A、f^S，并算出 G(Ω) ----
    result = solve_joint_offloading_and_allocation(config, network, channel_model, beam_hopping_plan)
    best_objective = float(result["max_lambda"])            # G(Ω) = max_i λ_i
    history: List[float] = [best_objective] if record_history else []   # record_history为是否记录历史的标志，默认开启
    accepted_swaps = 0                                      # 实际被接受的 swap 次数
    attempted_swaps = 0                                     # 实际构造出可行交换的次数

    # swap 的随机过程用独立的随机数发生器，种子由 config 决定，保证可复现
    rng = random.Random(config.random_seed + 1000)

    # ---- 第 11-15 行：swap 主循环 ----
    for _ in range(swap_budget):
        # 第 12 行：随机选一颗卫星的两个时隙，交换它们各自的一个被照明小区
        candidate_plan = propose_swap(config, beam_hopping_plan, rng)
        if candidate_plan is None:
            # 这次抽样不构成可行交换（会违反 C1 或去冗余规则），不消耗 swap 预算的判定
            if record_history:
                history.append(best_objective)  # ！！不增加 attempted_swaps 计数，但增加历史记录是否合适，难道因为history与swap_budget有关？
            continue
        attempted_swaps += 1

        # 第 13 行：对交换后的匹配重新用 Algorithm 3 求最优 A 和 f^S
        candidate_result = solve_joint_offloading_and_allocation(
            config, network, channel_model, candidate_plan
        )
        candidate_objective = float(candidate_result["max_lambda"])

        # 第 14 行：定义 3 的 swap-blocking pair 判据 —— 交换后目标严格下降就接受
        if candidate_objective < best_objective - config.tolerance:
            beam_hopping_plan = candidate_plan
            result = candidate_result
            best_objective = candidate_objective
            accepted_swaps += 1
        if record_history:
            history.append(best_objective)

    # ---- 汇总输出 ----
    access_matrix = result["access_matrix"]                 # A = [a_i]，0-1 关联
    allocation = result["allocation"]                       # f^S_{s,i}，单位 cycle/s
    latency_result = result["latency_result"]
    # 每个时隙每颗卫星实际点亮的小区数（应恒等于 N_K，用来核对约束 C1）
    served_cells_per_satellite_slot = [
        [sum(beam_hopping_plan[t][s]) for s in range(config.num_satellites)]
        for t in range(config.num_slots_per_BHP)
    ]

    return {
        "beam_hopping_plan": beam_hopping_plan,
        "access_matrix": access_matrix,
        "allocation": allocation,
        "objective_value": best_objective,                  # G(Ω) = max_i λ_i
        "max_lambda": best_objective,
        "maximum_latency_s": latency_result["maximum_latency_s"],
        "latencies_s": latency_result["latencies_s"],
        "normalized_latencies": latency_result["normalized_latencies"],
        "local_users": latency_result["local_users"],
        "offloaded_users": latency_result["offloaded_users"],
        "constraint_C4_ok": latency_result["constraint_C4_ok"],
        "bcd_history": result["bcd_history"],
        "history": history,
        "accepted_swaps": accepted_swaps,
        "attempted_swaps": attempted_swaps,
        "served_cells_per_satellite_slot": served_cells_per_satellite_slot,
        "elapsed_seconds": time.time() - start_time,
    }


def run_jbhco_mb(config: SimConfig, **kwargs) -> Dict[str, object]:
    """一步到位的便捷入口：建网络 → 建信道 → 跑 JBHCO-MB。"""
    network, channel_model = initialize_simulation(config)
    return proposed_jbhco_mb(config, network, channel_model, **kwargs)
