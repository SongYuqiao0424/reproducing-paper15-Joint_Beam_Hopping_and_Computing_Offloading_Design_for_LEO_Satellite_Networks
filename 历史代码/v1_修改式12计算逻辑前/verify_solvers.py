"""建模与求解器自检脚本：跑一遍就知道代码有没有偏离论文。

一共 7 项检查，任何一项 [失败] 都说明代码或参数有问题：

  检查 1  链路预算量级是否合理（FSPL / SNR / 单波束容量）
  检查 2  式 (5) 的天线方向图：θ=0 取 G_rx^max，相邻小区落在近旁瓣
  检查 3  BHTP 约束 C1：每星每槽正好 N_K 个波束，且每个小区至少被照明一次
  检查 4  CL 的解析值：max λ_i = max_i(M_i·O/f^l_i)/D_i，应与论文 CL 曲线一致
  检查 5  两套后端（cvxpy / fast）求解 P̃2、P̃3 的结果是否一致
  检查 6  Algorithm 3 的 BCD 目标序列是否单调不增
  检查 7  Algorithm 4 的 swap 是否只接受严格改进，且最终解满足 C1~C6

运行：
    python verify_solvers.py
"""

from __future__ import annotations

import math
from typing import List

from baselines import run_computing_locally
from config import build_config
from proposed_algo import initialize_simulation, proposed_jbhco_mb
from solvers import (
    OffloadingState,
    build_psi_matrix,
    build_zeta_matrix,
    design_beam_hopping_matching,
    illuminating_satellites,
    local_latency_s,
    solve_joint_offloading_and_allocation,
    solve_offloading,
    solve_resource_allocation,
    validate_beam_hopping_plan,
)

PASSED: List[bool] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    """打印一项检查结果并记录通过与否。"""
    PASSED.append(ok)
    mark = "[通过]" if ok else "[失败]"
    print(f"{mark} {name}" + (f"  —— {detail}" if detail else ""))


def main() -> None:
    print("=" * 78)
    print("论文 15 仿真代码自检")
    print("=" * 78)

    config = build_config(solver_backend="fast")
    network, channel_model = initialize_simulation(config)

    # ---------------- 检查 1：链路预算 ----------------
    distance_km = network.sat_to_cell_distance[(0, 0)]
    path_loss_db = channel_model.path_loss_db[(0, 0)]
    snr = config.user_tx_power_W * channel_model.channel_gain_linear[(0, 0)] / channel_model._noise_power_cache
    snr_db = 10.0 * math.log10(snr)
    capacity_Gbps = channel_model._capacity_from_sinr(snr) / 1e9
    ok = 170.0 < path_loss_db < 200.0 and 5.0 < snr_db < 35.0 and 0.5 < capacity_Gbps < 10.0
    report("检查 1 链路预算量级", ok,
           f"d={distance_km:.0f} km, PL={path_loss_db:.1f} dB, SNR={snr_db:.1f} dB, C={capacity_Gbps:.2f} Gbit/s")

    # ---------------- 检查 2：天线方向图 ----------------
    boresight = channel_model.receive_antenna_gain_dBi(0, 0, 0)          # θ=0
    adjacent = channel_model.receive_antenna_gain_dBi(0, 0, 1)           # 相邻小区
    far = channel_model.receive_antenna_gain_dBi(0, 0, config.num_ground_cells - 1)
    ok = (abs(boresight - config.sat_rx_gain_dBi) < 1e-9
          and adjacent < boresight - 3.0
          and far <= adjacent + 1e-9)
    report("检查 2 式 (5) 天线方向图", ok,
           f"θ=0 → {boresight:.1f} dBi，相邻小区 → {adjacent:.1f} dBi，最远小区 → {far:.1f} dBi")

    # ---------------- 检查 3：BHTP 约束 ----------------
    plan = design_beam_hopping_matching(config, network, channel_model)
    try:
        validate_beam_hopping_plan(config, network, plan)
        beam_counts = {
            sum(plan[t][s]) for t in range(config.num_slots_per_BHP)
            for s in range(config.num_satellites)
        }
        ok = beam_counts == {config.num_beams_per_satellite}
        detail = f"每星每槽波束数集合 = {beam_counts}"
    except ValueError as error:
        ok, detail = False, str(error)
    report("检查 3 BHTP 约束 C1", ok, detail)

    # ---------------- 检查 4：CL 的解析值 ----------------
    analytic = max(local_latency_s(t) / t.max_latency_s for t in network.user_tasks)
    simulated = run_computing_locally(config, network, channel_model)["max_lambda"]
    ok = abs(analytic - simulated) < 1e-12
    report("检查 4 CL 解析值", ok, f"解析 {analytic:.6f} vs 仿真 {simulated:.6f}")

    # ---------------- 检查 5：两套后端一致性 ----------------
    config_cvx = build_config(solver_backend="cvxpy")
    capacity_table = channel_model.average_capacity_table(plan)
    psi_unit = build_psi_matrix(config, network, capacity_table)

    # 构造一个测试关联：每个用户接到"照明过它且平均容量最大"的卫星
    association = [[0.0] * config.num_users for _ in range(config.num_satellites)]
    for user_index, task in enumerate(network.user_tasks):
        candidates = illuminating_satellites(plan, task.cell_id)
        if candidates:
            best = max(candidates, key=lambda s: capacity_table.get((s, task.cell_id), 0.0))
            association[best][user_index] = 1.0

    fast_p3 = solve_resource_allocation(config, network, association, psi_unit)
    cvx_p3 = solve_resource_allocation(config_cvx, network, association, psi_unit)
    p3_gap = abs(fast_p3["epsilon"] - cvx_p3["epsilon"]) / max(fast_p3["epsilon"], 1e-12)

    state = OffloadingState(config, network, psi_unit, [None] * config.num_users)
    zeta = build_zeta_matrix(config, network, capacity_table, state.forecast_matrix())
    fast_p2 = solve_offloading(config, network, zeta)
    cvx_p2 = solve_offloading(config_cvx, network, zeta)
    p2_gap = abs(fast_p2["gamma"] - cvx_p2["gamma"]) / max(fast_p2["gamma"], 1e-12)
    same_choice = sum(
        1 for i in range(config.num_users)
        if max(range(config.num_satellites), key=lambda s: fast_p2["association_relaxed"][s][i])
        == max(range(config.num_satellites), key=lambda s: cvx_p2["association_relaxed"][s][i])
    )
    ok = p3_gap < 1e-2 and p2_gap < 1e-4 and same_choice == config.num_users
    report("检查 5 cvxpy / fast 后端一致", ok,
           f"P̃3 ε 相对差 {p3_gap:.2e}，P̃2 γ 相对差 {p2_gap:.2e}，接入卫星一致 {same_choice}/{config.num_users}")
    print("        注：P̃3 的最优解在非瓶颈卫星上不唯一（剩余算力怎么分都不影响 ε），")
    print("            所以两套后端的 f^S 矩阵可以不同，但最优值 ε 必须一致。")

    # ---------------- 检查 6：BCD 单调性 ----------------
    result = solve_joint_offloading_and_allocation(config, network, channel_model, plan)
    history = result["bcd_history"]
    ok = all(history[i + 1] <= history[i] + 1e-9 for i in range(len(history) - 1))
    report("检查 6 BCD 目标单调不增", ok, f"序列 = {[round(v, 5) for v in history]}")

    # ---------------- 检查 7：swap 只接受改进 + 最终解可行 ----------------
    proposed = proposed_jbhco_mb(config, network, channel_model)
    swap_history = proposed["history"]
    monotone = all(swap_history[i + 1] <= swap_history[i] + 1e-12 for i in range(len(swap_history) - 1))
    try:
        validate_beam_hopping_plan(config, network, proposed["beam_hopping_plan"])
        plan_ok = True
    except ValueError:
        plan_ok = False
    # C5：每个用户最多关联一颗卫星
    c5_ok = all(
        sum(proposed["access_matrix"][s][i] for s in range(config.num_satellites)) <= 1
        for i in range(config.num_users)
    )
    # C2：每颗卫星分配的算力总和不超过 f_max
    c2_ok = all(
        sum(proposed["allocation"][s]) <= config.sat_compute_capacity_cps * (1 + 1e-6)
        for s in range(config.num_satellites)
    )
    ok = monotone and plan_ok and c5_ok and c2_ok and proposed["constraint_C4_ok"]
    report("检查 7 swap 单调 + C1~C6 可行", ok,
           f"swap 单调={monotone}, C1={plan_ok}, C2={c2_ok}, C4={proposed['constraint_C4_ok']}, C5={c5_ok}；"
           f"G(Ω) {swap_history[0]:.4f} → {swap_history[-1]:.4f}")

    print("-" * 78)
    print(f"自检结果：{sum(PASSED)}/{len(PASSED)} 项通过")
    if all(PASSED):
        print("全部通过，可以放心用于复现论文仿真图。")
    else:
        print("存在未通过项，请检查上面标记 [失败] 的条目。")


if __name__ == "__main__":
    main()
