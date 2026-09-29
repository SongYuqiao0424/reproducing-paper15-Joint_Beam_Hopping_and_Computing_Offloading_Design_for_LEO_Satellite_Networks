"""论文 15 Algorithm 1~4 的求解器实现。

本文件把论文第四节的公式逐条落成代码，章节结构如下：

  第 1 节  时延模型               式 (10)~(19)：T^l, T^U, T^P, T^D, T_i, λ_i
  第 2 节  资源分配状态对象        给定 0-1 关联时，增量式评估 max_i λ_i
  第 3 节  Algorithm 1 / P̃2      固定 f^S，求卸载关联 A 与上界 γ
  第 4 节  Algorithm 2 / P̃3      固定 A，求卫星计算资源 f^S 与上界 ε
  第 5 节  Algorithm 3            BCD 交替 + 0-1 关联恢复（步骤 7~17）
  第 6 节  Algorithm 4 前半段      匹配理论构造初始 BHTP（另含对照用的轮询 BHTP）
  第 7 节  Algorithm 4 后半段      swap 操作与 swap-blocking pair 判定

两套后端
--------
config.solver_backend 可取：
  "cvxpy"：严格按论文用凸优化建模 P̃2 / P̃3，交给 CVXPY 调 CLARABEL/SCS 求解，
           对应论文 Algorithm 1/2 里的 "Solve P̃2(P̃3) with CVX"。
  "fast" ：用与该凸问题严格等价的闭式解 / 二分解法（推导写在对应函数注释里）。
两者结果一致（verify_solvers.py 会逐点核对），但 "fast" 快 2~3 个数量级，
扫描仿真图时必须用它。
"""

from __future__ import annotations

import copy
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from channel_model import ChannelModel
from config import SimConfig, UserTask
from satellite_network import SatelliteNetwork

# ---------------------------------------------------------------------
# 类型别名，维度与论文变量一一对应
# ---------------------------------------------------------------------
BeamHoppingPlan = List[List[List[int]]]       # X^t_{s,i}：[N_w][N_S][N_I]，跳波束周期模式，时隙t卫星s是否照射用户i，元素 0/1
AccessMatrix = List[List[int]]                # a^s_i：[N_S][N_I]，用户i是否关联卫星s，元素 0/1（约束 C6）
RelaxedAccessMatrix = List[List[float]]       # 松弛后的 a^s_i ∈ [0,1]（式 (22b)）
ResourceMatrix = List[List[float]]            # f^S_{s,i}：[N_S][N_I]，卫星s为用户i分配的计算资源，单位 cycle/s
CapacityTable = Dict[Tuple[int, int], float]  # C^BH_{s,i}：链路容量？，单位 bit/s

_LIGHT_SPEED_M_S = 299792458.0                # c：光速，单位 m/s
_BISECTION_STEPS = 48                         # 二分次数；48 次可把区间压到 1e-14 量级


# =====================================================================
# 第 1 节：时延模型（论文式 (10)~(19)）
# =====================================================================
def local_latency_s(task: UserTask) -> float:
    """式 (10)：本地计算时延 T^l_i = O_i / f^l_i，单位 s。"""
    return task.required_cycles / max(task.local_cpu_GHz * 1e9, 1e-12)


def propagation_delay_s(network: SatelliteNetwork, sat_id: int, cell_id: int) -> float:
    """式 (13)/(17)：单程传播时延 d_i,s / c，单位 s。"""
    distance_m = network.sat_to_cell_distance[(sat_id, cell_id)] * 1000.0
    return distance_m / _LIGHT_SPEED_M_S


def communication_overhead_s(
    config: SimConfig,
    network: SatelliteNetwork,
    task: UserTask,
    sat_id: int,
    capacity_table: CapacityTable,
) -> float:
    """卸载总时延中与卫星算力无关的那一部分，单位 s。

        = T^{U,T}_i + T^{U,P}_i + T^{D,W}_i + T^{D,P}_i
        = M_i / C^BH_i  +  d_i,s/c  +  T_BH  +  d_i,s/c        （式 12,13,16,17）

    论文式 (16) 下方说明：处理结果必须在一个 BHP 内回传完，所以下行排队时延
    T^{D,W}_i 直接取 T_BH；下行数据量可忽略，因此没有下行传输时延项。
    若该卫星在整个 BHP 内从未照明这个小区，则 C^BH=0，返回 +inf 表示不可卸载。
    """
    capacity = capacity_table.get((sat_id, task.cell_id), 0.0)     # C^BH_{s,i}，单位 bit/s
    if capacity <= 0.0:
        return float("inf")
    uplink_transmission = task.data_size_Mbit * 1e6 / capacity      # 式 (12)
    propagation = propagation_delay_s(network, sat_id, task.cell_id)
    return uplink_transmission + 2.0 * propagation + config.beam_hopping_period_s


def build_psi_matrix(
    config: SimConfig,
    network: SatelliteNetwork,
    capacity_table: CapacityTable,
) -> List[List[float]]:
    """预计算 ψ 的"单位关联增量"，维度 [N_S][N_I]，无量纲。

    论文式 (28)：
        ψ_{i,s} = O_i/(D_i f^l_i) + Σ_s (a^s_i/D_i)·(M_i/C^BH_i + 2d_i,s/c + T^{D,W}_i − O_i/f^l_i)

    右边求和项中除 a^s_i 以外都是常数，这里先把这个常数算成矩阵：

        psi_unit[s][i] = (1/D_i)·( M_i/C^BH_{s,i} + 2 d_{i,s}/c + T_BH − T^l_i )
        #此处取值过于精细，C^BH_{s,i}精确到用户-卫星对的容量，
        #但原文仅使用用户对所有卫星的总容量C^BH_i，见式6，应该是每个时隙取照射的卫星计算，最终容量会大许多，
        #且即使在BHTP上卫星s未照射i，取值也不为inf，因为有其他卫星进行平均，C^BH_i不为0

    于是 ψ_{i,s} = L_i + Σ_s a^s_i·psi_unit[s][i]，其中 L_i = T^l_i/D_i。
    物理含义：把任务交给卫星 s 时，"通信与排队开销"相对"本地计算基准"的净增量。
    取值 inf 表示卫星 s 在这个 BHTP 下从未照明小区 i。
    """
    num_satellites = len(network.satellites)
    num_users = len(network.user_tasks)
    # 取值为inf是因为若卫星 s 在这个 BHTP 下从未照明小区 i，则C^BH_{s,i}为0，但是计算应当使用C^BH_i，其值一定不为0，可能需修改
    psi_unit = [[float("inf")] * num_users for _ in range(num_satellites)]
    for user_index, task in enumerate(network.user_tasks):
        local_time = local_latency_s(task)
        for sat_id in range(num_satellites):
            overhead = communication_overhead_s(config, network, task, sat_id, capacity_table)
            if math.isfinite(overhead):
                psi_unit[sat_id][user_index] = (overhead - local_time) / task.max_latency_s
    return psi_unit


def build_zeta_matrix(
    config: SimConfig,
    network: SatelliteNetwork,
    capacity_table: CapacityTable,
    resource: List[List[float]],
) -> List[List[float]]:
    """论文式 (24)：ζ_{i,s}，维度 [N_S][N_I]，无量纲。

        ζ_{i,s} = (1/D_i)( M_i/C^BH_i + 2d_i,s/c + O_i/f^S_{s,i} + T^{D,W}_i − O_i/f^l_i )

    比 psi_unit 多了卫星处理时延项 O_i/f^S_{s,i}，因为 Algorithm 1 里 f^S 是常数。
    ζ_{i,s} < 0 就表示"卸载到卫星 s 比本地算更快"。

    resource[s][i] 是"用户 i 在卫星 s 上可用的计算能力"，由
    OffloadingState.forecast_matrix() 提供，原因见该方法说明。
    """
    num_satellites = len(network.satellites)
    num_users = len(network.user_tasks)
    zeta = [[float("inf")] * num_users for _ in range(num_satellites)]
    for user_index, task in enumerate(network.user_tasks):
        local_time = local_latency_s(task)
        for sat_id in range(num_satellites):
            overhead = communication_overhead_s(config, network, task, sat_id, capacity_table)
            available = resource[sat_id][user_index]
            if math.isfinite(overhead) and available > 0.0:
                processing = task.required_cycles / available      # 式 (15)
                zeta[sat_id][user_index] = (overhead + processing - local_time) / task.max_latency_s
    return zeta


def selected_satellite(access_matrix: Sequence[Sequence[int]], user_index: int) -> Optional[int]:
    """读取用户 i 的 0-1 关联；返回 None 表示本地计算（a_i 全 0）。"""
    for sat_id, row in enumerate(access_matrix):
        if row[user_index] == 1:
            return sat_id
    return None


def illuminating_satellites(plan: BeamHoppingPlan, cell_id: int) -> List[int]:
    """返回在一个 BHP 内至少照明过小区 i 一次的卫星集合。

    论文规定"只有 X^t_{s,i}·a^s_i = 1 时用户 i 才能向卫星 s 卸载"，
    所以这个集合就是用户 i 可选的接入卫星范围。
    """
    num_satellites = len(plan[0])
    return [
        sat_id for sat_id in range(num_satellites)
        if any(slot_plan[sat_id][cell_id] for slot_plan in plan)
    ]


def evaluate_latency(
    config: SimConfig,
    network: SatelliteNetwork,
    channel_model: ChannelModel,
    access_matrix: Sequence[Sequence[int]],
    allocation: ResourceMatrix,
    plan: BeamHoppingPlan,
) -> Dict[str, object]:
    """按式 (18)(19) 计算每个用户的 T_i、λ_i 以及目标函数 max_i λ_i。

    返回字典字段：
        latencies_s          每个用户的总时延 T_i，单位 s
        normalized_latencies 时延满意度 λ_i = T_i/D_i
        max_lambda           论文目标 G(Ω) = max_i λ_i
        local_users          本地计算的用户编号
        offloaded_users      卸载上星的用户编号
        constraint_C4_ok     是否满足 C4（所有 λ_i ≤ 1）
    """
    capacity_table = channel_model.average_capacity_table(plan)
    latencies: List[float] = []
    normalized: List[float] = []
    local_users: List[int] = []
    offloaded_users: List[int] = []

    for user_index, task in enumerate(network.user_tasks):
        sat_id = selected_satellite(access_matrix, user_index)
        if sat_id is None:
            latency = local_latency_s(task)                        # 式 (18) 中 Σa=0 的分支
            local_users.append(task.user_id)
        else:
            overhead = communication_overhead_s(config, network, task, sat_id, capacity_table)
            available = allocation[sat_id][user_index]
            if math.isfinite(overhead) and available > 0.0:
                latency = overhead + task.required_cycles / available   # 式 (11)
                offloaded_users.append(task.user_id)
            else:
                latency = local_latency_s(task)                    # 该星实际无法服务 → 退回本地
                local_users.append(task.user_id)
        latencies.append(latency)
        normalized.append(latency / max(task.max_latency_s, 1e-12))

    return {
        "latencies_s": latencies,
        "normalized_latencies": normalized,
        "maximum_latency_s": max(latencies, default=0.0),
        "max_lambda": max(normalized, default=0.0),
        "local_users": local_users,
        "offloaded_users": offloaded_users,
        "constraint_C4_ok": all(value <= 1.0 + 1e-9 for value in normalized),
    }


def evaluate_objective(config, network, channel_model, access_matrix, allocation, beam_hopping_plan) -> float:
    """返回论文目标 G(Ω) = min max_i λ_i（式 (36)）。"""
    return float(evaluate_latency(
        config, network, channel_model, access_matrix, allocation, beam_hopping_plan
    )["max_lambda"])


# =====================================================================
# 第 2 节：资源分配状态对象
# =====================================================================
def _minimax_allocation_on_satellite(
    users: Sequence[int],
    psi_value: Sequence[float],     # ψ_i
    workload_coef: Sequence[float], # (O_i/D_i)
    budget_cps: float,              # f_max
) -> Tuple[float, Dict[int, float]]:
    """单颗卫星上的 minimax 计算资源分配，二分求解。

    (对于卸载用户，仅卸载卫星s的a_i,s为1，故P3中相关求和表述可省略)
    (f_i是f_i,s的略写，因为此时就是在计算卫星s的计算资源分配策略，i是所有选择卸载至卫星s的用户)
    (ε返回值为卫星 s 上所有用户的共同最坏时延满意度上界，故仅返回一个值，而f_i则需要返回字典)
    子问题P3：min ε  s.t. ψ_i + (O_i/D_i)/f_i ≤ ε ∀i∈U_s,  Σ_{i∈U_s} f_i ≤ f_max,  f_i ≥ 0

    对给定的 ε，约束等价于 f_i ≥ (O_i/D_i)/(ε − ψ_i)，于是
        "ε 可行" ⟺ Σ_i (O_i/D_i)/(ε − ψ_i) ≤ f_max
    左边关于 ε 单调递减，二分即可得最小 ε_s 与对应的精确分配。
    这就是论文 Algorithm 2 的二分法，只是把 CVX 的可行性判定换成了闭式不等式。

    返回 (ε_s, {用户下标: f^S_{s,i}(cycle/s)})；users 为空时返回 (0.0, {})。
    """
    if not users:   # 无用户不必分配资源
        return 0.0, {}
    lower = max(psi_value[i] for i in users)          # ε 必须严格大于所有 ψ_i

    def required(eps: float) -> float:                # 达到 eps 所需的总算力，单位 cycle/s
        return sum(workload_coef[i] / max(eps - psi_value[i], 1e-18) for i in users)

    upper = lower + 1.0
    for _ in range(200):                              # 先把上界撑到可行（任务极重时 ε 可能 >1）
        if required(upper) <= budget_cps:
            break
        upper = lower + (upper - lower) * 2.0
    low, high = lower, upper
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if required(middle) <= budget_cps:
            high = middle
        else:
            low = middle
    epsilon_s = high
    return epsilon_s, {i: workload_coef[i] / max(epsilon_s - psi_value[i], 1e-18) for i in users}


class OffloadingState:
    """给定 BHTP 时，描述"0-1 卸载关联 + 最优计算资源分配"的状态对象。

    把当前卸载和资源分配方案封装起来；当尝试移动一个用户时，只重新计算受影响卫星的资源分配，从而快速判断新的 max_i λ_i 是否更小。

    它把论文的 λ_i 表达式拆成两类（式 (27)(28)）:
        本地计算用户： λ_i = L_i = O_i/(D_i f^l_i)          —— 与 f^S 无关
        （对于卸载用户,仅卸载卫星s的a_i,s为1,故式27、28中求和可省略）
        卸载用户：     λ_i = ψ_i + (O_i/D_i)/f^S_{s,i}      —— 由所在卫星的 minimax 分配决定
    因为约束 C2 是按卫星分别写的、各卫星互不耦合，所以
        max_i λ_i = max( 本地用户的 L_i , 各卫星的 ε_s )

    这样"把某个用户从卫星 s 挪到 s'"只需重算 s 和 s' 两次二分，而不必把整个问题
    重解一遍 —— 这是 Algorithm 3 与 Algorithm 4 能跑快的关键。
    """

    def __init__(
        self,
        config: SimConfig,
        network: SatelliteNetwork,
        psi_unit: List[List[float]],    # 用于存储式28除a_i,s外常数部分计算结果，可见第一节 build_psi_matrix 函数
        assignment: Optional[Sequence[Optional[int]]] = None,   #各用户的接入卫星编号，本地卸载为None
    ):
        self.config = config
        self.network = network
        self.psi_unit = psi_unit
        self.num_satellites = len(network.satellites)
        self.num_users = len(network.user_tasks)
        self.budget = config.sat_compute_capacity_cps               # f_max，单位 cycle/s

        # L_i = T^l_i/D_i：本地计算的时延满意度；workload_i = O_i/D_i，单位 cycle/s
        self.local_lambda = [local_latency_s(t) / t.max_latency_s for t in network.user_tasks]
        self.workload = [t.required_cycles / t.max_latency_s for t in network.user_tasks]

        # assignment[i] = 用户 i 的接入卫星编号；None 表示本地计算
        self.assignment: List[Optional[int]] = (
            list(assignment) if assignment is not None else [None] * self.num_users
        )
        self._refresh_all()

    # ---------- 内部：重算 ----------
    def _psi_of(self, user_index: int, sat_id: Optional[int]) -> float:
        """用户 i 在指定接入方案下的 ψ_i（式 (28)）。"""
        if sat_id is None:
            return self.local_lambda[user_index]
        return self.local_lambda[user_index] + self.psi_unit[sat_id][user_index]

    def _refresh_all(self) -> None:
        """根据当前 assignment 重算各卫星的服务用户集合、ε_s 和 f^S。
           即根据用户选择卫星情况A，重新计算卫星计算资源分配f，属于算法2的内容
        """
        self.served: List[List[int]] = [[] for _ in range(self.num_satellites)] # 各卫星服务的小区列表
        self.psi_value = [0.0] * self.num_users # 保存原文式28中的ψ_i（原文是ψ_i,s，但应该不正确，对s求和后角标应该只有i）
        for user_index, sat_id in enumerate(self.assignment):
            self.psi_value[user_index] = self._psi_of(user_index, sat_id)
            if sat_id is not None:
                self.served[sat_id].append(user_index)
        self.epsilon = [0.0] * self.num_satellites
        self.allocation_per_sat: List[Dict[int, float]] = [{} for _ in range(self.num_satellites)]
        for sat_id in range(self.num_satellites):   # ！！！求解各卫星的ε和卫星计算资源分配；一个卫星的ε结果代表关联此卫星的各用户maxλ，所有卫星的ε结果最终可以代表全部用户的maxλ
            self.epsilon[sat_id], self.allocation_per_sat[sat_id] = _minimax_allocation_on_satellite(
                self.served[sat_id], self.psi_value, self.workload, self.budget
            )

    # ---------- 对外：目标值与增量评估 ----------
    def objective(self) -> float:
        """当前方案的 max_i λ_i，分别获取本地用户与卸载用户 max_i λ_i 再取其中较大值
           需要注意对于卸载用户此前已经根据其卸载卫星计算过各卫星的 λ_i，故取所有卫星中max λ_i也就是所有卸载用户中max λ_i
        """
        local_max = max(
            (self.local_lambda[i] for i, s in enumerate(self.assignment) if s is None), default=0.0
        )
        sat_max = max(
            (self.epsilon[s] for s in range(self.num_satellites) if self.served[s]), default=0.0
        )
        return max(local_max, sat_max)

    def user_lambdas(self) -> List[float]:
        """返回每个用户当前的 λ_i，供挑选"最差用户"使用。"""
        values: List[float] = []
        for user_index, sat_id in enumerate(self.assignment):
            if sat_id is None:
                values.append(self.local_lambda[user_index])
                continue
            allocated = self.allocation_per_sat[sat_id].get(user_index, 0.0)
            if allocated > 0.0:
                values.append(self.psi_value[user_index] + self.workload[user_index] / allocated)
            else:
                values.append(float("inf"))
        return values

    def evaluate_move(self, user_index: int, target: Optional[int]) -> float:
        """假设把用户 i 改判给 target（None=本地），返回改判后的 max_i λ_i；不改变状态。"""
        source = self.assignment[user_index]
        if source == target:
            return self.objective()
        if target is not None and not math.isfinite(self.psi_unit[target][user_index]):
            return float("inf")                       # 目标卫星根本没照明这个小区

        # 本地用户的最大 λ（把 user_index 的归属变化考虑进去）
        local_max = max(
            (self.local_lambda[i] for i, s in enumerate(self.assignment)
             if s is None and i != user_index),
            default=0.0,
        )
        if target is None:
            local_max = max(local_max, self.local_lambda[user_index])

        # 只有 source 和 target 两颗卫星的 ε 会变，其余直接沿用缓存值
        trial_psi = list(self.psi_value)
        trial_psi[user_index] = self._psi_of(user_index, target)
        sat_max = 0.0
        for sat_id in range(self.num_satellites):
            if sat_id == source:
                users = [i for i in self.served[sat_id] if i != user_index]
                if users:
                    value, _ = _minimax_allocation_on_satellite(
                        users, trial_psi, self.workload, self.budget)
                    sat_max = max(sat_max, value)
            elif sat_id == target:
                users = self.served[sat_id] + [user_index]
                value, _ = _minimax_allocation_on_satellite(
                    users, trial_psi, self.workload, self.budget)
                sat_max = max(sat_max, value)
            elif self.served[sat_id]:
                sat_max = max(sat_max, self.epsilon[sat_id])
        return max(local_max, sat_max)

    def apply_move(self, user_index: int, target: Optional[int]) -> None:
        """真正执行改判，并刷新受影响卫星的资源分配。"""
        if self.assignment[user_index] == target:
            return
        self.assignment[user_index] = target
        self._refresh_all()

    # ---------- 对外：导出论文变量 ----------
    def access_matrix(self) -> AccessMatrix:
        """导出 0-1 关联矩阵 A = [a^s_i]。"""
        matrix: AccessMatrix = [[0] * self.num_users for _ in range(self.num_satellites)]
        for user_index, sat_id in enumerate(self.assignment):
            if sat_id is not None:
                matrix[sat_id][user_index] = 1
        return matrix

    def allocation_matrix(self) -> ResourceMatrix:
        """导出计算资源分配 f^S_{s,i}，单位 cycle/s。"""
        matrix: ResourceMatrix = [[0.0] * self.num_users for _ in range(self.num_satellites)]
        for sat_id in range(self.num_satellites):
            for user_index, value in self.allocation_per_sat[sat_id].items():
                matrix[sat_id][user_index] = value
        return matrix

    def forecast_matrix(self) -> List[List[float]]:
        """预测矩阵 forecast[s][i]：用户 i 若接入卫星 s 能分到多少算力（cycle/s）。

        为什么需要它
        ------------
        论文 Algorithm 1 的 ζ_{i,s} 里含 O_i/f^S_{s,i}，而 f^S 来自上一步的
        Algorithm 2。但 Algorithm 2 只会给"已经关联到 s 的用户"分配算力，其余
        (s,i) 对的 f^S = 0 → ζ = +∞ → 用户永远无法切换到那颗卫星，BCD 会被初始
        关联锁死（这正是原始代码里"所有用户挤在同一颗星上、结果还不如本地计算"
        的根本原因）。

        正确做法是让 ζ 回答"如果我加入这颗星，我能分到多少"：把用户 i 临时放进
        U_s 重做一次 minimax 分配，取它分到的那一份（只需要将 ）。这样 Algorithm 1 就成为严格
        的最优响应步骤，BCD 才能在各卫星之间真正做负载均衡。

        目前是对 策略确定卸载至卫星s的用户+单个假设卸载至卫星s的用户 重新计算资源分配
        这样会导致这些假设用户过于乐观，结果有多个用户选择接入，但实际分配时却远远不够，导致新策略被拒
        """
        forecast = [[0.0] * self.num_users for _ in range(self.num_satellites)]
        for sat_id in range(self.num_satellites):
            base_users = self.served[sat_id]
            for user_index in range(self.num_users):
                if not math.isfinite(self.psi_unit[sat_id][user_index]):
                    # psi_unit为式28右边求和项除了a_i,s的部分，若其值为inf表示C^BH_i,s为0，即该星整个BHTP没照明这个小区，
                    # 用户i也不可能卸载此卫星，无需模拟分配计算资源
                    continue                                   
                if user_index in base_users:
                    # 若此用户在A中确定卸载至卫星，则此前计算过资源分配。直接使用无需预测
                    forecast[sat_id][user_index] = self.allocation_per_sat[sat_id].get(user_index, 0.0)
                    continue
                # 现在计算的用户，在BHTP中曾被卫星s照射，但并未选择卸载至此卫星
                # 原本计算此用户ψ时，psi_of输入为(i,None),此时输入卫星s即假设其卸载至卫星s，重新获取ψ
                trial_psi = list(self.psi_value)
                trial_psi[user_index] = self._psi_of(user_index, sat_id)
                # 输入“策略确定卸载至卫星s的用户+此假设卸载至卫星s的用户”，简化运算，计算此用户的预测f
                _, per_user = _minimax_allocation_on_satellite(
                    base_users + [user_index], trial_psi, self.workload, self.budget
                )
                forecast[sat_id][user_index] = per_user.get(user_index, 0.0)
        return forecast


# =====================================================================
# 第 3 节：Algorithm 1 —— 计算卸载子问题 P2 / P̃2
# =====================================================================
def solve_offloading(
    config: SimConfig,
    network: SatelliteNetwork,
    zeta_matrix: List[List[float]],
) -> Dict[str, object]:
    """Algorithm 1：固定 f^S，求解 P̃2（式 (26)）。

        min_A γ
        s.t. O_i/(D_i f^l_i) + Σ_s a^s_i ζ_{i,s} ≤ γ     (26a)
             γ ≤ 1                                        (26b) 即 C4
             Σ_s a^s_i ≤ 1, 0 ≤ a^s_i ≤ 1                 (26c) 即 C5 + 松弛
    """
    if config.solver_backend == "cvxpy":
        return _solve_offloading_cvxpy(config, network, zeta_matrix)
    return _solve_offloading_fast(config, network, zeta_matrix)


def _solve_offloading_fast(
    config: SimConfig,
    network: SatelliteNetwork,
    zeta_matrix: List[List[float]],
) -> Dict[str, object]:
    """P̃2 的闭式最优解。

    推导：P̃2 的约束按用户 i 完全解耦（只有 γ 把它们连起来），
    而 γ* = max_i ( L_i + Σ_s a^s_i ζ_{i,s} )。要让最大值最小，
    只需让每个用户各自的表达式最小：

        min_{a_i} Σ_s a^s_i ζ_{i,s}    s.t. Σ_s a^s_i ≤ 1, a^s_i ≥ 0

    这是单纯形上的线性规划，最优解必在顶点取到：
        若 min_s ζ_{i,s} < 0 → a^{s*}_i = 1（s* = argmin_s ζ），否则整行为 0（本地计算）。

    结论：P̃2 的松弛最优解天然就是 0-1 的，论文 Algorithm 1 的二分法只是在数值上
    逼近同一个 γ*，两者结果一致 —— 这也解释了为什么 Algorithm 3 第 7-17 步的
    "0-1 恢复"在实现上几乎不损失性能。
    """
    num_satellites = len(network.satellites)
    num_users = len(network.user_tasks)
    association: RelaxedAccessMatrix = [[0.0] * num_users for _ in range(num_satellites)]
    gamma = 0.0 # γ
    for user_index, task in enumerate(network.user_tasks):
        local_term = local_latency_s(task) / task.max_latency_s          # L_i
        best_sat, best_zeta = None, 0.0 # ζ
        for sat_id in range(num_satellites):    # 对每个用户遍历所有卫星，选择最小ζ的卫星
            value = zeta_matrix[sat_id][user_index]
            if math.isfinite(value) and value < best_zeta:
                best_sat, best_zeta = sat_id, value
        if best_sat is not None:
            association[best_sat][user_index] = 1.0
        gamma = max(gamma, local_term + best_zeta)
    return {"association_relaxed": association, "gamma": float(gamma), "status": "closed_form"}


def _solve_offloading_cvxpy(
    config: SimConfig,
    network: SatelliteNetwork,
    zeta_matrix: List[List[float]],
) -> Dict[str, object]:
    """用 CVXPY 严格建模 P̃2（论文 Algorithm 1 第 4 行 "Solve P̃2 with CVX"）。"""
    import cvxpy as cp
    import numpy as np

    num_satellites = len(network.satellites)
    num_users = len(network.user_tasks)

    # ζ 里的 inf 表示"该卫星无法服务该用户"，先替换成 0 再用掩码强制对应 a = 0
    zeta_array = np.array(zeta_matrix, dtype=float)
    feasible_mask = np.isfinite(zeta_array)
    zeta_clean = np.where(feasible_mask, zeta_array, 0.0)
    local_terms = np.array([local_latency_s(t) / t.max_latency_s for t in network.user_tasks])

    association = cp.Variable((num_satellites, num_users), nonneg=True)   # a^s_i
    gamma = cp.Variable(nonneg=True)                                      # γ
    upper_bound = gamma <= 1                                              # (26b)
    constraints = [
        association <= 1,                                                 # 式 (22b)
        cp.sum(association, axis=0) <= 1,                                 # C5
        local_terms + cp.sum(cp.multiply(zeta_clean, association), axis=0) <= gamma,   # (26a)
        upper_bound,
    ]
    if (~feasible_mask).any():
        constraints.append(association[~feasible_mask] == 0)
    problem = cp.Problem(cp.Minimize(gamma), constraints)
    problem = _run_cvxpy(problem, upper_bound)
    return {
        "association_relaxed": association.value.tolist(),
        "gamma": float(gamma.value),
        "status": problem.status,
    }


# =====================================================================
# 第 4 节：Algorithm 2 —— 计算资源分配子问题 P3 / P̃3
# =====================================================================
def solve_resource_allocation(
    config: SimConfig,
    network: SatelliteNetwork,
    association: Sequence[Sequence[float]],
    psi_unit: List[List[float]],
) -> Dict[str, object]:
    """Algorithm 2：固定 A，求解 P̃3（式 (30)）。

        min_{f^S} ε
        s.t. ψ_{i,s} + Σ_s a^s_i O_i/(D_i f^S_{s,i}) ≤ ε      (30a)
             Σ_i f^S_{s,i} ≤ f_max                             C2
             f^S_{s,i} ≥ 0,  ε ≤ 1                             C3 + (30b)
    """
    if config.solver_backend == "cvxpy":
        return _solve_resource_allocation_cvxpy(config, network, association, psi_unit)
    return _solve_resource_allocation_fast(config, network, association, psi_unit)


def _assignment_from_matrix(
    association: Sequence[Sequence[float]], num_users: int
) -> List[Optional[int]]:
    """把（可能是分数的）关联矩阵转成"每个用户的接入卫星"列表。

    BCD 中 P̃2 的解天然是 0-1 的（见 _solve_offloading_fast 的推导），
    所以这里几乎总是精确转换；若外部传入分数解，则取权重最大的卫星作近似，
    这正是论文 Algorithm 3 第 10-12 行"找最大的 a^{s̃}_i 并置 1"的做法。
    """
    assignment: List[Optional[int]] = [None] * num_users
    for user_index in range(num_users):
        best_sat, best_weight = None, 1e-9
        for sat_id, row in enumerate(association):
            weight = float(row[user_index])
            if weight > best_weight:
                best_sat, best_weight = sat_id, weight
        assignment[user_index] = best_sat
    return assignment


def _solve_resource_allocation_fast(
    config: SimConfig,
    network: SatelliteNetwork,
    association: Sequence[Sequence[float]],
    psi_unit: List[List[float]],
) -> Dict[str, object]:
    """P̃3 的精确解：各卫星预算独立，逐星 minimax 分配再取最大 ε。

    全局最小 ε = max( 本地用户的 λ_i , 各卫星各自的最小 ε_s )，
    因为约束 C2 是按卫星分别写的，不同卫星之间没有耦合。
    """
    num_users = len(network.user_tasks)
    assignment = _assignment_from_matrix(association, num_users)
    # 关联到"没照明该小区"的卫星是不可行的（违反 X·a=1），按本地计算处理
    assignment = [
        sat if sat is not None and math.isfinite(psi_unit[sat][i]) else None
        for i, sat in enumerate(assignment)
    ]
    state = OffloadingState(config, network, psi_unit, assignment)
    return {
        "allocation": state.allocation_matrix(),
        "epsilon": state.objective(),
        "status": "bisection",
        "state": state,
    }


def _solve_resource_allocation_cvxpy(
    config: SimConfig,
    network: SatelliteNetwork,
    association: Sequence[Sequence[float]],
    psi_unit: List[List[float]],
) -> Dict[str, object]:
    """用 CVXPY 严格建模 P̃3（论文 Algorithm 2 第 4 行 "Solve P̃3 with CVX"）。

    优化变量取 q_{s,i} = f^S_{s,i}/1e9（单位 Gcycle/s）而不是 cycle/s，
    避免 1e9 量级造成数值病态；返回前再乘回 1e9 恢复论文变量。
    式 (30a) 中的 1/f 用 cp.inv_pos 表达，使问题被识别为凸（论文 Lemma 1）。
    """
    import cvxpy as cp
    import numpy as np

    num_satellites = len(network.satellites)
    num_users = len(network.user_tasks)

    association_array = np.array([[float(v) for v in row] for row in association])
    psi_array = np.array(psi_unit, dtype=float)
    feasible_mask = np.isfinite(psi_array)
    psi_clean = np.where(feasible_mask, psi_array, 0.0)
    effective_a = np.where(feasible_mask, association_array, 0.0)        # 不可服务的关联置 0

    local_terms = np.array([local_latency_s(t) / t.max_latency_s for t in network.user_tasks])
    workload = np.array([t.required_cycles / t.max_latency_s / 1e9 for t in network.user_tasks])

    resource_GHz = cp.Variable((num_satellites, num_users), nonneg=True)  # q_{s,i}
    epsilon = cp.Variable(nonneg=True)                                    # ε
    # ψ_{i,s}（式 (28)）：本地基准项 + 通信开销净增量
    psi_expression = local_terms + np.sum(effective_a * psi_clean, axis=0)
    # Σ_s a^s_i O_i/(D_i f^S_{s,i})：式 (30a) 的倒数项，用 inv_pos 保持凸性
    processing_expression = cp.sum(
        cp.multiply(effective_a * workload[None, :], cp.inv_pos(resource_GHz)), axis=0
    )
    upper_bound = epsilon <= 1                                            # (30b) 即 C4
    constraints = [
        cp.sum(resource_GHz, axis=1) <= config.sat_compute_capacity_GHz,  # C2
        resource_GHz >= 1e-9,                                             # C3（下限保证 inv_pos 有界）
        psi_expression + processing_expression <= epsilon,                # (30a)
        upper_bound,
    ]
    problem = cp.Problem(cp.Minimize(epsilon), constraints)
    problem = _run_cvxpy(problem, upper_bound)
    # 没被任何用户关联的 (s,i) 位置在论文里没有意义，清零以便与快速后端对齐
    allocation_array = resource_GHz.value * 1e9
    allocation_array = np.where(effective_a > 1e-9, allocation_array, 0.0)
    return {
        "allocation": allocation_array.tolist(),
        "epsilon": float(epsilon.value),
        "status": problem.status,
    }


def _run_cvxpy(problem, upper_bound_constraint=None):
    """调用 CVXPY，按 CLARABEL → SCS → SCIPY 的顺序回退求解。

    若因为 γ ≤ 1 / ε ≤ 1（也就是 C4）导致不可行，说明该场景下无论怎么调度都有
    用户超时。此时去掉这条上界再解一次，把"C4 被违反"留给上层 evaluate_latency
    的 constraint_C4_ok 字段报告，而不是直接中断整条曲线。
    返回真正被求解的 problem 对象（可能是去掉上界后的新对象）。
    """
    import cvxpy as cp

    installed = cp.installed_solvers()
    candidates = [problem]
    if upper_bound_constraint is not None:
        candidates.append(cp.Problem(
            problem.objective,
            [c for c in problem.constraints if c is not upper_bound_constraint],
        ))
    for candidate in candidates:
        for solver in ("CLARABEL", "SCS", "SCIPY"):
            if solver not in installed:
                continue
            try:
                candidate.solve(solver=solver, verbose=False)
            except Exception:                      # 某个后端数值失败就换下一个
                continue
            if candidate.status in ("optimal", "optimal_inaccurate"):
                return candidate
    raise RuntimeError(f"CVXPY 求解失败，最终状态：{problem.status}")


# =====================================================================
# 第 5 节：Algorithm 3 —— BCD 联合求解 + 0-1 关联恢复
# =====================================================================
def solve_joint_offloading_and_allocation(
    config: SimConfig,
    network: SatelliteNetwork,
    channel_model: ChannelModel,
    plan: BeamHoppingPlan,
) -> Dict[str, object]:
    """Algorithm 3：给定 BHTP，联合求解计算卸载 A 与计算资源分配 f^S。

    与论文 Algorithm 3 伪代码的对应关系：
        第 1-2 行   给定 BHTP，取 A 与 f^S 的可行初值
        第 3-6 行   BCD 交替：Algorithm 2 更新 f^S → Algorithm 1/逐用户最优响应 更新 A，直到收敛
        第 7-17 行  按卸载比例降序把松弛解恢复成 0-1 解，并逐个检查 C1~C6

    返回字典含 access_matrix / allocation / max_lambda / bcd_history 等。
    """
    num_users = len(network.user_tasks)
    capacity_table = channel_model.average_capacity_table(plan)      # C^BH_{s,i}，包括此BHTP下各卫星-小区对的BH平均容量
    psi_unit = build_psi_matrix(config, network, capacity_table)     # 式 (28) 的单位增量

    # ---- 第 2 行：A、f^S 的可行初值 ----
    # 取"全本地计算"（A = 0）作为初值：它必然满足 C1~C6，同时也是一个明确的性能
    # 基准 —— 后续每一步 BCD 只有比它更好才会被接受。
    state = OffloadingState(config, network, psi_unit, [None] * num_users)
    best_objective = state.objective()  # max λ_i，即优化目标，需要最小化此变量

    # ---- 第 3-6 行：BCD 交替 ----
    # bcd_history[0] 记录第 2 行给出的初值（全本地）的目标，之后每一项是一轮 BCD 迭代
    # 结束后的 max_i λ_i；论文图 2a 画的就是这条序列。
    bcd_history: List[float] = [best_objective]
    for _ in range(config.max_bcd_iterations):
        # 第 4 行：给定 A，用 Algorithm 2 更新 f^S。
        #   状态对象内部已按 minimax 把 f^S 分好；forecast_matrix 额外给出 ζ 所需的
        #   "若接入卫星 s 可获得的算力"。
        forecast = state.forecast_matrix()
        # 第 5 行：给定 f^S，用 Algorithm 1 更新 A
        zeta_matrix = build_zeta_matrix(config, network, capacity_table, forecast)
        offloading_result = solve_offloading(config, network, zeta_matrix)
        # 若 solve_offloading 函数使用cvxpy求解，则需 _assignment_from_matrix 对其离散化
        proposed = _assignment_from_matrix(offloading_result["association_relaxed"], num_users)
        # 按用户顺序依次排列其选择的卫星，若本地卸载或选择的卫星并未在BHTP中照射该小区，则为None
        proposed = [
            sat if sat is not None and math.isfinite(psi_unit[sat][i]) else None
            for i, sat in enumerate(proposed)
        ]

        # P̃2 是"所有用户同时做最优响应"，当大家都涌向同一颗卫星时会出现拥塞过冲。
        # 产生此现象原因为， forecast_matrix 函数计算预测资源分配时，是单独对每个卫星假设上星，分配量较多过于乐观，可能导致最终多卫星选择进而实际分配量较少
        # 因此加一道单调性保护：整体更新只有确实降低了真实目标才接受；否则退化为
        # 逐用户最优响应（仍属块坐标下降，只是把每个 a_i 视作一个块），保证 BCD 的
        # 目标序列单调不增，与论文图 2a 的收敛形态一致。
        candidate_state = OffloadingState(config, network, psi_unit, proposed)
        candidate_objective = candidate_state.objective()
        if candidate_objective < best_objective - config.tolerance:
            state, best_objective = candidate_state, candidate_objective
        else:   
            # 若新优化目标值收敛或恶化，则进行逐个尝试本地计算和其他卫星；如果没有任何动作能继续降低目标，立刻break
            # 逐用户最优响应 与 上方算法1整体更新用户关联卫星A 完全无关，是独立的优化步骤，即上述方法结果不接受时采用此方法
            # ！！！原文中 逐用户最优响应 并不在BCD中使用，仅在BCD结束后使用一次，此处加入为了修正 forecast_matrix 函数的乐观预测问题，也可以删除
            # _sequential_best_response 函数中会修改state，无需额外赋值
            improved = _sequential_best_response(config, state, plan, network)  
            # _sequential_best_response 函数中也会判断策略变动是否会超过阈值，若超过阈值则会更新state，并令improved等于新优化目标值
            # 若其内部第一次尝试改变策略就不超过阈值，表示原策略为局部最优，则state不会变动，且improved = best_objective。
            if improved >= best_objective - config.tolerance:   
                # 此时代表本轮BCD中Algorithm 1与逐用户最优响应均收敛，结束BCD，策略A与优化目标值λ取值为上一轮BCD结果
                bcd_history.append(best_objective)
                break                                   
            best_objective = improved                   # 若未收敛则记录最新优化目标值λ，继续BCD
        bcd_history.append(best_objective)

    # ---- 第 7-17 行：0-1 恢复 + 逐用户可行性检查 ----
    # 状态对象里的 assignment 本身已是 0-1 解（P̃2 的最优解天然整数），这里再做一次
    # 逐用户扫描，确认不存在"换一颗星或改本地就能更好"的违反情形，对应论文
    # 第 10-16 行"取最大的 a^{s̃}_i；若不满足约束就把 s̃ 从 S' 中去掉再试下一个"。
    # ！！！由于BCD中添加了 逐用户最优响应 故此处可能可以删除，若不删除且此处确实再次进行优化，则实际优化目标值与 bcd_history 最后的值不同
    _sequential_best_response(config, state, plan, network)

    access_matrix = state.access_matrix()
    allocation = state.allocation_matrix()
    latency_result = evaluate_latency(config, network, channel_model, access_matrix, allocation, plan)
    return {
        "access_matrix": access_matrix,
        "allocation": allocation,
        "assignment": list(state.assignment),
        "max_lambda": float(latency_result["max_lambda"]),
        "latency_result": latency_result,
        "bcd_history": bcd_history,
    }


def _sequential_best_response(
    config: SimConfig,
    state: OffloadingState,
    plan: BeamHoppingPlan,
    network: SatelliteNetwork,
) -> float:
    """逐用户最优响应扫描：反复改判当前 λ 最大的用户，直到无法再改进。

    对应论文 Algorithm 3 第 9-17 行 "until constraints C1−C6 hold" 那层循环：
      - 找出当前 λ 最大的用户（它决定了目标 max_i λ_i）；
      - 依次尝试：(a) 改为本地计算；(b) 改接到另一颗照明过它的卫星；
      - 取使 max_i λ_i 下降最多的动作；若没有任何动作能改进就停止。

    每次尝试都通过 OffloadingState.evaluate_move() 做增量评估（只重算受影响的两颗
    卫星），因此整轮扫描的代价很低。返回扫描结束后的目标值。
    """
    current = state.objective()
    for _ in range(config.max_repair_iterations):
        lambdas = state.user_lambdas()
        # 根据 lambdas[i] 比较大小，返回最大值对应index
        worst_user = max(range(len(lambdas)), key=lambda i: lambdas[i])
        cell_id = network.user_tasks[worst_user].cell_id

        # 可选择关联的卫星列表，即BHTP中至少照射小区一次的卫星；None表示本地计算
        actions: List[Optional[int]] = [None] + illuminating_satellites(plan, cell_id)
        best_action, best_value = None, current
        found = False
        for action in actions:
            # 新选择的卫星与原A中的关联卫星相同，跳过
            if action == state.assignment[worst_user]:
                continue
            # 若不同则计算更改卫星后，λ的变化，变化大于阈值则更新
            value = state.evaluate_move(worst_user, action)
            if value < best_value - config.tolerance:
                best_action, best_value, found = action, value, True
        if not found:
            break                                   # 未找到新动作，已是局部最优
        # 采用新动作/用户关联卫星矩阵A，并重算卫星资源分配f以及优化目标λ
        # ！！！从循环逻辑看，只要第一个卫星的动作/改变策略成功，state会根据 逐用户最优响应 逻辑改变策略
        # ！！！反之，仅当第一个卫星的动作/改变策略失败，state才会保持不变
        state.apply_move(worst_user, best_action)
        current = state.objective()
    return current


# =====================================================================
# 第 6 节：Algorithm 4 前半段 —— BHTP 设计
# =====================================================================
def design_beam_hopping_matching(
    config: SimConfig,
    network: SatelliteNetwork,
    channel_model: ChannelModel,
) -> BeamHoppingPlan:
    """Algorithm 4 第 2-8 行：多对多匹配生成初始 BHTP。

    论文第四节 B 的匹配模型：
        V = {v^s_i|v^s_i = s*N_I +i,i∈I,s ∈ S}   卫星-小区关联，v^s_i 表示"卫星 s 照明小区 i"
        J = {J_t|t = 0,1,...,N_w-1}     一个 BHP 内的 N_w 个时隙
        偏好 U^V_{i,s}(Ω) = C^t_i，U^J_t(Ω) = C^t_i   （式 (31)(32)） ,表示V/J在当前关联Ω下对J/V的偏好值
        约束 定义 1 第 3 条：每颗卫星在每个时隙最多匹配 N_K 个小区（即 C1）

    论文另外规定的两条"偏好置零"规则在这里实现为硬性规则：
      1) 若 v^s_i 已匹配 J_t，则 J_t 对任意 s'≠s 的 v^{s'}_i 偏好为 0
         （同一时隙重复照明同一小区不会提升容量，属于浪费波束）；
      2) 若卫星 s 在 J_t 已有 N_K 个小区，则 J_t 对该星其余小区 v 的偏好为 0（C1）。

    算法实现：延迟接受（deferred acceptance）。每个 v^s_i 沿偏好列表由高到低向时隙
    提案，时隙对同一颗卫星只保留偏好最高的 N_K 个，被拒者继续向下一个时隙提案，
    直到所有偏好列表用尽 —— 也就是论文所说的
    "each satellite has explicitly designated the N_K cells for each time slot"。
    """
    num_slots = config.num_slots_per_BHP            # N_w
    num_satellites = len(network.satellites)        # N_S
    num_cells = len(network.ground_cells)           # N_I
    beams = config.num_beams_per_satellite          # N_K

    # 表示每个时隙每颗卫星保留的 N_K 个小区，用于实现用户规则 2，即避免同一颗卫星在同一时隙照明超过 N_K 个小区
    holders: List[List[List[int]]] = [[[] for _ in range(num_satellites)] for _ in range(num_slots)]  
    # 表示每个时隙已被点亮的小区，用于实现用户规则1，即避免同一时隙重复照明同一小区
    occupied_cells: List[set] = [set() for _ in range(num_slots)]      

    def preference_value(sat_id: int, cell_id: int, slot_id: int) -> float:
        """U^V_{i,s}(J_t) = C^t_i：把小区 i 放进该时隙能拿到的容量，单位 bit/s。

        干扰源取"该时隙当前已保留的全部关联"，这就是论文第 6-7 行
        "Calculate U 并更新偏好列表"所描述的动态偏好。
        """
        active = [(s, c) for s in range(num_satellites) for c in holders[slot_id][s]]
        # active.append((sat_id, cell_id)) 貌似与下方“holders[slot_id][sat_id].append(cell_id)”重复了
        sinr = channel_model.compute_sinr_for_user(sat_id, cell_id, active)
        return channel_model._capacity_from_sinr(sinr)

    # ---- 第 2 行：构造每个 v^s_i 的初始偏好列表 ----
    # 静态几何下各时隙的基础容量相同，用 (卫星, 小区) 决定的轮转量做确定性打破并列，
    # 让不同小区分散到不同时隙，而不是所有 v 都挤向同一个 J_t。
    preference_lists: Dict[Tuple[int, int], List[int]] = {}
    for sat_id in range(num_satellites):
        for cell_id in range(num_cells):
            offset = (cell_id + sat_id * beams) % num_slots # 轮转量，保证不同卫星-小区的偏好起始时隙错开，sat_id * beams没有什么特别含义
            preference_lists[(sat_id, cell_id)] = [(offset + k) % num_slots for k in range(num_slots)] # 每个 v^s_i 的偏好时隙排序列表

    # ---- 第 3-8 行：延迟接受主循环 ----
    next_index = {key: 0 for key in preference_lists}     # 每个 用户-卫星对v 下一个要提案的时隙序号(初始均为偏好列表中第一个时隙)
    pending: List[Tuple[int, int]] = sorted(              # 对准容量高的关联优先提案，由于偏好容量仅与 用户-卫星对v 有关，与时隙无关，所以此处直接对所有 用户-卫星对v 容量排序，容量大的无论哪个时隙均优先挑选
        preference_lists.keys(),
        key=lambda key: channel_model.boresight_capacity[key],  # 按容量排序，列表内容为排序后的(卫星, 小区)元组
        reverse=True,
    )
    while pending:
        still_pending: List[Tuple[int, int]] = []
        for sat_id, cell_id in pending:
            index = next_index[(sat_id, cell_id)]         # 当前 用户-卫星对v 要提案的时隙在偏好列表中的索引
            if index >= num_slots:
                continue                                  # 偏好列表已用尽
            slot_id = preference_lists[(sat_id, cell_id)][index]    # 当前 用户-卫星对v 要提案的实际时隙
            next_index[(sat_id, cell_id)] = index + 1
            # 第 4 行：若该时隙这个小区已被别的卫星点亮，则拒绝并等待向下一个时隙提案
            if cell_id in occupied_cells[slot_id]:
                still_pending.append((sat_id, cell_id))
                continue
            # 第 5 行：记录当前时隙卫星照射用户、当前时隙已被照射用户，用于判断约束；
            holders[slot_id][sat_id].append(cell_id)
            occupied_cells[slot_id].add(cell_id)
            if len(holders[slot_id][sat_id]) > beams:
                # 第 6-7 行：按照当前时隙已经完成匹配情况，重算偏好并重排，令时隙保留偏好最高的 v；超过 N_K 就拒掉最差的那个
                ranked = sorted(
                    holders[slot_id][sat_id],
                    key=lambda c: preference_value(sat_id, c, slot_id),
                    reverse=True,
                )
                holders[slot_id][sat_id] = ranked[:beams]
                for rejected in ranked[beams:]:     # 超过 N_K 的小区被拒绝，等待向下一个时隙提案
                    occupied_cells[slot_id].discard(rejected)
                    still_pending.append((sat_id, rejected))
        pending = still_pending

    # ---- 兜底补齐：保证 C1 取等号（每星每槽正好 N_K 个波束）----
    for slot_id in range(num_slots):
        for sat_id in range(num_satellites):
            _fill_slot_to_beam_capacity(
                holders, occupied_cells, slot_id, sat_id, beams, num_cells,
                rank_key=lambda c, s=sat_id: channel_model.boresight_capacity[(s, c)],
            )

    plan = _holders_to_plan(holders, num_slots, num_satellites, num_cells)
    validate_beam_hopping_plan(config, network, plan)
    return plan


def design_polling_beam_hopping(
    config: SimConfig,
    network: SatelliteNetwork,
    channel_model: ChannelModel,
) -> BeamHoppingPlan:
    """对照算法 JBHCO-PB / JBHCO-PG 使用的轮询（polling）BHTP。

    论文第五节 B：
      "polling strategy is used for periodic BHTP design, in which each satellite
       illuminates the cells yielding the maximum capacity"（来自参考文献 [17]）

    实现：每颗卫星先把所有小区按"对准容量从大到小"排序，再沿这个有序表以 N_K 为
    步长循环轮询 —— 第 t 个时隙点亮表中接下来的 N_K 个小区。不同卫星错开起始位置，
    尽量避免同一时隙重复照明同一小区。
    与匹配方案的本质区别：轮询是固定节拍，既不看干扰，也不响应业务分布。
    """
    num_slots = config.num_slots_per_BHP
    num_satellites = len(network.satellites)
    num_cells = len(network.ground_cells)
    beams = config.num_beams_per_satellite

    holders: List[List[List[int]]] = [[[] for _ in range(num_satellites)] for _ in range(num_slots)]
    occupied_cells: List[set] = [set() for _ in range(num_slots)]

    for sat_id in range(num_satellites):
        ranked_cells = sorted(
            range(num_cells),
            key=lambda c: channel_model.boresight_capacity[(sat_id, c)],
            reverse=True,
        )
        cursor = (sat_id * beams) % num_cells                 # 各卫星错开起始位置
        for slot_id in range(num_slots):
            picked, scanned = 0, 0
            while picked < beams and scanned < num_cells:
                cell_id = ranked_cells[cursor % num_cells]
                cursor += 1
                scanned += 1
                if cell_id in occupied_cells[slot_id]:
                    continue                                  # 该槽已被别的卫星点亮，跳过
                holders[slot_id][sat_id].append(cell_id)
                occupied_cells[slot_id].add(cell_id)
                picked += 1
            # N_S·N_K > N_I 时同槽小区不够分，按 C1 要求允许冗余照明补满
            _fill_slot_to_beam_capacity(
                holders, occupied_cells, slot_id, sat_id, beams, num_cells,
                rank_key=lambda c, s=sat_id: channel_model.boresight_capacity[(s, c)],
            )

    plan = _holders_to_plan(holders, num_slots, num_satellites, num_cells)
    validate_beam_hopping_plan(config, network, plan)
    return plan


def _fill_slot_to_beam_capacity(
    holders: List[List[List[int]]],
    occupied_cells: List[set],
    slot_id: int,
    sat_id: int,
    beams: int,
    num_cells: int,
    rank_key,
) -> None:
    """把某个 (时隙, 卫星) 的波束补满到 N_K 个，保证约束 C1 取等号。

    补齐顺序分两轮：
      第 1 轮：优先挑该时隙尚未被任何卫星点亮的小区（不产生冗余照明）；
      第 2 轮：若第 1 轮不够，再允许挑已被别的卫星点亮的小区。

    为什么需要第 2 轮：C1 要求每颗卫星每个时隙必须点亮 N_K 个小区，当
    N_S·N_K > N_I 时（例如论文图 8 扫描到 N_S=7、N_K=4、N_I=25，
    7×4=28 > 25），同一时隙内不可能让所有波束都对准互不相同的小区，
    此时冗余照明是唯一可行解。论文的"同槽不重复照明"本来也只是匹配阶段的
    偏好置零规则（偏好为 0 而不是禁止），并非硬约束，所以这样处理与论文一致。
    """
    shortage = beams - len(holders[slot_id][sat_id])
    if shortage <= 0:
        return
    already = set(holders[slot_id][sat_id])
    # 当前时隙未被任何卫星照射小区为空闲小区，计算所有空闲小区与当前卫星容量并排序，挑选容量高的补全卫星波束照射
    free_cells = sorted(
        (c for c in range(num_cells) if c not in occupied_cells[slot_id]),
        key=rank_key, reverse=True,
    )
    for cell_id in free_cells[:shortage]:
        holders[slot_id][sat_id].append(cell_id)
        occupied_cells[slot_id].add(cell_id)
        already.add(cell_id)

    # 若第 1 轮不够，再允许挑已被别的卫星点亮的小区
    shortage = beams - len(holders[slot_id][sat_id])
    if shortage <= 0:
        return
    redundant_cells = sorted(
        (c for c in range(num_cells) if c not in already), key=rank_key, reverse=True
    )
    for cell_id in redundant_cells[:shortage]:
        holders[slot_id][sat_id].append(cell_id)
        occupied_cells[slot_id].add(cell_id)


def _holders_to_plan(
    holders: List[List[List[int]]],
    num_slots: int,
    num_satellites: int,
    num_cells: int,
) -> BeamHoppingPlan:
    """把 holders[t][s] = [小区列表] 转成论文式 (1) 的 0-1 张量 X^t_{s,i}。"""
    plan: BeamHoppingPlan = [
        [[0] * num_cells for _ in range(num_satellites)] for _ in range(num_slots)
    ]
    for slot_id in range(num_slots):
        for sat_id in range(num_satellites):
            for cell_id in holders[slot_id][sat_id]:
                plan[slot_id][sat_id][cell_id] = 1
    return plan


def validate_beam_hopping_plan(
    config: SimConfig,
    network: SatelliteNetwork,
    plan: BeamHoppingPlan,
) -> None:
    """校验 BHTP 的三条硬性要求，不满足直接抛异常，便于尽早发现建模错误。

      1) 张量维度必须是 [N_w][N_S][N_I]；
      2) C1：Σ_i X^t_{s,i} = N_K，每颗卫星每个时隙正好点亮 N_K 个小区；
      3) 每个小区在一个 BHP 内至少被照明一次，否则它连卸载机会都没有
         （论文第三节 "each ground cell must be served at least once within a single BHP"）。
    """
    expected = (config.num_slots_per_BHP, config.num_satellites, config.num_ground_cells)
    actual = (len(plan), len(plan[0]) if plan else 0, len(plan[0][0]) if plan and plan[0] else 0)
    if actual != expected:
        raise ValueError(f"BHTP 维度错误：期望 {expected}，实际 {actual}")

    for slot_id, slot_plan in enumerate(plan):
        for sat_id, satellite_row in enumerate(slot_plan):
            if sum(satellite_row) != network.satellites[sat_id].beam_capacity:
                raise ValueError(
                    f"BHTP 违反 C1：时隙 {slot_id}、卫星 {sat_id} 点亮了 {sum(satellite_row)} "
                    f"个小区，应为 {network.satellites[sat_id].beam_capacity}"
                )

    uncovered = [
        cell_id for cell_id in range(config.num_ground_cells)
        if not any(slot_plan[sat_id][cell_id]
                   for slot_plan in plan for sat_id in range(config.num_satellites))
    ]
    if uncovered:
        raise ValueError(f"BHTP 未在一个 BHP 内覆盖这些小区：{uncovered}")


# =====================================================================
# 第 7 节：Algorithm 4 后半段 —— swap 操作
# =====================================================================
def propose_swap(
    config: SimConfig,
    plan: BeamHoppingPlan,
    rng: random.Random,
) -> Optional[BeamHoppingPlan]:
    """Algorithm 4 第 12 行：随机构造一个 swap matching Ω^{t,t'}_{i,i'}（定义 2）。

    做法：随机挑一颗卫星 s 和它的两个时隙 J_t、J_t'，再从 V^s_t、V^s_t' 各随机挑一个
    已点亮的小区 i、i'，把它们的时隙对调：
        (v^s_i, J_t), (v^s_i', J_t')  →  (v^s_i, J_t'), (v^s_i', J_t)

    返回 None 表示这次随机抽样不构成可行交换（下面三种情况），调用方跳过即可：
      - i == i'：交换没有意义；
      - i' 已在 J_t 被同一颗卫星点亮（或 i 已在 J_t'）：交换后该星波束数会掉，违反 C1；
      - 交换后某个小区在整个 BHP 内一次都没被照明，失去卸载机会。（貌似不存在此情况）

    注意：论文定义 3 只要求 Ω^{t,t'}_{i,i'} 满足 C1~C6。"同一时隙不重复照明同一小区"
    属于初始匹配阶段的偏好置零规则，不是硬约束，所以这里不再限制 —— 否则可行交换
    比例会低到 swap 几乎无法进行。
    """
    num_slots = len(plan)
    num_satellites = len(plan[0])
    if num_slots < 2:
        return None

    sat_id = rng.randrange(num_satellites)
    slot_a, slot_b = rng.sample(range(num_slots), 2)
    # 卫星在这两个时隙照射小区的列表
    cells_a = [c for c, v in enumerate(plan[slot_a][sat_id]) if v]
    cells_b = [c for c, v in enumerate(plan[slot_b][sat_id]) if v]
    if not cells_a or not cells_b:  # 某时隙无照射小区，无法交换
        return None
    cell_a, cell_b = rng.choice(cells_a), rng.choice(cells_b)
    if cell_a == cell_b:    # 交换的小区相同，无法交换
        return None
    if plan[slot_a][sat_id][cell_b] or plan[slot_b][sat_id][cell_a]:    # 小区在交换后的时隙已被照射
        return None                                        # 会导致波束数下降，违反 C1

    candidate = copy.deepcopy(plan)
    candidate[slot_a][sat_id][cell_a] = 0
    candidate[slot_a][sat_id][cell_b] = 1
    candidate[slot_b][sat_id][cell_b] = 0
    candidate[slot_b][sat_id][cell_a] = 1

    # 初始plan满足覆盖前提下，不存在此种情况
    # for cell_id in (cell_a, cell_b):                       # 每个小区仍须至少被照明一次
    #     if not any(candidate[t][s][cell_id]
    #                for t in range(num_slots) for s in range(num_satellites)):
    #         return None
        
    return candidate


def swap_improve_beam_hopping(
    config: SimConfig,
    network: SatelliteNetwork,
    channel_model: ChannelModel,
    current_plan: BeamHoppingPlan,
    access_matrix: AccessMatrix,
    allocation: ResourceMatrix,
    current_objective: float,
) -> Optional[BeamHoppingPlan]:
    """兼容旧接口：连续尝试若干次 swap，返回第一个严格更优的 BHTP，没有则返回 None。

    新代码请直接用 proposed_algo.proposed_jbhco_mb()，那里完整实现了
    Algorithm 4 第 11-15 行的 N_max 次 swap 主循环并记录收敛历史。
    """
    rng = random.Random(config.random_seed + int(abs(current_objective) * 1e9) % 100000)
    for _ in range(config.max_swap_iterations):
        candidate = propose_swap(config, current_plan, rng)
        if candidate is None:
            continue
        result = solve_joint_offloading_and_allocation(config, network, channel_model, candidate)
        if result["max_lambda"] + config.tolerance < current_objective:
            return candidate
    return None
