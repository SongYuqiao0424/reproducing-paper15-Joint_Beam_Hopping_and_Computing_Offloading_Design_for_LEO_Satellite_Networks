"""论文 15《Joint Beam Hopping and Computing Offloading Design for LEO Satellite
Networks》的仿真配置与任务参数。

命名尽量与论文第三节（系统模型）和表 II（系统参数）保持一致。
每个核心变量第一次出现时，都在同一行注明：论文符号 / 物理含义 / 单位。

本文件只负责"参数"和"任务生成"，不含任何算法逻辑，方便各仿真图脚本
通过 build_config() 批量改参数做扫描实验。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ======================================================================
# 1. 单个用户任务的数据结构（论文式 (2)：R_i = {M_i, O_i, D_i}）
# ======================================================================
@dataclass
class UserTask:
    """一个地面小区聚合后的用户任务。

    论文假设"每个 ground cell 只有一个用户"（第三节脚注 1），
    因此 user_id 与 cell_id 一一对应。
    """

    user_id: int              # i：用户编号，用于索引关联变量 a_i^s。
    cell_id: int              # i：用户所在地面小区编号（与 user_id 相同）。
    x_km: float               # 用户地面投影的东向坐标，单位 km。
    y_km: float               # 用户地面投影的北向坐标，单位 km。
    data_size_Mbit: float     # M_i：任务输入数据量，单位 Mbit。
    cpu_cycles_per_bit: float # 处理 1 bit 数据所需 CPU 周期数，单位 cycle/bit。
    local_cpu_GHz: float      # f_i^l：用户本地计算能力，单位 Gcycle/s。
    max_latency_s: float      # D_i：任务最大容忍时延，单位 s。

    @property
    def required_cycles(self) -> float:
        """O_i：任务所需的总 CPU 周期数，单位 cycle。

        论文式 (10) 写作 T_i^l = O_i / f_i^l，而表 II 给的是"处理 1 bit 所需
        周期数 = 100 cycle"，所以 O_i = M_i(bit) x 100(cycle/bit)。
        用表 II 默认值核算：M_i=10 Mbit 时 O_i=1e9 cycle，f_i^l=1 GC/s，
        得 T^l=1 s，λ=T/D=1/6≈0.167，与论文图 4/图 7/图 9 中 CL 曲线
        恒为 0.167 完全吻合，可确认这一理解正确。
        """
        return self.data_size_Mbit * 1e6 * self.cpu_cycles_per_bit

    # ---- 下面两个别名仅为兼容旧绘图脚本中的 task.x / task.y ----
    @property
    def x(self) -> float:
        return self.x_km

    @property
    def y(self) -> float:
        return self.y_km


# ======================================================================
# 2. 全局仿真配置（默认值 = 论文表 II）
# ======================================================================
@dataclass
class SimConfig:
    """仿真全局配置，默认值全部按照论文表 II 设置。"""

    # ---------- 网络规模 ----------
    num_satellites: int = 5            # N_S：LEO 卫星数量（论文表 II：5）。
    num_ground_cells: int = 25         # N_I：地面小区数量（论文表 II：25）。
    num_users: int = 25                # 用户数 = 小区数（一小区一用户）。
    num_beams_per_satellite: int = 4   # N_K：每星每时隙可同时照明的小区数（表 II：4）。

    # ---------- 波束跳跃时间结构 ----------
    slot_duration_s: float = 0.002     # Δt：单个 BH 时隙长度，单位 s（表 II：2 ms）。
    num_slots_per_BHP: int = 8         # N_w：一个 BHP 内的时隙数（表 II：8）。
    beam_hopping_period_s: float = 0.016  # T_BH = N_w · Δt，单位 s（由上两项推出）。

    # ---------- 几何参数 ----------
    satellite_altitude_km: float = 1200.0  # h：卫星轨道高度，单位 km（表 II：1200）。
    earth_radius_km: float = 6371.0        # R_E：地球平均半径，单位 km。
    cell_spacing_km: Optional[float] = None   # 相邻小区中心间距，单位 km；None 表示按 3 dB 波束角自动推算。
    satellite_spread_radius_km: float = 300.0 # 各卫星星下点所在圆的半径，表征卫星分布范围，单位 km（论文未给轨道，取紧凑子网络尺度）。

    # ---------- 链路与天线参数 ----------
    carrier_frequency_GHz: float = 30.0    # f_c：Ka 波段载频，单位 GHz（表 II：30）。
    system_bandwidth_GHz: float = 0.4      # B：系统带宽，单位 GHz（表 II：0.4）。
    noise_density_dBm_per_Hz: float = -174.0  # σ²：噪声谱密度，单位 dBm/Hz（表 II：-174）。
    shadowing_std_dB: float = 0.0          # σ_SF：阴影衰落标准差，单位 dB；论文未给数值，默认关闭。
    rain_attenuation_dB: float = 0.0       # 雨衰，单位 dB；可按 ITU-R P.618 结果替换。
    beamwidth_3dB_deg: float = 1.7647      # 3 dB 波束角，单位 degree（表 II）；仅作记录，式 (5) 用孔径半径。
    antenna_radius_m: float = 0.165        # a：卫星接收天线圆孔径半径，单位 m（表 II：0.165）。
    sat_rx_gain_dBi: float = 38.5          # G_rx^max：卫星最大接收增益，单位 dBi（表 II：38.5）。
    user_tx_gain_dBi: float = 43.2         # G^tx：用户发射天线增益，单位 dBi（表 II：43.2）。
    user_tx_power_W: float = 2.0           # P_u：用户发射功率，单位 W（表 II：2）。
    sidelobe_floor_dBi: float = -10.0      # 旁瓣增益下限，单位 dBi；抑制贝塞尔方向图零点造成的"干扰完全消失"数值假象。

    # ---------- 计算能力与任务参数 ----------
    sat_compute_capacity_GHz: float = 5.0  # f_max：每颗卫星计算能力，单位 Gcycle/s（表 II：5）。
    user_local_compute_GHz: float = 1.0    # f_i^l：用户本地计算能力，单位 Gcycle/s（表 II：1）。
    cpu_cycles_per_bit: float = 100.0      # 处理 1 bit 所需 CPU 周期，单位 cycle/bit（表 II：100）。
    task_data_size_range_Mbit: Tuple[float, float] = (5.0, 10.0)  # M_i 取值范围，单位 Mbit（表 II：5~10）。
    max_latency_s: float = 6.0             # D_i：任务最大容忍时延，单位 s（第五节：6 s）。

    # ---------- 算法超参数 ----------
    max_bcd_iterations: int = 8        # Algorithm 3 的 BCD 外层上限；论文图 2a 显示 8 次内收敛。
    max_swap_iterations: int = 100     # N_max：Algorithm 4 的 swap 次数；论文图 2b 取 100。
    max_repair_iterations: int = 60    # Algorithm 3 步骤 7-17 的可行性修复上限（见 solvers 说明）。
    tolerance: float = 1e-6            # ε_A / ε̃：收敛判据。
    solver_backend: str = "cvxpy"      # "cvxpy"=用 CVXPY 严格建模凸子问题；"fast"=用等价的精确二分解析解。
    random_seed: int = 7               # 随机种子，保证任务生成 / swap 随机过程可复现。

    # ---------- 由 __post_init__ 生成的派生数据 ----------
    user_positions: List[Tuple[float, float]] = field(default_factory=list)  # 小区中心坐标缓存，单位 km。
    user_tasks: List[UserTask] = field(default_factory=list)                 # 任务集合 {R_i}。

    # ------------------------------------------------------------------
    # 在初始化SimConfig后自动调用此函数
    def __post_init__(self) -> None:
        """参数自洽性处理 + 生成默认拓扑坐标和任务。"""
        # T_BH 与 Δt、N_w 必须自洽：以 Δt 和 N_w 为准重算 T_BH。
        self.num_slots_per_BHP = max(1, int(self.num_slots_per_BHP))
        self.beam_hopping_period_s = self.num_slots_per_BHP * self.slot_duration_s
        # 论文采用"一小区一用户"模型，强制任务数量和小区数量一致。
        self.num_users = self.num_ground_cells
        if self.solver_backend not in ("cvxpy", "fast"):
            raise ValueError("solver_backend 只能是 'cvxpy' 或 'fast'")
        if not self.user_positions:
            self.user_positions = self._generate_user_positions()
        if not self.user_tasks:
            self.user_tasks = self._generate_user_tasks()

    # ---------- 单位换算属性 ----------
    @property
    def system_bandwidth_Hz(self) -> float:
        """B：系统带宽的 Hz 表示，供香农公式式 (8) 使用。"""
        return self.system_bandwidth_GHz * 1e9

    @property
    def carrier_frequency_Hz(self) -> float:
        """f_c：载频的 Hz 表示，供 FSPL 式 (4) 和波数 κ 使用。"""
        return self.carrier_frequency_GHz * 1e9

    @property
    def sat_compute_capacity_cps(self) -> float:
        """f_max：卫星计算能力的 cycle/s 表示。"""
        return self.sat_compute_capacity_GHz * 1e9

    # ---------- 拓扑与任务生成 ----------
    @property
    def effective_cell_spacing_km(self) -> float:
        """相邻小区中心间距，单位 km。

        论文没有直接给地面小区尺寸，但给了 3 dB 波束角 θ_3dB = 1.7647° 和
        卫星高度 h = 1200 km，而论文明确说"one beam precisely illuminates a
        ground cell"（一个波束正好照亮一个小区）。因此小区尺寸就应等于波束
        在地面的 3 dB 足印直径：

            spacing = 2 · h · tan(θ_3dB / 2) ≈ 2 × 1200 × tan(0.882°) ≈ 37 km

        这个尺度非常关键：只有让小区间距与波束足印一致，相邻小区之间的离轴角
        才会落在天线方向图的主瓣/近旁瓣区域，式 (6) 的同频干扰才有量级可言，
        BHTP 设计（哪些小区同时被点亮）才会真正影响容量。若把小区摊开到几百
        公里，离轴角远超主瓣，干扰会衰减到可以忽略，BHTP 就退化成无关紧要的变量。
        """
        if self.cell_spacing_km is not None:
            return self.cell_spacing_km
        half_beamwidth_rad = math.radians(self.beamwidth_3dB_deg) / 2.0
        return 2.0 * self.satellite_altitude_km * math.tan(half_beamwidth_rad)

    @property
    def coverage_radius_km(self) -> float:
        """地面仿真区域的半边长，单位 km；由小区间距和小区数量推出。"""
        side = math.ceil(math.sqrt(max(self.num_ground_cells, 1)))
        return 0.5 * side * self.effective_cell_spacing_km

    def _generate_user_positions(self) -> List[Tuple[float, float]]:
        """在方形地面区域内生成规则排布的小区中心位置，单位 km。

        论文只说有 N_I 个地面小区，没给具体经纬度，这里用正方形网格作规则抽象：
        先算一边放几个小区，再按"行、列"逐个排布，间距取波束足印直径（见
        effective_cell_spacing_km）。
        """
        side = math.ceil(math.sqrt(self.num_ground_cells))   # 网格一边的小区个数
        spacing = self.effective_cell_spacing_km             # 相邻小区中心间距，单位 km
        half_extent = 0.5 * side * spacing
        positions: List[Tuple[float, float]] = []
        for cell_id in range(self.num_ground_cells):
            row, column = divmod(cell_id, side)              # 把一维编号映射到二维网格
            x_km = -half_extent + (column + 0.5) * spacing
            y_km = -half_extent + (row + 0.5) * spacing
            positions.append((x_km, y_km))
        return positions

    def _generate_user_tasks(self) -> List[UserTask]:
        """生成与地面小区一一对应的任务 R_i = {M_i, O_i, D_i}。

        M_i 在 task_data_size_range_Mbit 上均匀采样（论文表 II：5~10 Mbit）。
        用固定 random_seed，保证同一 seed 下不同算法看到完全相同的任务集合，
        这样四条曲线之间的比较才是公平的。

        在一个BHTP中，每个用户仅有一个任务，根据BHTP平均容量得到处理时延
        """
        low, high = self.task_data_size_range_Mbit
        rng = random.Random(self.random_seed)
        tasks: List[UserTask] = []
        for user_id, (x_km, y_km) in enumerate(self.user_positions):
            data_size = rng.uniform(low, high)               # M_i，单位 Mbit
            tasks.append(
                UserTask(
                    user_id=user_id,
                    cell_id=user_id,
                    x_km=x_km,
                    y_km=y_km,
                    data_size_Mbit=data_size,
                    cpu_cycles_per_bit=self.cpu_cycles_per_bit,
                    local_cpu_GHz=self.user_local_compute_GHz,
                    max_latency_s=self.max_latency_s,
                )
            )
        return tasks


# ======================================================================
# 3. 供各仿真图脚本使用的配置工厂
# ======================================================================
def build_config(**overrides: Any) -> SimConfig:
    """按关键字参数覆盖表 II 默认值，生成一个新的 SimConfig。

    用法示例（论文图 4：扫描卫星计算能力 f_max）：
        cfg = build_config(sat_compute_capacity_GHz=7.0)

    注意：user_positions / user_tasks 故意不允许由外部传入，一律在
    __post_init__ 中按最新的 num_ground_cells、random_seed 重新生成，
    避免扫描 N_I 时坐标和任务数量对不上。
    """
    invalid = {"user_positions", "user_tasks"}
    if invalid & set(overrides):
        raise ValueError("user_positions / user_tasks 由配置内部生成，请勿直接传入")
    return SimConfig(**overrides)


def describe_config(config: SimConfig) -> Dict[str, Any]:
    """返回一个便于写入 JSON 结果文件的参数字典（只挑关键项）。"""
    return {
        "num_satellites_NS": config.num_satellites,
        "num_ground_cells_NI": config.num_ground_cells,
        "num_beams_NK": config.num_beams_per_satellite,
        "num_slots_Nw": config.num_slots_per_BHP,
        "slot_duration_s": config.slot_duration_s,
        "beam_hopping_period_s": config.beam_hopping_period_s,
        "system_bandwidth_GHz": config.system_bandwidth_GHz,
        "sat_compute_capacity_GHz": config.sat_compute_capacity_GHz,
        "user_local_compute_GHz": config.user_local_compute_GHz,
        "cpu_cycles_per_bit": config.cpu_cycles_per_bit,
        "task_data_size_range_Mbit": list(config.task_data_size_range_Mbit),
        "max_latency_s": config.max_latency_s,
        "max_swap_iterations_Nmax": config.max_swap_iterations,
        "solver_backend": config.solver_backend,
        "random_seed": config.random_seed,
    }
