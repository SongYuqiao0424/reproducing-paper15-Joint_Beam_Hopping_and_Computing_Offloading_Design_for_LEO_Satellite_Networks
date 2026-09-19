"""论文 15 的卫星-地面上行信道与容量模型。

实现的物理链路关系（对应论文式 (3)~(9)）：

    d_i,s  --式(4)-->  FSPL
                       + 阴影衰落 SF + 雨衰            = PL_{s,i}
    θ      --式(5)-->  卫星接收天线增益 G^rx_{s,i}
    式(3)              h_{s,i} = G^tx · PL_{s,i} · G^rx_{s,i}
    式(6)              SINR_i^t = X^t_{s,i} P_u h_{s,i} / (σ² + Σ_{j≠i} X^t_{s,j} P_u h_{s,j})
    式(8)              C_i^t = B log2(1 + SINR_i^t)
    式(9)              C_i^BH = (1/T_BH) Σ_t C_i^t Δt

其中 BHTP 变量 X^t_{s,i} 决定"第 t 个时隙卫星 s 有没有照明小区 i"，
所以信道容量是随 BHTP 变化的 —— 这正是论文里 BHTP 设计与卸载决策耦合的根源。

与原始版本相比修正的三处建模问题（详见各函数内注释）：
  (1) 式 (5) 的 θ 是"用户相对其服务波束中心"的离轴角，服务链路应取 θ=0
      的最大增益 G_rx^max，而不是用户相对星下点的角度；
  (2) 式 (6) 分母的同频干扰来自同一卫星上其他波束里的用户 j，
      应该用"干扰用户 j 的路径损耗 × j 相对受害波束 i 中心的离轴增益"，
      而不是受害用户自己的路径损耗；
  (3) 离轴角用卫星处两条视线方向的三维夹角计算，而不是地面横向距离的平面近似。
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from config import SimConfig
from satellite_network import SatelliteNetwork

# 类型别名：BHTP 张量 X，维度 [N_w][N_S][N_I]，分别为跳波束周期时隙数、卫星数、小区数，元素取 0/1。
BeamHoppingPlan = List[List[List[int]]]


class ChannelModel:
    """静态几何下的多卫星多波束上行信道模型。"""

    SPEED_OF_LIGHT_M_S = 299792458.0  # c：光速，单位 m/s。

    # ==================================================================
    # 0. 初始化
    # ==================================================================
    def __init__(self, config: SimConfig, network: SatelliteNetwork):
        self.config = config
        self.network = network

        # ---- 基础链路量（与 BHTP 无关，只依赖几何，一次算好反复用）----
        self.path_loss_db: Dict[Tuple[int, int], float] = {}        # PL_{s,i}：链路总损耗，单位 dB。
        self.channel_gain_linear: Dict[Tuple[int, int], float] = {} # h_{s,i}（θ=0 波束对准时），无量纲。
        self.boresight_capacity: Dict[Tuple[int, int], float] = {}  # 单链路无干扰容量，单位 bit/s，用于偏好排序。

        # ---- 随 BHTP 变化的量 ----
        self.sinr_matrix: Dict[Tuple[int, int], float] = {}         # SINR_{s,i}：最近一次计算的信干噪比。
        self.capacity_matrix: Dict[Tuple[int, int], float] = {}     # C_{s,i}：最近一次计算的瞬时容量，bit/s。
        self.slot_capacity: Dict[Tuple[int, int, int], float] = {}  # C^t_{s,i}：时隙容量，bit/s。
        self.average_capacity: Dict[int, float] = {}                # C_i^BH：一个 BHP 内的平均容量，bit/s。

        # ---- 内部缓存：离轴增益和"某 BHTP 下的平均容量表"----
        # 离轴增益只依赖三元组 (卫星, 干扰用户所在小区, 波束中心小区)，
        # 在 swap 搜索中会被反复查询上万次，必须缓存，否则贝塞尔函数会成为瓶颈。
        self._beam_gain_cache: Dict[Tuple[int, int, int], float] = {}
        # plan_capacity_cache数据类型中Dict[Tuple[int, int], float]很好理解，是不同卫星-小区对的BHTP平均容量；
        # 前面的Tuple为BHTP（整体）形成的元组，可以直接输入BHTP获取此跳波束模式下的各小区-卫星对的平均容量，即可存储多种BHTP的情况
        self._plan_capacity_cache: Dict[Tuple, Dict[Tuple[int, int], float]] = {}   

        self._rng = random.Random(config.random_seed)
        self._noise_power_cache = self._noise_power_W(config.system_bandwidth_Hz)  # σ²·B，单位 W。
        self._initialize_link_statistics()

    def _initialize_link_statistics(self) -> None:
        """预计算所有卫星-小区对的路径损耗、对准增益和无干扰容量。"""
        for satellite in self.network.satellites:
            for cell in self.network.ground_cells:
                key = (satellite.sat_id, cell.cell_id)
                distance_m = self.network.sat_to_cell_distance[key] * 1000.0
                path_loss_db = self._calculate_path_loss_db(distance_m)
                self.path_loss_db[key] = path_loss_db
                # 服务链路：波束正对本小区，式 (5) 取 θ=0 分支 → G_rx^max。
                gain = self._link_gain(
                    path_loss_db=path_loss_db,
                    receive_gain_dBi=self.config.sat_rx_gain_dBi,
                )
                self.channel_gain_linear[key] = gain
                sinr_no_interference = self.config.user_tx_power_W * gain / self._noise_power_cache
                self.boresight_capacity[key] = self._capacity_from_sinr(sinr_no_interference)
                self.sinr_matrix[key] = 0.0
                self.capacity_matrix[key] = 0.0

    # ==================================================================
    # 1. 路径损耗与天线增益
    # ==================================================================
    def _calculate_path_loss_db(self, distance_m: float) -> float:
        """PL_{s,i}：FSPL(自由空间路径损耗) + 阴影衰落 + 雨衰，单位 dB。

        论文式 (4)：FSPL = (4π d_i,s f_c / c)²，下面写成它的 dB 形式。
        阴影衰落 SF ~ N(0, σ_SF²)（论文未给 σ_SF，config 默认 0）。
        雨衰按 ITU-R P.618 计算，这里作为常数参数留出接口。
        """
        fspl_db = 20.0 * math.log10(
            4.0 * math.pi * max(distance_m, 1.0)
            * self.config.carrier_frequency_Hz / self.SPEED_OF_LIGHT_M_S
        )
        shadowing_db = (
            self._rng.gauss(0.0, self.config.shadowing_std_dB)
            if self.config.shadowing_std_dB > 0.0 else 0.0
        )
        return fspl_db + shadowing_db + self.config.rain_attenuation_dB

    def _link_gain(self, path_loss_db: float, receive_gain_dBi: float) -> float:
        """式 (3)：h = G^tx · PL · G^rx，全部换成线性值相乘，无量纲。"""
        tx_gain_linear = 10.0 ** (self.config.user_tx_gain_dBi / 10.0)
        path_gain_linear = 10.0 ** (-path_loss_db / 10.0)
        rx_gain_linear = 10.0 ** (receive_gain_dBi / 10.0)
        return tx_gain_linear * path_gain_linear * rx_gain_linear

    def _off_axis_angle_rad(self, sat_id: int, user_cell_id: int, beam_center_cell_id: int) -> float:
        """θ：用户所在小区相对某个波束中心方向的离轴角，单位 rad。

        做法：取卫星指向"用户小区"和指向"波束中心小区"的两个单位方向向量，
        用点积求夹角。这样在用户和波束中心都远离星下点时依然准确，
        比"地面横向距离 / 卫星高度"的平面近似更可靠。
        """
        satellite = self.network.get_satellite_by_id(sat_id)
        user_cell = self.network.get_cell_by_id(user_cell_id)
        beam_cell = self.network.get_cell_by_id(beam_center_cell_id)
        if satellite is None or user_cell is None or beam_cell is None:
            return math.pi / 2.0
        ux, uy, uz = satellite.direction_to_cell(user_cell)
        bx, by, bz = satellite.direction_to_cell(beam_cell)
        cosine = max(-1.0, min(1.0, ux * bx + uy * by + uz * bz))
        return math.acos(cosine)

    def receive_antenna_gain_dBi(self, sat_id: int, user_cell_id: int, beam_center_cell_id: int) -> float:
        """论文式 (5)：卫星接收天线增益 G^rx_{s,i}，单位 dBi。

            θ = 0      →  G_rx^max
            0 < θ ≤ 90 →  4 G_rx^max |J1(κ a sinθ) / (κ a sinθ)|²
        其中 κ = 2π f_c / c 为波数，a 为天线圆孔径半径，J1 为第一类一阶贝塞尔函数。

        参数 beam_center_cell_id 就是"服务波束的 bore-sight 方向"落在哪个小区：
        - 计算期望信号时传入用户自己的小区 → θ=0 → 取最大增益；
        - 计算同频干扰时传入受害波束的中心小区 → θ>0 → 取旁瓣增益。
        """
        cache_key = (sat_id, user_cell_id, beam_center_cell_id)
        cached = self._beam_gain_cache.get(cache_key)
        if cached is not None:
            return cached
        if user_cell_id == beam_center_cell_id:
            gain_dBi = self.config.sat_rx_gain_dBi
        else:
            theta_rad = self._off_axis_angle_rad(sat_id, user_cell_id, beam_center_cell_id)
            wave_number = 2.0 * math.pi * self.config.carrier_frequency_Hz / self.SPEED_OF_LIGHT_M_S  # κ
            argument = wave_number * self.config.antenna_radius_m * math.sin(theta_rad)  # κ a sinθ
            if abs(argument) < 1e-12:
                pattern_ratio = 0.5          # lim_{x→0} J1(x)/x = 1/2
            else:
                pattern_ratio = self._bessel_j1(argument) / argument
            max_gain_linear = 10.0 ** (self.config.sat_rx_gain_dBi / 10.0)
            gain_linear = 4.0 * max_gain_linear * pattern_ratio * pattern_ratio
            gain_dBi = 10.0 * math.log10(max(gain_linear, 1e-20))
            # 贝塞尔方向图存在零点，理论上会让某些角度的干扰精确为 0。
            # 实际天线有旁瓣包络，这里加一个可配置下限避免数值上的"干扰凭空消失"。
            gain_dBi = max(gain_dBi, self.config.sidelobe_floor_dBi)
        self._beam_gain_cache[cache_key] = gain_dBi
        return gain_dBi

    @staticmethod
    def _bessel_j1(argument: float) -> float:
        """第一类一阶贝塞尔函数 J1(x)，用级数 + 大宗量渐近式实现。

        自己实现是为了不强制依赖 SciPy；|x|>20 时级数项会溢出，
        改用渐近式 J1(x) ≈ sqrt(2/(πx)) cos(x - 3π/4)，精度足够刻画旁瓣量级。
        """
        if abs(argument) > 20.0:
            magnitude = math.sqrt(2.0 / (math.pi * abs(argument))) * math.cos(
                abs(argument) - 3.0 * math.pi / 4.0
            )
            return magnitude if argument >= 0.0 else -magnitude
        term = argument / 2.0
        result = term
        factorial_m = 1.0
        factorial_m_plus_one = 1.0
        half_argument_squared = (argument / 2.0) ** 2
        for order in range(1, 40):
            factorial_m *= order
            factorial_m_plus_one *= order + 1
            term *= -half_argument_squared
            result += term / (factorial_m * factorial_m_plus_one)
            if abs(term) < 1e-18:
                break
        return result

    # ==================================================================
    # 2. 噪声、SINR 与容量
    # ==================================================================
    def _noise_power_W(self, bandwidth_Hz: float) -> float:
        """把噪声谱密度 σ²(dBm/Hz) 换算成带宽内的 AWGN 总功率，单位 W。"""
        noise_dBm = self.config.noise_density_dBm_per_Hz + 10.0 * math.log10(max(bandwidth_Hz, 1.0))
        return 10.0 ** ((noise_dBm - 30.0) / 10.0)

    def compute_sinr_for_user(
        self,
        sat_id: int,
        cell_id: int,
        active_links: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> float:
        """论文式 (6)：计算用户 i 在卫星 s 上的 SINR，无量纲。

        参数
        ----
        sat_id, cell_id : 受害链路，即"卫星 s 用一个波束对准小区 i 收这个用户"。
        active_links    : 当前时隙所有被照明的 (卫星, 小区) 对，也就是 X^t_{s,i}=1 的集合。

        干扰建模
        --------
        论文采用全频率复用，同一卫星的 N_K 个波束共用整个带宽 B，
        所以卫星 s 在收小区 i 的时候，会从自己其他波束里收到别的用户 j 的信号。
        干扰功率 = P_u × [G^tx · PL_{s,j} · G^rx(θ_{j→波束i})]
        注意这里用的是"干扰用户 j 的路径损耗"和"j 相对受害波束 i 中心的离轴角"。
        不同卫星是独立接收机，其他卫星收到的功率不会进入本卫星的分母。
        """
        desired_gain = self.channel_gain_linear[(sat_id, cell_id)]           # θ=0 的服务链路增益
        desired_power = self.config.user_tx_power_W * desired_gain
        interference_power = 0.0
        for other_sat_id, other_cell_id in (active_links or []):
            if other_sat_id != sat_id or other_cell_id == cell_id:
                continue
            interferer_gain = self._link_gain(
                path_loss_db=self.path_loss_db[(sat_id, other_cell_id)],     # 干扰用户 j 的 PL
                receive_gain_dBi=self.receive_antenna_gain_dBi(              # j 落在波束 i 旁瓣上的增益
                    sat_id, user_cell_id=other_cell_id, beam_center_cell_id=cell_id
                ),
            )
            interference_power += self.config.user_tx_power_W * interferer_gain
        return desired_power / max(self._noise_power_cache + interference_power, 1e-30)

    def _capacity_from_sinr(self, sinr: float) -> float:
        """论文式 (8)：C_i^t = B log2(1 + SINR)，单位 bit/s。

        论文采用全频率复用，每个波束都可以使用完整系统带宽 B。
        """
        return self.config.system_bandwidth_Hz * math.log2(1.0 + max(sinr, 0.0))

    # ==================================================================
    # 3. 与 BHTP 相关的容量计算
    # ==================================================================
    @staticmethod
    def _plan_signature(plan: BeamHoppingPlan) -> Tuple:
        """把 BHTP 压成一个可哈希的签名，用作平均容量表的缓存键。""" #即将BHTP由列表变为元组，用于当不同BHTP下平均容量字典检索的键
        return tuple(
            tuple(tuple(satellite_row) for satellite_row in slot_plan) for slot_plan in plan
        )

    def average_capacity_table(self, plan: BeamHoppingPlan) -> Dict[Tuple[int, int], float]:
        """按论文式 (9) 一次算出全部 (卫星 s, 小区 i) 的 C_i^BH，单位 bit/s。

        返回的字典只包含"至少被照明过一次"的 (s,i) 对：
            C^BH_{s,i} = (Δt / T_BH) · Σ_{t: X^t_{s,i}=1} C^t_{s,i}

        为什么要按 (s,i) 分别存：论文里 C_i^BH 是"用户 i 通过它的接入卫星"的
        平均容量，接入卫星 a_i^s 还没定下来之前，每个候选卫星都要各算一份。

        整表结果按 BHTP 签名缓存 —— Algorithm 4 的 swap 搜索会对同一个
        BHTP 反复求解 Algorithm 3，不缓存的话贝塞尔/对数运算会重复上万次。
        """
        signature = self._plan_signature(plan)
        cached = self._plan_capacity_cache.get(signature)
        # 若此前已计算过此BHTP（signature）的容量，则可以直接获取
        if cached is not None:
            return cached

        duty_ratio = self.config.slot_duration_s / self.config.beam_hopping_period_s  # Δt / T_BH，按公式9每个时隙容量仅计入时间占比部分
        table: Dict[Tuple[int, int], float] = {}    # 记录各卫星-小区对的平均容量
        for slot_plan in plan:
            # 当前时隙被照明的所有 (卫星, 小区) 对，即 X^t=1 的集合
            active_links = [
                (sat_id, cell_id)
                for sat_id, satellite_row in enumerate(slot_plan)
                for cell_id, illuminated in enumerate(satellite_row)
                if illuminated
            ]
            for sat_id, cell_id in active_links:
                sinr = self.compute_sinr_for_user(sat_id, cell_id, active_links)
                capacity = self._capacity_from_sinr(sinr)
                table[(sat_id, cell_id)] = table.get((sat_id, cell_id), 0.0) + capacity * duty_ratio
        # 缓存此BHTP下的容量，并将BHTP（signature）作为键存储，便于查找
        self._plan_capacity_cache[signature] = table
        # 缓存上限保护：swap 搜索会产生大量一次性 BHTP，避免字典无限增长。
        if len(self._plan_capacity_cache) > 512:
            self._plan_capacity_cache.pop(next(iter(self._plan_capacity_cache)))
        return table

    def update_for_beam_hopping(self, beam_hopping_plan: BeamHoppingPlan) -> Dict[int, float]:
        """按给定 BHTP 刷新 slot_capacity / sinr_matrix / capacity_matrix / average_capacity。

        这是给"想直接查看某个 BHTP 下各链路状态"的调用方用的接口
        （比如 main.py 打印诊断信息、绘图脚本画 SINR 分布）。
        算法内部计算平均容量请直接用更快的 average_capacity_table()。
        """
        self.slot_capacity.clear()
        for key in self.capacity_matrix:
            self.capacity_matrix[key] = 0.0
            self.sinr_matrix[key] = 0.0
        duty_ratio = self.config.slot_duration_s / self.config.beam_hopping_period_s
        self.average_capacity = {task.user_id: 0.0 for task in self.network.user_tasks}

        for slot_id, slot_plan in enumerate(beam_hopping_plan):
            active_links = [
                (sat_id, cell_id)
                for sat_id, satellite_row in enumerate(slot_plan)
                for cell_id, illuminated in enumerate(satellite_row)
                if illuminated
            ]
            for sat_id, cell_id in active_links:
                sinr = self.compute_sinr_for_user(sat_id, cell_id, active_links)
                capacity = self._capacity_from_sinr(sinr)
                self.slot_capacity[(slot_id, sat_id, cell_id)] = capacity
                self.sinr_matrix[(sat_id, cell_id)] = sinr
                self.capacity_matrix[(sat_id, cell_id)] = capacity
                # 这里的 average_capacity 是"所有卫星照明加总"的上界视角，
                # 真正参与时延计算的是 average_capacity_table() 里按接入卫星取的那一份。
                if cell_id in self.average_capacity:
                    self.average_capacity[cell_id] += capacity * duty_ratio
        return self.average_capacity

    def update_channel_matrix(self, active_satellite_set: Optional[List[int]] = None) -> None:
        """在"每颗卫星只对准各自小区、不考虑 BHTP"的参考场景下刷新容量矩阵。

        仅用于初始化阶段给出一个参考量级（例如 main.py 打印链路预算），
        不参与论文算法的时延计算。
        """
        active = set(active_satellite_set or [sat.sat_id for sat in self.network.satellites])
        for satellite in self.network.satellites:
            for cell in self.network.ground_cells:
                key = (satellite.sat_id, cell.cell_id)
                if satellite.sat_id not in active:
                    self.sinr_matrix[key] = 0.0
                    self.capacity_matrix[key] = 0.0
                    continue
                sinr = self.config.user_tx_power_W * self.channel_gain_linear[key] / self._noise_power_cache
                self.sinr_matrix[key] = sinr
                self.capacity_matrix[key] = self._capacity_from_sinr(sinr)

    # ---------- 查询接口 ----------
    def get_capacity(self, sat_id: int, cell_id: int) -> float:
        """返回最近一次刷新的瞬时容量 C_{s,i}，单位 bit/s。"""
        return self.capacity_matrix.get((sat_id, cell_id), 0.0)

    def get_sinr(self, sat_id: int, cell_id: int) -> float:
        """返回最近一次刷新的 SINR_{s,i}，无量纲。"""
        return self.sinr_matrix.get((sat_id, cell_id), 0.0)
