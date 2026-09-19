"""论文 15 的卫星网络拓扑生成模块。

拓扑包含四类对象：
    1) LEO 卫星节点 S = {1,...,N_S}；
    2) 地面小区 I = {1,...,N_I}；
    3) 每个小区聚合的一个用户（论文第三节脚注 1 的假设）；
    4) 卫星间链路 ISL（论文第三节：每颗卫星与相邻卫星保持 4 条 ISL）。

坐标系说明
----------
论文只给出"5 颗卫星组成的紧凑子网络 + 25 个地面小区"，没有给瞬时轨道根数。
本文件因此采用"局部地平面 + 高度 z"的抽象坐标系：
    x_km / y_km ：地面东向 / 北向坐标；
    z_km        ：卫星相对地面的高度（=论文的 h=1200 km）。
卫星到小区的距离用三维直线距离计算，与论文式 (4) 中的 d_i,s 含义一致。
如果后续要接真实轨道，只需替换 _generate_satellites()，算法接口不变。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import SimConfig, UserTask


# ======================================================================
# 1. 基本节点数据结构
# ======================================================================
@dataclass
class GroundCell:
    """地面小区 i（论文中每个小区聚合一个用户的业务需求）。"""

    cell_id: int          # i：地面小区编号。
    x_km: float           # 小区中心东向坐标，单位 km。
    y_km: float           # 小区中心北向坐标，单位 km。
    demand: float = 1.0   # 小区归一化业务需求，用于 BHTP 偏好排序时的权重。

    @property
    def x(self) -> float:
        return self.x_km

    @property
    def y(self) -> float:
        return self.y_km


@dataclass
class SatelliteNode:
    """LEO 卫星节点 s。"""

    sat_id: int          # s：卫星编号。
    x_km: float          # 卫星星下点东向坐标，单位 km。
    y_km: float          # 卫星星下点北向坐标，单位 km。
    z_km: float          # h：卫星高度，单位 km。
    beam_capacity: int   # N_K：该卫星同一时隙最大同时照明小区数。

    @property
    def x(self) -> float:
        return self.x_km

    @property
    def y(self) -> float:
        return self.y_km

    @property
    def z(self) -> float:
        return self.z_km

    def distance_to_ground(self, cell: GroundCell) -> float:
        """d_i,s：卫星到地面小区的三维直线距离，单位 km（论文式 (4) 用到）。"""
        dx = cell.x_km - self.x_km
        dy = cell.y_km - self.y_km
        dz = -self.z_km
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def direction_to_cell(self, cell: GroundCell) -> Tuple[float, float, float]:
        """返回卫星指向某小区的单位方向向量（无量纲），供计算离轴角 θ 使用。"""
        dx = cell.x_km - self.x_km
        dy = cell.y_km - self.y_km
        dz = -self.z_km
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        norm = max(norm, 1e-12)
        return (dx / norm, dy / norm, dz / norm)


# ======================================================================
# 2. 网络整体
# ======================================================================
@dataclass
class SatelliteNetwork:
    """静态（准静态）网络拓扑及其派生的几何关系矩阵。

    论文第三节说明：资源分配时长只有几个 BHP（数百毫秒），信道和用户关联
    可视为准静态，因此这里把拓扑当成一次生成、全程不变的静态结构。
    """

    config: SimConfig
    satellites: List[SatelliteNode] = field(default_factory=list)   # 卫星集合 S，相当于初始化satellites=[]。
    ground_cells: List[GroundCell] = field(default_factory=list)    # 小区集合 I。
    user_tasks: List[UserTask] = field(default_factory=list)        # 任务集合 {R_i}。
    sat_to_cell_distance: Dict[Tuple[int, int], float] = field(default_factory=dict)  # d_i,s，单位 km。
    satellite_neighbors: Dict[int, List[int]] = field(default_factory=dict)           # ISL 邻居表。

    def __post_init__(self) -> None:
        """按论文系统模型依次生成：卫星 → 小区 → 距离矩阵 → ISL。"""
        self.user_tasks = list(self.config.user_tasks)
        self._generate_satellites()
        self._generate_ground_cells()
        self._compute_geometry_matrix()
        self._generate_inter_satellite_links()

    # ---------- 2.1 卫星 ----------
    def _generate_satellites(self) -> None:
        """生成"紧凑子网络"的卫星星下点，采用环形均匀投影布局。

        论文第五节：把任意一颗卫星和它相邻的 4 颗卫星视作一个子网络，
        每颗卫星并行管理自己的子网络资源，所以仿真只用 5 颗卫星。
        这里把这 5 颗（或扫描时的 N_S 颗）卫星的星下点均匀摆在一个圆上，
        使它们对 25 个小区的几何关系互不相同但又都可见。
        """
        radius_km = self.config.satellite_spread_radius_km  # 星下点分布圆半径，单位 km
        for sat_id in range(self.config.num_satellites):
            angle = 2.0 * math.pi * sat_id / max(self.config.num_satellites, 1)
            self.satellites.append(
                SatelliteNode(
                    sat_id=sat_id,
                    x_km=radius_km * math.cos(angle),
                    y_km=radius_km * math.sin(angle),
                    z_km=self.config.satellite_altitude_km,
                    beam_capacity=self.config.num_beams_per_satellite,
                )
            )

    # ---------- 2.2 地面小区 ----------
    def _generate_ground_cells(self) -> None:
        """生成 N_I 个规则地面小区，一个小区对应一个用户。"""
        if len(self.config.user_positions) != self.config.num_ground_cells:
            raise ValueError(
                "user_positions 数量必须等于 num_ground_cells，"
                "以保证任务、小区、信道几何一一对应。"
            )
        for cell_id, (x_km, y_km) in enumerate(self.config.user_positions):
            self.ground_cells.append(GroundCell(cell_id=cell_id, x_km=x_km, y_km=y_km, demand=1.0))

    # ---------- 2.3 几何 ----------
    def _compute_geometry_matrix(self) -> None:
        """计算全部卫星-小区距离 d_i,s，单位 km。"""
        for satellite in self.satellites:
            for cell in self.ground_cells:
                self.sat_to_cell_distance[(satellite.sat_id, cell.cell_id)] = \
                    satellite.distance_to_ground(cell)

    # ---------- 2.4 ISL ----------
    def _generate_inter_satellite_links(self) -> None:
        """生成论文描述的 ISL 关系（每星最多 4 条：2 条同轨 + 2 条异轨）。

        本文的算法不含星间协同计算（论文明确说明 inter-satellite cooperative
        computing 不在考虑范围），ISL 只用于网关汇聚信息，因此这里只需要
        一个连通的邻居表，用于拓扑图绘制和后续扩展。
        实现：在环形布局上取每颗卫星左右各 2 个邻居，正好 4 条 ISL。
        """
        count = len(self.satellites)
        for sat_id in range(count):
            neighbors = set()   # 集合的作用是自动去重
            for offset in (-2, -1, 1, 2):
                if count > 1:
                    neighbors.add((sat_id + offset) % count)
            neighbors.discard(sat_id)
            self.satellite_neighbors[sat_id] = sorted(neighbors)

    # ---------- 2.5 查询接口 ----------
    def get_cell_by_id(self, cell_id: int) -> Optional[GroundCell]:
        """按小区编号 i 取地面小区对象（越界返回 None）。"""
        if 0 <= cell_id < len(self.ground_cells):
            return self.ground_cells[cell_id]
        return None

    def get_satellite_by_id(self, sat_id: int) -> Optional[SatelliteNode]:
        """按卫星编号 s 取卫星对象（越界返回 None）。"""
        if 0 <= sat_id < len(self.satellites):
            return self.satellites[sat_id]
        return None

    def build_task_to_cell_mapping(self) -> Dict[int, int]:
        """建立用户 i → 小区 i 的映射（本模型下是恒等映射，保留接口备用）。"""
        return {task.user_id: task.cell_id for task in self.user_tasks}

    def get_access_candidates(self, max_candidates: Optional[int] = None) -> Dict[int, List[int]]:
        """按几何距离从近到远，为每个用户给出候选接入卫星列表。

        论文中"任意小区都可能被多颗卫星覆盖，用户需要选一颗接入卫星"，
        这里的候选集就是这句话的实现；真正能不能用还要看 BHTP 有没有照明它。
        """
        candidate_count = max_candidates or len(self.satellites)
        candidates: Dict[int, List[int]] = {}
        for task in self.user_tasks:
            ranked = sorted(
                self.satellites,
                key=lambda satellite: self.sat_to_cell_distance[(satellite.sat_id, task.cell_id)],
            )
            candidates[task.user_id] = [satellite.sat_id for satellite in ranked[:candidate_count]]
        return candidates
