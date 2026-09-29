"""绘制仿真模型中的卫星、地面小区、用户与波束照明关系。

图中所有关系都来自当前仿真模型本身：
  左图：模型几何 + 候选接入关系（来自 SatelliteNetwork.get_access_candidates）
  右图：指定时隙的实际波束照明关系（来自 Algorithm 4 的初始 BHTP）+ ISL 连线

论文没有给瞬时轨道根数和真实波束指向数据，所以这里不画地面覆盖圆，
只画"卫星星下点 — 被照明小区"的连线，这是模型里真实存在的关系。

运行：
    python plot_topology.py                # 画第 1 个时隙
    python plot_topology.py --slot 3       # 画第 4 个时隙
    python plot_topology.py --polling      # 改画轮询 BHTP（对照算法用的那个）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from channel_model import ChannelModel
from config import SimConfig, build_config
from satellite_network import SatelliteNetwork
from solvers import (
    BeamHoppingPlan,
    design_beam_hopping_matching,
    design_polling_beam_hopping,
)


def configure_chinese_font() -> None:
    """尽量启用中文字体；找不到时 matplotlib 会退回默认字体，不影响出图。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def draw_entities(axis, network: SatelliteNetwork) -> None:
    """画地面小区、用户和卫星星下点三类实体。"""
    axis.scatter(
        [cell.x_km for cell in network.ground_cells],
        [cell.y_km for cell in network.ground_cells],
        s=130, marker="s", facecolor="#B7E4C7", edgecolor="#2D6A4F",
        linewidth=0.8, label="地面小区", zorder=3,
    )
    axis.scatter(
        [task.x_km for task in network.user_tasks],
        [task.y_km for task in network.user_tasks],
        s=22, color="#277DA1", label="用户（小区聚合）", zorder=4,
    )
    axis.scatter(
        [satellite.x_km for satellite in network.satellites],
        [satellite.y_km for satellite in network.satellites],
        s=240, marker="*", color="#F3722C", edgecolor="#7F2704",
        linewidth=0.8, label="LEO 卫星星下点", zorder=5,
    )
    # 小区和卫星编号，方便和日志/结果对照
    label_offset = 0.02 * max(1.0, network.config.coverage_radius_km)
    for cell in network.ground_cells:
        axis.text(cell.x_km + label_offset, cell.y_km + label_offset,
                  f"C{cell.cell_id}", fontsize=6.5, color="#386641")
    for satellite in network.satellites:
        axis.text(satellite.x_km + label_offset * 2, satellite.y_km + label_offset * 2,
                  f"S{satellite.sat_id}", fontsize=9, weight="bold")


def set_axis_limits(axis, network: SatelliteNetwork) -> None:
    """按所有实体的坐标范围设置显示区域，保持等比例。"""
    x_values = [cell.x_km for cell in network.ground_cells] + \
               [satellite.x_km for satellite in network.satellites]
    y_values = [cell.y_km for cell in network.ground_cells] + \
               [satellite.y_km for satellite in network.satellites]
    span = max(max(x_values) - min(x_values), max(y_values) - min(y_values), 1.0)
    margin = 0.08 * span
    axis.set_xlim(min(x_values) - margin, max(x_values) + margin)
    axis.set_ylim(min(y_values) - margin, max(y_values) + margin)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linestyle=":", alpha=0.45)
    axis.set_xlabel("局部地平面 x (km)")
    axis.set_ylabel("局部地平面 y (km)")


def draw_candidate_links(axis, network: SatelliteNetwork) -> None:
    """画"每个用户可以接入哪些卫星"的候选边（几何上可见的全部卫星）。"""
    candidates = network.get_access_candidates()
    for task in network.user_tasks:
        cell = network.get_cell_by_id(task.cell_id)
        for sat_id in candidates[task.user_id]:
            satellite = network.get_satellite_by_id(sat_id)
            axis.plot([satellite.x_km, cell.x_km], [satellite.y_km, cell.y_km],
                      color="#8D99AE", linewidth=0.5, alpha=0.18, zorder=1)


def draw_illuminated_links(axis, network: SatelliteNetwork, plan: BeamHoppingPlan, slot_id: int) -> None:
    """画指定时隙 BHTP 实际点亮的"卫星—小区"波束连线（X^t_{s,i}=1 的那些）。"""
    for sat_id, satellite in enumerate(network.satellites):
        for cell in network.ground_cells:
            if not plan[slot_id][sat_id][cell.cell_id]:
                continue
            axis.plot([satellite.x_km, cell.x_km], [satellite.y_km, cell.y_km],
                      color="#F8961E", linewidth=1.2, alpha=0.8, zorder=2)


def draw_inter_satellite_links(axis, network: SatelliteNetwork) -> None:
    """画 ISL 无向边（论文中每颗卫星有 4 条星间链路）。"""
    drawn = set()
    for sat_id, neighbors in network.satellite_neighbors.items():
        source = network.get_satellite_by_id(sat_id)
        for neighbor_id in neighbors:
            edge = tuple(sorted((sat_id, neighbor_id)))
            if edge in drawn:
                continue
            target = network.get_satellite_by_id(neighbor_id)
            axis.plot([source.x_km, target.x_km], [source.y_km, target.y_km],
                      color="#6C757D", linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)
            drawn.add(edge)


def plot_topology(config: SimConfig, save_path: Path, slot_id: int = 0, use_polling: bool = False) -> None:
    """画左右两幅子图：模型几何关系 / 指定时隙的波束照明关系。"""
    network = SatelliteNetwork(config)
    channel_model = ChannelModel(config, network)
    plan = (design_polling_beam_hopping if use_polling else design_beam_hopping_matching)(
        config, network, channel_model
    )
    if not 0 <= slot_id < config.num_slots_per_BHP:
        raise ValueError(f"slot_id 必须在 0 到 {config.num_slots_per_BHP - 1} 之间")

    configure_chinese_font()
    figure, (geometry_axis, bhtp_axis) = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    # ---- 左图：几何与候选接入关系 ----
    draw_candidate_links(geometry_axis, network)
    draw_entities(geometry_axis, network)
    set_axis_limits(geometry_axis, network)
    geometry_axis.set_title("模型几何与候选接入关系", fontsize=13, weight="bold")
    geometry_axis.legend(loc="upper right", fontsize=8)

    # ---- 右图：某个时隙的实际波束照明 + ISL ----
    draw_inter_satellite_links(bhtp_axis, network)
    draw_illuminated_links(bhtp_axis, network, plan, slot_id)
    draw_entities(bhtp_axis, network)
    set_axis_limits(bhtp_axis, network)
    scheme = "轮询 BHTP" if use_polling else "匹配 BHTP"
    bhtp_axis.set_title(f"{scheme} 第 {slot_id + 1} 个时隙的实际波束照明", fontsize=13, weight="bold")
    bhtp_axis.legend(handles=[
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#F3722C",
               markeredgecolor="#7F2704", markersize=12, label="LEO 卫星"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#B7E4C7",
               markeredgecolor="#2D6A4F", markersize=8, label="地面小区"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#277DA1",
               markersize=6, label="用户"),
        Line2D([0], [0], color="#F8961E", linewidth=1.4, label="当前时隙波束照明"),
        Line2D([0], [0], color="#6C757D", linestyle="--", label="星间链路 ISL"),
    ], loc="upper right", fontsize=8)

    figure.suptitle(
        f"LEO 卫星网络拓扑 | 卫星 {len(network.satellites)} 颗 | "
        f"地面小区 {len(network.ground_cells)} 个 | 每星每时隙 {config.num_beams_per_satellite} 个波束 | "
        f"小区间距 {config.effective_cell_spacing_km:.1f} km",
        fontsize=14, weight="bold",
    )
    figure.savefig(save_path, dpi=300)
    plt.close(figure)
    print(f"[出图] 拓扑图已保存到 {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制卫星网络拓扑与波束照明关系")
    parser.add_argument("--slot", type=int, default=0, help="要画第几个时隙（从 0 开始）")
    parser.add_argument("--polling", action="store_true", help="改画对照算法使用的轮询 BHTP")
    arguments = parser.parse_args()

    config = build_config(solver_backend="fast")
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(exist_ok=True)
    name = "topology_polling.png" if arguments.polling else "topology.png"
    plot_topology(config, output_dir / name, slot_id=arguments.slot, use_polling=arguments.polling)


if __name__ == "__main__":
    main()
