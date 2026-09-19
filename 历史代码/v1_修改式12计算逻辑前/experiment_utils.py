"""各仿真图脚本的公共工具：参数扫描、蒙特卡洛平均、结果缓存、统一绘图风格。

所有 figXX_*.py 都只做两件事：
    1) 声明"扫描哪个参数、取哪些值"；
    2) 调用本模块的 sweep_parameter() 和 plot_curves()。
这样九张图的脚本都很短，改参数也只需要改一行。

绘图风格与论文图 3~图 9 对齐
---------------------------
    JBHCO-MB : 黑色 实心方块 ■
    JBHCO-PB : 红色 实心圆点 ●
    CL       : 蓝色 实心上三角 ▲
    JBHCO-PG : 绿色 实心下三角 ▼
坐标轴文字统一用英文，和论文图一致，同时也避免不同系统缺中文字体时出现方框。
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")          # 无显示环境也能出图；如需交互式窗口可删掉这一行
import matplotlib.pyplot as plt

from baselines import run_computing_locally, run_jbhco_pb, run_jbhco_pg
from config import SimConfig, build_config
from proposed_algo import initialize_simulation, proposed_jbhco_mb

# ---------------------------------------------------------------------
# 常量：四种算法的名称与绘图样式
# ---------------------------------------------------------------------
ALGORITHM_ORDER = ["JBHCO-MB", "JBHCO-PB", "CL", "JBHCO-PG"]

PLOT_STYLE: Dict[str, Dict[str, Any]] = {
    "JBHCO-MB": {"color": "#000000", "marker": "s", "label": "JBHCO-MB"},
    "JBHCO-PB": {"color": "#E02020", "marker": "o", "label": "JBHCO-PB"},
    "CL":       {"color": "#2050E0", "marker": "^", "label": "CL"},
    "JBHCO-PG": {"color": "#12A050", "marker": "v", "label": "JBHCO-PG"},
}

# 结果缓存目录：同一组参数只跑一次，改绘图样式时无需重算
RESULT_DIR = Path(__file__).resolve().parent / "results"
FIGURE_DIR = Path(__file__).resolve().parent / "figures"


# =====================================================================
# 1. 单点仿真：在一组参数下跑指定的几个算法
# =====================================================================
def run_single_scenario(
    config: SimConfig,
    algorithms: Sequence[str] = tuple(ALGORITHM_ORDER),
) -> Dict[str, float]:
    """在给定配置下运行指定算法，返回 {算法名: max λ_i}。

    四种算法共用同一份网络拓扑、信道和任务集合，保证比较公平。
    """
    network, channel_model = initialize_simulation(config)
    values: Dict[str, float] = {}
    for name in algorithms:
        if name == "JBHCO-MB":
            result = proposed_jbhco_mb(config, network, channel_model, record_history=False)
        elif name == "JBHCO-PB":
            result = run_jbhco_pb(config, network, channel_model)
        elif name == "CL":
            result = run_computing_locally(config, network, channel_model)
        elif name == "JBHCO-PG":
            result = run_jbhco_pg(config, network, channel_model)
        else:
            raise ValueError(f"未知算法名 {name}")
        values[name] = float(result["max_lambda"])
    return values


def run_monte_carlo(
    overrides: Dict[str, Any],
    algorithms: Sequence[str] = tuple(ALGORITHM_ORDER),
    num_runs: int = 5,
    base_seed: int = 7,
) -> Dict[str, float]:
    """同一组参数下换若干组随机任务重复仿真，返回各算法 max λ_i 的均值。

    为什么要做蒙特卡洛：论文的目标是 max_i λ_i，它由"任务量最大的那个用户"主导，
    单次随机抽样的 M_i 最大值波动很大，曲线会出现锯齿。多跑几组随机种子求平均，
    才能得到论文图 3~图 9 那样平滑、可比较的曲线。
    """
    accumulated: Dict[str, List[float]] = {name: [] for name in algorithms}
    for run_index in range(num_runs):
        local_overrides = dict(overrides)
        local_overrides["random_seed"] = base_seed + run_index
        config = build_config(**local_overrides)
        values = run_single_scenario(config, algorithms)
        for name, value in values.items():
            accumulated[name].append(value)
    return {name: statistics.fmean(values) for name, values in accumulated.items()}


# =====================================================================
# 2. 参数扫描
# =====================================================================
def sweep_parameter(
    parameter_name: str,
    parameter_values: Sequence[Any],
    cache_name: str,
    algorithms: Sequence[str] = tuple(ALGORITHM_ORDER),
    num_runs: int = 5,
    base_overrides: Optional[Dict[str, Any]] = None,
    value_to_overrides: Optional[Callable[[Any], Dict[str, Any]]] = None,
    force_rerun: bool = False,
) -> Dict[str, List[float]]:
    """沿某个参数扫描，返回 {算法名: [各参数点的 max λ_i 均值]}。

    参数
    ----
    parameter_name    : 要扫描的 SimConfig 字段名，仅用于日志与缓存标识
    parameter_values  : 横坐标取值列表
    cache_name        : 缓存文件名（不含扩展名），结果存到 results/<cache_name>.json
    algorithms        : 要跑哪些算法
    num_runs          : 每个参数点的蒙特卡洛次数
    base_overrides    : 所有参数点共用的配置覆盖项（例如 solver_backend="fast"）
    value_to_overrides: 把横坐标值翻译成配置覆盖项的函数。默认是
                        {parameter_name: value}；像图 6 那样横坐标是"区间"时需要自定义
    force_rerun       : True 则忽略已有缓存，强制重算
    """
    RESULT_DIR.mkdir(exist_ok=True)
    cache_path = RESULT_DIR / f"{cache_name}.json"
    if cache_path.exists() and not force_rerun:
        with cache_path.open(encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("parameter_values") == list(map(str, parameter_values)) \
                and cached.get("num_runs") == num_runs:
            print(f"[缓存] 直接读取 {cache_path.name}（如需重算请加 --rerun）")
            return {name: cached["curves"][name] for name in algorithms}

    base_overrides = dict(base_overrides or {})
    base_overrides.setdefault("solver_backend", "fast")
    curves: Dict[str, List[float]] = {name: [] for name in algorithms}

    print(f"[扫描] 参数 {parameter_name}，共 {len(parameter_values)} 个点，"
          f"每点 {num_runs} 次蒙特卡洛")
    for index, value in enumerate(parameter_values):
        overrides = dict(base_overrides)
        overrides.update(
            value_to_overrides(value) if value_to_overrides else {parameter_name: value}
        )
        start = time.time()
        averaged = run_monte_carlo(overrides, algorithms, num_runs)
        for name in algorithms:
            curves[name].append(averaged[name])
        detail = "  ".join(f"{name}={averaged[name]:.4f}" for name in algorithms)
        print(f"  [{index + 1}/{len(parameter_values)}] {parameter_name}={value}  "
              f"{detail}  ({time.time() - start:.1f}s)")

    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "parameter_name": parameter_name,
                "parameter_values": list(map(str, parameter_values)),
                "num_runs": num_runs,
                "curves": curves,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[保存] 数值结果写入 {cache_path}")
    return curves


# =====================================================================
# 3. 绘图
# =====================================================================
def plot_curves(
    x_values: Sequence[Any],
    curves: Dict[str, Sequence[float]],
    x_label: str,
    y_label: str = r"The maximum $\lambda_i$",
    output_name: str = "figure.png",
    title: Optional[str] = None,
    x_tick_labels: Optional[Sequence[str]] = None,
    log_x: bool = False,
) -> Path:
    """按论文图 3~图 9 的风格画四条曲线并保存 PNG。

    参数
    ----
    x_values       : 横坐标数值（若是分类型横坐标，传 0,1,2,... 并配合 x_tick_labels）
    curves         : {算法名: 纵坐标序列}
    x_label/y_label: 坐标轴标题（英文，与论文一致）
    output_name    : 输出文件名，保存在 figures/ 目录
    x_tick_labels  : 自定义刻度标签，例如图 6 的 "0~5"、"5~10"
    log_x          : 横坐标是否用对数刻度（图 2b 的 N_max 用得上）
    """
    FIGURE_DIR.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    for name in ALGORITHM_ORDER:
        if name not in curves:
            continue
        style = PLOT_STYLE[name]
        axis.plot(
            x_values, curves[name],
            color=style["color"], marker=style["marker"], label=style["label"],
            linewidth=1.6, markersize=6, markerfacecolor=style["color"],
        )
    if log_x:
        axis.set_xscale("log")
    if x_tick_labels is not None:
        axis.set_xticks(list(x_values))
        axis.set_xticklabels(list(x_tick_labels))
    axis.set_xlabel(x_label, fontsize=11)
    axis.set_ylabel(y_label, fontsize=11)
    if title:
        axis.set_title(title, fontsize=11)
    axis.grid(True, linestyle=":", alpha=0.45)
    axis.legend(fontsize=10, framealpha=0.95)
    figure.tight_layout()
    output_path = FIGURE_DIR / output_name
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"[出图] {output_path}")
    return output_path


def plot_generic_series(
    series: Dict[str, Dict[str, Sequence[float]]],
    x_label: str,
    y_label: str,
    output_name: str,
    title: Optional[str] = None,
    log_x: bool = False,
) -> Path:
    """画"若干条自定义曲线"的通用函数，供图 2a / 图 2b 使用。

    series 的结构：{曲线名: {"x": 横坐标序列, "y": 纵坐标序列}}
    图 2 的四条曲线对应四种网络规模场景，配色与标记单独指定。
    """
    FIGURE_DIR.mkdir(exist_ok=True)
    palette = ["#000000", "#E02020", "#2050E0", "#12A050"]
    markers = ["s", "o", "^", "v"]
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    for index, (name, data) in enumerate(series.items()):
        axis.plot(
            data["x"], data["y"],
            color=palette[index % len(palette)], marker=markers[index % len(markers)],
            label=name, linewidth=1.6, markersize=5,
        )
    if log_x:
        axis.set_xscale("log")
    axis.set_xlabel(x_label, fontsize=11)
    axis.set_ylabel(y_label, fontsize=11)
    if title:
        axis.set_title(title, fontsize=11)
    axis.grid(True, linestyle=":", alpha=0.45)
    axis.legend(fontsize=9, framealpha=0.95)
    figure.tight_layout()
    output_path = FIGURE_DIR / output_name
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"[出图] {output_path}")
    return output_path


# =====================================================================
# 4. 命令行公共参数
# =====================================================================
def add_common_arguments(parser) -> None:
    """给每个 figXX 脚本加上统一的命令行选项。"""
    parser.add_argument("--runs", type=int, default=5,
                        help="每个参数点的蒙特卡洛次数（越大曲线越平滑，耗时线性增长）")
    parser.add_argument("--rerun", action="store_true",
                        help="忽略 results/ 下的缓存，强制重新仿真")
    parser.add_argument("--swaps", type=int, default=100,
                        help="N_max：JBHCO-MB 的 swap 次数，论文取 100")
