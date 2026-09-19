"""一键生成论文的全部仿真图（图 2a、2b、3~9）以及拓扑图。

每张图由一个独立脚本负责，本文件只是按顺序把它们调起来，
并在最后汇总耗时与产物路径。任何一张图单独重跑，直接运行对应的
figXX_*.py 即可，不需要动这个文件。

运行：
    python run_all_figures.py                    # 用缓存（已算过的图直接读结果重画）
    python run_all_figures.py --rerun            # 全部重新仿真
    python run_all_figures.py --runs 10          # 每个参数点做 10 次蒙特卡洛，曲线更平滑
    python run_all_figures.py --only fig3 fig8   # 只跑指定的几张图
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# 图脚本清单：(简称, 文件名, 说明)
FIGURE_SCRIPTS = [
    ("topology", "plot_topology.py",               "网络拓扑与波束照明关系（非论文图，用于检查建模）"),
    ("fig2a",    "fig2a_bcd_convergence.py",       "图 2(a) BCD 收敛曲线"),
    ("fig2b",    "fig2b_jbhcomb_convergence.py",   "图 2(b) JBHCO-MB 随 swap 次数收敛"),
    ("fig3",     "fig3_user_compute.py",           "图 3 不同用户本地算力 f^l_i"),
    ("fig4",     "fig4_satellite_compute.py",      "图 4 不同卫星算力 f_max"),
    ("fig5",     "fig5_cpu_cycles.py",             "图 5 不同每 bit CPU 周期数"),
    ("fig6",     "fig6_task_size.py",              "图 6 不同任务数据量 M_i"),
    ("fig7",     "fig7_bandwidth.py",              "图 7 不同系统带宽 B"),
    ("fig8",     "fig8_num_satellites.py",         "图 8 不同卫星数量 N_S"),
    ("fig9",     "fig9_num_cells.py",              "图 9 不同地面小区数量 N_I"),
]

# 这两个脚本不接受 --runs/--rerun/--swaps 之外的参数，单列出来便于组装命令行
SWEEP_SCRIPTS = {"fig3", "fig4", "fig5", "fig6", "fig7", "fig8", "fig9"}
CONVERGENCE_SCRIPTS = {"fig2a", "fig2b"}


def build_command(key: str, script: str, arguments: argparse.Namespace) -> list:
    """根据脚本类型拼出它能接受的命令行参数。"""
    command = [sys.executable, script]
    if key in SWEEP_SCRIPTS:
        command += ["--runs", str(arguments.runs), "--swaps", str(arguments.swaps)]
        if arguments.rerun:
            command.append("--rerun")
    elif key in CONVERGENCE_SCRIPTS:
        command += ["--runs", str(arguments.runs)]
        if arguments.rerun:
            command.append("--rerun")
        if key == "fig2b":
            command += ["--max-swaps", str(arguments.max_swaps)]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="一键生成论文全部仿真图")
    parser.add_argument("--runs", type=int, default=5, help="每个参数点的蒙特卡洛次数")
    parser.add_argument("--swaps", type=int, default=100, help="N_max：JBHCO-MB 的 swap 次数")
    parser.add_argument("--max-swaps", type=int, default=1000, help="图 2b 画到多少次 swap")
    parser.add_argument("--rerun", action="store_true", help="忽略缓存，全部重新仿真")
    parser.add_argument("--only", nargs="*", default=None,
                        help="只跑指定的图，例如 --only fig3 fig8")
    arguments = parser.parse_args()

    selected = [item for item in FIGURE_SCRIPTS
                if arguments.only is None or item[0] in arguments.only]
    if not selected:
        print(f"[错误] --only 里没有匹配到任何脚本，可选：{[k for k, _, _ in FIGURE_SCRIPTS]}")
        return

    work_dir = Path(__file__).resolve().parent
    print("=" * 78)
    print(f"准备生成 {len(selected)} 张图（蒙特卡洛 {arguments.runs} 次 / 点，N_max={arguments.swaps}）")
    print("=" * 78)

    total_start = time.time()
    timing = []
    for key, script, description in selected:
        print(f"\n>>> {key}：{description}")
        start = time.time()
        command = build_command(key, script, arguments)
        completed = subprocess.run(command, cwd=work_dir)
        elapsed = time.time() - start
        status = "成功" if completed.returncode == 0 else f"失败(返回码 {completed.returncode})"
        timing.append((key, description, elapsed, status))
        print(f"<<< {key} {status}，耗时 {elapsed:.1f} s")

    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    for key, description, elapsed, status in timing:
        print(f"  {key:<9s} {status:<6s} {elapsed:7.1f} s   {description}")
    print(f"\n总耗时 {time.time() - total_start:.1f} s")
    print(f"图片输出目录：{work_dir / 'figures'}")
    print(f"数值结果目录：{work_dir / 'results'}")


if __name__ == "__main__":
    main()
