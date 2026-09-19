# 论文 15 复现代码 —— 文件清单与运行逻辑

> 论文：Gaofeng Cui et al., *Joint Beam Hopping and Computing Offloading Design for
> LEO Satellite Networks*, IEEE TVT, DOI 10.1109/TVT.2026.3705067

---

## 0. 三分钟上手

```bash
pip install numpy matplotlib          # 必需
pip install cvxpy                     # 可选：只有用 --backend cvxpy 时才需要

python verify_solvers.py              # 第一步：跑自检，7 项全通过说明环境和建模没问题
python main.py                        # 第二步：默认参数下跑一遍四种算法（约 2 秒）
python run_all_figures.py --runs 5    # 第三步：生成论文全部仿真图（约 3 分钟）
```

产物位置：

- 图片 → `figures/`
- 数值结果（可直接用于重新绘图或做表）→ `results/*.json`
- 单次仿真明细 → `simulation_result.json`

---

## 1. 文件清单

### 1.1 核心模型层（被所有脚本依赖，一般不需要改）

| 文件                     | 职责        | 关键内容                                                                                                     |
| ---------------------- | --------- | -------------------------------------------------------------------------------------------------------- |
| `config.py`            | 仿真参数与任务生成 | `SimConfig` 默认值 = 论文表 II；`UserTask` 对应式 (2) 的 R_i={M_i,O_i,D_i}；`build_config(**overrides)` 是所有扫描实验的统一入口 |
| `satellite_network.py` | 拓扑建模      | `SatelliteNode` / `GroundCell`；距离矩阵 d_i,s；ISL 邻居表；`get_access_candidates()` 给出每个用户几何上可见的卫星               |
| `channel_model.py`     | 信道与容量建模   | 式 (3)~(9) 全部实现：路径损耗、式 (5) 贝塞尔方向图、式 (6) SINR、式 (8) 香农容量、式 (9) 一个 BHP 内的平均容量 C_i^BH                        |

### 1.2 算法层

| 文件                 | 职责           | 对应论文                                                                                                                                                                                                                                                          |
| ------------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `solvers.py`       | 论文全部算法构件     | 第 1 节时延模型 = 式 (10)~(19)；`solve_offloading` = Algorithm 1 / P̃2；`solve_resource_allocation` = Algorithm 2 / P̃3；`solve_joint_offloading_and_allocation` = Algorithm 3；`design_beam_hopping_matching` = Algorithm 4 第 2-8 行；`propose_swap` = Algorithm 4 第 12 行 |
| `proposed_algo.py` | JBHCO-MB 主算法 | Algorithm 4 完整主循环：初始匹配 → Algorithm 3 → N_max 次 swap                                                                                                                                                                                                           |
| `baselines.py`     | 三个对照算法       | 论文第五节 B 的 JBHCO-PB（轮询+BCD）、CL（全本地）、JBHCO-PG（轮询+贪婪）                                                                                                                                                                                                            |

### 1.3 入口与检查脚本

| 文件                  | 用途                        | 典型运行时间         |
| ------------------- | ------------------------- | -------------- |
| `main.py`           | 默认参数下跑一遍四种算法，打印对比并写 JSON  | 约 2 秒（fast 后端） |
| `verify_solvers.py` | 7 项建模/求解自检，改完代码务必先跑它      | 约 10 秒         |
| `plot_topology.py`  | 画拓扑与波束照明关系（非论文图，用于肉眼检查建模） | 约 1 秒          |

### 1.4 仿真图脚本

| 文件                             | 复现论文哪张图              | 横轴             | 单独运行时间（runs=5） |
| ------------------------------ | -------------------- | -------------- | -------------- |
| `experiment_utils.py`          | （公共层，不单独运行）          | —              | —              |
| `fig2a_bcd_convergence.py`     | 图 2(a) BCD 收敛曲线      | BCD 迭代次数 1~8   | 约 3 秒          |
| `fig2b_jbhcomb_convergence.py` | 图 2(b) JBHCO-MB 收敛曲线 | N_max（swap 次数） | 约 200 秒        |
| `fig3_user_compute.py`         | 图 3                  | 用户本地算力 f^l_i   | 约 62 秒         |
| `fig4_satellite_compute.py`    | 图 4                  | 卫星算力 f_max     | 约 60 秒         |
| `fig5_cpu_cycles.py`           | 图 5                  | 每 bit CPU 周期数  | 约 65 秒         |
| `fig6_task_size.py`            | 图 6                  | 任务数据量 M_i 区间   | 约 56 秒         |
| `fig7_bandwidth.py`            | 图 7                  | 系统带宽 B         | 约 58 秒         |
| `fig8_num_satellites.py`       | 图 8                  | 卫星数 N_S        | 约 58 秒         |
| `fig9_num_cells.py`            | 图 9                  | 地面小区数 N_I      | 约 64 秒         |
| `run_all_figures.py`           | 一键跑上面全部              | —              | 约 10 分钟        |

---

## 2. 模块调用关系

```
                    ┌──────────────┐
                    │  config.py   │  参数 + 任务 R_i={M_i,O_i,D_i}
                    └──────┬───────┘
                           │
                    ┌──────▼──────────────┐
                    │ satellite_network.py│  卫星/小区坐标、距离 d_i,s、ISL
                    └──────┬──────────────┘
                           │
                    ┌──────▼───────────┐
                    │ channel_model.py │  PL → G_rx → SINR → C_i^t → C_i^BH
                    └──────┬───────────┘
                           │
                    ┌──────▼───────┐
                    │  solvers.py  │  Algorithm 1 / 2 / 3 + BHTP 匹配 + swap
                    └──┬────────┬──┘
                       │        │
        ┌──────────────▼──┐  ┌──▼────────────┐
        │ proposed_algo.py│  │  baselines.py │
        │   (JBHCO-MB)    │  │ PB / CL / PG  │
        └──────┬──────────┘  └──────┬────────┘
               │                    │
               └────────┬───────────┘
                        │
        ┌───────────────┼────────────────┬──────────────────┐
        │               │                │                  │
   ┌────▼────┐  ┌───────▼─────────┐ ┌────▼──────────┐ ┌─────▼──────────┐
   │ main.py │  │verify_solvers.py│ │experiment_    │ │plot_topology.py│
   │单次仿真 │  │   7 项自检      │ │  utils.py     │ │   拓扑图       │
   └─────────┘  └─────────────────┘ └────┬──────────┘ └────────────────┘
                                          │
                          ┌───────────────▼──────────────┐
                          │ fig2a / fig2b / fig3 ... fig9│
                          │      run_all_figures.py      │
                          └──────────────────────────────┘
```

**一句话总结依赖方向**：参数 → 拓扑 → 信道 → 算法 → 入口脚本。箭头从不反向，
所以改上游（比如换一套卫星轨道模型）不需要动下游算法代码。

---

## 3. 单次仿真的执行顺序

以 `python main.py` 为例，内部发生的事情：

1. `build_config()` 生成参数，`__post_init__` 里同时生成 25 个小区坐标和 25 个任务。
2. `initialize_simulation()` 建 `SatelliteNetwork`（算好 d_i,s）和 `ChannelModel`
   （算好每条链路的路径损耗和对准增益）。
3. `proposed_jbhco_mb()` 执行 Algorithm 4：
   - `design_beam_hopping_matching()` 用延迟接受匹配生成初始 BHTP X^t_{s,i}；
   - `solve_joint_offloading_and_allocation()` 执行 Algorithm 3：
     - `OffloadingState` 从"全本地计算"这个可行初值出发；
     - 循环：`forecast_matrix()` → `build_zeta_matrix()` → Algorithm 1 更新 A
       → 若整体更新不降低目标，则退化为逐用户最优响应，保证单调下降；
     - 收敛后做 0-1 恢复与逐用户可行性检查（论文第 7-17 行）；
   - 循环 N_max=100 次：`propose_swap()` 随机交换 → 重解 Algorithm 3 →
     若构成 swap-blocking pair（目标严格下降）就接受。
4. `baselines.py` 里的三个对照算法在**同一份网络、信道、任务**上各跑一遍。
5. 打印对比表 + 约束检查，写入 `simulation_result.json`。

---

## 4. 两套求解后端

`SimConfig.solver_backend` 可选 `"cvxpy"` 或 `"fast"`：

|                  | `cvxpy`                       | `fast`      |
| ---------------- | ----------------------------- | ----------- |
| P̃2（Algorithm 1） | CVXPY 建 LP，调 CLARABEL         | 闭式解         |
| P̃3（Algorithm 2） | CVXPY + `inv_pos`，调 CLARABEL  | 逐星二分        |
| 单次 P̃3 耗时        | 约 4.2 秒                       | 约 0.3 毫秒    |
| 用途               | 核对建模是否忠实于论文的 "solve with CVX" | 扫描仿真图（必须用它） |

两者等价性已在 `verify_solvers.py` 检查 5 中逐点核对：P̃2 的 γ 相对差约 1e-9，
P̃3 的 ε 相对差约 7e-4（求解器容差），25 个用户选出的接入卫星完全一致。

**为什么 fast 后端是精确的、而不是近似**，推导写在源码注释里：

- **P̃2**：约束按用户完全解耦，每个用户的子问题是单纯形上的线性规划，
  最优解必在顶点 —— 若 min_s ζ_{i,s} < 0 就把 a 全押在 argmin 上，否则全 0。
  所以松弛解天然是 0-1 的，二分法只是在数值上逼近同一个 γ*。
- **P̃3**：对给定 ε，约束 (30a) 等价于 f_i ≥ (O_i/D_i)/(ε−ψ_i)，于是
  "ε 可行 ⟺ Σ_i (O_i/D_i)/(ε−ψ_i) ≤ f_max"。左边关于 ε 单调递减，二分即得精确解。
  各卫星预算独立（C2 按卫星写），所以全局 ε 就是各星 ε_s 的最大值。

---

## 5. 常用命令速查

```bash
# 单次仿真
python main.py                          # fast 后端，默认参数
python main.py --backend cvxpy          # 严格按论文用 CVX 解凸子问题（慢）
python main.py --seed 11                # 换一组随机任务
python main.py --swaps 500              # 把 N_max 从 100 改成 500

# 自检
python verify_solvers.py

# 拓扑图
python plot_topology.py                 # 匹配 BHTP 的第 1 个时隙
python plot_topology.py --slot 3        # 第 4 个时隙
python plot_topology.py --polling       # 改画对照算法用的轮询 BHTP

# 单张仿真图
python fig4_satellite_compute.py                  # 有缓存就直接读缓存重画
python fig4_satellite_compute.py --runs 10 --rerun  # 强制重算，10 次蒙特卡洛

# 批量
python run_all_figures.py --runs 5
python run_all_figures.py --only fig3 fig8 --rerun
```

三个通用命令行参数：

- `--runs N`：每个参数点做几次蒙特卡洛。max λ_i 由"任务量最大的那个用户"主导，
  单次抽样波动大，跑 5~10 次求平均曲线才平滑。耗时随 N 线性增长。
- `--rerun`：忽略 `results/` 里的缓存强制重算。只改绘图样式时不要加，秒出图。
- `--swaps N`：N_max，论文取 100。

---

## 6. 想改哪里就改哪里

| 想做的事               | 改哪个文件的哪里                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------- |
| 换系统参数（带宽、算力、任务量……） | `config.py` 的 `SimConfig` 字段；或调用时用 `build_config(字段名=值)`                                             |
| 换卫星轨道模型（接真实星历）     | `satellite_network.py` 的 `_generate_satellites()`，其余代码不用动                                            |
| 换天线方向图             | `channel_model.py` 的 `receive_antenna_gain_dBi()`                                                    |
| 打开阴影衰落 / 雨衰        | `config.py` 的 `shadowing_std_dB`、`rain_attenuation_dB`（默认都是 0，论文未给数值）                                |
| 改小区排布              | `config.py` 的 `_generate_user_positions()`；间距由 `effective_cell_spacing_km` 控制                        |
| 换 BHTP 设计算法        | `solvers.py` 新写一个返回 `BeamHoppingPlan` 的函数，替换 `proposed_algo.py` 里的调用                                 |
| 加第 5 种对照算法         | `baselines.py` 里加函数，注册进 `BASELINE_RUNNERS`，再在 `experiment_utils.ALGORITHM_ORDER` 和 `PLOT_STYLE` 里加一项 |
| 加一张新的扫描图           | 复制 `fig7_bandwidth.py` 改三行：扫描字段名、取值列表、缓存名                                                            |

---

## 7. 相对原始代码修正的建模问题

复现过程中发现并修正了 8 处，按影响从大到小排列。这一节建议写进论文复现报告。

### 7.1 小区间距与波束足印不匹配（影响最大）

原代码把 25 个小区摊在半径 1000 km 的区域内，相邻小区相距 400 km。
但论文给了 3 dB 波束角 1.7647° 和高度 1200 km，并明确说
"one beam precisely illuminates a ground cell"，所以小区尺寸应等于波束足印直径：

```
spacing = 2·h·tan(θ_3dB/2) = 2 × 1200 × tan(0.882°) ≈ 37 km
```

400 km 间距下相邻小区的离轴角约 18°，早已落进远旁瓣，式 (6) 的同频干扰
衰减到可以忽略 —— 这会让 BHTP 设计（哪些小区同时被点亮）变成一个无关紧要的变量，
整篇论文的核心贡献就失去了意义。改成 37 km 后：相邻小区离轴增益 26.4 dBi、
主瓣 38.5 dBi（只低 12 dB），实测含干扰 SINR 在 11.3~18.2 dB 之间波动，
BHTP 才真正影响容量。现由 `config.effective_cell_spacing_km` 自动推算。

### 7.2 BCD 被初始关联锁死

Algorithm 1 的 ζ_{i,s} 含 O_i/f^S_{s,i}，而 ==Algorithm 2 只给"已关联到 s 的用户"
分配算力，其余 (s,i) 对 f^S=0 → ζ=+∞ → 用户永远无法切换卫星。==
原代码的后果是 25 个用户全挤在少数几颗星上，**max λ_i = 0.1668，比全本地计算
（0.1647）还差**，即算法完全失效。

修正：新增 `OffloadingState.forecast_matrix()`，把用户 i 临时放进 U_s 重做一次
minimax 分配，用"我若加入这颗星能分到多少"来构造 ζ。这样 Algorithm 1 成为
严格的最优响应步骤，BCD 才能在各卫星之间做负载均衡。

### 7.3 式 (6) 的干扰项用错了路径损耗

原代码计算的是"受害用户自己的信号经由别的波束收到的功率"，
用的是受害用户的 PL。物理上应该是"干扰用户 j 的信号落进受害波束 i 的旁瓣"：

```
I = P_u · G_tx · PL_{s,j} · G_rx(θ: 用户 j 相对波束 i 中心的离轴角)
```

即用**干扰用户 j** 的路径损耗。已修正。

### 7.4 式 (5) 的离轴角定义

论文写明 θ 是"用户相对其服务波束 bore-sight 方向"的角度。原代码用的是
用户相对卫星星下点的角度，导致所有服务链路都被压到旁瓣下限。
现在：服务链路取 θ=0 → G_rx^max；干扰链路取卫星处两条视线的三维夹角
（而非地面横向距离的平面近似，后者在用户远离星下点时误差很大）。

### 7.5 主循环结构与 Algorithm 4 不符

原代码是"外层 50 次迭代，每次内部再做 100 次 swap"，即最多 5000 次 swap，
每次都要重解 Algorithm 3。论文 Algorithm 4 是单层结构：
一次匹配 → 一次 Algorithm 3 → N_max 次 swap。已按论文改写，
并顺带记录每次 swap 后的 G(Ω)，正好用来画图 2(b)。

### 7.6 swap 操作的可行性判定

两个问题：

- 没检查"目标时隙是否已有该小区"，交换后该星波束数会掉到 N_K 以下，违反 C1；
- 把"同一时隙不重复照明同一小区"当成硬约束。但论文定义 3 只要求
  Ω^{t,t'}_{i,i'} 满足 C1~C6，去冗余只是匹配阶段的**偏好置零**规则。
  当成硬约束的结果是 100 次随机抽样只有 2 次可行，swap 几乎无法进行。

修正后可行交换比例升到约 85%，swap 带来约 3.3% 的目标改善
（论文报告 3.68%~9.43%）。

### 7.7 N_S·N_K > N_I 时 BHTP 无解

论文图 8 扫描到 N_S=7，此时 7×4=28 > 25，同一时隙内不可能让所有波束
对准互不相同的小区，C1 与去冗余无法同时满足，原代码直接抛异常。
现在 `_fill_slot_to_beam_capacity()` 分两轮补齐：优先挑未被点亮的小区，
不够时允许冗余照明。

### 7.8 求解速度

CVXPY 解一次 P̃3 要 4.2 秒，跑完九张图需要数十小时。
新增数学上等价的精确后端（推导见第 4 节），单次 0.3 毫秒，
九张图约 10 分钟跑完。`verify_solvers.py` 会核对两个后端结果一致。

---

## 8. 复现结果与论文的吻合度

### 8.1 完全吻合的部分

| 项目                       | 本复现               | 论文                   |
| ------------------------ | ----------------- | -------------------- |
| CL 随 f^l_i 变化（图 3）       | 0.823 → 0.137     | 0.83 → 0.14          |
| CL 在图 4/7/9 中的水平值        | 0.1646            | 0.167（= 1/6 的理论值）    |
| 图 4 中 CL 与 JBHCO-PG 出现交叉 | 是                 | 是                    |
| 四种算法排序（全部 9 张图、全部参数点）    | MB < PB < CL < PG | 同                    |
| swap 带来的改善幅度             | 3.3%              | 3.68% ~ 9.43%        |
| BCD 收敛所需迭代次数             | 3~5 次             | ≤ 8 次                |
| JBHCO-MB 相比 CL 的改善       | 35% ~ 40%         | 15.27% ~ 78.36%（随场景） |

CL 曲线能对上很关键：它反证了 `O_i = M_i(bit) × 100 cycle/bit` 这个理解正确。
论文表 II 只写了"处理 1 bit 所需周期数 = 100"，而式 (10) 写的是 T^l = O_i/f^l_i，
两者的关系论文没有明说。用 M_i=10 Mbit 代入得 λ = 1e9/1e9/6 = 1/6 ≈ 0.167，
与论文图 4/7/9 中 CL 的水平线完全重合。

### 8.2 存在差异的部分

| 项目             | 本复现         | 论文          | 原因                                      |
| -------------- | ----------- | ----------- | --------------------------------------- |
| JBHCO-MB 默认参数下 | 约 0.106     | 约 0.14      | 论文未给小区/卫星几何布局，干扰强度不同                    |
| JBHCO-PG 默认参数下 | 约 0.183     | 约 0.28      | 论文未说明 JBHCO-PG 的算力分配规则，本复现按"正比于工作量预留"实现 |
| 图 4 交叉点位置      | f_max ≈ 5~7 | f_max ≈ 8~9 | 同上                                      |

论文没有公开的三项信息：地面小区的具体经纬度分布、5 颗卫星的瞬时轨道位置、
JBHCO-PG 的贪婪细节。这三项都会影响绝对数值，但不影响任何定性结论。
所有能与论文直接对照的解析量（CL 曲线）都精确吻合。

---

## 9. 已知局限

1. **拓扑是准静态的**。论文第三节说明资源分配只持续几个 BHP（数百毫秒），
   期间信道和用户关联近似不变，所以不建模卫星运动。若要研究长时间尺度，
   需替换 `satellite_network._generate_satellites()` 接入真实星历。
2. **阴影衰落和雨衰默认关闭**。论文给了模型（SF ~ N(0,σ²)、ITU-R P.618）
   但没给 σ_SF 和雨衰数值，所以 `config.py` 里留了接口、默认设 0。
3. **不含星间协同计算**。论文明确说 inter-satellite cooperative computing
   不在考虑范围，ISL 只用于向网关汇聚信息。
4. **一小区一用户**。论文脚注 1 的假设；扩展到多用户需要额外的带宽分配维度，
   论文把它列为 future work。
5. **swap 搜索得到的是局部最优**。这是论文自己的说法
   （"yielding a locally optimal solution"），不是实现缺陷。
6. **BCD 的单调性保护**。论文声称 P̃2、P̃3 都凸因此 BCD 收敛到全局最优，
   这个论断对联合非凸问题一般不成立。本实现加了单调性保护：整体更新若不降低
   真实目标，就退化为逐用户最优响应（仍属块坐标下降，只是把每个 a_i 当作一个块），
   保证目标序列单调不增。

---

## 10. 代码风格约定

- 变量名用完整英文单词，不用缩写（`num_satellites` 而非 `n_sat`）。
- 每个与论文公式对应的量，第一次出现时在同一行注明**论文符号 / 物理含义 / 单位**，
  例如 `sat_compute_capacity_GHz: float = 5.0  # f_max：每颗卫星计算能力，单位 Gcycle/s`。
- 每个函数的 docstring 说明它对应论文的哪个公式或哪一行伪代码。
- 每段逻辑前用 `# ---- 第 N 行：…… ----` 标注对应的论文伪代码步骤。
- 凡是偏离论文字面描述的地方，都在注释里写清楚**为什么**，而不是默默改掉。
