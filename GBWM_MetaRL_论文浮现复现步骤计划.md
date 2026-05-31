# 《A Meta Reinforcement Learning Approach to Goals-Based Wealth Management》浮现/复现步骤计划

本文档用于把论文从“读懂”推进到“可复现、可解释、可扩展”的工程计划。当前工作区论文文件为 `A Meta Reinforcement Learning Approach to GBWM.pdf`，论文元信息显示 arXiv ID 为 `2605.02300v1`，标题为 *A Meta Reinforcement Learning Approach to Goals-Based Wealth Management*。外部公开入口可参考 [arXiv:2605.02300](https://arxiv.org/abs/2605.02300)、[arXiv DOI](https://doi.org/10.48550/arXiv.2605.02300) 和期刊版 DOI `10.1016/j.jfds.2026.100186`。

## 1. 目标定义

### 1.1 最终目标

复现论文提出的 MetaRL 方法，用一个预训练的双智能体 PPO 模型，在新的 Goals-Based Wealth Management（GBWM）问题上做零样本推理，输出：

1. 当前目标是否达成的决策 `g(t)`。
2. 下一期投资组合选择 `p(t)`。
3. 与 Dynamic Programming（DP）基准相比的运行时间、决策热力图和 RL-Efficiency。

论文核心结果包括：

1. 训练一个 MetaRL 模型，而不是为每个新投资者问题重新训练。
2. 使用 26 维归一化状态输入，覆盖不同财富、时间期限、目标、注入资金和投资前沿。
3. 使用两个 PPO agent：`GoalAgent` 决定是否支付当前目标，`PortfolioAgent` 决定下一期投资组合。
4. 在 66 个测试场景上，RL-Efficiency 平均约 `0.978`。
5. 单步纯推理比 DP 快 100 倍以上。
6. 进一步扩展到 concurrent/partial goals 和 stochastic inflation。

### 1.2 复现层级

建议按三级递进，避免一开始就追求完整论文级复现：

| 层级 | 目标 | 通过标准 |
|---|---|---|
| Level 0：数学与环境复刻 | 实现 GBWM 环境、状态、动作、奖励、随机场景生成 | 能跑单个 episode，并输出财富、目标达成、组合选择轨迹 |
| Level 1：最小可用 MetaRL | 训练双 PPO agent，对少量测试用例产生合理策略 | 决策热力图与 DP 方向一致，RL-Efficiency 初步可计算 |
| Level 2：论文结果复现 | 复现 66 个测试场景、运行时间、RL-Efficiency、有效前沿鲁棒性 | 平均 RL-Efficiency 接近论文，误差来源可解释 |
| Level 3：扩展验证 | concurrent/partial goals、stochastic inflation | 能展示 MetaRL 比 DP 更容易扩展状态维度 |

## 2. 已从论文提取的关键事实

### 2.1 GBWM 问题参数

单个投资者问题由以下参数定义：

1. 时间期限 `T`，论文实验中每步 `h = 1` 年。
2. 初始财富 `W(0)`。
3. 未来注入资金 `I(t)`。
4. 目标成本 `C(t)`。
5. 目标效用 `U(t)`。
6. 可选投资组合数量 `P`。
7. 每个组合的期望收益 `mu_p` 和波动率 `sigma_p`。

### 2.2 动作与财富演化

每个时间步先做目标决策，再做组合决策：

1. `GoalAgent` 输出连续动作 `a_g(t) in [0, 1]`。
2. 若 `a_g(t) >= 0.5` 且 `W(t-) >= C(t)`，则 `g(t) = 1`，目标被达成。
3. 达成目标后财富更新：`W(t+) = W(t-) - g(t) * C(t)`。
4. `PortfolioAgent` 输出连续动作 `a_p(t) in [0, 1]`。
5. 将 `a_p(t)` 单调映射到离散组合 `p(t) in {0, ..., P-1}`。
6. 财富按 GBM 演化：

```text
W((t+1)-) = W(t+) * exp((mu_p - 0.5 * sigma_p^2) * h + sigma_p * Z * sqrt(h)) + I(t+1)
```

### 2.3 26 维状态空间

论文用 `K = 7` 个未来时间块：

```text
L = [[0], [1], [2], [3], [4,5], [6,7,8,9], [10:]]
```

状态维度为 `5 + 3K = 26`：

| 状态 | 维度 | 含义 |
|---|---:|---|
| `t_norm` | 1 | 当前时间除以时间期限 `T` |
| `W_min` | 1 | 当前财富除以悲观、保守折现下的未来目标成本和 |
| `W_max` | 1 | 当前财富除以乐观、激进折现下的未来目标成本和 |
| `U_agg` | 7 | 当前及未来目标效用按时间块聚合，并按效用总和归一化 |
| `C_min` | 7 | 乐观、激进折现后的目标成本按时间块聚合归一化 |
| `C_max` | 7 | 悲观、保守折现后的目标成本按时间块聚合归一化 |
| `g_sim` | 1 | 前向模拟给出的“当前目标应否达成”证据 |
| `p_sim` | 1 | 前向模拟给出的“应选择多激进组合”证据 |

### 2.4 奖励设计

奖励由外在奖励和内在奖励组成：

1. 外在奖励对应实际达成目标的归一化效用。
2. 最终时刻额外加入小的剩余财富奖励，使模型知道“钱多一点更接近最后目标”。
3. 内在奖励惩罚动作偏离 `g_sim` 和 `p_sim`。
4. 内在奖励系数 `rho` 从 `1` anneal 到 `0.25`。
5. 效用不做时间折现，RL discount rate 设为 `1.0`。

### 2.5 网络与训练配置

论文附录 B 给出的训练配置：

| 项目 | 配置 |
|---|---|
| 算法 | 自定义双 PPO |
| Agent | `GoalAgent` 和 `PortfolioAgent` |
| Actor hidden layers | `256 -> 64 -> 16` |
| Critic hidden layers | `64 -> 16` |
| Hidden activation | `tanh` |
| Actor output | `sigmoid`，输出 `[0,1]` |
| Critic output | linear |
| Learning rate | `1e-4` |
| PPO clip | `0.2` |
| Discount rate | `1.0` |
| Epochs | `1000` |
| Episodes per epoch | `500` |
| Random seeds | `5` |
| Training time | 论文报告约 4 小时，AWS `c7i.24xlarge` |
| 实现依赖 | Python 3.11、torch 2.1、numba 0.60、gym |

### 2.6 训练场景生成

每个 epoch 生成一个新场景，并用 500 个随机 episode 求解。核心规则：

1. `T` 从 `{5, ..., 50}` 均匀抽样。
2. 目标数量 `N_G` 按论文给定离散分布抽样，且最终时刻 `T` 一定有目标。
3. 每个非零目标时刻生成成本和效用：

```text
C(t) = 100 * u_1 * 1.03^t
U(t) = 0.3 * C(t) / 1.03^t + 25 * u_2
```

4. 初始财富在两个折现目标成本和之间均匀抽样。
5. 训练使用 5 个随机种子，推理时取 5 个模型输出动作的 median。

### 2.7 测试与论文指标

论文测试集：

1. 66 个手工设计场景。
2. 时间期限 `T` 范围为 3 到 100 年，均值约 38。
3. 目标数量范围为 1 到 60，均值约 16。
4. `P = 15` 个投资组合。
5. 33 个测试用例无注入资金，另外 33 个是对应的有注入资金版本。

主要指标：

| 指标 | 说明 |
|---|---|
| 单步决策时间 | 比较 RL inference 与 DP backward pass |
| 决策热力图 | 在 `(t, W)` 网格上比较 `g(t)` 和 `p(t)` |
| Expected attained utility | 通过 10,000 条 Monte Carlo 轨迹估计 |
| RL-Efficiency | RL attained utility / DP attained utility |
| Efficient frontier robustness | 换不同资本市场有效前沿后重新评估效率 |

论文报告的关键数值：

1. 纯组合决策 RL inference 均值约 `9.277 ms`。
2. 组合和目标同时决策 RL inference 均值约 `20.94 ms`。
3. DP backward pass 均值约 `2198 ms`。
4. 66 个测试场景平均 RL-Efficiency 约 `0.978`。
5. 论文还测试了 2022-2023、2012-2013、2020-2023、1988-2023、1997-2023 等有效前沿变化。

## 3. 准备阶段

### 3.1 推荐项目结构

建议在当前工作区创建如下工程结构：

```text
gbwm-metarl/
  README.md
  pyproject.toml
  src/
    gbwm/
      __init__.py
      config.py
      environment.py
      state_features.py
      scenario_generation.py
      efficient_frontier.py
      dp_baseline.py
      ppo.py
      agents.py
      inference.py
      metrics.py
      plotting.py
  experiments/
    00_smoke_test.py
    01_train_metarl.py
    02_eval_66_cases.py
    03_frontier_robustness.py
    04_partial_goals.py
    05_stochastic_inflation.py
  data/
    test_cases_66.yaml
    efficient_frontiers.yaml
  notebooks/
    paper_reading_notes.ipynb
  outputs/
    checkpoints/
    figures/
    tables/
    logs/
  tests/
    test_environment.py
    test_state_features.py
    test_scenario_generation.py
    test_dp_baseline.py
    test_metrics.py
```

### 3.2 环境依赖

优先用 Python 3.11，贴近论文实现：

```text
python = ">=3.11,<3.12"
torch = "2.1.*"
numpy
scipy
pandas
numba = "0.60.*"
gymnasium 或 gym
matplotlib
seaborn
pyyaml
tqdm
pytest
```

如果本地没有 Python，可使用 Codex 桌面内置 runtime。当前已验证内置 Python 可用，且已用 `pypdf` 成功提取论文文本。

### 3.3 可调用插件与用途

| 插件/能力 | 使用时机 | 产出 |
|---|---|---|
| Codex 内置 Python/PDF 能力 | 提取论文文本、检查章节、生成数据表草稿 | 论文结构、算法参数、测试用例 |
| Browser 插件 | 本地 Web 可视化、打开训练 dashboard、检查热力图页面 | 可交互查看图表 |
| Spreadsheets 插件 | 整理 66 个测试用例、实验指标、运行时间对比 | `.xlsx` 实验记录表 |
| GitHub 插件 | 如果后续发现或创建代码仓库，用于 issue、PR、CI | 版本化复现工程 |
| Documents 插件 | 如果需要把复现报告转成 Word/PDF | 可交付报告 |

### 3.4 外部资料核查

1. 优先以当前 PDF 为准。
2. 使用 arXiv 页面确认版本、日期、作者和 DOI。
3. 检查 ScienceDirect 页面是否有期刊版更新。
4. 搜索是否存在官方 GitHub 或作者补充材料；若存在，优先比对实现细节。
5. 若无官方代码，需要从论文附录 A、B、C 手工复现核心算法。

## 4. 详细实施步骤

### 阶段 0：论文拆解与规格冻结

目标：把论文内容转换成工程规格。

步骤：

1. 建立 `paper_spec.md`，记录所有公式、变量和符号。
2. 建立变量表，统一 `t`、`T`、`W(t-)`、`W(t+)`、`C(t)`、`U(t)`、`I(t)`、`mu`、`sigma`、`P`。
3. 明确数组下标从 `0` 还是 `1` 开始，工程实现中建议统一为 `0..T`，并写清楚与论文表达的转换。
4. 提取附录 C 的 66 个测试用例，录入 `data/test_cases_66.yaml`。
5. 提取或重建 baseline efficient frontier 的 15 个组合参数。
6. 冻结第一个复现范围：先只做“每期至多一个 all-or-nothing 目标”。

验收标准：

1. 每个核心公式都能在工程文档中定位。
2. 66 个测试用例至少完成 10 个手工录入和校验。
3. 环境输入、状态输入、动作输出、奖励定义没有悬空变量。

### 阶段 1：GBWM 环境实现

目标：实现论文式环境，不引入 PPO。

步骤：

1. 实现 `Scenario` 数据结构，包含 `T, W0, C, U, I, mu, sigma, P`。
2. 实现 `GBWMEnv.reset()`，返回初始财富和时间。
3. 实现 `goal_decision(a_g, W, C_t, a_thresh=0.5)`。
4. 实现 `portfolio_decision(a_p, P)`，把 `[0,1]` 映射到 `0..P-1`。
5. 实现 GBM 财富演化。
6. 实现 episode 记录：时间、财富、目标动作、目标决策、组合动作、组合决策、达成效用、随机冲击 `Z`。
7. 写 smoke test：固定 `Z=0`、固定动作，确认财富路径可手算。

验收标准：

1. 环境能跑完单个 episode。
2. `a_g < 0.5` 时不达成目标。
3. 财富不足时即使 `a_g >= 0.5` 也不达成目标。
4. `a_p=0` 映射到最保守组合，`a_p=1` 映射到最激进组合。
5. 注入资金只在对应未来时点加入。

### 阶段 2：状态特征实现

目标：实现 26 维 MetaRL 输入。

步骤：

1. 实现 `discount_vec(C[t:], p, z, mu, sigma, h=1)`。
2. 实现 `discount_sum(...)`。
3. 实现 `aggregate(vector, L)`，默认 `L = [[0], [1], [2], [3], [4,5], [6,7,8,9], [10:]]`。
4. 实现 `t_norm = t / T`。
5. 实现 `W_min` 和 `W_max`：
   - `W_min` 用保守组合 `p=0` 和悲观冲击 `z=-1`。
   - `W_max` 用激进组合 `p=P-1` 和乐观冲击 `z=1`。
6. 实现 `U_agg`、`C_min`、`C_max`。
7. 实现 `g_sim`：
   - 用 11 个标准正态 CDF 等分中点。
   - 比较强制优先当前目标与跳过当前目标的近似效用。
   - 用 logistic 转成 `[0,1]`。
8. 实现 `p_sim`：
   - 固定每个组合做近似前向模拟。
   - 取近似效用最大组合，并除以 `P-1`。
9. 将所有状态拼接成长度 26 的 `float32` 向量。

验收标准：

1. 每个状态向量长度恒为 26。
2. 对不同 `T`、不同剩余年限，状态长度不变。
3. 大部分状态值位于 `[0,1]` 或在合理范围内。
4. 单元测试覆盖无未来目标、只有最终目标、当前财富极低、当前财富极高四类边界。

### 阶段 3：训练场景生成

目标：复刻论文 Algorithm 5。

步骤：

1. 实现 `generate_scenario(mu, sigma, seed)`。
2. `T` 从 5 到 50 均匀抽样。
3. 按论文分布抽样 `N_G`：

```text
p(1)=0.22, p(2)=0.15, p(3)=0.12, p(4)=0.10, p(5)=0.06,
p(6)=0.05, p(7)=0.04, p(8)=0.03, p(9)=0.02, p(10)=0.01,
p(T)=0.20
```

4. 保证最终时刻 `T` 有一个目标。
5. 生成目标成本和效用。
6. 生成初始财富区间。
7. episode 内初始财富再乘以 `Uniform(0.8, 1.2)` 做随机化。

验收标准：

1. 抽样 10,000 次后，`T` 分布接近均匀。
2. `N_G` 分布接近论文配置。
3. 最终时刻目标存在率为 100%。
4. 训练场景不会生成负成本、负效用或空目标。

### 阶段 4：DP 基准实现

目标：得到可对照的最优策略和 value function。

步骤：

1. 先实现简化 DP：状态为 `(t, W_grid)`。
2. 设计财富网格，覆盖测试用例常见财富区间。
3. 对每个时间步做 backward pass。
4. 对每个财富格点枚举：
   - `g in {0,1}`，且财富不足时禁止 `g=1`。
   - `p in {0, ..., P-1}`。
5. 对 GBM 下一期财富做数值积分或离散化近似。
6. 保存最优 `g`、最优 `p` 和 value。
7. 输出 DP 决策热力图。

验收标准：

1. 对简单单目标场景，DP 策略符合直觉。
2. DP value 随财富非下降。
3. 对同一场景重复运行结果一致。
4. 可为至少 3 个论文测试用例生成热力图。

注意：

论文 DP 细节未完全展开，复现时要明确记录自己的积分、网格和插值方案。DP 是评估基准，误差会直接影响 RL-Efficiency，因此需要单独验证。

### 阶段 5：双 PPO 实现

目标：实现论文中的 `GoalAgent` 和 `PortfolioAgent`。

步骤：

1. 实现 actor-critic 网络类。
2. Actor：
   - 输入 26。
   - hidden layers `256, 64, 16`。
   - `tanh` 激活。
   - `sigmoid` 输出连续动作。
3. Critic：
   - 输入 26。
   - hidden layers `64, 16`。
   - `tanh` 激活。
   - linear 输出 value。
4. 实现 PPO buffer，分别记录两个 agent 的状态、动作、logprob、reward-to-go、advantage。
5. 按论文逻辑处理奖励：
   - `GoalAgent` 包含当前时刻外在奖励和内在奖励。
   - `PortfolioAgent` 不获得当前目标达成的外在奖励，只获得之后外在奖励和自己的内在奖励。
6. 实现 PPO clipped objective。
7. 实现每个 epoch 后统一更新网络。
8. 训练 5 个 seed 的模型。
9. 推理时对 5 个模型输出取 median。

验收标准：

1. 单 seed 能训练 10 个 epoch 且 loss 不为 NaN。
2. 随机小场景下平均 episode reward 随训练上升。
3. `a_g` 和 `a_p` 始终在 `[0,1]`。
4. 5 seed 推理聚合可用。

### 阶段 6：最小复现训练

目标：在本地资源可承受范围内跑通端到端。

建议先不要直接跑 1000 epoch x 500 episode x 5 seeds。先分三档：

| 档位 | 配置 | 目的 |
|---|---|---|
| smoke | 2 seeds, 5 epochs, 20 episodes | 检查代码路径 |
| mini | 3 seeds, 50 epochs, 100 episodes | 看学习趋势 |
| paper-like | 5 seeds, 1000 epochs, 500 episodes | 尝试论文级结果 |

验收标准：

1. smoke 档 10 分钟内完成。
2. mini 档能生成训练曲线。
3. paper-like 档能保存 checkpoint、日志和配置快照。

### 阶段 7：66 个测试用例复现

目标：生成论文主表和关键图。

步骤：

1. 完整录入 Appendix C 的 66 个测试场景。
2. 对每个场景运行 DP backward pass。
3. 对每个场景运行 RL inference 热力图。
4. 每个场景跑 10,000 条 Monte Carlo 轨迹。
5. 用同一批随机轨迹比较 RL 策略和 DP 策略。
6. 计算 RL-Efficiency。
7. 汇总均值、标准差、最小值、四分位数、最大值。
8. 复刻论文 Table 1、Table 2。
9. 选择 case 20 和 case 57 复刻决策热力图。

验收标准：

1. 66 个场景全部有结果。
2. Monte Carlo 随机种子固定，可重复。
3. 平均 RL-Efficiency 与论文 `0.978` 的差距可解释。
4. 单步推理时间和 DP 时间都用同一硬件测量，不能直接与论文 AWS 结果混用。

### 阶段 8：有效前沿鲁棒性测试

目标：验证论文的资本市场 regime shift 结论。

步骤：

1. 先使用 baseline efficient frontier。
2. 构造或提取另外 5 条有效前沿：
   - 2022-2023。
   - 2012-2013。
   - 2020-2023。
   - 1988-2023。
   - 1997-2023。
3. 对每条有效前沿调整初始财富，使 DP 最优效用 / 总目标效用均值保持在 `0.63-0.64`。
4. 不重新训练 MetaRL，直接测试迁移表现。
5. 汇总每条有效前沿的平均 RL-Efficiency。

验收标准：

1. 每条有效前沿包含 15 个组合。
2. 初始财富缩放过程有日志。
3. 输出对应论文 Table 3 的结果表。

### 阶段 9：扩展一：Concurrent 与 Partial Goals

目标：展示 MetaRL 对更复杂目标结构的扩展。

步骤：

1. 扩展目标数据结构，从每年一个目标改成每年多个候选目标。
2. 支持 partial goals：每个目标有多个成本-效用版本。
3. 重新定义 `GoalAgent` 输出或动作映射：
   - 可以先实现论文式扩展。
   - 若论文细节不足，明确记录自己的建模假设。
4. 构造 CP1-CP4 四个测试用例。
5. 比较 DP time、RL inference time、DP/RL time ratio、RL-Efficiency。

验收标准：

1. 至少 CP1 能完整运行。
2. 能输出类似论文 Table 4 的结果。
3. 能解释 RL 在复杂目标组合下的速度优势。

### 阶段 10：扩展二：Stochastic Inflation

目标：验证 DP 维度灾难与 MetaRL 状态扩展能力。

步骤：

1. 给环境加入通胀状态。
2. 扩展目标成本折现逻辑，参考论文 Algorithm 6。
3. 将 DP 状态从 `(t, W)` 扩成 `(t, W, inflation_state_1, inflation_state_2)` 的概念验证版本。
4. 记录 DP 网格规模如何爆炸。
5. 重新训练含通胀状态的 MetaRL。
6. 对比训练/推理耗时与策略表现。

验收标准：

1. 能跑一个 stochastic inflation 场景。
2. 能定量说明 DP 为什么不可行或成本极高。
3. MetaRL 扩展状态后仍能训练和推理。

## 5. 实验记录模板

每次实验必须保存：

```yaml
experiment_id:
date:
git_commit:
paper_version: arXiv 2605.02300v1
hardware:
python_version:
torch_version:
numba_version:
random_seed:
training:
  seeds:
  epochs:
  episodes_per_epoch:
  learning_rate:
  clip:
  discount:
  rho_start:
  rho_end:
scenario:
  source:
  T:
  P:
  number_of_goals:
  number_of_infusions:
evaluation:
  mc_paths:
  wealth_grid_size:
  dp_runtime:
  rl_runtime:
  rl_efficiency:
outputs:
  checkpoint:
  tables:
  figures:
notes:
```

## 6. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 论文没有开源代码 | 需要手工复现大量细节 | 先复刻核心环境，再逐步补 DP 和 PPO |
| DP 基准实现偏差 | RL-Efficiency 不可信 | 给 DP 单独写测试，记录网格和积分方案 |
| 有效前沿数据不完整 | Table 3 难以严格复现 | 先用合成有效前沿做结构复现，再找真实数据 |
| 本地算力不足 | 1000 epoch x 5 seeds 时间过长 | 分 smoke、mini、paper-like 三档 |
| `g_sim/p_sim` 计算重 | 训练很慢 | 用 numba 加速，并缓存可复用中间结果 |
| PPO 自定义实现不稳定 | 训练曲线发散 | 先用小场景做 reward sanity check，再扩大 |
| 下标和时点混乱 | 财富和目标错位 | 明确 `t-`、`t+`、`t+1` 的实现约定 |

## 7. 建议里程碑

| 周期 | 里程碑 | 交付物 |
|---|---|---|
| Day 1 | 论文规格冻结 | `paper_spec.md`、变量表、公式表 |
| Day 2-3 | 环境与状态特征 | `environment.py`、`state_features.py`、单元测试 |
| Day 4 | 场景生成与测试用例录入 | `scenario_generation.py`、部分 `test_cases_66.yaml` |
| Day 5-6 | DP baseline | `dp_baseline.py`、简单场景 heatmap |
| Day 7-9 | 双 PPO smoke/mini 训练 | checkpoint、训练曲线 |
| Day 10-12 | 66 cases 评估 | Table 1/2 风格结果 |
| Day 13 | 有效前沿鲁棒性 | Table 3 风格结果 |
| Day 14+ | partial goals / inflation 扩展 | 扩展实验报告 |

## 8. 当前下一步清单

优先从最小闭环开始：

1. 创建 `gbwm-metarl/` 项目骨架。
2. 写 `Scenario` 数据结构。
3. 实现 `goal_decision`、`portfolio_decision`、GBM 财富更新。
4. 实现 26 维状态特征。
5. 用一个手工场景跑通 episode。
6. 再开始 DP 和 PPO。

## 9. 完成判定

如果目标是“论文级复现”，完成标准应为：

1. 代码能从零训练 MetaRL。
2. 66 个测试用例可重复评估。
3. 能输出 RL inference 与 DP 的运行时间表。
4. 能输出 RL-Efficiency 汇总表。
5. 能生成至少两个案例的决策热力图。
6. 能说明与论文数值差异来自硬件、网格、随机种子、有效前沿、DP 近似或实现细节。
7. 所有实验配置、随机种子、checkpoint、图表和日志可追踪。

如果目标是“理解并准备复现”，完成标准应为：

1. 本文档中的阶段 0 到阶段 3 完成。
2. 环境和状态特征有单元测试。
3. smoke episode 能跑通。
4. 后续 DP/PPO 任务拆分清楚，可以逐项实现。
