# U1 L-CNN 计划

## 总目标

先为 `2D U(1)` 格点规范理论搭一条可验证、可扩展的 `L-CNN` 管线。这条管线需要做到：

1. 生成一批可信的 `U(1)` 基线配置。
2. 以统一的数据格式读取并表示这些配置。
3. 在不破坏局域规范对称性的前提下构造模型。
4. 先完成一个最小可训练的监督任务，再过渡到 diffusion / score model。
5. 在扩大规模前，先把“物理量正确”和“网络结构正确”这两件事分开验证清楚。

当前第一阶段的目标不是直接做 diffusion 采样，而是先把 `Monte Carlo + preprocess + gauge utils + 最小 L-CNN` 这四块真正接通。

## 范围

第一轮只覆盖：

- 理论：`2D U(1)` 纯规范理论
- 晶格：先固定 `L x L`
- phase 0：可信的 Monte Carlo 基线
- phase 1：数据表示与预处理
- phase 2：规范对称性工具与测试
- phase 3：最小 `L-CNN` 监督任务
- phase 4：与普通模型做对照
- phase 5：再向 score network / diffusion 过渡

第一轮暂不做：

- 非阿贝尔规范群
- 费米子
- 大规模生产级 ensemble
- 在单一 `beta` 都没验证好之前就做跨 `beta` 泛化

## 为什么从 U(1) 开始

`U(1)` 是把整条链路做正确的最合适起点，因为：

- 群结构简单，便于审查每一步定义。
- link 变量既可用 angle，也可用 `cos/sin` 表示。
- gauge transformation 很容易数值实现与测试。
- plaquette、Wilson loop、拓扑量都比较容易 debug。
- 一旦网络写错，更容易判断是表示问题、transport 问题，还是读出层问题。

## Phase 0：可信的 Monte Carlo 基线

### 0.1 理论定义

在周期边界条件下使用 Wilson 作用量：

$$
S[U] = -\beta \sum_x \mathrm{Re}\, U_{x,\square},
$$

其中 plaquette 为

$$
U_{x,\square}
=
U_{x,0}\,
U_{x+\hat 0,1}\,
U^\dagger_{x+\hat 1,0}\,
U^\dagger_{x,1}.
$$

对 `U(1)`，将 link 写成

$$
U_{x,\mu}=e^{i\phi_{x,\mu}},
$$

则 plaquette angle 为

$$
\theta_{x,\square}
=
\phi_{x,0}
+ \phi_{x+\hat 0,1}
- \phi_{x+\hat 1,0}
- \phi_{x,1},
$$

作用量变成

$$
S[\phi] = -\beta \sum_x \cos \theta_{x,\square}.
$$

### 0.2 MC 算法

第一版先使用局域 Metropolis 更新。它不是最快的，但最容易验证。

对每个 link：

1. 提议

$$
\phi'_{x,\mu} = \phi_{x,\mu} + \delta,\qquad \delta\sim \mathrm{Uniform}[-\Delta,\Delta]
$$

2. 只根据与该 link 相邻的两个 plaquette 计算局域作用量差。
3. 按

$$
P_{\mathrm{acc}} = \min(1,e^{-\Delta S})
$$

接受或拒绝。

### 0.3 初始 run 目标

starter ensemble 建议：

- `L = 8`
- `beta = 1.0`
- thermalization sweeps `>= 500`
- 相邻保存之间 sweeps `>= 20`
- 保存配置数 `>= 64`

这不是生产数据，而是后续全链路调试的可信起点。

### 0.4 输出位置

生成模块放在：

- `U1/config_gen/`

至少输出：

- 原始配置
- 元数据
- 观测量诊断结果

不要把生成逻辑散落到仓库别处。

### 0.5 必测观测量

至少测量：

- 平均 plaquette

$$
\langle P \rangle = \left\langle \frac{1}{L^2}\sum_x \cos \theta_{x,\square}\right\rangle
$$

- `W(1,1)`
- `W(2,2)`
- 拓扑电荷估计

$$
Q = \frac{1}{2\pi}\sum_x \mathrm{PV}(\theta_{x,\square})
$$

其中 `PV` 表示主值化到 `(-\pi,\pi]`
- 平均接受率
- 每次测量时的 plaquette 历史

### 0.6 sanity benchmark

使用无限体积极限下的参考值

$$
\langle P \rangle_{\mathrm{ref}} \approx \frac{I_1(\beta)}{I_0(\beta)}
$$

这不是 starter run 的有限体积严格等式，但足够作为强 sanity check。

### 0.7 结束条件

只有在以下条件都满足时，Phase 0 才算结束：

1. 平均 plaquette 与解析参考做过比较。
2. 诊断文件保存了 plaquette、Wilson loops、拓扑量和接受率。
3. 生成脚本输出清晰的 `PASS/FAIL` 总结。
4. 如果观测量明显异常，该 run 被视为不可信。

## Phase 1：数据表示与预处理

### 1.1 数据源

后续全部训练数据都基于 Phase 0 的可信 ensemble。

每个配置至少要能关联到：

- 原始 link 变量
- plaquette 场
- 平均 plaquette
- 若干 Wilson loops
- 拓扑量
- 元数据：`beta`、`L`、轨迹编号、chain 编号

### 1.2 表示选择

网络输入第一版采用：

- 每条 `U(1)` link 用两个通道表示：
  - `cos(phi_{x,\mu})`
  - `sin(phi_{x,\mu})`

这样可以避免原始 angle 在 `±pi` 附近的 branch cut 问题。

内部建议保留两套表示：

- 测量代码继续用 angle 表示
- 网络输入与保存数据用 `cos/sin`

### 1.3 张量布局

先统一采用：

- `shape = (batch, dir * 2, L, L)`

在 `2D` 时即：

- `shape = (batch, 4, L, L)`

通道顺序固定为：

- `cos_x`
- `sin_x`
- `cos_y`
- `sin_y`

### 1.4 预处理任务

预处理模块必须完成：

1. 读取原始配置。
2. 转成固定张量格式。
3. 检查周期边界与 shape 约定。
4. 按 decorrelated block 切分 train / val / test。
5. 保证可逆地回到 angle 表示。

## Phase 2：规范对称性工具与测试

在任何模型训练前，先把 gauge operation 写成独立模块并测试。

### 2.1 Gauge transformation

实现局域 `U(1)` gauge transformation：

- site field：`alpha_x`
- link 变换：

$$
U'_{x,\mu} = e^{i\alpha_x} U_{x,\mu} e^{-i\alpha_{x+\hat\mu}}
$$

在 angle 表示下，就是对 link angle 加上起点 site phase，减去终点 site phase。

### 2.2 Parallel transport helper

实现与后续 `L-CNN` 直接相关的工具：

- 最近邻 forward transport
- 最近邻 backward transport
- 短路径上的 path-ordered transport
- 由 link 构造 plaquette / Wilson loop 的 helper

### 2.3 必须通过的测试

单元测试至少验证：

- plaquette 在 gauge transformation 下不变
- Wilson loop 不变
- 拓扑量不变
- 被 transport 后的局域对象按协变方式变换
- 数值误差稳定在机器精度附近

这些测试必须在任何模型训练前先通过。

## Phase 3：最小 U(1) L-CNN

这一部分是当前计划里最关键、也是最容易写得太抽象的地方。下面把它拆成“我们到底要建什么”。

### 3.1 这一阶段真正的目标

这一阶段不是做 diffusion，也不是直接生成新配置。目标只是：

1. 用一个最小网络从局部规范结构中提取信息。
2. 让这个网络对局域 gauge transformation 保持正确的协变性或不变性。
3. 用一个简单监督任务确认这套结构真的在工作。

第一版只回答一个问题：

- “如果输入配置做一次任意随机 gauge transformation，网络输出会不会保持应有的不变性？”

### 3.2 第一版训练任务建议

最适合第一版的不是全局生成，而是局域监督任务。

优先顺序建议：

1. 预测局域 plaquette 标量场 `cos(theta_{x,\square})`
2. 预测小 Wilson loop 的局域或全局平均
3. 回归单个配置的平均 plaquette

原因很简单：

- 这些目标有明确物理意义。
- 能直接判断模型是否学到了 gauge-invariant 结构。
- 即使模型失败，也容易定位是输入、transport，还是 readout 出了问题。

### 3.3 第一版的输入到底是什么

这是原计划里最容易让人困惑的地方。

第一版不要一上来让网络直接从“任意抽象特征”开始。先把输入分成两层概念：

1. 原始输入
   - 原始配置仍然是 `U(1)` link field
   - 保存格式可为 angle 或 `cos/sin`
2. 网络内部使用的对象
   - `L-CNN` 的核心不是直接对 gauge-invariant 标量做普通卷积
   - 而是对“在 gauge transformation 下按确定方式变换的 covariant feature”做卷积

对 `U(1)`，第一版最简单的 covariant feature 应定义在 site 上，写成复数场：

$$
\psi_x \in \mathbb{C}
$$

并满足变换律

$$
\psi_x \to \psi'_x = e^{i\alpha_x}\psi_x.
$$

这样做的好处是：

- `U(1)` 情况下，site-covariant 特征就是“带一个相位”的复数。
- transport 写法非常直观。
- 后续 invariant readout 也简单，比如 `|\psi_x|^2` 或 `\psi_x^\ast \chi_x`。

### 3.4 什么是 `L-Conv`

第一版 `L-Conv` 的本质是：

1. 取 base site `x` 附近的若干邻居特征。
2. 用 link variable 把邻居特征 parallel transport 到同一个 base site `x`。
3. 把这些已经被搬运到同一个 site 的量做线性组合。

如果 `psi_x` 是 site-covariant 特征，满足 `psi_x -> g_x psi_x`，其中 `g_x = e^{i\alpha_x}`，那么：

- forward 邻居 `x + \hat\mu` 搬回 `x`：

$$
\tilde\psi^{(+\mu)}_x = U_{x,\mu}\,\psi_{x+\hat\mu}
$$

- backward 邻居 `x - \hat\mu` 搬回 `x`：

$$
\tilde\psi^{(-\mu)}_x = U^\dagger_{x-\hat\mu,\mu}\,\psi_{x-\hat\mu}
$$

这两个量都在 gauge transformation 下按

$$
\tilde\psi_x \to g_x \tilde\psi_x
$$

协变，因此它们已经处在同一个“参考点”上，可以安全相加。

所以最小 `L-Conv` 可以写成：

$$
h^{\mathrm{out}}_x
=
w_0 \psi_x
+ \sum_{\mu}
w^{(+)}_\mu \tilde\psi^{(+\mu)}_x
+ \sum_{\mu}
w^{(-)}_\mu \tilde\psi^{(-\mu)}_x.
$$

这里的权重 `w` 是全局共享参数，不依赖 site。

### 3.5 为什么这不是普通 CNN

普通 CNN 会直接把邻居像素加权相加，但这里不能直接这么做，因为：

- 邻居 site 上的 covariant feature 属于不同的局域 gauge frame。
- 不先 transport 到同一个 base site，直接相加在物理上没有意义。
- `L-Conv` 的关键不是“邻域求和”，而是“先对齐局域规范参考系，再求和”。

这就是 `L-CNN` 与普通 CNN 的本质区别。

### 3.6 第一版非线性怎么选

普通 `ReLU` 直接作用在复数 covariant feature 上不自然。第一版建议用只依赖模长的 equivariant 非线性，例如：

$$
f(\psi_x) = \psi_x \cdot \sigma(|\psi_x|)
$$

或者使用 `modReLU` 风格：

$$
f(\psi_x) =
\begin{cases}
\dfrac{\psi_x}{|\psi_x|}\,\mathrm{ReLU}(|\psi_x| + b), & |\psi_x| > 0 \\
0, & |\psi_x| = 0
\end{cases}
$$

因为它只改变模长，不改变相位，所以仍保持协变性。

### 3.7 什么是 `L-Bilin`

`L-Bilin` 是把多个协变通道做局域双线性组合。

若 `h_x^{(a)}` 和 `h_x^{(b)}` 都按相同方式协变：

$$
h_x^{(a)} \to g_x h_x^{(a)},\qquad
h_x^{(b)} \to g_x h_x^{(b)},
$$

那么下面这个量天然 gauge-invariant：

$$
I_x^{(a,b)} = \left(h_x^{(a)}\right)^\ast h_x^{(b)}.
$$

因此 `L-Bilin` 在 `U(1)` 第一版里可以理解为：

- 先得到若干 site-covariant 复数通道
- 再在同一 site 上做 `conj(h_a) * h_b`
- 得到实数或复数的 gauge-invariant 局域标量特征

如果目标是预测标量物理量，`L-Bilin` 往往是从 covariant 表示进入 invariant readout 的最直接桥梁。

### 3.8 第一版 readout 怎么做

读出层分成两种：

1. 局域标量输出
   - 用于预测每个 site 上的 plaquette 或局域 invariant
   - 输出 shape 可为 `(batch, 1, L, L)`
2. 全局标量输出
   - 用于预测平均 plaquette、全局 Wilson loop
   - 对局域 invariant feature 先做 spatial average，再接小 MLP 或线性层

第一版建议先做局域输出，再加一个简单的空间平均来得到全局输出。

### 3.9 第一版最小结构

建议的最小网络结构如下：

1. 输入：原始 links 与由 links 构造的局域对象
2. 初始 covariant feature：一个简单的 site-covariant 复数场表示
3. 一层 `L-Conv`
4. 一个 equivariant 非线性
5. 可选第二层 `L-Conv`
6. `L-Bilin` / invariant contraction
7. 标量 readout head

第一版不要深。目标是调试，不是追求表达能力。

### 3.10 第一版到底要验证什么

这一阶段的验证至少包括三组：

1. 结构验证
   - 对输入做随机 gauge transformation
   - 检查 covariant 中间特征是否按预期变换
   - 检查 invariant 输出是否保持不变
2. 学习验证
   - 在简单监督任务上 loss 能下降
   - 验证集误差能稳定收敛
3. 对照验证
   - 同样任务下，对比一个 symmetry-blind baseline
   - 检查 gauge-transformed validation 样本上，普通模型是否波动更大

### 3.11 第一版应当输出哪些日志

训练和验证脚本至少记录：

- train / val loss
- target observable 的 `MAE` 或 `RMSE`
- gauge robustness 指标：
  - `|f(U) - f(U^g)|`
- seed、学习率、batch size、模型宽度
- 输入数据路径与版本

### 3.12 Phase 3 的完成标准

只有在以下条件满足时，Phase 3 才算完成：

1. 有一个可以运行的最小 `L-CNN` 实现。
2. 中间 covariant 特征与最终 invariant 输出都通过 gauge 检查。
3. 在简单监督任务上成功训练。
4. 对随机 gauge transformation 后的样本，输出稳定。
5. 相比普通 CNN 或浅层基线，显示出明确的 gauge robustness 优势。

## Phase 4：与普通模型对照

`L-CNN` 不能只证明“能跑”，还要证明它不是无意义复杂化。

至少和一个普通基线比较：

- plain CNN
- 或浅层 MLP

在相同训练集和目标下比较：

- 预测误差
- gauge robustness
- 数据效率
- 对 gauge-transformed 验证样本的稳定性

## Phase 5：把 L-CNN 接到 score network

只有在监督任务阶段确认结构正确后，才开始把它改造成 score model。

### 5.1 训练目标

网络输入：

- 带噪配置
- 噪声等级或时间步嵌入

网络输出：

- 与输入局域结构匹配的 score field

对 `U(1)` 第一版，可以先在数值上更稳定的欧氏变量表示里做 denoising score matching。

### 5.2 时间嵌入

加入 noise level / diffusion time 的条件分支，使模型能在不同噪声尺度下工作。

### 5.3 验证

至少检查：

- 不同噪声级别的 score 误差
- 去噪一致性
- 未校正 reverse process 的样本质量
- score field 在 gauge transformation 下的行为是否符合预期

## Phase 6：采样管线

### 6.1 先做未校正逆过程

先运行 reverse diffusion 或 annealed Langevin，不要立刻加 Metropolis 校正。

先看：

- 局域观测量分布
- 轨迹稳定性
- 小噪声阶段的失稳模式

### 6.2 再加校正

在未校正过程足够稳定后，再加入：

- `MALA`
- 或带 Metropolis 校正的 annealed Langevin

### 6.3 物理验证

只有当生成样本能重现实验基线中的以下量时，采样阶段才算可信：

- plaquette 分布
- 若干 Wilson loops
- 拓扑相关量
- 接受率与自相关诊断

## Phase 7：泛化研究

只有在单一设置跑通后，才做：

- 新的 lattice size
- 邻近 `beta`
- 从一个 `L` 向另一个 `L` 的迁移

这些实验应与主线隔离，避免把失败原因混在一起。

## 工程检查单

- 定义唯一的 canonical configuration format
- 保持 preprocess / postprocess 可逆
- 先写 gauge transformation tests
- 把 observables 放在共享测量模块中
- 把训练脚本和采样脚本分开
- 完整记录超参数与随机种子
- checkpoint 同时保存验证观测量

## 推荐里程碑

### 里程碑 1

可信的 `U(1)` Monte Carlo 生成器、诊断与 gauge tests 全部通过

### 里程碑 2

最小 `L-CNN` 能预测 plaquette / 小 Wilson loop，且对 gauge transformation 稳定

### 里程碑 3

带时间条件的 `L-CNN` 作为 score network 成功训练

### 里程碑 4

reverse diffusion 能生成合理的未校正样本

### 里程碑 5

校正后的采样能重现实验基线观测量

## 现在的立即下一步

按当前仓库状态，最直接的后续动作应是：

1. 保持 `U1/config_gen/`、`U1/preprocess/`、`U1/gauge_utils/` 三部分可复现。
2. 定义第一版 `L-CNN` 所需的 covariant feature 类型。
3. 实现最近邻 transport 与最小 `L-Conv`。
4. 实现 invariant readout。
5. 先做一个 gauge-check 脚本，验证中间层协变、输出层不变。
6. 选一个最简单监督任务，例如局域 plaquette 回归。
7. 训练最小模型，再与普通基线对照。

## 实用原则

对 `U(1)` 来说，最主要的风险不是模型太小，而是：

- 数据表示偷偷破坏了规范结构
- transport 的定义写错
- 把不同局域 gauge frame 下的量直接相加
- 用了不合适的非线性或读出层

所以整个计划必须优先保证“结构正确可检查”，然后才是“模型更强、更深、能生成样本”。
