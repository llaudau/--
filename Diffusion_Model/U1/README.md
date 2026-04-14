# U1 模块说明

## 当前已经完成的工作

这个 `U1` 文件夹目前已经完成了整条工作流的第一阶段，具体包括：

- 一份 `U(1)` 的 `L-CNN` / diffusion 准备计划
- 一个可信的 `2D U(1)` Monte Carlo 初始生成器
- 一个可复用的可观测量测量脚本
- 一个独立的 `cos/sin` 预处理模块
- 一个独立的规范对称性工具模块，以及相应的不变性 / 协变性测试
- 一个最小 `U(1) L-CNN` 的 `numpy` 参考实现
- 一批已经生成并通过基础检查的 starter ensemble

当前目标还不是直接训练 diffusion model，而是先把后续 `preprocess`、`gauge utils` 和 `L-CNN` 依赖的基础数据管线做正确、做可验证。

## 当前目录结构

- `plan/u1_lcnn_plan.md`
  - `U(1)` 的分阶段计划
  - 包含 Monte Carlo 阶段的详细方案
  - 写明了 Wilson 作用量、MC 策略、可观测量和必须做的检查

- `config_gen/generate_u1_mc.py`
  - 用局域 Metropolis 更新生成 `2D U(1)` 纯规范场配置
  - 把配置和诊断结果写到 `config_gen/output/`

- `config_gen/measure_observables.py`
  - 读取已经保存的 ensemble
  - 测量规范不变量
  - 用平均 plaquette 的解析基准做 sanity check

- `config_gen/output/u1_mc_configs.npz`
  - 生成出的原始配置

- `config_gen/output/u1_mc_diagnostics.json`
  - 由生成脚本直接写出的诊断结果

- `config_gen/output/u1_mc_measured.json`
  - 由独立测量脚本重新测量后写出的诊断结果

- `preprocess/preprocess_u1.py`
  - 读取原始 `U(1)` ensemble
  - 把 link angle 转成 `cos/sin` 特征表示
  - 检查从 `cos/sin` 回到 angle 的可逆性
  - 按 block 切分训练 / 验证 / 测试集

- `preprocess/output/u1_features_all.npz`
  - 全量预处理后的特征数据

- `preprocess/output/u1_train.npz`
- `preprocess/output/u1_val.npz`
- `preprocess/output/u1_test.npz`
  - 用于训练的切分数据集

- `preprocess/output/preprocess_summary.json`
  - 预处理元数据和可逆性检查结果

- `gauge_utils/gauge_utils_u1.py`
  - `U(1)` 的局域 gauge transformation 工具
  - plaquette、Wilson loop、拓扑量和 transport helper

- `gauge_utils/test_gauge_utils_u1.py`
  - 数值不变性 / 协变性测试脚本

- `gauge_utils/gauge_utils_test_results.json`
  - 当前测试运行保存下来的结果

- `lcnn/u1_lcnn_numpy.py`
  - 最小 `U(1)` `L-CNN` 参考实现
  - 包括 site-covariant feature、最近邻 transport、`L-Conv`、equivariant 非线性和 invariant readout

- `lcnn/validate_u1_lcnn.py`
  - 检查最小 `L-CNN` 前向传播中的协变性 / 不变性是否成立

- `lcnn/u1_lcnn_torch.py`
  - `PyTorch` 版最小 `U(1)` `L-CNN`
  - 直接从 link 的 `cos/sin` 特征构造初始 covariant seed，并实现可训练的 `L-Conv` / readout

- `lcnn/validate_u1_lcnn_torch.py`
  - `PyTorch` 版 `L-CNN` 的 gauge 协变 / 不变性验证脚本

- `lcnn/train_local_plaquette_lcnn.py`
  - 用真实预处理数据训练 `PyTorch L-CNN` 做局域 plaquette 回归

- `lcnn/train_local_plaquette_baseline.py`
  - 对照用的普通 `PyTorch CNN` baseline

## 当前采用的物理设定

目前使用的是：

- 理论：`2D U(1)` 纯规范理论
- 作用量：Wilson 作用量

$$
S[\phi] = -\beta \sum_x \cos \theta_{x,\square}
$$

并采用周期边界条件。

link 变量用角度表示：

$$
U_{x,\mu}=e^{i\phi_{x,\mu}}.
$$

plaquette angle 写成：

$$
\theta_{x,\square}
=
\phi_{x,0}
+\phi_{x+\hat 0,1}
-\phi_{x+\hat 1,0}
-\phi_{x,1}.
$$

## 当前已经实现的 Monte Carlo 方法

当前生成器使用的是局域 Metropolis 更新。

对每一条 link：

1. 提出一个有界区间内的随机角度扰动
2. 计算与该 link 相邻的两个 plaquette 带来的局域作用量变化
3. 按如下概率接受或拒绝

$$
P_{\mathrm{acc}}=\min(1,e^{-\Delta S})
$$

这个实现是故意选得比较简单、容易审查的版本。它的目标是先给后续模块提供一个可信基准，而不是一开始就追求最高效率。

## 当前测量的可观测量

生成脚本和独立测量脚本现在都会检查：

- 平均 plaquette
- Wilson loop `W(1,1)`
- Wilson loop `W(2,2)`
- 拓扑电荷估计
- 平均接受率

当前用来检查平均 plaquette 的参考值是：

$$
\langle P \rangle_{\mathrm{ref}} \approx \frac{I_1(\beta)}{I_0(\beta)},
$$

这里的 `I_0` 和 `I_1` 是第一类修正 Bessel 函数。

这个值目前被当作 `2D U(1)` 纯规范理论下的 sanity benchmark 使用。对当前这个小体积 debug lattice，它不是被当成严格的有限体积精确值，而是作为一个非常有用的基准参考。

## 已经生成的 starter run

当前这批已经生成的 ensemble 参数是：

- `L = 8`
- `beta = 1.0`
- thermalization sweeps: `600`
- 两次保存配置之间的 sweeps: `25`
- 保存的配置数：`64`
- proposal width: `0.9`
- seed: `12345`

## 当前已经测得的结果

从这批生成出的 ensemble 得到：

- 平均 plaquette: `0.447899`
- 参考 plaquette: `0.446390`
- 绝对误差: `0.001509`
- 代码中使用的容差: `0.080000`
- 平均 `W(1,1)`: `0.447899`
- 平均 `W(2,2)`: `0.020131`
- 平均拓扑电荷: `-0.187500`
- 平均接受率: `0.831701`

当前这一步的 observable check 返回的是：

- `PASS`

## 为什么这一步重要

在开始写 `preprocess` 或 `L-CNN` 层之前，必须先有一套配置来源，它应当满足：

- 有明确物理含义
- 容易审查
- 可复现
- 至少已经通过一个可信的可观测量基准检查

当前这批 starter ensemble 的作用就是这个。

## 已经完成的预处理步骤

`preprocess` 模块现在已经实现并测试通过。

### 它做了什么

它读取：

- `config_gen/output/u1_mc_configs.npz`

该原始配置的 shape 是：

- `(N, 2, L, L)`

其中：

- channel `0`：x 方向 link angle
- channel `1`：y 方向 link angle

然后把每个配置转成更适合网络输入的特征张量：

- `(N, 4, L, L)`

通道顺序是：

- `cos_x`
- `sin_x`
- `cos_y`
- `sin_y`

### 为什么选这个表示

对 `U(1)` 来说，直接回归角度变量有 `±π` 的 branch cut 问题。用 `cos/sin` 表示可以避开这个不连续点，同时保留完整的群信息。

### 可逆性检查

预处理脚本会通过：

- `atan2(sin, cos)`

把角度重建回来，并计算 wrap 之后的最大重建误差。

当前结果：

- 最大重建误差：`0.0`
- 重建检查结果：`PASS`

这说明对当前数据集，这个表示在数值上是无损的。

### 当前采用的数据切分

当前切分是按 block 来做的，目的是尽量减少把彼此相邻、仍可能相关的 Monte Carlo 样本随意拆到 train / validation / test 里。

当前切分结果是：

- 总样本数：`64`
- train：`48`
- validation：`8`
- test：`8`
- block size：`8`

### 预处理输出文件

- `preprocess/output/u1_features_all.npz`
- `preprocess/output/u1_train.npz`
- `preprocess/output/u1_val.npz`
- `preprocess/output/u1_test.npz`
- `preprocess/output/preprocess_summary.json`

## 已经完成的 gauge utility 步骤

`gauge_utils` 模块现在也已经实现并测试通过。

### 它目前提供的功能

在 `gauge_utils/gauge_utils_u1.py` 里，目前已经包括：

- angle wrapping helper
- 随机局域 gauge angle 生成
- link angle 上的局域 `U(1)` gauge transformation
- plaquette angle 计算
- 平均 plaquette 测量
- 拓扑电荷测量
- Wilson loop 测量
- 一个简单的 site-local phase 对象 forward transport helper

### 测试了什么

测试脚本检查了：在一次随机局域 gauge transformation 之后，

- plaquette angle 是否保持不变
- average plaquette 是否保持不变
- `W(1,1)` 是否保持不变
- `W(2,2)` 是否保持不变
- topological charge 是否保持不变
- 一个被 transport 的局域对象是否按协变方式变换

### 当前测试结果

测试脚本：

- [test_gauge_utils_u1.py](/Users/wangkehe/Git_repository/Diffusion_Model/U1/gauge_utils/test_gauge_utils_u1.py)

已经成功运行，并写出了：

- [gauge_utils_test_results.json](/Users/wangkehe/Git_repository/Diffusion_Model/U1/gauge_utils/gauge_utils_test_results.json)

当前数值结果是：

- plaquette angle diff: `2.665e-15`
- average plaquette diff: `1.665e-16`
- Wilson `1x1` diff: `1.665e-16`
- Wilson `2x2` diff: `5.551e-17`
- topological charge diff: `6.661e-16`
- transport covariance diff: `3.511e-16`

最终结果：

- `PASS`

这些误差都在机器精度量级，这正是预期结果。

### 为什么这一步重要

这一步是第一次直接检查：代码是否真的尊重局域规范对称性，而不是只是“会处理数组”。

如果没有这一层基础模块，后面任何 `L-CNN` 实现都很难让人信任，因为我们将缺少一个可审查的基础层来支撑：

- gauge transformation
- gauge-invariant measurement
- 基于 transport 的 equivariant operation

所以这一步正好把“原始数据处理”与“对称性保持模型搭建”之间的缺口补上了。

## 下一步计划

下一步最自然的是：

- 开始搭第一版最小 `U(1)` `L-CNN`

这一步将基于已经准备好的 `cos/sin` 特征，并且会使用当前 `gauge_utils` 模块中的局域规范操作。具体目标应包括：

- 在原始 angle 表示上实现局域 gauge transformation 接口
- 数值验证 gauge invariance / equivariance
- 定义第一版局域 `L-Conv` 风格操作
- 把这些工具和预处理好的数据对接起来

## 备注

- 生成模块只放在了 `U1/config_gen/` 下，符合你的要求。
- 预处理模块单独放在了 `U1/preprocess/` 下。
- 规范对称性工具模块单独放在了 `U1/gauge_utils/` 下。
- 运行这些脚本所需的 `numpy` 已经安装在工作项目对应的虚拟环境里。
- 当前实现的定位是“第一版可信基准”，不是最终高效率采样器。
