# U1 L-CNN 模块说明

这个目录现在放的是第一版最小 `U(1) L-CNN` 参考实现。

它的定位不是“完整训练框架”，而是先把下面几件核心事情写清楚、写成代码：

- `site-covariant feature` 到底长什么样
- 最近邻 `parallel transport` 到底怎么写
- `L-Conv` 为什么不是普通卷积
- 复数 covariant feature 应该用什么样的非线性
- 怎样从 covariant feature 构造 gauge-invariant 标量读出

## 当前文件

- `u1_lcnn_numpy.py`
  - 用 `numpy` 写的最小前向参考实现
  - 包含：
    - `U(1)` link angle 到复数 link 的转换
    - 随机局域 gauge transformation
    - site-covariant 复数特征
    - forward / backward transport
    - `LConvLayer`
    - `EquivariantModReLU`
    - `InvariantReadout`
    - 两层的 `MinimalU1LCNN`

- `validate_u1_lcnn.py`
  - 随机生成：
    - 一个 `U(1)` link 配置
    - 一个 site-covariant 复数特征场
    - 一个随机初始化的最小 `L-CNN`
    - 一个随机 gauge transformation
  - 然后检查：
    - 中间 hidden feature 是否按协变方式变换
    - 局域标量输出是否保持不变
    - 全局标量输出是否保持不变

- `u1_lcnn_torch.py`
  - `PyTorch` 版最小 `U(1) L-CNN`
  - 使用实部 / 虚部分离表示复数 covariant feature
  - 从 link 配置构造 canonical open Wilson line seed 作为初始 site-covariant feature
  - 包含：
    - `LConv2d`
    - `EquivariantModReLU`
    - `InvariantReadout`
    - `LocalU1LCNN`

- `validate_u1_lcnn_torch.py`
  - 检查 `PyTorch` 版 `L-CNN` 的 hidden covariance 和输出 invariance

- `train_local_plaquette_lcnn.py`
  - 在真实 `u1_train.npz / u1_val.npz / u1_test.npz` 上训练 `L-CNN`

- `train_local_plaquette_baseline.py`
  - 在同一任务上训练普通 `CNN` baseline，供对照使用

## 这版代码解决了什么

这版代码的作用是把 `Phase 3` 里最抽象的部分先 concretize：

- `L-Conv` 的数学结构
- `equivariant` 非线性的选择
- `invariant readout` 的具体构造
- gauge check 的写法

这样后面即使要迁移到 `PyTorch`，也已经有一份小而清楚的参考实现可以对照。

## 这版代码还没做什么

当前还没有：

- 参数优化与反向传播
- 与普通 `CNN` 的对照实验
- diffusion / score network

现在已经有了真实数据上的最小监督训练，但这里仍然不是最终训练系统，因为还没有：

- 更系统的超参数搜索
- 更严格的物理验证集合
- diffusion / score network
