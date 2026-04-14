# 第一版 U(1) Score Model 说明

## 这版模型学的是什么

这版开始做的是时间条件的 denoising score matching。

输入：

- 干净配置 `x_0` 的 `cos/sin` 表示
- 加噪后的 `x_t = x_0 + \sigma \epsilon`
- 噪声等级 `sigma`

输出：

- 与输入同 shape 的 score 场，shape 为 `(batch, 4, L, L)`

监督目标：

$$
\nabla_{x_t} \log p(x_t \mid x_0)
=
-\frac{x_t - x_0}{\sigma^2}
=
-\frac{\epsilon}{\sigma}
$$

因此这次学到的已经不再是“如何从 link 构造 plaquette”，而是：

- 在给定噪声等级下，预测每个局域输入分量应该往哪个方向移动，才能更接近干净样本。

## 但它还不是最终物理版本

这一点必须说清楚。

当前这版是为了先把 score-network 的训练对象、输出 shape、时间条件路径和损失函数搭起来。

它还不是最终的规范理论 diffusion model，因为：

1. 噪声是加在 `cos/sin` 的欧氏表示上的。
2. 带噪样本会偏离严格的 `U(1)` 群流形。
3. 因此这里学到的是欧氏化表示下的 score，不是最终严格流形上的 score。

## 为什么仍然值得先做

因为这一步可以先验证几件关键事情：

- 模型输出能否和输入保持同 shape
- 时间条件路径是否正常工作
- DSM loss 是否稳定下降
- `L-CNN` backbone 是否能自然接到 score head 上

如果这一步都不稳定，就没必要立刻上更严格的角变量 / 流形噪声版本。

## 当前网络结构

1. 对 noisy `cos/sin` 输入做规范化投影，回到每条 link 的单位圆上。
2. 用投影后的 link 构造 `L-CNN` backbone 的 covariant feature。
3. 提取 invariant context。
4. 用 `sigma` 的时间嵌入调制 context 和输出头。
5. 输出同 shape 的 score 场。

核心实现：

- [u1_score_model_torch.py](/Users/wangkehe/Git_repository/Diffusion_Model/U1/lcnn/u1_score_model_torch.py)
- [train_u1_score_model.py](/Users/wangkehe/Git_repository/Diffusion_Model/U1/lcnn/train_u1_score_model.py)

## 和前面的 plaquette 回归有什么区别

前面的 `L-CNN` 回归器：

- 输入：干净 link 配置
- 输出：局域 plaquette 标量
- 任务：学习 gauge-invariant observable

现在这版 score model：

- 输入：带噪配置 + `sigma`
- 输出：每个输入分量的 score
- 任务：学习去噪方向场

所以这一步才真正开始接近 diffusion / score modeling。
