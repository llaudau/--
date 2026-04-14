# Wrapped Angle Score Model 说明

## 这版比前一版更接近什么

前一版 score model 输出的是 `cos/sin` 欧氏表示里的 4 通道 score。

这一版改成：

- 输入：noisy link 的 `cos/sin` 表示
- 输出：每条 link angle 的切向 score，shape 为 `(batch, 2, L, L)`

也就是说：

- channel `0`：`x` 方向 link angle 的 score
- channel `1`：`y` 方向 link angle 的 score

这比直接在 `cos/sin` 嵌入空间里学 4 通道方向更接近 `U(1)` 的真实几何。

## 当前采用的近似

噪声过程写成：

$$
\phi_t = \mathrm{wrap}(\phi_0 + \sigma \epsilon)
$$

并用 wrapped difference

$$
\Delta \phi = \mathrm{wrap}(\phi_t - \phi_0)
$$

构造目标：

$$
s^\star(\phi_t, \sigma) \approx -\frac{\Delta \phi}{\sigma^2}
$$

这在小噪声 regime 下是合理的局部切空间近似，因此这版特地把：

- `sigma_max` 收紧到 `0.20`

目的是减少绕圈带来的多值问题。

## 为什么这版仍然不是最终答案

它比欧氏版更物理，但仍然不是严格完备版本，因为：

1. 对圆上的精确 score，严格来说应考虑 wrapped normal 的周期求和。
2. 当前目标是小噪声局部近似，不是完整解析 score。
3. reverse sampling 还没有真正实现。

## 为什么还是值得做

因为这一步把最关键的对象先改对了：

- 输出不再是嵌入空间方向
- 输出直接对应每条 link angle 的切向去噪方向

这使后面做真正的 `U(1)` reverse diffusion / Langevin 更自然。

## 实现位置

- [u1_angle_score_model_torch.py](/Users/wangkehe/Git_repository/Diffusion_Model/U1/lcnn/u1_angle_score_model_torch.py)
- [train_u1_angle_score_model.py](/Users/wangkehe/Git_repository/Diffusion_Model/U1/lcnn/train_u1_angle_score_model.py)
