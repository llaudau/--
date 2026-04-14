# U(1) L-CNN 对照实验报告

日期：`2026-04-06`

## 目的

验证两件事：

1. 增大样本数后，`L-CNN` 的优势是否仍然稳定。
2. 把 lattice 从原来的 `L=8` 扩大到更大的 `L=12` 后，这套方法是否仍然可行。

本次对照全部使用同一个监督任务：

- 输入：`U(1)` link 的 `cos/sin` 特征
- 目标：局域 plaquette 标量场 `cos(theta_{x,\square})`

模型对照：

- 普通 `PyTorch CNN`
- `PyTorch U(1) L-CNN`

## 数据集

### 实验 A：`L = 8`

- 生成配置数：`256`
- thermalization sweeps：`1000`
- skip sweeps：`30`
- proposal width：`0.9`
- seed：`2026040601`
- train / val / test：`176 / 32 / 48`

观测量检查：

- avg plaquette：`0.448993`
- reference：`0.446390`
- abs error：`0.002603`
- independent check：`PASS`

相关文件：

- [u1_mc_diagnostics.json](/Users/wangkehe/Git_repository/Diffusion_Model/U1/experiments/l8_n256/config_gen/u1_mc_diagnostics.json)
- [u1_mc_measured.json](/Users/wangkehe/Git_repository/Diffusion_Model/U1/experiments/l8_n256/config_gen/u1_mc_measured.json)
- [preprocess_summary.json](/Users/wangkehe/Git_repository/Diffusion_Model/U1/experiments/l8_n256/preprocess/preprocess_summary.json)

### 实验 B：`L = 12`

- 生成配置数：`192`
- thermalization sweeps：`1000`
- skip sweeps：`30`
- proposal width：`0.9`
- seed：`2026040602`
- train / val / test：`128 / 32 / 32`

观测量检查：

- avg plaquette：`0.445692`
- reference：`0.446390`
- abs error：`0.000698`
- independent check：`PASS`

相关文件：

- [u1_mc_diagnostics.json](/Users/wangkehe/Git_repository/Diffusion_Model/U1/experiments/l12_n192/config_gen/u1_mc_diagnostics.json)
- [u1_mc_measured.json](/Users/wangkehe/Git_repository/Diffusion_Model/U1/experiments/l12_n192/config_gen/u1_mc_measured.json)
- [preprocess_summary.json](/Users/wangkehe/Git_repository/Diffusion_Model/U1/experiments/l12_n192/preprocess/preprocess_summary.json)

## 模型设置

普通 CNN：

- 3 个 `3x3` 卷积层 + `GELU`
- 1 个 `1x1` 输出层

`L-CNN`：

- 初始 covariant seed：由 canonical open Wilson line 和 gauge-invariant scalar 组合构造
- 2 层 `L-Conv`
- `EquivariantModReLU`
- invariant readout

相关实现：

- [u1_lcnn_torch.py](/Users/wangkehe/Git_repository/Diffusion_Model/U1/lcnn/u1_lcnn_torch.py)
- [train_local_plaquette_baseline.py](/Users/wangkehe/Git_repository/Diffusion_Model/U1/lcnn/train_local_plaquette_baseline.py)
- [train_local_plaquette_lcnn.py](/Users/wangkehe/Git_repository/Diffusion_Model/U1/lcnn/train_local_plaquette_lcnn.py)

## 结构验证

`PyTorch L-CNN` 的 gauge 检查结果：

- hidden covariance residual：`5.9006e-07`
- local invariant residual：`2.9802e-08`
- 结果：`PASS`

文件：

- [validation_torch_results.json](/Users/wangkehe/Git_repository/Diffusion_Model/U1/lcnn/validation_torch_results.json)

## 结果汇总

### `L = 8`, `256` configs

普通 CNN：

- best val loss：`0.3683545`
- test loss：`0.3631353`
- test MAE：`0.4970060`
- val gauge prediction MAE：`4.6941e-02`
- test gauge prediction MAE：`4.6664e-02`

文件：

- [plain_cnn_local_plaquette_metrics.json](/Users/wangkehe/Git_repository/Diffusion_Model/U1/experiments/l8_n256/models/plain/plain_cnn_local_plaquette_metrics.json)

`L-CNN`：

- best val loss：`0.0039710`
- test loss：`0.0049586`
- test MAE：`0.0490295`
- val gauge prediction MAE：`3.0619e-07`
- test gauge prediction MAE：`2.9116e-07`

文件：

- [u1_lcnn_local_plaquette_metrics.json](/Users/wangkehe/Git_repository/Diffusion_Model/U1/experiments/l8_n256/models/lcnn/u1_lcnn_local_plaquette_metrics.json)

### `L = 12`, `192` configs

普通 CNN：

- best val loss：`0.3563669`
- test loss：`0.3331352`
- test MAE：`0.4822211`
- val gauge prediction MAE：`2.8006e-02`
- test gauge prediction MAE：`2.7829e-02`

文件：

- [plain_cnn_local_plaquette_metrics.json](/Users/wangkehe/Git_repository/Diffusion_Model/U1/experiments/l12_n192/models/plain/plain_cnn_local_plaquette_metrics.json)

`L-CNN`：

- best val loss：`0.0030242`
- test loss：`0.0025808`
- test MAE：`0.0340350`
- val gauge prediction MAE：`3.0659e-07`
- test gauge prediction MAE：`3.1929e-07`

文件：

- [u1_lcnn_local_plaquette_metrics.json](/Users/wangkehe/Git_repository/Diffusion_Model/U1/experiments/l12_n192/models/lcnn/u1_lcnn_local_plaquette_metrics.json)

## 结论

结论很明确：

1. 在样本数明显增大后，`L-CNN` 的优势没有消失，反而非常稳定。
2. 在更大的 `L=12` lattice 上，`L-CNN` 仍然正常工作，并且比普通 CNN 更好。
3. 普通 CNN 虽然能学到一点局域结构，但对随机 gauge transformation 的输出漂移仍在 `1e-2` 到 `1e-1` 量级。
4. `L-CNN` 的 gauge 漂移稳定在 `1e-7` 量级，已经接近数值误差。
5. 在两个 lattice 上，`L-CNN` 的 test MAE 都比普通 CNN 小一个数量级左右。

更具体地说：

- `L=8` 时，test MAE 从 `0.4970` 降到 `0.0490`
- `L=12` 时，test MAE 从 `0.4822` 降到 `0.0340`

这说明这版 `L-CNN` 不是只在一个小样本、小 lattice 上“碰巧有效”，而是在更强一点的统计条件和更大的 lattice 上依然成立。

## 当前局限

这份报告说明了“方法可行”，但还不能说“已经充分完备”。

当前仍然存在的限制：

- 只测试了一个 `beta = 1.0`
- 只做了局域 plaquette 回归
- 还没有测试更大的 lattice，例如 `L=16+`
- 还没有做 score network / diffusion
- 当前 MC 生成器是透明优先的 debug 版本，不适合立刻扩到很大的生产规模

## 建议下一步

最合理的后续动作是：

1. 再加一个更难一点的监督目标，例如局域 `2x2` Wilson loop 或全局 observable。
2. 测试跨 lattice 泛化，例如在 `L=8` 训练、`L=12` 验证。
3. 给 `L-CNN` 增加 time embedding，开始过渡到 score network。
