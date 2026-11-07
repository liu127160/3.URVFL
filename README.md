# URVFL：纵向联邦学习隐蔽数据重建攻击复现报告

> 论文：**URVFL: Undetectable Data Reconstruction Attack on Vertical Federated Learning**（NDSS 2025）  
> 作者：Duanyi Yao、Songze Li、Xueluan Gong、Sizai Hou、Gaoning Pan  
> 论文页面：[NDSS 2025](https://www.ndss-symposium.org/ndss-paper/urvfl-undetectable-data-reconstruction-attack-on-vertical-federated-learning/)  
> 预印本：[arXiv:2404.19582](https://arxiv.org/abs/2404.19582)  
> 官方代码：[duanyiyao/URVFL](https://github.com/duanyiyao/URVFL)  
> 复现者：刘明鑫

## 应用场景

URVFL适用于评估纵向联邦学习中的数据泄露风险。例如，银行和电商拥有同一批用户的不同特征，双方在不交换原始数据的情况下共同训练模型。若持有标签并负责返回梯度的主动客户端是恶意参与方，它可能利用少量与正式任务特征类型、标签体系相同的完整辅助数据，尝试重建被动客户端的私有特征。

因此，该方法主要用于纵向联邦学习的隐私安全研究和防御能力评估。

## 理论依据

- **编码器—解码器重建：** 编码器把目标特征转换为参考嵌入，解码器学习从嵌入恢复目标特征。
- **对抗式分布对齐：** DAC引导被动客户端产生的嵌入逐渐接近参考编码器的嵌入分布，并利用类别标签保持不同类别的结构。
- **梯度反向传播：** 主动客户端虽然不能直接修改被动客户端的底层模型，但可以通过返回特制的嵌入梯度，间接改变该模型的训练方向。

## 方法及步骤

本文分为主动客户端和被动客户端。主动客户端包括底层模型、标签、编码器、解码器、DAC（带辅助分类器的判别器）、用户的部分特征，以及辅助数据（一小批与正式任务**特征类型和标签体系相同的完整样本**）。被动客户端拥有底层模型和用户的另一部分特征。

刚开始先使用辅助数据进行预训练，包括：辅助数据中的主动端特征 → 攻击者底层模型 → 主动端嵌入；辅助数据中的目标特征 → 编码器 → 参考嵌入；主动端嵌入＋参考嵌入 → 解码器 → 重建目标特征（即被动客户端所持有的那类输入数据）。目标是让编码器学会产生“参考嵌入”，让解码器学会根据这种参考嵌入恢复目标数据，同时让攻击者底层模型学会提取主动端特征。对于标准URVFL，这三个模型预训练完成后被冻结。

之后进行纵向联邦学习，被动客户端将其底层模型产生的嵌入上传给主动客户端。主动客户端通过DAC计算两种损失，分别产生用于更新被动客户端底层模型的恶意梯度，以及用于更新DAC自身的梯度：

1. 将被动客户端嵌入输入DAC，并故意设置目标标签为“编码器来源＋真实类别”，计算恶意损失 \(L_M\)。将得到的嵌入梯度传给被动客户端，使其嵌入分布逐渐接近固定编码器的嵌入分布。
2. 使用正确映射训练DAC：编码器嵌入对应“编码器来源＋真实类别”，被动客户端嵌入对应“被动客户端来源＋真实类别”。计算 \(L_D\) 并更新DAC，提高其识别嵌入类别和区分嵌入来源的能力。

两步相互对抗并反复进行。其中，\(L_M\) 用于更新被动客户端底层模型，\(L_D\) 用于更新DAC自身。

最后，被动客户端产生的嵌入分布逐渐接近固定编码器的嵌入分布。此时，攻击者将同一样本的被动客户端嵌入和主动客户端底层模型产生的嵌入拼接后输入解码器。由于解码器在预训练阶段已经学会根据这种嵌入形式恢复目标特征，因此可以重建被动客户端持有的私有数据。

## 1. 复现工作概述

本项目复现论文提出的纵向联邦学习（Vertical Federated Learning，VFL）恶意数据重建攻击 **URVFL** 和同步版本 **URVFL_sync**，并使用 **FSHA** 作为对照攻击，考察三种方法的数据重建能力、嵌入分布对齐效果以及绕过 **Gradient Scrutinizer（GS）** 检测的能力。

当前完成的内容包括：

- 在CIFAR-10上运行FSHA、URVFL和URVFL_sync无检测训练；
- 固定随机种子为327，统一保存各方法的结果数组；
- 运行FSHA、URVFL和URVFL_sync的GS检测训练；
- 保存GS检测分数和梯度统计量；
- 修复公开代码中URVFL_sync接入GS时的配置字段错误；
- 补充结果绘图脚本，生成嵌入距离曲线、最终指标表和GS检测分数曲线。

当前复现范围为 **CIFAR-10、两个参与方、单个随机种子327**。

本次使用的主要配置如下。

| 项目 | 当前设置 |
| --- | --- |
| 数据集 | CIFAR-10 |
| 客户端数量 | 2 |
| 特征划分 | 将32×32图像沿宽度平均切分给两个客户端 |
| 攻击者 / 目标客户端 | client_1 / client_2 |
| 随机种子 | 327 |
| 辅助样本数 | 4545 |
| 训练批大小 / 测试批大小 | 128 / 64 |
| URVFL与URVFL_sync攻击训练周期| 25 epochs |
| FSHA攻击训练周期 | 15 epochs |
| GS检测训练周期 | 8 epochs |
| 学习率 | 1e-4 |
| 嵌入维度 | 16 |
| 当前代码中的GS阈值 | 0.7 |

## 2. 方法简述

### 2.1 VFL与威胁模型

在VFL中，各参与方持有同一批样本的不同特征。被动客户端使用底层模型将本地特征编码为嵌入并发送给主动客户端；主动客户端持有标签和顶层模型，计算任务损失后，将嵌入梯度返回给被动客户端。

URVFL假设主动客户端是恶意参与方。攻击者能够接收被动客户端上传的嵌入，并控制返回给被动客户端的梯度，但不能直接查询被动客户端的底层模型，也不能向其注入自定义输入。攻击目标是在这种限制下重建被动客户端的私有特征，同时避免被现有恶意训练检测方法发现。

### 2.2 URVFL

URVFL首先利用带有完整特征和标签的辅助数据训练参考编码器、参考底层模型和解码器。攻击阶段，真实被动客户端产生目标嵌入，参考模型产生带类别信息的参考嵌入。

方法的核心是带辅助分类器的判别器（Discriminator with Auxiliary Classifier，DAC）：

1. DAC区分参考嵌入与目标嵌入；
2. DAC同时利用标签保持不同类别的嵌入结构；
3. 攻击者根据DAC损失计算针对目标嵌入的恶意梯度；
4. 被动客户端使用该梯度更新自身底层模型，使目标嵌入逐渐接近参考嵌入分布；
5. 攻击者最终使用在参考嵌入上训练的解码器重建被动方特征。

类别条件对齐一方面改善了解码器的重建能力，另一方面使恶意梯度更接近正常分类任务产生的梯度，从而提高对SG和GS的隐蔽性。

### 2.3 URVFL_sync

普通URVFL在预训练结束后固定参考编码器、参考底层模型和解码器，攻击阶段主要更新被动客户端模型与DAC。URVFL_sync则在攻击过程中继续使用辅助数据同步更新参考编码器、参考底层模型、解码器和DAC。

sync并没有获得更多被动客户端原始数据。它的作用是让**参考嵌入**、DAC和解码器随着训练过程共同变化，减少固定参考模型与动态目标嵌入之间的不适配。

### 2.4 GS检测

GS根据同类别和不同类别样本的嵌入梯度统计关系计算检测分数。当前实现从约第100个训练批次开始，每隔50个批次更新一次分数，并对最近5个分数求平均；当平均分低于阈值时，判定出现攻击并停止恶意训练。

论文将GS阈值设置为0.8，但当前公开代码在 `defense.py` 中使用0.7。本次结果沿用当前代码的0.7，因此与论文表格不是严格同配置重复实验。

### 2.5 评价指标

- **Recon MSE（越低越好）**：重建数据与真实数据之间的均方误差；
- **Emb MSE（越低越好）**：参考嵌入与目标嵌入之间的均方距离；
- **Emb Cos（越低越好）**：两类嵌入之间的余弦距离；
- **PSNR（越高越好）**：重建图像的峰值信噪比；
- **SSIM（越高越好）**：重建图像与真实图像的结构相似度；
- **GS score**：GS检测分数，当前代码中低于0.7表示更加可疑。

## 3. 复现结果与分析

### 3.1 无检测训练结果

![FSHA、URVFL和URVFL_sync嵌入距离曲线](cifar10/figures/attack_curves_seed327.png)

![最终攻击指标](cifar10/figures/attack_table_seed327.png)

| 方法 | 结果来源 | Recon MSE ↓ | Emb MSE ↓ | Emb Cos ↓ | PSNR ↑ | SSIM ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FSHA | 论文Table II/III | 0.0317±0.0065 | 0.4657±0.1286 | 0.1610±0.0285 | 19.7955±2.2895 | 0.5312±0.0298 |
| FSHA | 本次复现 | 0.032871 | 0.478440 | 0.104483 | 20.8614 | 0.609241 |
| URVFL | 论文Table II/III | 0.0302±0.0011 | 0.5679±0.0834 | 0.1303±0.0131 | 21.0666±0.0165 | 0.5927±0.0002 |
| URVFL | 本次复现 | 0.031942 | 0.616937 | 0.140835 | 20.9854 | 0.571823 |
| URVFL_sync | 论文Table II/III | 0.0176±0.0139 | 0.2793±0.1868 | 0.0675±0.0549 | 26.2360±0.0030 | 0.8125±0.0001 |
| URVFL_sync | 本次复现 | **0.012065** | **0.234185** | **0.033162** | **25.2136** | **0.801032** |

本次结果与论文在总体趋势上基本一致：URVFL与FSHA的最终重建MSE接近，而URVFL_sync在重建MSE、嵌入距离、PSNR和SSIM上表现最好。URVFL的嵌入距离并不一定始终小于FSHA，因此不能仅凭Emb MSE判断最终重建能力；重建效果还取决于参考编码器、解码器和类别条件对齐共同学到的表示。

论文表格报告多次实验的均值和标准差，本次结果只有seed=327的一次运行，因此只能比较数值范围和趋势，不能据此判断本次实现优于论文。

### 3.2 GS检测结果

![GS检测分数](cifar10/figures/detection_scores_seed327.png)

| 方法 | GS记录数 | 分数范围 | 平均分 | 是否触发GS | 终止位置 |
| --- | ---: | ---: | ---: | --- | --- |
| FSHA | 6 | 0.0095–0.0151 | 0.0129 | 是 | 约第350个训练批次 |
| URVFL | 55 | 0.9600–1.0000 | 0.9936 | 否 | 完成8个检测epoch |
| URVFL_sync | 55 | 0.9220–1.0000 | 0.9785 | 否 | 完成8个检测epoch |

FSHA的GS分数持续低于0.7，最近5次分数的平均值满足检测条件，因此攻击在约第350个训练批次被停止。URVFL与URVFL_sync的分数始终高于当前阈值，均运行到8个检测epoch结束。这复现了论文Figure 6的核心现象：GS能够识别FSHA，但难以识别URVFL及其同步版本。

当前绘图脚本根据代码的记录规则，将GS横轴还原为实际训练批次：第一次记录约在第100批次，此后每50批次记录一次。因此55个分数的最后一个横坐标为2800；完整的8个epoch约包含2840个训练批次。

### 3.3 GS条件下的重建结果

| 方法 | 结果来源 | Recon MSE ↓ | Emb MSE ↓ | Emb Cos ↓ | PSNR ↑ | SSIM ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FSHA-GS | 论文Table II/III | 0.1312±0.0150 | 1.5940±0.0684 | 0.3133±0.0230 | 15.1751±0.5590 | 0.3083±0.0053 |
| FSHA-GS | 本次复现 | 0.152127 | 1.982220 | 0.312764 | 14.2269 | 0.289493 |
| URVFL-GS | 论文Table II/III | 0.0302±0.0011 | 0.5679±0.0834 | 0.1303±0.0131 | 21.0666±0.0165 | 0.5927±0.0002 |
| URVFL-GS | 本次复现 | 0.022291 | 0.384818 | 0.087248 | 22.5485 | 0.677400 |
| URVFL_sync-GS | 论文Table II/III | 0.0176±0.0139 | 0.2793±0.1868 | 0.0675±0.0549 | 26.2360±0.0030 | 0.8125±0.0001 |
| URVFL_sync-GS | 本次复现 | 0.016698 | 0.291498 | 0.063219 | 23.8020 | 0.733291 |

FSHA被GS较早终止，因此重建质量明显下降。URVFL和URVFL_sync没有触发GS，仍能获得较好的重建结果。由于本次使用的GS阈值、随机种子和运行次数与论文并不完全一致，表中数值用于检查趋势，不应视为对论文均值的严格重复。

## 4. 快速启动指南

### 4.1 环境准备

当前代码在WSL/Linux与NVIDIA CUDA环境下运行，Python版本为3.9。依赖版本已经写入 `requirements.txt`。

```bash
conda create -n urvfl python=3.9
conda activate urvfl
pip install -r requirements.txt
```

若出现 `pkg_resources is deprecated`，它是setuptools相关警告，不会直接导致训练失败；需要消除警告时可以暂时使用 `setuptools<81`。若出现cuDNN执行计划警告，但训练仍继续并产生结果，则可记录环境差异后继续观察。

### 4.2 数据路径与配置

首先将 `cifar10/cifar_data_pre.py` 中CIFAR-10的 `root` 修改为本机数据目录，例如：

```python
trainset = datasets.CIFAR10(root="/path/to/URVFL/data", train=True, download=True, transform=transform_train)
testset = datasets.CIFAR10(root="/path/to/URVFL/data", train=False, download=True, transform=transform_test)
```

主要超参数位于 `cifar10/config_cifar.json`：

- `save=true`：重新预训练参考编码器、参考底层模型和解码器；
- `save=false`：加载已有预训练模型，使用前应确认脚本中的 `save_seed` 与模型文件一致；
- `epochs`：URVFL和URVFL_sync攻击轮数；
- `fsha_epochs`：FSHA攻击周期；
- `gs_epochs`：GS检测训练周期；
- `aux_size`：辅助数据数量；
- `sg_defense`、`gs_defense`：检测触发后是否停止攻击。

当前 `sync_cifar.py` 将辅助样本数直接写为4545，而不是读取 `config['aux_size']`。修改辅助数据规模时需要同步修改该脚本，或者将其统一改为读取配置。

### 4.3 运行无检测攻击

从项目根目录执行：

```bash
mkdir -p logs

python cifar10/fsha_cifar.py 2>&1 | tee logs/fsha_327.log
python cifar10/urvfl_cifar.py 2>&1 | tee logs/urvfl_327.log
python cifar10/sync_cifar.py 2>&1 | tee logs/sync_327.log
```

其中URVFL在 `save=true` 时先完成20个epoch的预训练，再运行25个攻击epoch。执行URVFL的GS检测前，应先确保 `cifar10/saved_model/` 中存在与 `save_seed` 对应的编码器、解码器和参考底层模型。

### 4.4 运行GS检测

在 `cifar10/detection_cifar.py` 中选择攻击与检测方法：

```python
atk_type = ["fsha"]      # 可改为 "urvfl" 或 "sync"
defenses = ["gs"]
```

然后分别运行并保存日志：

```bash
python cifar10/detection_cifar.py 2>&1 | tee logs/gs_fsha_327.log
python cifar10/detection_cifar.py 2>&1 | tee logs/gs_urvfl_327.log
python cifar10/detection_cifar.py 2>&1 | tee logs/gs_sync_327.log
```

每次运行前需要把 `atk_type` 改为对应方法。当前脚本默认设置为 `sync` 与 `gs`。

### 4.5 生成图表

```bash
python cifar10/plot_results.py --seed 327
```

默认读取 `cifar10/attack_result/`，并在 `cifar10/figures/` 生成：

- `attack_curves_seed327.png`；
- `attack_table_seed327.csv`；
- `attack_table_seed327.png`；
- `detection_scores_seed327.png`。

结果数组统一采用以下行顺序：

```text
Emb MSE, Emb cosine distance, Recon MSE, PSNR, SSIM
```

## 5. 遇到的困难及解决办法

### 5.1 URVFL_sync接入GS时缺少 `target_weight`

**困难：** `defense.py` 的sync分支读取 `config['target_weight']`，但 `config_cifar.json` 中没有该字段，导致GS-SYNC在第一个批次报错退出。

**解决办法：** 参考无检测的URVFL_sync实现，将解码器输入由带 `target_weight` 的拼接改为参考编码嵌入与参考底层嵌入的直接拼接，使检测分支与独立sync训练保持一致。

### 5.2 GS完整运行后的返回值与调用端不一致

**困难：** 当URVFL或URVFL_sync一直未触发GS、完整运行8个epoch后，`gradient_scru()`原本只返回梯度统计数组，而 `detection_cifar.py` 按7个返回值解包，出现 `expected 7, got 4` 等错误。

**解决办法：** 统一提前终止和完整结束两条路径的返回结构，完整返回检测分数、嵌入距离、重建指标和梯度统计量。

## 6. 可复现性与限制说明

1. 当前只完成CIFAR-10、seed=327的一次运行，而论文表格报告均值和标准差；本结果只能说明总体趋势基本一致。
2. 当前代码的GS阈值为0.7，与论文使用的0.8不同；两者不是严格同配置比较。
3. 公开仓库目前主要提供CIFAR-10实现，本项目没有复现MNIST、Tiny ImageNet、Credit和RT-IOT2022结果。
4. `set_seed()`设置了PyTorch、NumPy和Python随机种子，但没有完全固定CUDA与cuDNN的确定性行为，不同硬件可能产生小幅差异。
5. 无检测曲线保存的是每个epoch的汇总值，无法恢复论文Figure 7中每个攻击迭代的原始轨迹。
