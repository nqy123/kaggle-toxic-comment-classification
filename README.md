# Kaggle Jigsaw 有毒评论多标签分类

![Result Card](assets/result_card.png)

## 项目一句话

对评论同时预测 toxic、obscene、threat、insult 等多个标签。

这个项目不是简单跑一个 baseline，而是围绕 **数据清洗 -> 特征工程 -> 稳定验证 -> 模型融合 -> 线上结果复盘** 做成一条完整建模链路。核心目标是：让模型不仅分数高，而且每一步为什么有效都能讲清楚。

## 当前结果

| 项目 | 内容 |
| --- | --- |
| Competition | `jigsaw-toxic-comment-classification-challenge` |
| Metric | `Mean ROC-AUC` |
| Best Submission | `outputs/submission_nbsvm30_keras0_transformer_e3_70.csv` |
| Best Score | Private AUC 0.98611 |
| Validation / Extra | Public AUC 0.98619 |
| Status | 接近 top 20%，无 test_labels 泄露 |

## 数据清洗

- 轻度清洗换行、URL、HTML 实体，保留辱骂词、大小写和拼写变体中的信号。
- 严格不使用 post-competition `test_labels.csv`，保证方案合法。
- 多标签目标独立建模，同时用统一验证方式比较各模型。

## 特征工程亮点

- NB-SVM 使用 word 1-2 gram 和 char 3-5 gram，能抓住脏词变体和拼写规避。
- Keras BiGRU-CNN 学习句子顺序，但单模较弱，后期被更强 Transformer 替代。
- 3 epoch DistilBERT 捕捉上下文毒性，最终与 NB-SVM 形成传统 NLP + 深度语义互补。

这部分是项目最重要的地方：特征不是随便堆出来的，而是尽量贴近业务或数据生成逻辑。我的思路是先问“这个变量为什么会影响目标”，再把这个想法翻译成模型能理解的数值、类别、比例、交叉或序列表示。

## 模型方法

- NB-SVM Logistic Regression 作为强 sparse baseline。
- Kaggle GPU 训练 Keras BiGRU-CNN 和 DistilBERT。
- 最终选用 30% NB-SVM + 70% DistilBERT E3，去掉拖分的 Keras。

验证上尽量使用 OOF 思路，避免只看一次线上提交。融合也不是机械平均，而是根据 OOF、public/private 表现和模型互补性来选择。

## 结果分析

- 分数从 NB-SVM private 0.97937 提升到 0.98611，主要来自更强的 3 epoch Transformer。
- 后期实验证明 Keras 在强 Transformer 面前反而拖分，及时移除是关键决策。
- 这个项目展示了不是所有模型都该融合，融合权重必须由验证和 leaderboard 共同校准。

## 如何复现

安装依赖：

```bash
pip install -r requirements.txt
```

复现时先从 Kaggle 下载原始数据到 README 或脚本约定的数据目录。部分仓库为了保持轻量，只保留最佳提交文件、实验日志和核心说明；如果仓库中存在 `src/`、`notebooks/` 或 `kaggle_kernel_*`，优先从这些入口运行训练。

常见入口示例：

```bash
python src/train_best.py
# 或在 Kaggle 上运行 kaggle_kernel_* 中的 GPU kernel
```

如果当前项目只保留了最佳产物，则可直接查看 `outputs/` 中的 OOF、prediction、submission 和实验摘要文件。

## 未来改进方向

- 训练 5-fold Transformer OOF，而不是单 holdout。
- 尝试 RoBERTa/DeBERTa/toxic-bert，多模型 rank/prob 融合。
- 增加 max_len 到 192/256，改善长评论截断问题。

## 项目价值

这个项目可以体现三类能力：

- **建模能力**：能从 baseline 走到调参、融合和线上验证。
- **特征工程能力**：能把业务直觉、数据分布和模型输入连接起来。
- **复盘能力**：能说明为什么涨分、为什么不涨，以及下一步该往哪里优化。
