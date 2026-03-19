# ResNet34 特征相似度分析说明

## 1. 这次分析做了什么

本目录保存的是对 `insertion_seq_network_input/` 中 14 张按距离命名的图片做的 `ResNet34` 特征相似度分析结果。

- 使用模型: `ImageNet` 预训练 `ResNet34`
- 特征位置: 去掉最后分类层后的 backbone 输出特征
- 图片范围: `-0.8m` 到 `+0.5m`
- 主要目的: 看不同插入距离下，图像在 `ResNet34` 特征空间里是否仍然有可区分的变化

## 2. 核心结论

- 相邻 `0.1m` 图像的平均特征余弦相似度约为 `0.9882`
- 相邻图像整体都很像，但不是完全不变
- 特征相似度会随着距离差增大而逐步下降，说明 `ResNet34` 仍然编码了距离变化
- `+0.4m -> +0.5m` 的特征相似度是 `0.9856`，并没有比其它相邻区间显著更高
- 从这组图本身来看，"到 0.5m 后视觉特征几乎没变化" 这个判断没有被明显支持

更直接地说:

- 视觉变化是存在的
- 但这些变化属于连续、小幅、强相关的变化
- 如果强化学习策略在 `0.5m` 左右失效，原因可能不只是 backbone 看不见，还可能和策略、奖励设计、动作精度需求、时序信息利用方式有关

## 3. 目录里的文件分别是什么

### 图像结果

- `adjacent_similarity.png`
  - 看相邻 `0.1m` 图片之间的相似度变化
  - 上半图: 特征余弦相似度和像素余弦相似度
  - 下半图: 相邻两张图在特征空间中的变化幅度

- `feature_similarity_heatmap.png`
  - 所有图片两两之间的 `ResNet34` 特征余弦相似度热力图
  - 越亮表示越相似

- `pixel_similarity_heatmap.png`
  - 原始像素向量之间的相似度热力图
  - 用来和模型特征做对比

- `reference_similarity.png`
  - 以 `-0.5m` 和 `+0.0m` 为参考图，查看它们与其它距离图像的相似度变化

### 数据文件

- `summary.json`
  - 核心统计结果摘要

- `adjacent_similarity.csv`
  - 每一对相邻图片的详细数值
  - 包括特征余弦相似度、特征 L2 距离、像素余弦相似度

- `ordered_samples.json`
  - 本次参与分析的图片顺序和距离标签

- `feature_similarity_matrix.npy`
  - `ResNet34` 特征相似度矩阵

- `pixel_similarity_matrix.npy`
  - 像素相似度矩阵

- `feature_vectors.npy`
  - 每张图片提取出来的特征向量

## 4. 如何查看输出内容

当前目录:

```bash
cd /Users/sull/Documents/test_resnet
```

### 先看图

最推荐先看这三张:

```bash
open analysis_outputs/resnet34_similarity/adjacent_similarity.png
open analysis_outputs/resnet34_similarity/feature_similarity_heatmap.png
open analysis_outputs/resnet34_similarity/reference_similarity.png
```

如果你想直接打开整个输出文件夹:

```bash
open analysis_outputs/resnet34_similarity
```

### 看摘要数字

```bash
cat analysis_outputs/resnet34_similarity/summary.json
```

### 看相邻距离的详细数值

```bash
cat analysis_outputs/resnet34_similarity/adjacent_similarity.csv
```

如果想看得整齐一点:

```bash
column -s, -t < analysis_outputs/resnet34_similarity/adjacent_similarity.csv
```

### 查看 `.npy` 数组文件

```bash
python3 - <<'PY'
import numpy as np

feature_sim = np.load('analysis_outputs/resnet34_similarity/feature_similarity_matrix.npy')
pixel_sim = np.load('analysis_outputs/resnet34_similarity/pixel_similarity_matrix.npy')
features = np.load('analysis_outputs/resnet34_similarity/feature_vectors.npy')

print('feature_similarity_matrix shape =', feature_sim.shape)
print('pixel_similarity_matrix shape   =', pixel_sim.shape)
print('feature_vectors shape           =', features.shape)
PY
```

## 5. 如何重新生成这份输出

本次分析脚本在项目根目录:

- `analyze_resnet34_similarity.py`

重新运行:

```bash
cd /Users/sull/Documents/test_resnet
TORCH_HOME=/tmp/torch_cache python3 analyze_resnet34_similarity.py --weights default
```

输出默认会写到:

- `analysis_outputs/resnet34_similarity/`

如果你后面想和随机初始化的 `ResNet34` 对比，建议换一个输出目录，避免覆盖:

```bash
python3 analyze_resnet34_similarity.py --weights none --output-dir analysis_outputs/resnet34_similarity_random
```

## 6. 建议的下一步

如果你想更接近真实问题，下一步最有价值的是:

- 对你当前强化学习模型自己的视觉编码器做同样的特征相似度分析
- 检查 `0.5m` 前后策略输出是否变得不稳定
- 看是否需要引入时序信息或距离/位姿辅助输入，而不是只依赖单帧图像
