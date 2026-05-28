# 目标检测——从零实现 YOLO

> 检测是在特征图的每个位置上运行的分类加回归，最后用非极大值抑制清理结果。

**类型：** 动手实现
**语言：** Python
**前置知识：** Phase 4 第3课（CNN），Phase 4 第4课（图像分类），Phase 4 第5课（迁移学习）
**预计时间：** ~75分钟

## 学习目标

- 解释将检测转化为密集预测问题的网格加锚框设计，并说明输出张量中每个数字的含义
- 计算框之间的交并比，并从零实现非极大值抑制
- 在预训练骨干上构建一个最小的 YOLO 风格检测头，包括分类、目标性和框回归损失
- 读取检测指标行（precision@0.5、recall、mAP@0.5、mAP@0.5:0.95），并决定下一步调整哪个旋钮

## 问题所在

分类说「这张图像是一只狗」。检测说「有一只狗在像素 (112, 40, 280, 210)，有一只猫在 (400, 180, 560, 310)，画面中没有其他东西。」这一个结构变化——预测可变数量的标记框而不是每张图像一个标签——正是每个自动驾驶系统、每个安防产品、每个文档布局解析器和每个工厂视觉流水线所依赖的。

检测也是视觉中所有工程权衡同时出现的地方。你需要准确的框（回归头），需要每个框的正确类别（分类头），需要模型知道什么都没有时不输出（目标性分数），还需要每个真实物体恰好一个预测（非极大值抑制）。漏掉其中任何一个，流水线就会漏检物体、报告幻觉框，或者对同一个物体在略微不同的位置预测十五次。

YOLO（You Only Look Once，Redmon 等，2016）是使这一切能实时运行的设计，通过一个卷积网络的单次前向传播完成所有工作，同样的结构决策至今仍是现代检测器的骨干（YOLOv8、YOLOv9、YOLO-NAS、RT-DETR）。学会核心，每个变体都变成了对相同零件的重新排列。

## 核心概念

### 检测作为密集预测

分类器每张图像输出 C 个数字。YOLO 风格检测器每张图像输出 `S × S × (5 + C)` 个数字，其中 S 是空间网格大小。

```mermaid
flowchart LR
    IMG["输入 416x416 RGB"] --> BB["骨干网络<br/>（ResNet, DarkNet, ...）"]
    BB --> FM["特征图<br/>（C_feat, 13, 13）"]
    FM --> HEAD["检测头<br/>（1x1 卷积）"]
    HEAD --> OUT["输出张量<br/>（13, 13, B*(5+C)）"]
    OUT --> DEC["解码<br/>（网格 + sigmoid + exp）"]
    DEC --> NMS["非极大值抑制"]
    NMS --> RESULT["最终检测框"]

    style IMG fill:#dbeafe,stroke:#2563eb
    style HEAD fill:#fef3c7,stroke:#d97706
    style NMS fill:#fecaca,stroke:#dc2626
    style RESULT fill:#dcfce7,stroke:#16a34a
```

`S × S` 个网格单元中的每一个预测 `B` 个框。每个框有：

- 4 个数字描述几何：`tx, ty, tw, th`
- 1 个数字是目标性分数：「这个单元中心是否有物体？」
- C 个数字是类别概率

每个单元合计：`B × (5 + C)`。对于 `S=13, B=2, C=20` 的 VOC，每个单元有 50 个数字。

### 为什么需要网格和锚框

纯回归会将每个物体的 `(x, y, w, h)` 预测为绝对坐标。这对卷积网络很难，因为平移图像不应该以相同幅度平移所有预测——每个物体在空间上是锚定的。网格通过将每个真实框分配给其中心所在的网格单元来解决这个问题；只有该单元对该物体负责。

锚框解决第二个问题。一个 3×3 的卷积无法轻易从 16 像素感受野特征单元回归出一个 500 像素宽的框。相反，我们为每个单元预定义 `B` 个先验框形状（锚框），并预测与每个锚框的小偏差。模型学习选择正确的锚框并微调它，而不是从零回归。

```
锚框先验（416x416 输入的示例）：

  小:   (30,  60)
  中:   (75,  170)
  大:   (200, 380)

在每个网格单元，每个锚框发出 (tx, ty, tw, th, obj, c_1, ..., c_C)。
```

现代检测器通常在不同分辨率上使用带有不同锚框集的 FPN——浅层高分辨率图上用小锚框，深层低分辨率图上用大锚框。同样的想法，更多尺度。

### 解码预测

原始的 `tx, ty, tw, th` 不是框坐标；它们是在绘制之前需要变换的回归目标：

```
中心 x  = (sigmoid(tx) + cell_x) * stride
中心 y  = (sigmoid(ty) + cell_y) * stride
宽度    = anchor_w * exp(tw)
高度    = anchor_h * exp(th)
```

`sigmoid` 将中心偏移量保持在单元内。`exp` 让宽度从锚框自由缩放而不会出现符号翻转。`stride` 将网格坐标缩放回像素。这个解码步骤在自 v2 以来的每个 YOLO 版本中都相同。

### IoU（交并比）

检测中两个框之间的通用相似度指标：

```
IoU(A, B) = 面积(A ∩ B) / 面积(A ∪ B)
```

IoU = 1 表示完全重合；IoU = 0 表示无重叠。预测框与真实框之间的 IoU 决定预测是否算作真正例（通常 IoU >= 0.5）。两个预测之间的 IoU 是 NMS 用于去重的依据。

### 非极大值抑制

在相邻锚框上训练的卷积网络通常会为同一物体预测重叠的框。NMS 保留置信度最高的预测，并删除与被选框 IoU 超过阈值的其他任何预测。

```
NMS(boxes, scores, iou_threshold):
    按分数降序排列框
    keep = []
    while boxes 不为空:
        选取最高分框，加入 keep
        删除与被选框 IoU > iou_threshold 的所有框
    return keep
```

典型阈值：目标检测为 0.45。近期检测器用 `soft-NMS`、`DIoU-NMS` 替换标准 NMS，或直接学习抑制（RT-DETR），但结构目的相同。

### 损失函数

YOLO 损失是三个损失的加权求和：

```
L = lambda_coord * L_box(pred, target, 目标存在处)
  + lambda_obj   * L_obj(pred, 1,     目标存在处)
  + lambda_noobj * L_obj(pred, 0,     目标不存在处)
  + lambda_cls   * L_cls(pred, target, 目标存在处)
```

只有包含物体的单元对框回归和分类损失有贡献。没有物体的单元只对目标性损失有贡献（教模型保持沉默）。`lambda_noobj` 通常较小（约 0.5），因为绝大多数单元都是空的，否则它们会主导总损失。

现代变体将 MSE 框损失换成 CIoU / DIoU（直接优化 IoU），用焦点损失处理类别不平衡，用质量焦点损失平衡目标性。三组件结构不变。

### 检测指标

准确率不能迁移到检测。四个能用的数字：

- **Precision@IoU=0.5** — 被计为正例的预测中有多少是真正正确的。
- **Recall@IoU=0.5** — 真实物体中有多少被找到了。
- **AP@0.5** — IoU 阈值 0.5 处的精确率-召回率曲线面积；每个类别一个数字。
- **mAP@0.5:0.95** — IoU 阈值 0.5、0.55、...、0.95 的 AP 平均值。COCO 指标；最严格也最具信息量。

报告全部四个。在 mAP@0.5 上强但在 mAP@0.5:0.95 上弱的检测器，定位粗糙而不精确；用更好的框回归损失修复。高精确率低召回率的检测器过于保守；降低置信度阈值或增加目标性权重。

## 动手实现

### 第1步：IoU

整节课的主力函数。处理 `(x1, y1, x2, y2)` 格式的两个框数组。

```python
import numpy as np

def box_iou(boxes_a, boxes_b):
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-8, None)
```

返回一个 `(N_a, N_b)` 的成对 IoU 矩阵。通过将其中一个数组设为 `(1, 4)` 形状，可以与单个真实框对比使用。

### 第2步：非极大值抑制

```python
def nms(boxes, scores, iou_threshold=0.45):
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[[i]], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return np.array(keep, dtype=np.int64)
```

确定性，来自排序的 `O(N log N)`，与 `torchvision.ops.nms` 在相同输入上的行为一致。

### 第3步：框编码与解码

在像素坐标和网络实际回归的 `(tx, ty, tw, th)` 目标之间转换。

```python
def encode(box_xyxy, cell_x, cell_y, stride, anchor_wh):
    x1, y1, x2, y2 = box_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    tx = cx / stride - cell_x
    ty = cy / stride - cell_y
    tw = np.log(w / anchor_wh[0] + 1e-8)
    th = np.log(h / anchor_wh[1] + 1e-8)
    return np.array([tx, ty, tw, th])


def decode(tx_ty_tw_th, cell_x, cell_y, stride, anchor_wh):
    tx, ty, tw, th = tx_ty_tw_th
    cx = (sigmoid(tx) + cell_x) * stride
    cy = (sigmoid(ty) + cell_y) * stride
    w = anchor_wh[0] * np.exp(tw)
    h = anchor_wh[1] * np.exp(th)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
```

测试：先编码一个框再解码——你应该得到非常接近原始值的结果（在 sigmoid 逆变换不完全可逆时有微小差异）。

### 第4步：一个最小的 YOLO 检测头

一个特征图上的 1×1 卷积，reshape 为 `(B, S, S, num_anchors, 5 + C)`。

```python
import torch
import torch.nn as nn

class YOLOHead(nn.Module):
    def __init__(self, in_c, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_c, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        n, _, h, w = x.shape
        y = self.conv(x)
        y = y.view(n, self.num_anchors, 5 + self.num_classes, h, w)
        y = y.permute(0, 3, 4, 1, 2).contiguous()
        return y
```

输出形状：`(N, H, W, num_anchors, 5 + C)`。最后一维保存 `[tx, ty, tw, th, obj, cls_0, ..., cls_{C-1}]`。

### 第5步：真实标签分配

对于每个真实框，决定哪个 `(单元, 锚框)` 负责它。

```python
def assign_targets(boxes_xyxy, classes, anchors, stride, grid_size, num_classes):
    num_anchors = len(anchors)
    target = np.zeros((grid_size, grid_size, num_anchors, 5 + num_classes), dtype=np.float32)
    has_obj = np.zeros((grid_size, grid_size, num_anchors), dtype=bool)

    for box, cls in zip(boxes_xyxy, classes):
        x1, y1, x2, y2 = box
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        gx, gy = int(cx / stride), int(cy / stride)
        bw, bh = x2 - x1, y2 - y1

        ious = np.array([
            (min(bw, aw) * min(bh, ah)) / (bw * bh + aw * ah - min(bw, aw) * min(bh, ah))
            for aw, ah in anchors
        ])
        best = int(np.argmax(ious))
        aw, ah = anchors[best]

        target[gy, gx, best, 0] = cx / stride - gx
        target[gy, gx, best, 1] = cy / stride - gy
        target[gy, gx, best, 2] = np.log(bw / aw + 1e-8)
        target[gy, gx, best, 3] = np.log(bh / ah + 1e-8)
        target[gy, gx, best, 4] = 1.0
        target[gy, gx, best, 5 + cls] = 1.0
        has_obj[gy, gx, best] = True
    return target, has_obj
```

锚框选择是「与真实框的最佳形状 IoU」——一个匹配 YOLOv2/v3 分配策略的廉价代理。v5 及更高版本使用更复杂的策略（任务对齐匹配、动态 k），是对同一想法的精化。

### 第6步：三个损失函数

```python
def yolo_loss(pred, target, has_obj, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_cls=1.0):
    has_obj_t = torch.from_numpy(has_obj).bool()
    target_t = torch.from_numpy(target).float()

    # 框回归损失：只在有物体的单元上
    box_pred = pred[..., :4][has_obj_t]
    box_true = target_t[..., :4][has_obj_t]
    loss_box = torch.nn.functional.mse_loss(box_pred, box_true, reduction="sum")

    # 目标性损失
    obj_pred = pred[..., 4]
    obj_true = target_t[..., 4]
    loss_obj_pos = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[has_obj_t], obj_true[has_obj_t], reduction="sum")
    loss_obj_neg = torch.nn.functional.binary_cross_entropy_with_logits(
        obj_pred[~has_obj_t], obj_true[~has_obj_t], reduction="sum")

    # 分类损失：只在有物体的单元上
    cls_pred = pred[..., 5:][has_obj_t]
    cls_true = target_t[..., 5:][has_obj_t]
    loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(
        cls_pred, cls_true, reduction="sum")

    total = (lambda_coord * loss_box
             + lambda_obj * loss_obj_pos
             + lambda_noobj * loss_obj_neg
             + lambda_cls * loss_cls)
    return total, {"box": loss_box.item(), "obj_pos": loss_obj_pos.item(),
                   "obj_neg": loss_obj_neg.item(), "cls": loss_cls.item()}
```

五个超参数，每个 YOLO 教程要么硬编码要么扫描。比例很重要：`lambda_coord=5, lambda_noobj=0.5` 反映了原始 YOLOv1 论文，至今仍是合理的默认值。

### 第7步：推理流水线

解码原始检测头输出，应用 sigmoid/exp，基于目标性阈值过滤，然后 NMS。

```python
def postprocess(pred_tensor, anchors, stride, img_size, conf_threshold=0.25, iou_threshold=0.45):
    pred = pred_tensor.detach().cpu().numpy()
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    num_anchors = len(anchors)

    boxes, scores, classes = [], [], []
    for gy in range(grid_h):
        for gx in range(grid_w):
            for a in range(num_anchors):
                tx, ty, tw, th, obj, *cls = pred[0, gy, gx, a]
                score = sigmoid(obj) * sigmoid(np.array(cls)).max()
                if score < conf_threshold:
                    continue
                cls_idx = int(np.argmax(cls))
                cx = (sigmoid(tx) + gx) * stride
                cy = (sigmoid(ty) + gy) * stride
                w = anchors[a][0] * np.exp(tw)
                h = anchors[a][1] * np.exp(th)
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                scores.append(float(score))
                classes.append(cls_idx)

    if not boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep], classes[keep]
```

这是完整的评估路径：检测头 → 解码 → 阈值过滤 → NMS。

## 实际使用

`torchvision.models.detection` 提供具有相同概念结构的生产级检测器。加载预训练模型只需三行。

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
model.eval()
with torch.no_grad():
    predictions = model([torch.randn(3, 400, 600)])
print(predictions[0].keys())
print(f"boxes:  {predictions[0]['boxes'].shape}")
print(f"scores: {predictions[0]['scores'].shape}")
print(f"labels: {predictions[0]['labels'].shape}")
```

对于实时推理流水线，`ultralytics`（YOLOv8/v9）是标准：`from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model(img)`。模型在内部处理解码和 NMS，返回与你上面构建的相同的 `boxes / scores / labels` 三元组。

## 练习

1. **(简单)** 实现 `box_iou`，在 1000 个随机框对上与 `torchvision.ops.box_iou` 对比。验证最大绝对差小于 `1e-6`。
2. **(中等)** 将 `yolo_loss` 移植到使用 `CIoU` 框损失而非 MSE 的版本。在 100 张合成图像的数据集上，展示 CIoU 在相同轮次内收敛到比 MSE 更好的最终 mAP@0.5:0.95。
3. **(困难)** 实现多尺度推理：在三种分辨率下将同一图像通过模型，合并框预测，最后运行一次 NMS。在保留集上测量与单尺度推理相比的 mAP 提升。

## 关键术语

| 术语 | 通常的说法 | 准确含义 |
|------|-----------|---------|
| 锚框 (Anchor) | 「框先验」 | 每个网格单元的预定义框形状，网络从中预测偏差而非绝对坐标 |
| IoU | 「重叠度」 | 两个框的交并比；检测中的通用相似度度量 |
| NMS | 「去重」 | 保留最高分预测并删除超过阈值的重叠框的贪心算法 |
| 目标性 (Objectness) | 「这里有东西吗」 | 每个锚框、每个单元的标量，预测该单元中心是否有物体 |
| 网格步幅 (Grid stride) | 「下采样因子」 | 每个网格单元的像素数；416 像素输入配 13 网格检测头，步幅为 32 |
| mAP | 「平均精确率」 | 精确率-召回率曲线下面积的平均值，在类别上（以及 COCO 中还有 IoU 阈值上）取平均 |
| AP@0.5 | 「PASCAL VOC AP」 | IoU 阈值 0.5 时的平均精确率；指标的宽松版本 |
| mAP@0.5:0.95 | 「COCO AP」 | IoU 阈值 0.5..0.95 步长 0.05 的平均；严格版本，也是当前社区标准 |

## 延伸阅读

- [YOLOv1: You Only Look Once (Redmon et al., 2016)](https://arxiv.org/abs/1506.02640) — 开山之作；每个后续 YOLO 都是这个结构的精化
- [YOLOv3 (Redmon & Farhadi, 2018)](https://arxiv.org/abs/1804.02767) — 引入多尺度 FPN 风格检测头的论文；图解最清晰
- [Ultralytics YOLOv8 文档](https://docs.ultralytics.com) — 当前生产参考；涵盖数据集格式、数据增强、训练方案
- [目标检测图解指南 (Jonathan Hui)](https://jonathan-hui.medium.com/object-detection-series-24d03a12f904) — 最佳的检测器大家族通俗讲解；对理解 DETR、RetinaNet、FCOS 和 YOLO 的关系极有价值
