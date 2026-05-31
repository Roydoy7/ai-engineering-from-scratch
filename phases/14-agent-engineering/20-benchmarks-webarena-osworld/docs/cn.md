# 基准测试：WebArena 与 OSWorld（Benchmarks: WebArena and OSWorld）

> WebArena 测试跨四个自托管应用的网页智能体能力。OSWorld 测试跨 Ubuntu、Windows、macOS 的桌面智能体能力。发布时（2023-2024），两者均显示出最佳智能体与人类之间的巨大差距。差距正在缩小；失败模式尚未改变。

**类型：** 学习  
**语言：** Python（标准库）  
**前置知识：** Phase 14 · 19（SWE-bench、GAIA）  
**预计时间：** 约 60 分钟

## 学习目标

- 描述 WebArena 的四个自托管应用，以及为什么基于执行的评估很重要。
- 解释为什么 OSWorld 使用真实 OS 截图而非无障碍 API。
- 说出 OSWorld 的两个主要失败模式：GUI 定位和操作知识。
- 总结 OSWorld-G 和 OSWorld-Human 在基础基准测试之上新增了什么。

## 问题所在

通用智能体可以调用工具。它们能在浏览器上通过 20 次点击完成购物结算吗？它们能仅用键盘和鼠标配置 Linux 机器吗？这些是 WebArena 和 OSWorld 回答的问题。

## 核心概念

### WebArena（Zhou 等人，ICLR 2024）

- 跨四个自托管网页应用的 812 个长期任务：购物网站、论坛、类 GitLab 开发工具、商业 CMS。
- 加上实用工具：地图、计算器、草稿本。
- 通过 gym API 进行基于执行的评估——订单是否已下，issue 是否已关闭，CMS 页面是否已更新？
- 发布时：最佳 GPT-4 智能体达到 14.41% 成功率，而人类为 78.24%。

自托管框架很重要——该基准测试不会不稳定，因为目标应用是固定且可重现的。

### 扩展

- **VisualWebArena** — 视觉定位任务，成功取决于解读图像（截图作为一等观察对象）。
- **TheAgentCompany**（2024 年 12 月）— 增加终端 + 编码；更类似真实的远程工作环境。

### OSWorld（Xie 等人，NeurIPS 2024）

- 跨 Ubuntu、Windows、macOS 的 369 个真实计算机任务。
- 对真实应用进行自由形式的键盘和鼠标控制。
- 1920×1080 截图作为观察对象。
- 发布时：最佳模型 12.24%，而人类为 72.36%。

### 主要失败模式

1. **GUI 定位（GUI grounding）。** 像素 → 元素映射。模型难以在 1920×1080 中可靠地定位 UI 元素。
2. **操作知识（Operational knowledge）。** 哪个菜单有该设置，哪个键盘快捷键，哪个偏好面板。人类多年积累的知识尾部。

### 后续工作

- **OSWorld-G** — 564 个样本的定位套件 + Jedi 训练集。将定位与规划分解，让你可以分别测量它们。
- **OSWorld-Human** — 手动精选的黄金动作轨迹。显示顶级智能体比必要步骤多用 1.4-2.7 倍（轨迹效率差距）。

### 为什么这很重要

Claude computer use、OpenAI CUA、Gemini 2.5 Computer Use（第 21 课）都在 WebArena 和 OSWorld 形状的工作负载上训练。基准测试是目标；生产模型是发布的答案。

### 基准测试在哪里出错

- **仅截图评估。** OSWorld 是截图驱动的；在 OSWorld 上评估使用 DOM 或无障碍 API 的智能体会错过定位挑战。
- **忽略轨迹长度。** 仅对成功率评分会遗漏 OSWorld-Human 揭示的 1.4-2.7 倍步骤低效。
- **过时的自托管应用。** WebArena 的应用固定特定版本；未重新整理就更新会破坏可比性。

## 构建它

`code/main.py` 实现了一个玩具网页智能体框架：

- 一个最小化的"购物应用"状态机：list_items、add_to_cart、checkout。
- 3 个任务的黄金轨迹。
- 一个尝试每个任务的脚本化智能体。
- 基于执行的评估器（状态检查）和轨迹效率指标（步骤数 vs 黄金）。

运行：

```
python3 code/main.py
```

输出：每个任务的成功率和轨迹效率，镜像 OSWorld-Human 的方法论。

## 使用它

- **WebArena Verified** 在内部集群上自托管用于持续评估。
- **OSWorld** 在 VM 集群中用于桌面智能体。
- **计算机使用智能体**（第 21 课）——Claude、OpenAI CUA、Gemini——都在这样的工作负载上训练。
- **你自己的产品流程** — 为你的前 20 个任务捕获黄金轨迹；每周对智能体运行它们。

## 交付它

`outputs/skill-web-desktop-harness.md` 构建一个带有基于执行评估和轨迹效率指标的网页/桌面智能体框架。

## 练习

1. 用第二个应用（论坛）扩展玩具框架。为其编写 3 个任务加黄金轨迹。
2. 添加每个任务的轨迹效率报告。在你的玩具上，智能体比黄金多用 1x、2x 还是 3x？
3. 实现一个"干扰"工具——一个黄金轨迹从未使用的工具。脚本化智能体是否会受到诱惑？
4. 阅读 OSWorld-G。如何在你自己的评估中将定位失败与规划失败分离？
5. 阅读 WebArena 的应用 README。当你升级某个固定的应用版本时会发生什么问题？

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| WebArena | "网页智能体基准测试" | 跨 4 个自托管应用的 812 个任务；gym 风格评估 |
| VisualWebArena | "视觉 WebArena" | 视觉定位的 WebArena；截图是观察对象 |
| OSWorld | "桌面智能体基准测试" | 在真实 Ubuntu/Windows/macOS 上的 369 个任务 |
| GUI grounding（GUI 定位） | "像素到元素映射" | 模型在 1920×1080 中定位 UI 元素 |
| Operational knowledge（操作知识） | "OS 知识" | 哪个菜单、哪个快捷键、哪个偏好面板 |
| OSWorld-G | "定位套件" | 564 个纯定位样本 + 训练集 |
| OSWorld-Human | "黄金轨迹" | 手动专家动作序列，用于测量效率 |
| Trajectory efficiency（轨迹效率） | "超过黄金的步骤数" | 智能体步骤数除以人类最小步骤数 |

## 延伸阅读

- [Zhou 等人，WebArena（arXiv:2307.13854）](https://arxiv.org/abs/2307.13854) — 四应用网页基准测试
- [Xie 等人，OSWorld（arXiv:2404.07972）](https://arxiv.org/abs/2404.07972) — 跨 OS 桌面基准测试
- [Anthropic，介绍 computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude 的基准测试形状能力
- [OpenAI，Computer-Using Agent](https://openai.com/index/computer-using-agent/) — OSWorld 和 WebArena 数字
