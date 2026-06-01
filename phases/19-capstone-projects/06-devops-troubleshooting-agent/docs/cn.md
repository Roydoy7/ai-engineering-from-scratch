# 压轴项目 06——Kubernetes DevOps 故障排查智能体（Capstone 06 — DevOps Troubleshooting Agent for Kubernetes）

> AWS 的 DevOps Agent 正式上线，Resolve AI 发布了 K8s 操作手册，NeuBird 演示了语义监控，Metoro 将 AI SRE 与每个服务的 SLO 绑定。生产形态已经确立：告警 webhook 触发，智能体读取遥测数据，遍历 K8s 对象图，对根因假设进行排名，并在 Slack 上发送带批准按钮的简报。默认只读。每个修复操作都由人类把关。本压轴项目就是这个智能体，针对 20 个合成事件进行评估，并在三个共享案例上与 AWS 的 Agent 进行比较。

**类型：** 压轴项目  
**语言：** Python（智能体），TypeScript（Slack 集成）  
**前置知识：** Phase 11（LLM 工程）、Phase 13（工具和 MCP）、Phase 14（智能体）、Phase 15（自主系统）、Phase 17（基础设施）、Phase 18（安全）  
**涉及的阶段：** P11 · P13 · P14 · P15 · P17 · P18  
**预计时间：** 30 小时

## 问题所在

2025-2026 年的 SRE 叙事变成了："AI 智能体分流事件，人类批准修复。"AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 都在生产中采用这种形态。智能体读取 Prometheus 指标、Loki 日志、Tempo 追踪、kube-state-metrics 和 K8s 对象的知识图谱。它在五分钟内生成带遥测引用的排名根因假设。在没有通过 Slack 获得人类明确批准的情况下，它绝不执行破坏性命令。

大部分艰难工作在于范围限定和安全，而非推理。智能体需要一个默认只读的 RBAC 界面、一个强化的 MCP 工具服务器，以及每个被考虑但未执行命令的审计日志。它需要知道何时超出能力范围并进行升级。而且它运行成本必须足够低，OOM 杀死级联不会产生 5000 美元的智能体账单。

## 核心概念

智能体在知识图谱上操作。节点是 K8s 对象（Pod、Deployment、Service、Node、HPA、PVC）加上遥测来源（Prometheus 系列、Loki 流、Tempo 追踪）。边编码所有权（Pod -> ReplicaSet -> Deployment）、调度（Pod -> Node）和观察（Pod -> Prometheus 系列）。图通过 kube-state-metrics 同步保持新鲜，并在每次告警时重新采样。

当告警触发时，智能体从受影响的对象开始进行根因分析。它遍历边，拉取相关遥测切片（最近 15 分钟），并起草假设。假设按证据排名：有多少遥测引用支持它，有多近，有多具体。前 3 个假设带图路径可视化和修复操作批准按钮发送到 Slack。

修复是有门控的。允许的默认操作是只读的。破坏性操作（缩容、回滚、删除 Pod）需要 Slack 批准；ArgoCD 回滚钩子需要智能体永远不持有的认证令牌。审计日志记录智能体*考虑*的每个命令——不仅仅是执行的——所以审查过程能捕获险些失手的情况。

## 架构

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI 接收器
           |
           v
   LangGraph 根因分析智能体
           |
           +---- 只读 MCP 工具 ----+
           |                      |
           v                      v
   K8s 知识图谱              遥测切片
     (Neo4j / kuzu)      Prometheus, Loki, Tempo
   所有权 + 调度            最近 15m，有范围
           |
           v
   假设排名（证据权重）
           |
           v
   Slack 简报 + 批准按钮
           |
           v（批准后）
   ArgoCD 回滚钩子 / PagerDuty 升级
           |
           v
   审计日志：考虑过的 vs 执行的，每个命令
```

## 技术栈

- 可观测性来源：Prometheus、Loki、Tempo、kube-state-metrics
- 知识图谱：K8s 对象 + 遥测边的 Neo4j（托管）或 kuzu（嵌入）
- 智能体：LangGraph，每工具允许列表，默认只读
- 工具传输：FastMCP over StreamableHTTP；破坏性工具的单独服务器在批准门后
- 模型：Claude Sonnet 4.7 用于根因推理，Gemini 2.5 Flash 用于日志摘要
- 修复：ArgoCD 回滚 webhook，PagerDuty 升级，Slack 批准卡片
- 审计：仅追加结构化日志（考虑过的，执行的，批准的，结果）
- 部署：K8s deployment，带自己的窄 RBAC 角色；单独命名空间

## 构建它

1. **图摄入。** 每 30 秒将 kube-state-metrics 同步到 Neo4j/kuzu。节点：Pod、Deployment、Node、Service、PVC、HPA。边：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测覆盖边：OBSERVED_BY（Pod 被 Prometheus 系列观察）。

2. **告警接收器。** FastAPI 端点，接受 PagerDuty 或 Alertmanager webhook。提取受影响的对象和 SLO 违规。

3. **只读工具界面。** 通过 FastMCP 封装 kubectl、Prometheus 查询、Loki logql、Tempo traceql。每个工具有窄 RBAC 动词（"get"、"list"、"describe"）。默认服务器中没有"delete"、"exec"、"scale"。

4. **根因智能体。** LangGraph，三个节点：`sample` 拉取最近 15 分钟的遥测切片，`walk` 查询相邻对象的图，`hypothesize` 起草带遥测引用的排名根因候选。

5. **证据评分。** 每个假设的分数 = 时效性 * 特异性 * 图路径长度倒数 * 引用数。返回前 3 个。

6. **Slack 简报。** 发布带假设、图路径可视化（服务器端渲染的子图图像）和最多一个修复操作批准按钮的附件。

7. **修复门控。** 破坏性工具（缩容、回滚、删除）位于批准令牌后面的第二个 MCP 服务器上。只有在 Slack 卡片被人类批准后，智能体才能调用它们。

8. **审计日志。** 仅追加 JSONL：对每个候选命令，记录它是否被考虑，是否被执行，谁批准了它。每日发送到 S3。

9. **合成事件套件。** 构建 20 个场景：OOMKill 级联、DNS 抖动、HPA 颠簸、PVC 填满、噪音邻居、有故障的 sidecar、错误 ConfigMap 发布、证书轮换、镜像拉取退避等。在根因准确性和假设生成时间上对智能体评分。

## 使用它

```
webhook: alert.pagerduty.com -> checkout-api SLO 违规，错误率 14%
[图]    受影响：Deployment checkout-api（3 个 Pod，节点 ip-10-2-3-4）
[遍历]  邻居：ReplicaSet checkout-api-abc，Service checkout-api，
           14 分钟前的最近发布
[采样]  prometheus error_rate 14%，上升趋势；loki /api/v2/pay 上的 500
[假设]  #1 不良发布：最新镜像 checkout-api:v2.41 在 /healthz 失败
          引用：deploy.yaml（第 42 次修订），prometheus errorRate，loki 500 堆栈
[slack]  [回滚到 v2.40]  [升级]  [忽略]
          （需要批准；智能体不会单方面回滚）
```

## 交付它

`outputs/skill-devops-agent.md` 是可交付成果。给定 K8s 集群和告警来源，智能体生成排名根因假设和 Slack 门控修复流程。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 场景套件上的 RCA 准确率 | 20 个合成事件中 ≥80% 正确根因 |
| 20 | 安全性 | 在审计日志中，破坏性操作守护从不在没有 Slack 批准的情况下触发 |
| 20 | 假设生成时间 | 从告警到 Slack 简报 p50 低于 5 分钟 |
| 20 | 可解释性 | 每个假设都有图路径和遥测引用 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端工作 |
| **100** | | |

## 练习

1. 在 AWS DevOps Agent 演示的同三个事件上运行你的智能体。发布并排比较。报告智能体在哪里存在分歧。

2. 添加"险些失手"审计，标记智能体*考虑*的任何本会在没有批准的情况下执行破坏性操作的命令。在一周内测量险些失手率。

3. 将假设模型从 Claude Sonnet 4.7 换为自托管的 Llama 3.3 70B。测量 RCA 准确率差距和每事件美元成本。

4. 构建因果过滤器：区分相关遥测峰值和真正的根因。在 20 个场景标签上训练一个小型分类器。

5. 添加回滚演习：ArgoCD 针对使用相同清单的暂存集群进行回滚。在 Slack 批准按钮之前，在实时集群中验证回滚计划。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| K8s knowledge graph（K8s 知识图谱） | "集群图" | 节点 = K8s 对象 + 遥测系列；边 = 所有权，调度，观察 |
| Read-only-by-default（默认只读） | "范围 RBAC" | 智能体的服务账户只有 get/list/describe 动词；破坏性动词在批准后面的单独服务器中 |
| Audit log（审计日志） | "考虑过的 vs 执行的" | 每个候选命令的仅追加记录，它是否运行，谁批准 |
| Hypothesis ranking（假设排名） | "证据分数" | 时效性 × 特异性 × 图路径长度倒数 × 引用数 |
| Slack approval card（Slack 批准卡片） | "HITL 门控" | 带修复按钮的交互式 Slack 消息；智能体在人类点击之前无法继续 |
| Telemetry citation（遥测引用） | "证据指针" | 支持声明的 Prometheus 查询、Loki 选择器或 Tempo 追踪 URL |
| MTTR | "解决时间" | 从告警触发到 SLO 恢复的挂钟时间 |

## 延伸阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 标准性 2026 参考
- [Resolve AI K8s 故障排查](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 竞争参考
- [NeuBird 语义监控](https://www.neubird.ai) — 语义图方法
- [Metoro AI SRE](https://metoro.io) — SLO 优先生产框架
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 集群状态来源
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 参考智能体编排器
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 服务器框架
- [ArgoCD 回滚](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 有门控的修复目标
