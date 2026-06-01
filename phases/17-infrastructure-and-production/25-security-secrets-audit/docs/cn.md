# 安全——密钥、API 密钥轮换、审计日志、护栏（Security — Secrets, API Key Rotation, Audit Logs, Guardrails）

> 通过集中式保险库（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除密钥蔓延。永远不要将凭证存储在配置文件、VCS 中的 env 文件或电子表格中。使用 IAM 角色而非静态密钥；CI/CD 使用 OIDC。AI 网关模式是 2026 年的解决方案：应用→网关→模型供应商，网关在运行时从保险库提取凭证。在保险库中轮换，所有应用在几分钟内获取新凭证——无需重新部署，不再有 Slack 上的"谁有新密钥"消息。轮换策略 ≤ 90 天；在每次提交时使用 TruffleHog / GitGuardian / Gitleaks 扫描。零信任：MFA、SSO、RBAC/ABAC、短期令牌、设备态势。PII 清洗使用实体识别在转发前屏蔽 PHI/PII；一致性分词（Mesh 方法）将敏感值映射到稳定占位符，使 LLM 保留代码/关系语义。网络出站：LLM 服务在专用 VPC/VNet 子网中，仅允许 `api.openai.com`、`api.anthropic.com` 等；阻止所有其他出站。2026 年的事件驱动因素：Vercel 供应链攻击通过被入侵的 CI/CD 凭证在数千个客户部署中泄露了 env 变量。

**类型：** 学习  
**语言：** Python（标准库，玩具 PII 清洗器 + 审计日志写入器）  
**前置知识：** Phase 17 · 19（AI 网关）、Phase 17 · 13（可观测性）  
**预计时间：** 约 60 分钟

## 学习目标

- 列举四种密钥管理反模式（VCS 中的配置文件、硬编码 env、电子表格、静态密钥），并说出其替代方案。
- 解释 AI 网关从保险库提取的模式作为 2026 年生产标准。
- 实现带有一致性分词的 PII 清洗器（相同值 → 相同占位符）使语义得以保留。
- 说出 2026 年 Vercel 供应链事件以及它关于 CI/CD 凭证卫生的教训。

## 问题所在

一个实习生提交了带 API 密钥的 `.env` 文件。他们快速删除了它。密钥已经在 git 历史中——GitGuardian 扫描发现了它，你的轮换流程是"在 Slack 上通知团队，更新 40 个配置文件，重新部署所有服务"。8 小时后，你的服务一半上线了，另一半在等待部署窗口。

另一个问题，用户提示词包含"我的社保号是 123-45-6789"。提示词发给了 OpenAI。你有 BAA，但你的内部政策是在转发前屏蔽 PII。你没有做到。

还有一个问题，你的 EKS 集群的 LLM Pod 可以访问任何互联网主机。有人通过 DNS 查询向攻击者控制的域名泄露数据。什么都没有阻止它。

LLM 服务的安全必须处理所有三个向量。由保险库支持的凭证。PII 清洗。网络出站过滤。审计日志。

## 核心概念

### 集中式保险库 + IAM 角色拉取

**保险库**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。单一事实来源。

**IAM 角色**：应用/网关通过其 IAM 身份进行认证，而非静态密钥。保险库返回令牌有效期内的密钥。

**AI 网关模式**：网关在请求时从保险库拉取 `OPENAI_API_KEY`。在保险库中轮换；下一个请求获取新密钥。无需重新部署。

### 轮换策略 ≤ 90 天

所有 API 密钥、保险库根令牌、CI/CD 凭证。尽可能自动轮换。手动轮换需要记录和跟踪。

### 密钥扫描

- **TruffleHog** — 在提交上进行正则表达式 + 熵检测。
- **GitGuardian** — 商业产品，高准确率。
- **Gitleaks** — 开源，在 CI 中运行。

在每次提交时运行。检测到新密钥时阻止 PR。

### 零信任态势

- 所有账户必须使用 MFA。
- 通过 SAML/OIDC 的 SSO。
- RBAC（基于角色）或 ABAC（基于属性）用于细粒度访问。
- 短期令牌（小时，而非天）。
- 设备态势——只有带磁盘加密的公司设备。

### PII / PHI 清洗

在提示词离开你的基础设施之前：

1. 实体识别（spaCy NER、Presidio、商业产品）。
2. 屏蔽匹配实体：`"我的社保号是 123-45-6789"` → `"我的社保号是 [SSN_TOKEN_A3F]"`。
3. 一致性分词（Mesh 方法）：相同值映射到相同占位符，使 LLM 保留关系。
4. 可选的对 LLM 响应进行反向映射。

静态正则表达式过滤捕获基本模式；NER 捕获更多。两者都使用。

### 输入 + 输出护栏

输入：阻止已知越狱、禁止话题；按用户速率限制。

输出：对泄露密钥（拒绝上下文中的 API 密钥模式、邮箱模式）进行正则表达式清洗，对政策违规进行分类器检测。

### 网络出站白名单

LLM 服务在专用子网中：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、保险库端点。
- 其他一切：丢弃。
- 通过仅允许列表的 DNS 解析器（避免 DNS 隧道泄露）。

### 审计日志

每次 LLM 调用的不可变日志，包含：
- 时间戳。
- 用户/租户。
- 提示词哈希（非原始提示词，出于隐私考虑）。
- 模型 + 版本。
- Token 计数。
- 成本。
- 响应哈希。
- 任何护栏触发。

根据监管要求保留（SOC 2 为 1 年，HIPAA 为 6 年）。

### 2026 年 Vercel 事件

供应链攻击：被入侵的 CI/CD 凭证在数千个客户部署中泄露了 env 变量。教训：CI/CD 凭证等同于生产凭证。存储在保险库中。范围要窄。积极轮换。

### 你应该记住的数字

- 轮换策略：≤ 90 天。
- 每次提交扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD 凭证被入侵 → 数千个客户 env 变量泄露。
- 审计日志保留：SOC 2 = 1 年，HIPAA = 6 年。

## 使用它

`code/main.py` 实现了带一致性分词的玩具 PII 清洗器和仅追加的审计日志。

## 交付它

本课产出 `outputs/skill-llm-security-plan.md`。给定监管范围和当前状态，规划保险库迁移、清洗器、出站、审计日志。

## 练习

1. 运行 `code/main.py`。发送两个引用相同社保号的提示词。确认两者获得相同的占位符。
2. 为调用 OpenAI + Anthropic + Weaviate 的 vLLM-on-EKS 部署设计网络出站策略。
3. 你在 git 历史中发现了一个 2 年前的密钥。正确的响应是什么——轮换密钥、清洗历史，还是两者都做？论证。
4. 你的审计日志每天增长 10 GB。设计保留层（热 30 天、温 12 个月、冷 6 年）。
5. 论证反向分词（将真实值替换回 LLM 响应）是否值得复杂性，还是保持占位符可见更好。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| Vault（保险库） | "密钥存储" | 集中式凭证管理服务 |
| IAM role（IAM 角色） | "基于身份的认证" | 应用承担的角色；返回短期凭证 |
| OIDC for CI/CD | "云签发令牌" | CI 中无静态密钥——通过 OIDC 进行身份验证 |
| TruffleHog / GitGuardian / Gitleaks | "密钥扫描器" | 提交时密钥检测 |
| RBAC / ABAC | "访问控制" | 基于角色 vs 基于属性 |
| PII scrubbing（PII 清洗） | "数据屏蔽" | 删除或分词化敏感实体 |
| Consistent tokenization（一致性分词） | "稳定占位符" | 相同值每次映射到相同令牌 |
| Mesh approach（Mesh 方法） | "Mesh 分词" | 语义保留的分词模式 |
| Egress whitelist（出站白名单） | "出站允许列表" | 只有允许的域名可达 |
| Audit log（审计日志） | "不可变历史" | 用于合规的仅追加记录 |

## 延伸阅读

- [Doppler — 高级 LLM 安全](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — 用密钥引用管理 LLM API 密钥](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM 护栏最佳实践](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — 2026 年密钥管理最佳实践](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII 检测和匿名化
- [HashiCorp Vault 文档](https://developer.hashicorp.com/vault/docs)
