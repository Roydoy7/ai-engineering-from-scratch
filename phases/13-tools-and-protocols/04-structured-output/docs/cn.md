# 结构化输出——JSON Schema、Pydantic、Zod、约束解码（Structured Output — JSON Schema, Pydantic, Zod, Constrained Decoding）

> "礼貌地请求模型返回 JSON"即使在前沿模型上也有 5% 到 15% 的失败率。结构化输出通过约束解码弥补了这一差距：模型被从根本上阻止输出会违反模式的 token。OpenAI 的严格模式、Anthropic 的模式类型化工具使用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type` 和 Zod 的 `.parse` 是同一思想的五种表现形式。本章构建模式验证器和严格模式契约，供学习者在每个生产提取流水线中使用。

**类型：** 构建  
**语言：** Python（标准库，JSON Schema 2020-12 子集）  
**前置知识：** Phase 13 · 02（函数调用深度解析）  
**预计时间：** 约 75 分钟

## 学习目标

- 使用正确的约束（枚举、最小/最大值、必填字段、正则模式）为提取目标编写 JSON Schema 2020-12。
- 解释严格模式和约束解码相较于"生成后验证"提供了哪些不同的保证。
- 区分三种失效模式：解析错误、模式违规、模型拒绝。
- 提交一个带类型化修复和类型化拒绝处理的提取流水线。

## 问题所在

一个读取采购订单邮件的智能体需要将自由文本转化为 `{customer, line_items, total_usd}`。有三种方案。

**方案一：提示模型输出 JSON。** "请以 JSON 格式回复，包含 customer、line_items、total_usd 字段。"在前沿模型上有 85% 到 95% 的成功率。以六种方式失败：缺少花括号、尾随逗号、类型错误、幻觉字段、在 token 上限处截断、泄漏"以下是您的 JSON："之类的散文。

**方案二：生成后验证。** 自由生成、解析、对模式验证、失败时重试。可靠但代价高昂——每次重试都要付费，截断 bug 每次发生都需要额外一轮。

**方案三：约束解码。** 提供商在解码时强制执行模式。无效 token 从采样分布中被屏蔽。输出保证能够解析，保证能够验证。失效模式收缩为一种：拒绝（模型判定输入不符合模式）。

2026 年所有前沿提供商都提供某种形式的方案三。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}` 加上模型拒绝时响应中的 `refusal` 字段。
- **Anthropic。** 对 `tool_use` 输入执行模式，`stop_reason: "refusal"` 不存在，但无工具调用的 `end_turn` 是信号。
- **Gemini。** 请求级 `responseSchema`；2026 年 Gemini 为特定类型提供 token 级语法约束。
- **Pydantic AI。** `output_type=InvoiceModel` 输出类型化为 `InvoiceModel` 的结构化 `RunResult`。
- **Zod（TypeScript）。** 运行时解析器，根据 Zod 模式验证提供商输出；与 OpenAI 的 `beta.chat.completions.parse` 配合使用。

共同主线：声明一次模式，端到端强制执行。

## 核心概念

### JSON Schema 2020-12——通用语言

每个提供商都接受 JSON Schema 2020-12。最常用的构件：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名到子模式的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许值的封闭集合。
- `minimum` / `maximum`（数字），`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用于每个数组元素的子模式。
- `additionalProperties`：`false` 禁止额外字段（默认值因模式而异）。

OpenAI 严格模式增加了三个要求：每个属性必须列在 `required` 中，所有地方都要有 `additionalProperties: false`，且不能有未解析的 `$ref`。违反这些规则，API 会在请求时返回 400。

### Pydantic——Python 绑定

Pydantic v2 通过 `model_json_schema()` 从数据类形状的模型生成 JSON Schema。Pydantic AI 在此基础上封装，让你只需写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

智能体框架会在边缘将模式翻译为 OpenAI 严格模式、Anthropic `input_schema` 或 Gemini `responseSchema`。模型的输出以类型化的 `Invoice` 实例返回。验证错误会引发带类型化错误路径的 `ValidationError`。

### Zod——TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TypeScript 的等价物。OpenAI 的 Node SDK 暴露了 `zodResponseFormat(Invoice)`，将其翻译为 API 的 JSON Schema 有效载荷。

### 拒绝

严格模式无法强迫模型回答。如果输入不符合模式（"这封邮件是一首诗，不是发票"），模型会发出包含原因的 `refusal` 字段。你的代码必须将其作为一等结果处理，而非失败。拒绝还有用作安全信号的价值：被要求从受保护内容邮件中提取信用卡号的模型，会返回附带安全原因的拒绝。

### 开放权重的约束解码

开放权重实现使用三种技术。

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：从模式构建确定性有限自动机；在每一步，屏蔽会违反 FSM 的 token 的 logit。
2. **带 JSON 解析器的 logit 屏蔽**：与模型同步运行流式 JSON 解析器；在每一步计算有效下一 token 集合。
3. **带验证器的推测解码**：廉价的草稿模型提出 token，验证器强制执行模式。

商业提供商在幕后选择其中一种。2026 年的技术水平是：对短结构化输出比普通生成更快，对长输出速度基本相同。

### 三种失效模式

1. **解析错误。** 输出不是有效的 JSON。在严格模式下不可能发生。在非严格提供商上仍可能出现。
2. **模式违规。** 输出能解析但违反了模式。在严格模式下不可能发生。在严格模式之外很常见。
3. **拒绝。** 模型拒绝。必须作为类型化结果处理。

### 重试策略

当你处于严格模式之外时（Anthropic 工具使用、非严格 OpenAI、旧版 Gemini），恢复模式是：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

通常一次重试就够了。三次重试能捕获弱模型的偶发错误。超过三次是模式有问题的信号：模型对某些输入无法满足模式，需要修改提示或模式。

### 小模型支持

约束解码适用于小模型。在结构化任务上，带语法强制的 3B 参数开放模型优于带原始提示的 70B 参数模型。这是结构化输出在生产中重要的主要原因：它将可靠性与模型大小解耦。

## 动手使用

`code/main.py` 在标准库中提供了一个最小化的 JSON Schema 2020-12 验证器（类型、必填字段、枚举、最小/最大值、正则模式、数组元素、额外属性）。它封装了一个 `Invoice` 模式，并通过验证器运行假的 LLM 输出，演示解析错误、模式违规和拒绝路径。在生产中，将假输出替换为任何提供商的真实响应即可。

要关注的内容：

- 验证器返回带路径和消息的类型化 `[ValidationError]` 列表。这是你希望呈现给重试提示的形状。
- 拒绝分支不重试。它记录并返回类型化的拒绝。Phase 14 · 09 将拒绝用作安全信号。
- `additionalProperties: false` 检查在对抗性测试输入上触发，展示了严格模式如何堵死幻觉字段的入口。

## 输出产物

本章生成 `outputs/skill-structured-output-designer.md`。给定自由文本提取目标（发票、支持工单、简历等），该技能生成兼容严格模式的 JSON Schema 2020-12 和镜像它的 Pydantic 模型，并附带类型化拒绝和重试处理存根。

## 练习

1. 运行 `code/main.py`。添加第四个测试用例，其 `total_usd` 为负数。确认验证器以 `minimum` 约束路径拒绝了它。

2. 扩展验证器以支持带判别器的 `oneOf`。常见情况：`line_item` 是产品或服务之一，用 `kind` 字段标记。严格模式在这里有微妙的规则；查看 OpenAI 的结构化输出指南。

3. 将同一 Invoice 模式写成 Pydantic BaseModel，并将 `model_json_schema()` 输出与手写模式进行比较。找出 Pydantic 默认设置而手写版本缺少的那个字段。

4. 测量拒绝率。构建十个不应该可提取的输入（歌词、数学证明、空白邮件），用真实提供商在严格模式下运行它们。统计拒绝与幻觉输出的数量。这是你用于拒绝感知重试的基准数据。

5. 从头到尾阅读 OpenAI 的结构化输出指南。找出它在严格模式中明确禁止但普通 JSON Schema 允许的那个构件。然后设计一个非本质性地使用该禁止构件的模式，并将其重构为严格模式兼容的版本。

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| JSON Schema 2020-12 | "模式规范" | 每个现代提供商都支持的 IETF 草案模式方言。 |
| 严格模式（Strict mode） | "保证模式" | 通过约束解码强制执行模式的 OpenAI 标志。 |
| 约束解码（Constrained decoding） | "Logit 屏蔽" | 解码时强制执行，屏蔽无效下一 token 的机制。 |
| 拒绝（Refusal） | "模型拒绝" | 输入无法适配模式时的类型化结果。 |
| 解析错误（Parse error） | "无效 JSON" | 输出无法解析为 JSON；在严格模式下不可能。 |
| 模式违规（Schema violation） | "形状错误" | 能解析但违反了类型/必填字段/枚举/范围约束。 |
| `additionalProperties: false` | "不允许额外字段" | 禁止未知字段；OpenAI 严格模式中必须设置。 |
| Pydantic BaseModel | "类型化输出" | 输出并验证 JSON Schema 的 Python 类。 |
| Zod 模式（Zod schema） | "TypeScript 输出类型" | 用于提供商输出验证的 TypeScript 运行时模式。 |
| 语法强制（Grammar enforcement） | "开放权重约束解码" | 基于 FSM 的 logit 屏蔽，如 outlines/guidance 中的实现。 |

## 延伸阅读

- [OpenAI — 结构化输出](https://platform.openai.com/docs/guides/structured-outputs) — 严格模式、拒绝和模式要求
- [OpenAI — 引入结构化输出](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024 年 8 月发布文章，解释解码保证
- [Pydantic AI — 输出](https://ai.pydantic.dev/output/) — 序列化到各提供商的类型化 `output_type` 绑定
- [JSON Schema — 2020-12 发布说明](https://json-schema.org/draft/2020-12/release-notes) — 规范原文
- [Microsoft — Azure OpenAI 中的结构化输出](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 企业部署说明和严格模式注意事项
