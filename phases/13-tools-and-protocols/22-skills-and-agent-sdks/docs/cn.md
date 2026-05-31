# 技能与智能体 SDK——Anthropic Skills、AGENTS.md、OpenAI Apps SDK（Skills and Agent SDKs — Anthropic Skills, AGENTS.md, OpenAI Apps SDK）

> MCP 回答"有哪些工具存在"。技能回答"如何完成一项任务"。2026 年的技术栈将两者叠加使用。Anthropic 的智能体技能（开放标准，2025 年 12 月发布）以 SKILL.md 的形式发布，支持渐进式展开。OpenAI 的 Apps SDK 是 MCP 加上组件元数据。AGENTS.md（目前已被 60,000+ 个仓库采用）位于仓库根目录，作为项目级智能体上下文。本章说明各层覆盖的内容，并构建一个可跨智能体传递的最小化 SKILL.md + AGENTS.md 包。

**类型：** 学习  
**语言：** Python（标准库，SKILL.md 解析器和加载器）  
**前置知识：** Phase 13 · 07（MCP 服务器）  
**预计时间：** 约 45 分钟

## 学习目标

- 区分三个层次：AGENTS.md（项目上下文）、SKILL.md（可复用的知识方法）、MCP（工具）。
- 编写带有 YAML 前言和渐进式展开的 SKILL.md。
- 以文件系统方式将技能加载到智能体运行时。
- 将技能与 MCP 服务器和 AGENTS.md 组合，使一个包在 Claude Code、Cursor 和 Codex 中都能工作。

## 问题所在

一位工程师将发版说明撰写工作流提炼为多步骤提示："读取最新合并的 PR。按领域分组。各自总结。按团队风格撰写变更日志条目。发布到 Slack 草稿。"他们把这个工作流放在 Notion 文档里供团队使用。

现在他们想在 Claude Code、Cursor 和 Codex CLI 中使用这个工作流。每个智能体加载指令的方式各不相同：Claude Code 的斜杠命令、Cursor 的规则、Codex 的 `.codex.md`。工程师不得不复制三份工作流并分别维护。

AGENTS.md 和 SKILL.md 共同解决了这个问题：

- **AGENTS.md** 位于仓库根目录。每个兼容的智能体在会话开始时读取它。"这个项目是怎么运作的？约定是什么？哪些命令运行测试？"
- **SKILL.md** 是一个可移植的包：YAML 前言（名称、描述）+ Markdown 正文 + 可选资源。支持技能的智能体按名称按需加载它们。
- **MCP**（Phase 13 · 06-14）处理技能需要调用的工具。

三个层次，一个可移植的制品。

## 核心概念

### AGENTS.md（agents.md）

2025 年底发布，截至 2026 年 4 月已被 60,000+ 个仓库采用。一个文件位于仓库根目录。格式：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

智能体在会话开始时读取此文件，并用它来校准针对该项目的行为。2026 年所有主流代码智能体都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的智能体技能（2025 年 12 月作为开放标准发布）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

前言声明技能的身份。正文是技能加载时展示给模型的提示内容。

### 渐进式展开

技能可以引用子资源，智能体仅在需要时才获取它们。示例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 写道"风格规则见 style-guide.md"。智能体仅在技能实际运行时才拉取 style-guide.md。这避免了将模型可能不需要的细节充斥进提示。

### 文件系统发现

智能体运行时扫描已知目录中的 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- 项目 `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

按文件夹名称和前言 `name` 字段加载。Claude Code、Anthropic Claude Agent SDK 和 SkillKit（跨智能体）都遵循此模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）在会话开始时加载技能，将其作为可调用的"智能体"暴露在运行时中。当用户调用技能时，智能体循环会分发给对应技能。

### OpenAI Apps SDK

2025 年 10 月发布；直接构建于 MCP 之上。将 OpenAI 之前的 Connectors 和 Custom GPT Actions 统一到单一开发者接口。Apps SDK 应用由以下部分组成：

- 一个 MCP 服务器（工具、资源、提示）。
- 加上 ChatGPT UI 的组件元数据。
- 加上可选的 MCP Apps `ui://` 资源，用于交互式界面。

相同的协议，更丰富的用户体验。

### 通过 SkillKit 实现跨智能体可移植性

SkillKit 等跨智能体分发工具将单一 SKILL.md 转换为 32+ 个 AI 智能体（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）的原生格式。一个真相来源，多个消费者。

### 三层技术栈

| 层次 | 文件 | 加载时机 | 用途 |
|------|------|----------|------|
| AGENTS.md | 仓库根目录 | 会话开始时 | 项目级约定 |
| SKILL.md | 技能目录 | 调用技能时 | 可复用的工作流 |
| MCP 服务器 | 外部进程 | 需要工具时 | 可调用的操作 |

三者协同组合：智能体在会话开始时读取 AGENTS.md，用户调用技能，技能指令包含 MCP 工具调用，智能体通过 MCP 客户端分发调用。

## 动手使用

`code/main.py` 提供了一个标准库 SKILL.md 解析器和加载器。它在 `./skills/` 下发现技能，解析 YAML 前言和 Markdown 正文，并生成以技能名称为键的字典。然后模拟一个按名称调用 `release-notes-writer` 的智能体循环。

要关注的内容：

- 用最小标准库解析器（无需 `pyyaml` 依赖）解析 YAML 前言。
- 技能正文原文存储；调用时智能体将其追加到系统提示前。
- 通过 `read_subresource` 函数演示渐进式展开，按需拉取引用的文件。

## 输出产物

本章生成 `outputs/skill-agent-bundle.md`。给定一个工作流，该技能生成组合的 SKILL.md + AGENTS.md + MCP 服务器蓝图包，可跨智能体移植。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个技能，确认加载器能识别它。

2. 为本课程仓库编写一个 AGENTS.md。包括测试命令、代码风格约定和 Phase 13 的心智模型。

3. 将团队内部文档中的一个多步骤工作流移植到 SKILL.md。在 Claude Code 中验证它能正常加载。

4. 手动将该技能翻译为 Cursor 和 Codex 的原生规则格式。统计格式间的差异——这就是 SkillKit 自动化的翻译工作量。

5. 阅读 Anthropic 智能体技能博客文章。找出 Claude Agent SDK 中本章加载器没有覆盖的一个功能。（提示：智能体子调用。）

## 关键术语

| 术语 | 通俗说法 | 准确含义 |
|------|----------|----------|
| SKILL.md | "技能文件" | YAML 前言加 Markdown 正文，由智能体运行时加载。 |
| AGENTS.md | "仓库根目录智能体上下文" | 会话开始时读取的项目级约定文件。 |
| 渐进式展开（Progressive disclosure） | "延迟加载子资源" | 技能正文按需引用拉取的文件。 |
| 前言（Frontmatter） | "顶部 YAML 块" | `---` 分隔符中的元数据（名称、描述）。 |
| Claude Agent SDK | "Anthropic 的技能运行时" | `@anthropic-ai/claude-agent-sdk`，加载技能并路由调用。 |
| OpenAI Apps SDK | "MCP + 组件元数据" | OpenAI 构建于 MCP 之上的开发接口，附带 ChatGPT UI 钩子。 |
| 技能发现（Skill discovery） | "文件系统扫描" | 遍历已知目录寻找 SKILL.md，按名称索引。 |
| 跨智能体可移植性（Cross-agent portability） | "一个技能多个智能体" | 通过 SkillKit 类工具将一个 SKILL.md 转换为 32+ 个智能体格式。 |
| 智能体技能（Agent Skill） | "可移植的知识方法" | MCP 工具概念之外的可复用任务模板。 |
| Apps SDK | "MCP 加 ChatGPT UI" | 基于 MCP 统一 Connectors 和 Custom GPTs 的开发接口。 |

## 延伸阅读

- [Anthropic — 智能体技能公告](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布
- [Anthropic — 智能体技能文档](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 基于 MCP 的 ChatGPT 开发者平台
- [agents.md](https://agents.md/) — AGENTS.md 格式和采用列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方技能示例
