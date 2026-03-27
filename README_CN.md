# EbbingContext

[![CI](https://github.com/liainseptember-boop/ebbingcontext/actions/workflows/ci.yml/badge.svg)](https://github.com/liainseptember-boop/ebbingcontext/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Coverage: 83%](https://img.shields.io/badge/coverage-83%25-brightgreen.svg)]()

**基于艾宾浩斯遗忘曲线的 LLM Agent 上下文管理引擎**

> 让 AI Agent 像人脑一样管理记忆：该记住的牢牢记住，该遗忘的优雅释放。

[English](./README.md) | [架构设计详解](./ARCHITECTURE.md) | [在线演示 →](https://liainseptember-boop.github.io/ebbingcontext/)

---

## 问题

大语言模型 Agent 的上下文管理面临三重困境：

| 困境 | 表现 | 后果 |
|------|------|------|
| **上下文窗口有限** | 即使 128K tokens 也会被 agentic loop 耗尽 | 关键信息被截断丢失 |
| **注意力分布不均** | "Lost in the Middle" — 模型对中间内容关注度骤降 | 重要信息被忽视 |
| **全量保留低效** | Tool 返回 40+ 字段只有 5 个相关，无差别累积 | 噪声淹没关键上下文 |

现有方案——手动裁剪 Tool 输出、progressive summarization、显式 prompt 管理——全都依赖开发者的手动决策。每个 Agent 项目重复造轮子。

## 解决方案

EbbingContext 将艾宾浩斯遗忘曲线的认知科学原理工程化，构建自适应的上下文管理引擎：

```
人脑记忆规律                    EbbingContext 映射
─────────────                  ─────────────────
新学知识快速遗忘         →      新信息初始衰减速率高
反复复习强化记忆         →      被多次引用的信息强度自动提升
重要记忆衰减更慢         →      高重要性信息的衰减系数更低
遗忘释放认知资源         →      低强度信息移出上下文窗口
短期记忆 → 长期记忆      →      高频高重要性信息迁移至持久层
```

## 核心特性

- **三种衰减策略** — Pin / Decay-Recoverable / Decay-Irreversible，按信息丢失后果自动分类
- **双维度衰减** — 会话内按 token 距离，跨会话按物理时间
- **三层存储** — Active / Warm / Archive，强度阈值驱动自动迁移
- **审计链** — Archive 保留完整衰减路径、引用关系和快照上下文
- **位置编排** — 高强度记忆自动放在上下文开头和结尾，应对 "Lost in the Middle"
- **安全控制** — 双标签分类（衰减策略 + 敏感等级），Agent 间传递按权限过滤
- **模型无关** — 内置轻量分类模型开箱即用，可切换到开发者自己的模型

## 与现有方案对比

| 维度 | FIFO | RAG | Mem0 | MemGPT | **EbbingContext** |
|------|------|-----|------|--------|-------------------|
| **遗忘机制** | 无（截断） | 无 | 简单 TTL | 无 | 自适应指数衰减 |
| **重要性区分** | ❌ | ❌ | ✅ LLM 评分 | ❌ | ✅ 重要性×频率×时间 |
| **召回增强** | ❌ | ❌ | ⚠️ 有限 | ❌ | ✅ 每次检索自动增强 |
| **层级管理** | 单层 | 单层 | 单层 | 双层 | 三层 + 审计链 |
| **安全控制** | ❌ | ❌ | ❌ | ❌ | ✅ 敏感等级 + 权限 |
| **位置编排** | ❌ | ❌ | ❌ | ❌ | ✅ 应对 Lost in Middle |

## MCP Tool 接口

| Tool | 功能 |
|------|------|
| `store_memory` | 存储信息（LLM 决定存不存，系统决定怎么分类） |
| `recall_memory` | 检索相关记忆（按 相似度 × 强度 排序） |
| `pin_memory` | 标记不可遗忘（LLM 显式操作优先于系统判定） |
| `forget_memory` | 显式移除（移入 Archive 而非删除） |
| `inspect_memory` | 查看记忆状态（强度、分类、衰减路径） |
| `transfer_memory` | Agent 间传递记忆（按敏感等级和权限过滤） |

## 快速开始

### 安装

```bash
# 核心包（内置轻量嵌入模型，开箱即用）
pip install ebbingcontext

# 使用本地 BGE-M3 嵌入模型（生产推荐）
pip install ebbingcontext[bge]

# 使用 OpenAI 嵌入 + LLM 分类兜底
pip install ebbingcontext[openai]

# 所有可选依赖
pip install ebbingcontext[bge,openai]
```

### Python API

```python
from ebbingcontext import MemoryEngine

engine = MemoryEngine()

# 存储 — 系统自动分类衰减策略、敏感等级、重要性
engine.store("用户偏好：喜欢简洁的代码风格", importance=0.9)
engine.store("API key: sk-xxx", source_type="tool")  # 自动分类为 SENSITIVE

# 召回 — 按 相似度 × 记忆强度 排序
results = engine.recall("代码风格", top_k=5)
for scored in results:
    print(f"{scored.item.content} (分数: {scored.final_score:.2f})")

# 组装 Prompt — 自动管理 token 预算
prompt = engine.recall_for_prompt(
    query="代码风格",
    total_window=128000,
    system_prompt="你是一个有用的助手。",
)
print(f"使用 {prompt.total_tokens} tokens，包含 {prompt.memories_included} 条记忆")

# Pin / Forget / Inspect
engine.pin(results[0].item.id)
info = engine.inspect(results[0].item.id)
engine.forget(results[0].item.id)
```

### MCP 服务器

```bash
# 启动（内置分类模型，开箱即用）
ebbingcontext serve

# 指定配置
ebbingcontext serve --config config.yaml
```

### 持久化

```python
from ebbingcontext import MemoryEngine, EbbingConfig, load_config

# 启用持久化存储（JSON + ChromaDB + SQLite）
config = load_config("config.yaml")  # 设置 storage.persist: true
engine = MemoryEngine.from_config(config)

# 数据跨会话保持
engine.store("这条记忆会在重启后依然存在")
```

## Demo

交互式演示，实时观察记忆如何衰减、被引用后如何回升、如何在三层存储间迁移：

**[→ 在线演示](https://liainseptember-boop.github.io/ebbingcontext/)**

## 性能基准

标准基准测试评估（进行中）：

| 基准 | 指标 | 目标 |
|------|------|------|
| LoCoMo | Multi-hop F1 | ≥ 29.0 |
| MSC | RP@10 | ≥ 75.0 |
| LTI-Bench | 55% 存储下关键事实保留率 | ≥ 80% |

## 文档

- [架构设计详解 → ARCHITECTURE.md](./ARCHITECTURE.md)
- [配置参考](./docs/config.md)
- [API 文档](./docs/api.md)

## 参考

- Ebbinghaus, H. (1885). *Über das Gedächtnis*
- Zhong et al. (2024). *MemoryBank: Enhancing LLMs with Long-Term Memory* (AAAI 2024)
- Wei et al. (2026). *FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory*
- Park et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*

## License

MIT License
