# vLLM 源码学习路线（Roadmap）

> 适用对象：已在 AutoDL(RTX 3090)上跑通过 `vllm serve` + Gradio 部署、想深入源码的初学者。
> 仓库根目录为 `/Users/shm/WorkBuddy/vllm`（已通过 gitclone.com 镜像浅克隆 `--depth 1`，commit `dc1b4a6`，2025-04-14 的 V0 版本快照）。
> ⚠️ 路径随版本变化较大，本文所有路径已按本仓库实际结构核对。若日后 `git pull` 升级，路径可能变动，以实际目录为准。
> 学习原则：**先跑通、再读核心数据结构、然后追请求生命周期、最后啃执行与分布式**。不要从头到尾读，要带着问题读。

---

## 0. 先建立全局认知（只读，不写代码）

目标：搞清楚 vLLM 到底解决了什么、核心创新是什么。

1. `README.md` —— 看 Features 列表，知道它支持哪些模型/量化/并行。
2. `docs/source/design/arch_overview.md` —— **必读**，官方架构总览（PagedAttention、continuous batching、块管理）；PagedAttention 内核细节见 `docs/source/design/kernel/paged_attention.md`，前缀缓存设计见 `docs/source/design/automatic_prefix_caching.md`。
3. 两篇奠基材料（网络上搜）：
   - vLLM 论文 *PagedAttention for Efficient LLM Serving* (SOSP 2022)
   - vLLM 官方博客 *Easy, Fast, and Cheap LLM Serving with PagedAttention*
4. 你之前整理的 `prefix_caching_explained.md` 和 `vllm-autodl-deploy.md` —— 复习部署经验，把"跑起来"和"源码"对上号。

**一句话心智模型**：vLLM 把 KV Cache 切成固定大小的「块(block)」，像操作系统管理内存页一样按需分配/回收，从而支持连续批处理(continuous batching)和前缀缓存(prefix caching)，大幅提升吞吐。

---

## 1. 跑起来、摸到 API 两层（动手）

目标：理解 vLLM 对外暴露的两种入口，以及它们背后的引擎是同一个。

- 离线批量入口：`vllm/entrypoints/llm.py` 里的 `LLM` 类（`generate()`）。
  - 配套例子：`examples/offline_inference/`（先跑最小的，`examples/offline_inference/llm_engine_example.py` 或 `examples/offline_inference/neural_search.py` 这种简单脚本）。
- 在线服务入口：`vllm/entrypoints/openai/api_server.py` + `vllm/entrypoints/openai/serving_chat.py`。
  - 你部署时用过的 `vllm serve` 命令就走到这里。
- 关键认知：两个入口最终都驱动同一个核心引擎 `LLMEngine`（见 Phase 3）。`LLM` 是"同步薄封装"，`AsyncLLMEngine` 是"asyncio 封装 + 后台步进循环"。

**今天先做的**：用 `examples/offline_inference/` 里一个小脚本在**本地 Mac** 跑个小模型（如 `facebook/opt-125m` 或 `Qwen/Qwen2-0.5B`），确认本地能 import vllm（注意：Mac 默认 MPS/CPU 后端，GPU 内核用不了，但逻辑能跑通）。

---

## 2. 核心数据结构（精读，这是地基）

按这个顺序读，理解一次请求在内存里长什么样：

1. `vllm/sequence.py` —— `Sequence` / `SequenceGroup` / `SequenceData` / `SequenceStatus`。
   - 一个 `SequenceGroup` = 一个用户请求（可能含多条采样序列）；状态机 `SequenceStatus` 决定了它在调度里的身份。
2. `vllm/sampling_params.py` —— `SamplingParams`，请求级的采样参数（temperature、top_p、max_tokens…）。
3. `vllm/inputs/` 目录 —— `data.py`（`TextPrompt` / `TokensPrompt`）、`preprocess.py`、`parse.py`。
   - 理解文本/词元如何被预处理成引擎能消费的结构。

**检验**：合上文件，能画出"一个请求从 HTTP 进来 → 变成 SequenceGroup → 进调度器"的对象关系图吗？不能就回头重读。

---

## 3. 引擎与请求生命周期（全文重点，反复读）

这是 vLLM 的心脏。顺着"一个请求的一生"读：

1. `vllm/engine/llm_engine.py` —— `LLMEngine`
   - `add_request()`：外部请求如何进入引擎。
   - `step()`：**引擎主循环**。每步：向调度器要批 → 交给 Worker 执行 → 拿回输出 → 处理完成/抢占。
   - `encode_request`、`_process_model_outputs` 也值得看。
2. `vllm/engine/async_llm_engine.py` —— `AsyncLLMEngine`
   - 用 `asyncio.Queue` + 后台 `run_engine_loop` 把 `LLMEngine.step()` 包成异步服务。在线服务靠它。
3. `vllm/core/scheduler.py` —— **调度器，连续批处理的核心**
   - `schedule()`：每步决定哪些序列进这一批（running / waiting / swapped）。
   - 抢占(preemption)与恢复：KV 块不够时如何换出(swap)到 CPU、再换回。
   - 连续批处理(continuous batching)就体现在这里——不同请求在不同步完成、随时加入/离开批次。
4. `vllm/core/block_manager.py` —— **PagedAttention 的块管理**
   - 逻辑块 ↔ 物理块的映射表；块的分配、回收、拷贝。
   - 前缀缓存(prefix caching)如何复用已有物理块（和你那份 `prefix_caching_explained.md` 对照读）。

**检验**：能口述 `step()` 里"调度→执行→后处理"三步各自调用了谁、数据怎么流动吗？

---

## 4. Worker 与模型执行（啃硬件相关的部分）

引擎决定"跑什么"，Worker 决定"怎么跑"。

1. `vllm/worker/worker.py` —— `Worker`
   - `execute_model()`：接收调度器给的 `ModelInput`，驱动一次前向。
   - 多 GPU 时，每个 GPU 一个 Worker 进程。
2. `vllm/worker/model_runner.py` —— `ModelRunner`
   - `prepare_model_input()`：把多个序列拼成 batch 的张量。
   - 调 `model.forward()`，收集采样器输出。
3. `vllm/worker/cache_engine.py` —— **KV Cache 内存管理**
   - 在 GPU/CPU 上按块分配连续的 KV 张量；把 `block_manager` 的逻辑块映射到真实显存。
4. `vllm/attention/` 目录
   - `selector.py`：根据环境选 attention 后端。
   - `ops/paged_attn.py`：PagedAttention 的核心实现（KV 按块寻址的注意力）。
   - `layer.py`：注意力层（`Attention` 类）如何调用后端。
   - `backends/flash_attn.py`、`xformers.py`、`torch_sdpa.py` 等：不同后端实现。
5. `vllm/model_executor/models/llama.py` —— 挑**一个**模型看实现
   - 看 `LlamaForCausalLM` 如何组合 `vllm/model_executor/layers/` 里的注意力层、MLP、词嵌入。
   - 想加新模型，从这里和 `docs/developer/` 学。

---

## 5. 在线服务 / OpenAI 兼容层

把 Phase 1 + Phase 3 串起来：

- `vllm/entrypoints/openai/api_server.py` —— FastAPI 应用，路由定义。
- `vllm/entrypoints/openai/serving_chat.py` / `serving_completion.py` / `serving_engine.py` —— 把 HTTP 请求转成 `EngineArgs` + `AsyncLLMEngine.generate()`。
- `vllm/entrypoints/openai/cli_args.py` —— 你部署时那些命令行参数(`--tensor-parallel-size`、`--gpu-memory-utilization` 等)的解析处。
- 调用链：`curl → FastAPI 路由 → ServingXXX → AsyncLLMEngine.generate() → LLMEngine.step()`。

---

## 6. 分布式推理（多卡 / 多机）

你之前问过 production-stack 多节点，这里先打基础：

1. `vllm/distributed/parallel_state.py` —— 初始化进程组、拿到 rank/world_size。
2. `vllm/distributed/communication_op.py` —— 集合通信封装（all-reduce 等）。
3. `vllm/model_executor/parallel_utils/` —— 张量并行层（`ColumnParallelLinear` / `RowParallelLinear`）。
4. `docs/source/serving/` 下的分布式服务文档 —— 张量并行(TP) / 流水线并行(PP) 概念与使用。
5. 多机编排：**另开仓库** `vllm-project/production-stack`（你之前看过 00-a 多节点教程），用 Kubernetes + Ray 做跨节点调度，vLLM 自身只负责单节点内的并行。

---

## 7. 进阶专题（按需深入）

- 量化：`vllm/model_executor/layers/quantization/`（AWQ / GPTQ / FP8 / BitsAndBytes），配合 `docs/source/models/` 下的量化说明。
- 投机解码(speculative decoding)：`vllm/spec_decode/`。
- 分块预填充(chunked prefill)：`docs/source/design/` 下相关设计文档 + scheduler 相关逻辑。
- 性能剖析：`docs/source/performance/` 下的相关文档，`benchmarks/`。
- 自定义模型/插件：`docs/source/contributing/overview.md`（开发总览）、`docs/source/design/plugin_system.md`（插件系统），新增模型见 `vllm/model_executor/models/` 与 `docs/source/` 中 models 说明、`vllm/plugins/`。
- CUDA 内核：`csrc/`（PagedAttention 等 C++/CUDA 实现），想改性能看这里。

---

## 8. 给初学者的「今天起步清单」

1. ✅ 仓库已克隆到 `/Users/shm/WorkBuddy/vllm`。
2. 读 `docs/models/architecture.md`（30 分钟）。
3. 跑 `examples/offline_inference/` 里一个小脚本（本地 Mac 用小模型验证 import 正常）。
4. 精读 `vllm/sequence.py` 和 `vllm/sampling_params.py`，画一张对象关系图。
5. 下一份目标：读懂 `vllm/engine/llm_engine.py` 的 `step()` 与 `vllm/core/scheduler.py` 的 `schedule()`。

> 节奏建议：Phase 0–3 是"必须打通"的内功，每天啃 1–2 个文件，配 `vllm-autodl-deploy.md` 的实战经验对照，理解会快很多。遇到不懂的 CUDA/分布式细节先跳过，主线（请求生命周期）通了再回头补。

---

## 附：常用仓库路径速查

| 关注点 | 路径 |
|---|---|
| 离线入口 | `vllm/entrypoints/llm.py` |
| 在线服务 | `vllm/entrypoints/openai/api_server.py` |
| 核心引擎 | `vllm/engine/llm_engine.py`、`vllm/engine/async_llm_engine.py` |
| 调度器 | `vllm/core/scheduler.py` |
| 块管理(PagedAttention) | `vllm/core/block_manager.py` |
| 序列数据结构 | `vllm/sequence.py` |
| 输入预处理 | `vllm/inputs/` |
| Worker | `vllm/worker/worker.py`、`model_runner.py`、`cache_engine.py` |
| 注意力(PagedAttention 内核) | `vllm/attention/ops/paged_attn.py`、`vllm/attention/backends/`、`vllm/attention/layer.py` |
| 模型实现 | `vllm/model_executor/models/` |
| 分布式 | `vllm/distributed/` |
| 量化 | `vllm/model_executor/layers/quantization/` |
| 架构文档 | `docs/source/design/arch_overview.md`、`docs/source/design/kernel/paged_attention.md` |
| 例子 | `examples/` |
| 基准测试 | `benchmarks/` |
| CUDA 内核 | `csrc/` |
