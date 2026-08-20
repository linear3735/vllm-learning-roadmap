# vLLM 学习路线图 · 第 1 层：用起来

> 目标：能独立用 vLLM 做离线推理、起 OpenAI 兼容服务，并讲清 PagedAttention 与连续批处理（continuous batching）。
> 周期：1–2 周，每天 1–2 小时。面向 vLLM 初学者（当前水平：刚了解）。
> 定位：这是通往 AReaL / RL-Kernel / production-stack 三项目的共同底座，先吃透这一层。

## 设备分工（很重要）
- **本机 Mac**：用来读代码、看文档、写示例、理解概念。vLLM 真跑推理需要 CUDA GPU，Mac 上跑不动大模型。
- **GPU 云**：workspace 里的 `vllm-autodl-deploy.md` 是现成的上云入口（AutoDL RTX 3090 单机实战），第 6–7 天用。

## 前置自查
- [ ] Python 3.9+ 基础（函数 / 类 / 虚拟环境 / pip）
- [ ] 知道什么是 LLM、token、prompt（不需要会训练）
- [ ] PyTorch 不要求熟练（第 2 层再补）
- [ ] 一台能联网的机器（Mac 看文档，GPU 云跑服务）

## 每日计划

### Day 1 — 建立地图
- 读官方文档首页：https://docs.vllm.ai （What is vLLM / Features 两节）
- 看 vLLM 仓库 README 的 Quickstart
- 自测：能用一句话向别人解释 vLLM 是什么、解决什么痛点

### Day 2 — 核心概念（本周重点）
- **PagedAttention**：把 KV cache 像操作系统内存一样「分页」管理，避免显存碎片，显存利用率大幅提升。
- **Continuous batching（连续批处理）**：请求来一个就塞进正在跑的 batch，不等凑齐，token 粒度调度。
- **KV cache 是什么、为什么是显存大户**：每生成一个 token 都要缓存历史注意力，序列越长越占显存。
- 自测：在纸上画出「请求进来 → 调度 → 逐 token 生成」的流程草图。

### Day 3 — 读懂离线推理代码（本机即可）
```python
from vllm import LLM, SamplingParams

prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64)

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.prompt, "->", output.outputs[0].text)
```
- 搞懂三个对象：`LLM`（推理引擎）、`SamplingParams`（采样参数）、`outputs`（结果结构）。
- 注：Mac 上跑这个可能需要 GPU 云；本机先读懂代码与 API 即可。

### Day 4 — 起 OpenAI 服务 + 用 SDK 调
服务端（GPU 云上执行）：
```bash
vllm serve facebook/opt-125m --port 8000
```
客户端（任意机器）：
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```
- 理解「vLLM 提供 OpenAI 兼容 API」——这正是 production-stack 能直接包它的原因。

### Day 5 — 配置与性能初探
- 常用启动参数：
  - `--tensor-parallel-size`：多卡张量并行（单卡填 1）
  - `--gpu-memory-utilization`：显存利用率上限（如 0.9）
  - `--max-model-len`：最大上下文长度（如 8192）
  - `--dtype`：精度（half / bfloat16）
  - `--enable-prefix-caching`：前缀缓存（和 RL-Kernel 的 prefix 概念呼应）
- 浏览仓库 `examples/` 目录，看别人怎么用。

### Day 6–7 — 上云实跑（按 vllm-autodl-deploy.md）
参考 workspace 里的 `vllm-autodl-deploy.md`（AutoDL RTX 3090 单机实战），关键步骤：
1. AutoDL 控制台开机 → JupyterLab → Terminal。
2. 验证环境：`nvidia-smi`、`nvcc --version`、`python --version`。
3. 建虚拟环境并安装：`python -m venv .venv && source .venv/bin/activate && pip install vllm`（慢可加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`）。
4. 起服务：
   ```bash
   vllm serve Qwen/Qwen2.5-7B-Instruct \
     --tensor-parallel-size 1 \
     --gpu-memory-utilization 0.9 \
     --max-model-len 8192
   ```
5. 新开 Terminal 测试：`curl http://localhost:8000/v1/models` 和 `/v1/chat/completions`。
6. 模型下载慢：先 `export HF_ENDPOINT=https://hf-mirror.com` 再重启服务。
7. 本机 Mac 访问：用 AutoDL 实例详情页的「快捷指令」SSH 端口转发，或「自定义服务」绑域名。
8. 可选：按文档部署 OpenWebUI 做可视化聊天界面。
9. 省钱：用完点「停止」（非销毁），14 天内数据盘保留。

## 毕业自测（第 1 层达标标准）
1. vLLM 和直接用 transformers 推理，最大区别是什么？
2. PagedAttention 解决了什么显存问题？
3. 用一行命令怎么起一个 OpenAI 兼容服务？
4. `--tensor-parallel-size` 是干什么的？
5. 你能不能照着 `vllm-autodl-deploy.md` 在云上把一个 7B 模型跑通并 curl 出回答？

> 5 条都能答/做出来，第 1 层就算毕业。

## 下一步（别急，先毕业再走）
- **第 2 层：懂架构** —— 顺着代码主线读：
  `vllm/engine`（引擎循环）→ `vllm/core/scheduler.py` + `block_manager`（调度与 KV 块管理）→ `vllm/model_executor`（模型执行）。
- 第 2 层直接服务于你后面要碰的 **AReaL**（怎么把 vLLM 当后端）和 **production-stack**（怎么在 K8s 上部署它）。
- **第 3 层：抠内核**（attention backend / 自定义算子 / 钩子）→ 等学到 **RL-Kernel** 再回头。
