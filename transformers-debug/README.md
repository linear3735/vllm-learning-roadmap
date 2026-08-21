# 调试 transformers（opt-125m）学习笔记

> 学习方法来源：志鹏老师——**先学会 debug transformers 把整个模型推理跑通，再 debug vLLM**。
> 核心套路：**差分调试（differential debugging）**——用 transformers 当可信参考，逐层对比 vLLM，第一个 `max abs error` 突变的层就是 bug。
> 本仓库路径：`transformers-debug/`

---

## 0. 环境

- 机器：Apple M5 MacBook Pro，**无 NVIDIA GPU**（CUDA / Triton 跑不了，只能做正确性审查）
- Conda 环境 `dbg`（Python 3.11）：`torch 2.13.0` + `transformers 5.15.1`
- HuggingFace 镜像（绕过直连超时）：
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  export HF_HUB_ENDPOINT=https://hf-mirror.com
  ```
- 模型：`facebook/opt-125m`（12 层 decoder，hidden 768，12 头 × 64 维，词表 50272）

### 生成参考轨迹 `ref_trace.pt`

`run_opt.py` 用 forward hook 干净地 dump 每层输出，**不修改任何库源码**，作为对比 vLLM 的「尺子」：

```bash
/opt/miniconda3/envs/dbg/bin/python run_opt.py
# 产出 ref_trace.pt，26 个 key：layer0_attn ... layer11_attn, layer0_hidden ... layer11_hidden, final_norm, logits
```

各层张量形状均为 `(1, 6, 768)`，logits 为 `(1, 6, 50272)`。

---

## 1. VS Code 调试配置与踩坑

`.vscode/launch.json` 关键点：

```json
{
  "name": "debug opt",
  "type": "debugpy",
  "request": "launch",
  "program": "${workspaceFolder}/run_opt.py",
  "python": "/opt/miniconda3/envs/dbg/bin/python",
  "console": "internalConsole",
  "justMyCode": false,
  "env": { "HF_ENDPOINT": "https://hf-mirror.com" }
}
```

| 配置项 | 作用 / 踩坑 |
|---|---|
| `"python"` 而非 `"pythonPath"` | `debugpy` 类型**不认 `pythonPath`**，必须用 `"python"` 指定解释器，否则掉回系统 `/usr/local/bin/python3` |
| `"justMyCode": false` | 否则断点进不了 `transformers` 库源码（默认只停在你自己的代码） |
| `"console": "internalConsole"` | 绕过终端 conda 激活干扰；调试输出在 DEBUG CONSOLE 面板看 |
| 断点行号 | `transformers/models/opt/modeling_opt.py` 的 **151 行** `query_states = self.q_proj(hidden_states) * self.scaling` |

**三个最坑的点：**

1. **受限模式（Restricted Mode）**：`settings.json` 里 `security.workspace.trust.untrustedFiles: "open"` 会让 VS Code 把工作区当不可信，**`.vscode/launch.json` 根本不加载** → 按 F5 适配器偶尔起、程序永远跑不起来。关掉 `security.workspace.trust.enabled: false` 即可。
2. **Mac 的 F1–F12 默认是媒体键**：调试要 `Fn + F5/F9/F10/F11`，或系统设置里开「将 F1、F2 等键用作标准功能键」。实在不行，**直接点工具栏的调试按钮（▶ 继续 / ⬆ 步过 / ⬇● 步入 / ⬆● 步出）**，不受键盘冲突影响。
3. **Call Stack 帧选错 → `self` 指向错误对象**：在 Debug Console 敲 `self.q_proj` 报 `'OPTDecoderLayer' object has no attribute 'q_proj'`，是因为 Call Stack 当前选中的是 `OPTDecoderLayer.forward` 帧（那里的 `self` 是 Layer，没有 `q_proj`）。点回 `OPTAttention.forward` 帧即可。

---

## 2. 调试会话解剖（停在 151 行时眼前 5 个区域）

| 区域 | 干嘛的 |
|---|---|
| **代码区** | 断点 = **下一行待执行**。停在 151 行时 `q_proj` 还没算、`query_states` 还不存在；`hidden_states` 是上一行算好的**输入** |
| **VARIABLES** | 自动列出当前栈帧的局部变量，如 `hidden_states: Tensor (1,6,768)` |
| **DEBUG CONSOLE** | 实时求值任意 Python 表达式（`hidden_states.shape`、`self.scaling` 等），变量名直接可用 |
| **CALL STACK** | 调用链 `model() → OPTDecoderLayer.forward → OPTAttention.forward`，点任意帧跳去看那层变量 |
| **步进控制** | `F10` 步过（不进函数）/ `F11` 步入（钻进函数）/ `F5` 继续（到下一断点） |

---

## 3. 断点的第一性原理

断点不是让程序变慢，而是在指定行插一个「检查点」。

- **Python 层**：debugpy 用 `sys.settrace` 给每个栈帧注册 trace 函数。解释器**每准备执行一行**都先调它（line event），它查这行有没有断点，有就暂停、把控制权交给 IDE。
- **最小复现**：

```python
import sys
def trace(frame, event, arg):
    if event == 'line' and frame.f_lineno == 151:
        print("hit 151, hidden_states =", frame.f_locals.get('hidden_states'))
        import pdb; pdb.set_trace()
    return trace
sys.settrace(trace)
```

VS Code 的 Variables / Call Stack / Debug Console 只是把这个机制包装成了图形界面。

---

## 4. opt-125m 推理链路（OPTAttention.forward）

### 4.1 输入与 Q/K/V 投影

| 行 | 代码 | 在干嘛 |
|---|---|---|
| 144 | `bsz, tgt_len, _ = hidden_states.size()` | 解包形状 → 1 / 6 / 768 |
| 151 | `query_states = self.q_proj(hidden_states) * self.scaling` | Q 投影（Linear 768→768）+ 缩放 |
| 152 | `query_states.view(bsz,-1,num_heads,head_dim).transpose(1,2)` | 拆多头 → `(1,12,6,64)` |
| 154–155 | `k_proj` / `v_proj` | K、V 投影（**同样输入，不同权重**） |
| 167 | `attention_interface(self, Q, K, V, ...)` | 算注意力 `softmax(Q·Kᵀ/√d)·V` |
| 178–179 | `reshape` + `out_proj` | 拼回 `(1,6,768)` 再过最后一个 Linear |

`q_proj` / `k_proj` / `v_proj` 三个都是 `nn.Linear(768, 768)`，**共享输入 `hidden_states`，但权重矩阵完全不同** —— 这就是同一句话能算出「查询 / 键 / 值」三种角色的原因。`nn.Linear` 内部就是 `F.linear(x, W, b) = x @ W.T + b`，没有魔法。

`self.scaling = self.head_dim ** -0.5 = 1/√64 ≈ 0.125`，**在 Q 上提前乘**（实现取舍，与标准公式 `Q·Kᵀ/√d` 数学等价）。

### 4.2 为什么 `view` 后还要 `transpose`

```python
query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
# 单独 view → (1, 6, 12, 64)   head 在第 3 位
# 再 transpose(1,2) → (1, 12, 6, 64)  交换 token↔head 两维
```

- `view(1, -1, 12, 64)`：`-1` 让 PyTorch 自己算出 `6`，形状 `[batch, token, head, dim]`。
- `transpose(1, 2)`：交换第 1、2 维（token 与 head），变成 `[batch, head, token, dim]`。
- **为什么非要 transpose**：PyTorch 注意力核要求 `(batch, head, seq, dim)` 布局。head 在第 1 维时，同一 head 的 `(seq, dim)` 在内存里连续，GPU 做 `Q·Kᵀ` 矩阵乘能连续读取、最快。

### 4.3 注意力后端分发器

```python
attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
    self.config._attn_implementation, eager_attention_forward
)
```

- 这是**策略模式**：按 `config._attn_implementation` 字符串从注册表挑一个函数。opt-125m 默认是 `None` → 注册表里查不到 → 返回 `default`，即 `eager_attention_forward`。
- 多后端算的是同一套数学 `softmax(QKᵀ/√d)·V`，只是「怎么算」不同：

| 后端 | 怎么算 | 能否单步 |
|---|---|---|
| **eager** | 纯 PyTorch matmul+softmax，最慢最通用 | ✅ 全 Python，最好调试 |
| sdpa | PyTorch 融合算子 | ⚠️ 部分 C++ |
| flash_attention_2/3/4 | FlashAttention，省显存快 | ❌ CUDA 内核（Mac 跑不了） |

- `get_interface` 本质就是带校验的 `dict.get(name, default)`：名字既非 `"eager"` 又没注册 → 直接 `KeyError` 报错（**不是静默兜底**）。

### 4.4 eager_attention_forward 真身（modeling_opt.py:75–94）

```python
key = self.k_proj(hidden_states)
value = self.v_proj(hidden_states)
query_states = query_states.view(bsz, num_heads, tgt_len, head_dim)
key_states = key_states.view(bsz, num_heads, src_len, head_dim)
value_states = value_states.view(bsz, num_heads, src_len, head_dim)
scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * scaling
scores = scores + attention_mask          # 因果掩码：只能看自己及之前的 token
attn_weights = torch.softmax(scores, dim=-1)
attn_output = torch.matmul(attn_weights, value_states)
```

> 进 `eager_attention_forward` 时若停在 `module.py` 的 `__getattr__`，是 PyTorch 在从 `_modules` 取子模块，正常，按 `F10` 步过即可回到 `modeling_opt.py`。

---

## 5. (1, 6, 768) 张量含义

| 维度 | 名字 | 值 | 含义 |
|---|---|---|---|
| 第 1 维 `1` | `bsz` (batch size) | 1 | 这批同时喂了几个样本（只写了一句话） |
| 第 2 维 `6` | `tgt_len` (target length) | 6 | 一句话几个 token |
| 第 3 维 `768` | `embed_dim` | 768 | 每个 token 用多长的向量表示 |

整句翻译：**1 个样本，6 个 token，每个 token 用 768 维向量表示**。第 3 维从头到尾都在，含义从「词向量」逐步变成「被注意力充分混合过的语义表示」。

```python
x = torch.zeros(1, 6, 768)   # 就是一个三维数组的「格子坐标」[样本, 词位置, 维度]
x[0, 2, 100] = 3.14          # 第 0 个样本、第 2 个 token、第 100 维
```

---

## 6. 差分调试闭环（下一步接 vLLM）

1. **设断点**于某层 attention / `lm_head` 等关键节点（已会在 `modeling_opt.py:151` 停）。
2. **看张量**：Variables / Debug Console 看 `shape / dtype / 抽样数值`，判断有无 NaN、量级是否合理。
3. **对照 `ref_trace.pt`**：transformers 是可信参考，vLLM 是待验证实现，每层输出应几乎一致。
4. **判一致**：一致 → 进下一层；不一致 → 这就是分歧点（bug 藏在这一层）。**第一个开始飘的层即锁定目标**。

---

## 7. Debug Console 实操命令清单

```python
# 投影本身的形状与参数（需 Call Stack 在 OPTAttention.forward 帧）
self.scaling                      # → 0.125
self.num_heads, self.head_dim    # → 12, 64
self.q_proj.weight.shape         # → torch.Size([768, 768])
self.q_proj.weight[0, :5]        # 看权重矩阵第一行前 5 个数

# 数据流怎么变
hidden_states.shape              # (1, 6, 768)  进 q_proj 之前
query_states.shape               # F10 走完 151 行后 → (1, 6, 768)
# 再 F10 走完 152 行 → (1, 12, 6, 64)  拆多头成功

# 与参考轨迹对比（差分调试）
import torch
ref = torch.load("ref_trace.pt")
ref["layer0_attn"].shape         # 应和你跑出的 query/attn 形状一致
```

**条件断点（以后查「某层崩了」特别有用）**：右键断点红点 → `Edit Condition`，如 `self.layer_idx == 5` 只在第 5 层停，或 `hidden_states.isnan().any()` 只在出现 NaN 时停。

---

## 8. 复现步骤（Mac）

1. `conda create -n dbg python=3.11 -y && conda activate dbg`
2. `pip install torch transformers`
3. `git clone` 本项目，`File → Open Folder` 打开 `transformers-debug/`（用绝对路径，别用 `~/`）
4. 左下角状态栏选 Python 解释器 → `/opt/miniconda3/envs/dbg/bin/python`
5. 打开 `transformers/models/opt/modeling_opt.py`，在 **151 行**左侧 gutter 点红点
6. 先 `python run_opt.py` 生成 `ref_trace.pt`（或从缓存加载，不依赖网络）
7. `Cmd+Shift+D` → 选 `debug opt` → 点绿色 ▶（或 `Fn+F5`）启动，停在 151 行开始探索
