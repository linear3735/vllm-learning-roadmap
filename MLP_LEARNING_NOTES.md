# vLLM 学习笔记：MLP / SwiGLU / 张量并行 / RMSNorm

> 整理自 2026-08-01 的讲解。源码位置：`vllm/model_executor/models/llama.py`（LlamaMLP）、`vllm/model_executor/layers/layernorm.py`（RMSNorm）。

## 1. MLP 基础

MLP（Multi-Layer Perceptron，多层感知机）= 把神经元堆叠成多层、相邻层**全连接**的前馈网络。

通用单层隐藏层 MLP：
```
y = W₂ · ReLU(W₁ · x)
```
- `W₁·x`：第一次线性变换
- `ReLU(·)`：非线性
- `W₂·(...)`：第二次线性变换

核心：没有非线性激活，堆多少层都等价于单层线性变换，学不了曲线边界。隐藏层 + 非线性 → 可逼近任意连续函数（通用近似定理）。

## 2. LlamaMLP = SwiGLU 门控 FFN（源码 L58–95）

骨架与通用 MLP 相同（两次线性变换夹一层非线性），但有三处升级：

```python
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, ...):
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,   # 一次投影出 2·d
            ...)
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            reduce_results=True, ...)
        # hidden_act 硬编码为 silu，否则报错
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)   # [*, hidden] → [*, 2·d]
        x = self.act_fn(x)            # silu(gate) ⊙ up → [*, d]
        x, _ = self.down_proj(x)      # [*, d] → [*, hidden]
        return x
```

- **门控（gated）**：第一次投影其实是 `gate_proj` 和 `up_proj` **融合**成 `gate_up_proj`（输出 `2·d` 宽）。数学上等价于两个独立投影拼接，融合只是省一次 kernel 启动的工程优化。
- **SwiGLU 激活**：`SiluAndMul` 把 `2·d` 沿末维切成两半 `[gate, up]`，算 `silu(gate) ⊙ up`：
  ```
  h = SwiGLU(x) = silu(W_gate·x) ⊙ (W_up·x)
  ```
  `silu(gate)` 输出恒在 (0,1)，相当于"软阀门/调光开关"：按 0~1 比例逐通道调节信息流，比单 ReLU 表达力更强（所以第一投影必须是 `2·d`）。
- **无偏置 + 仅 silu**：线性层 `bias=False`，归一化交给 RMSNorm；`hidden_act` 硬编码 silu。

维度示例（Llama-7B：hidden=4096, intermediate=11008）：
```
x:            [tokens, 4096]
gate_up_proj: [tokens, 22016]      # 2×11008
silu_and_mul: [tokens, 11008]
down_proj:    [tokens, 4096]
```
每层 MLP 参数量 ≈ `3·hidden·intermediate`（gate+up+down）。

## 3. 张量并行（TP）

TP = 把一层的大矩阵乘拆到多张 GPU 合作算。线性层 `Y = X · A`（A 为 `[输入维, 输出维]`）：

- **按列切（Column Parallel）**：切 A 的**输出**维。X **复制**到每卡，每卡产出一份**输出**切片，最后拼接。→ 对应 `gate_up_proj`（MergedColumnParallelLinear）。
- **按行切（Row Parallel）**：切 A 的**输入**维（同时 X 按特征维切开）。每卡用自己那块 X 和对应 A 的行算**部分和**，最后 **all-reduce 求和**。→ 对应 `down_proj`（RowParallelLinear，`reduce_results=True`）。

**为什么是「列并行 → 行并行」这个顺序**：`gate_up` 产出的输出分片，正好就是 `down_proj` 想要的已切开输入，两层之间**零通信**，整层只在 `down_proj` 末尾 all-reduce 一次（Megatron-LM 经典布局）。`tp=N` 即拆到 N 张卡。

> 矩阵乘法的两种视角：列视角 `Y[:,j] = X·A[:,j]`（→ 列并行，X 复制）；行视角（求和轴）`Y = Σ X[:,k-block]·A[k-block,:]`（→ 行并行，需 all-reduce）。

## 4. RMSNorm（源码 layernorm.py:82）

公式（注释直接给出）：
```
x → w * x / sqrt(E[x²] + eps)
```

与 LayerNorm 区别：RMSNorm **砍掉均值中心化**（不做 `x − mean`），只保留"除以均方根缩放"。LLaMA/Gemma/Mistral 等现代 LLM 均用，几乎不掉点且更省算力。

`forward_native` 等价步骤（L111）：
```python
x = x.to(torch.float32)                 # ① 升精度（防 fp16/bf16 平方溢出）
if residual is not None:
    x = x + residual.to(float32)        # ② 残差相加
    residual = x.to(orig_dtype)         #   残差存回原 dtype（pre-norm 和）
variance = x.pow(2).mean(dim=-1, keepdim=True)  # ③ 均方 E[x²]
x = x * torch.rsqrt(variance + eps)     # ④ 除均方根
x = x.to(orig_dtype)                    # ⑤ 转回原 dtype
x = x * self.weight                     # ⑥ 乘可学习权重
```

`forward_cuda`（推理热路径，L149）调用手写融合 kernel `fused_add_rms_norm`，把**「残差相加 + 归一化 + 乘权重」三步融进一个 GPU kernel**，少一次显存读写、少一次 kernel 启动。

`var_hidden_size`（默认 None）：只对 hidden 前一部分做归一化统计（部分 MoE 模型用）。

## 5. 残差（residual）与 eps

**残差连接（skip connection）**：`output = F(x) + x`。`F(x)` 是子层（MLP/attention）学到的变换，`x` 直接绕过子层加到输出。因为 `F(x)` 学的是"输出相对于输入的**残差/增量**"，所以叫残差。作用：给梯度开"高速公路"，缓解梯度消失，让深层网络可训。LLaMA 每个 decoder layer 即 `x = x + attention(norm(x))` / `x = x + mlp(norm(x))`，那个 `+x` 就是残差连接；vLLM 把残差存在 `residual` 张量、层间累加。

**eps（ε，极小量）**：在归一化分母 `sqrt(var + eps)` 中作平滑项，防止分母为 0 或极端小导致数值爆炸。典型 `1e-5`/`1e-6`（LLaMA 用 `1e-6`），正常数值下几乎无影响，仅兜底。同样用于 Adam、softmax 等数值稳定场景。

## 6. Decoder Layer 中的实际顺序（LlamaDecoderLayer）

```
hidden, residual = input_layernorm(hidden, residual)   # RMSNorm + 残差累加（融合）
hidden = self_attn(positions, hidden)
hidden, residual = post_attention_layernorm(hidden, residual)
hidden = mlp(hidden)     # 即上面的 LlamaMLP（SwiGLU）
# MLP 输出先挂起，在下一层 input_layernorm 处才真正加回残差流
```
