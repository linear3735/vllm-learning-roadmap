# Flash Attention 学习笔记

> 核心结论：**Flash Attention 是一种 IO 感知（IO-aware）的精确注意力算法**。它不近似、不丢精度，只是把计算顺序重排，让 GPU 从「被显存带宽卡死」变成「真正在算东西」。

---

## 1. 标准 Attention 的瓶颈在哪？

对单个注意力头，输入 Q/K/V 的形状都是 `(seq_len, head_dim)`。标准做法是先算分数矩阵 `S = Q·Kᵀ`，再做 Softmax，再乘 V：

```
S = Q @ K^T          # (N, N)   N = seq_len
P = softmax(S)       # (N, N)   归一化
O = P @ V            # (N, N) @ (N, d) = (N, d)
```

**问题在中间那个 `N×N` 矩阵。** 当 `seq_len = 4096`、`head_dim = 128` 时：

```
S 的大小 = 4096 × 4096 × 2 bytes（fp16）≈ 64 MB
```

而 GPU 片上缓存 SRAM 只有约 20 MB（多个 SM 共享），装不下这个矩阵。所以 `S` 和 `P` 必须**写回 HBM（显存），算完再读回来**——这一来一回就是瓶颈。

标准 Attention 的 HBM 读写次数约 **3 次**（写 S、读回 S 算 softmax、写 O），且中间矩阵占 `O(N²)` 显存。序列越长越慢、越爆显存。

---

## 2. SRAM 和 HBM 是什么？

它们是 GPU 上**两层不同速度的存储器**，本质上是计算机体系结构里经典的「内存金字塔」：

| 存储器 | 位置 | 容量 | 带宽 | 角色 |
|--------|------|------|------|------|
| **SRAM**（共享内存 / on-chip） | 计算单元（SM）内部 | 极小（~20–40 MB，全 GPU 共享） | ~19 TB/s | 极快的小缓存 |
| **HBM**（高带宽显存） | GPU 板卡上的显存 | 大（几十 GB） | ~0.9–3 TB/s | GPU 的「内存」 |

**一句话**：HBM 是「大但慢」的显存；SRAM 是「小但快」的片上缓存，比 HBM 快约一个数量级。

- 标准 Attention 把 `N×N` 矩阵反复在 HBM 里搬进搬出 → **卡在 HBM 带宽上**。
- Flash Attention 把小块数据搬进 SRAM，算完即丢、绝不写回 HBM → **绕过 HBM 瓶颈**。

---

## 3. Flash Attention 的核心思想：Tiling + 在线 Softmax

Flash Attention 不存储完整的 `N×N` 注意力矩阵，而是：

1. **Tiling（分块）**：把 Q 按行分成若干块（块大小 `Br`），把 K/V 按列分成若干块（块大小 `Bc`）。每一块小到能放进 SRAM。
2. 外层循环遍历 K/V 的列块，内层循环遍历 Q 的行块。对每个 `(Q块, K/V块)` 组合，在 SRAM 内算局部分数、局部 Softmax、累加输出。
3. **算完即丢**：中间矩阵从不在 HBM 落盘，只在 SRAM 里临时存在。

这样 HBM 读写次数从 ~3 次降到 **1 次**（只写最终输出 O），额外显存从 `O(N²)` 降到 `O(1)`。

---

## 4. 最精妙的部分：在线 Softmax（Online Softmax）

标准 Softmax 的难点：要对第 `i` 行算归一化，得先知道整行的全局最大值 `m = maxⱼ S[i,j]`。但 Flash Attention 是**分块**处理的——处理第 1 块 K/V 时，根本不知道后面块里会不会有更大的分数。

**解法：维护两个运行统计量，每来一个新块就做一次「重缩放」：**
- `m` = 到目前为止看到的最大分数（运行最大值）
- `l` = 到目前为止 Softmax 分母的累加（运行归一化因子）
- `O` = 到目前为止的加权输出累加

每处理一个新块 `j`（对应局部分数 `S_ij`、局部值 `V_j`）：

```
m_new   = max(m_old, rowmax(S_ij))              # 更新最大值
α       = exp(m_old − m_new)                    # 重缩放因子，α ≤ 1
l_new   = α · l_old + Σ exp(S_ij − m_new)       # 分母随新基准重缩放
O_new   = α · O_old + exp(S_ij − m_new) · V_j    # 输出随新基准重缩放
```

**直觉**：每当发现更大的分数（`m` 变大），就把之前累积的 `O` 和 `l` 乘以 `α = exp(旧max − 新max)` 一起「缩小」到新基准，再加上新块的贡献。最终：

```
O_final = O_new / l_new
```

这与标准 Softmax **逐字符等价**——但全程只需 `O(1)` 额外内存，不存任何中间矩阵。

**为什么 `α` 成立（数学本质）**：`e^(S−m_old) = e^(S−m_new) · e^(m_old−m_new)`。指数相减等于乘法，所以一次乘法就把整个历史对齐到新最大值，无需重新遍历旧数据。

---

## 5. 数值走查（最直观的理解）

设某行分数 `S = [2, 5, 3]`，分块处理：块1 = `[2, 5]`，块2 = `[3]`。

**初始**：`m = −∞`，`l = 0`，`O = 0`

**处理块1 = [2, 5]**：
```
rowmax = 5  →  m_new = max(−∞, 5) = 5
α = exp(−∞ − 5) = 0  →  O_old·α = 0，l_old·α = 0
exp(2−5)=0.0498,  exp(5−5)=1.0
l_new = 0 + 0.0498 + 1.0 = 1.0498
O_new = 0.0498·V₁ + 1.0·V₂
```
此时最大值一直是 5，所以 `α=1`（因为没有旧累积需要缩放）——看不出重缩放在起作用。

**处理块2 = [3]**：
```
rowmax = 3  →  m_new = max(5, 3) = 5（不变）
α = exp(5 − 5) = 1
exp(3−5) = 0.1353
l_new = 1·1.0498 + 0.1353 = 1.1851
O_new = 1·(0.0498·V₁ + 1.0·V₂) + 0.1353·V₃
```
到这里 `m=5, l=1.1851`，结果和标准 Softmax 一致。

### 关键续例：再来一个「6」

块3 = `[6]`，它**刷新了运行最大值**：

```
rowmax = 6  →  m_new = max(5, 6) = 6
α = exp(5 − 6) = exp(−1) ≈ 0.3679   ← 注意 α < 1！
exp(6−6) = 1.0

l_new = 0.3679 · 1.1851 + 1.0 = 0.4361 + 1.0 = 1.4361
O_new = 0.3679·(0.0498·V₁ + 1.0·V₂ + 0.1353·V₃) + 1.0·V₄
      = 0.0183·V₁ + 0.3679·V₂ + 0.0498·V₃ + 1.0·V₄
```

**发生了什么**：

| 项 | 旧权重（相对 max=5） | ×α=0.3679 后（相对 max=6） |
|----|------|------|
| V₁（分数2） | 0.0498 | **0.0183** |
| V₂（分数5） | 1.0 | **0.3679** |
| V₃（分数3） | 0.135 | **0.0498** |
| V₄（分数6） | — | **1.0**（新主角） |

旧累积输出被整体乘以 0.3679「缩水」，新块 V₄ 以权重 1 加入。权重分布重新洗牌——V₂ 从绝对主角退到 0.3679，V₄ 成了新的最大权重项。

**验证精确性**：
- 在线结果：`l_final = 1.4361`，`O_new = 0.0183·V₁ + 0.3679·V₂ + 0.0498·V₃ + V₄`
- 标准 Softmax（整行已知 m=6）：`l = 0.0183+0.3679+0.0498+1 = 1.4360`，`O` 完全一致 ✓

所以 `α` 这个重缩放因子，就是 Online Softmax 能在「不全存、不重算」前提下精确等价于标准 Softmax 的根本原因。

---

## 6. 性能对比（为什么快）

| 维度 | Standard Attention | Flash Attention |
|------|-------------------|-----------------|
| HBM 读写次数 | ~3 次（写 S、读回 S、写 O） | **1 次**（只写 O） |
| 中间矩阵存储 | N² 全量存 HBM | **不存储**，片上算完即丢 |
| Softmax 方式 | 需完整行后计算 | **在线增量更新** m, l |
| 实际加速（4K 序列） | 基准 | **~7–8x** |
| 显存节省 | O(N²) 额外开销 | **O(1)** 额外开销 |

Flash Attention 通过「分块计算 + 在线 Softmax」，把 GPU 从「等显存数据搬来搬去」中解放出来。这也是为什么 vLLM、TensorRT-LLM、llama.cpp 等主流框架都默认使用 Flash Attention 作为核心 kernel。

---

## 7. 与 vLLM `flash_attn.py` 的对应

vLLM 的注意力实现是多层后端体系：

| 文件 | 作用 |
|------|------|
| `flash_attn.py` | **Flash Attention-2 前端入口**，调用 `flash_attn_varlen_func` |
| `flashinfer.py` | **FlashInfer 后端**（更优的 kernel 融合） |
| `flashmla.py` | **Flash MLA**，DeepSeek MLA 架构专用优化 |
| `torch_sdpa.py` | PyTorch 原生 `F.scaled_dot_product_attention` fallback |

`flash_attn_varlen_func` 的关键参数正好对应上面的概念：

```
q=query, k=key, v=value,
cu_seqlens_q=..., cu_seqlens_k=...   ← 变长序列的分块边界（对应 Tiling 的 Br/Bc）
max_seqlen_q=..., max_seqlen_k=...     ← 最大序列长度
soft_scale=softmax_scale                ← 1/sqrt(d)
window_size=window_size                 ← 滑动窗口（可选）
```

**`cu_seqlens`（cumulative sequence lengths）是 Flash Attention 处理变长序列的关键**：vLLM 用 continuous batching 把多个不同长度的请求拼成一个大 batch。`cu_seqlens` 记录了每个请求在拼接后 batch 中的起始位置，让 kernel 知道每个请求的 Q/K/V 从哪里开始分块、到哪里结束——这样不同长度的序列能在同一个 kernel launch 里正确处理，而不会互相污染。

---

## 8. Flash Attention 2 / 3 的改进（一句话了解）

- **FA2**：在 kernel 层面减少不必要的 HBM 读写、优化 warp 间通信（让不同 warp 负责不同 K/V 列块但共享 Q 行块），进一步逼近计算峰值。
- **FA3**：用更激进的 warp-specialization 和 Hopper（H100）的 TMA / WGMMA 指令，把 SRAM<->HBM 搬运与计算更好地 overlap。

> 三大支柱串起来：**Tiling** 决定数据按块进 SRAM（不存 N²）；**Online Softmax** 用 `m / l / O` 三个 O(1) 统计量增量累积，靠 `α` 重缩放保证精确性；二者合力实现 **IO 最小化**，把 GPU 从 HBM 带宽瓶颈里解放出来。
