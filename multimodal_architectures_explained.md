# 多模态模型架构与核心组件详解（vLLM-Omni 视角）

> 整理自 vLLM-Omni 架构图与 ViT / DiT / AR / [CLS] 基础概念，用于 AI Infra 面试与源码研读。

## 一、三种多模态架构范式

vLLM-Omni 将当前 any-to-any 多模态模型归纳为三类推理时（inference-time）流水线：

### (a) Thinker-Talker — 思考与说话分离
- 代表：Qwen2.5-Omni / Qwen3-Omni / Ming-Omni
- 流水线：`Encoders → Thinker(AR) → Talker(AR) → Vocoder(DiT) → 音频`
- 要点：Thinker 做深层推理（文本形式的中间结果），Talker 把语义转成语音控制 token，Vocoder(DiT) 逐步去噪合成波形。
- 任务：any-to-any（文本 + 音频输出）。

### (b) AR + DiT Pipeline — 语言模型指挥画笔
- 代表：GLM-Image / Qwen-Image / Wan2.2 / FLUX / MiniMax H3
- 流水线：`Encoders(CLIP) → LLM(AR) → Decoder(DiT) → 图像/视频`
- 要点：LLM 输出中间 token（图像的压缩语义表示），DiT 以该 token + 文本条件逐步去噪出像素。DiT 是主输出模态。
- 任务：text-to-image / text-to-video。

### (c) AR + Specialized Generator — 双专家 MoT
- 代表：BAGEL / Hunyuan Image 3.0 / GR00T-N1
- 流水线：`Encoders → {Und. Expert(AR) ⇄ Gen. Expert(Rectified Flow)}`(KV 共享) `→ 图像`
- 要点：Understanding Expert 做 AR 推理并保留每层 KV cache；Generation Expert 直接复用其 KV 作为 cross-attention 的 memory，用 Rectified Flow（流匹配）少步数（10–28 步）生成像素。
- 任务：t2i / i2i / i2t 等。

### 对比表
| 维度 | (a) Thinker-Talker | (b) AR+DiT | (c) AR+Spec.Gen |
|------|-------------------|------------|-----------------|
| AR 模块 | 2 个串联 (Thinker+Talke r) | 1 个 (LLM) | 1 个 (Und. Expert) |
| 生成模块 | Vocoder (DiT, 仅音频) | Decoder (DiT, 主输出) | Gen. Expert (RF/Flow) |
| 模块通信 | token 序列传递 | token 序列传递 | **KV Cache 共享** |
| 扩散步数 | 多 (音频细粒度) | 中 (20–50) | **少 (10–28, RF 加速)** |
| 典型延迟 | 高 | 中 | 较低 |

vLLM-Omni 用 **OmniStage** 抽象把三者统一成 `Encoder → AR → Generation` 的通用调度框架，每个 stage 可独立分配 GPU、独立做 continuous batching。

---

## 二、ViT（Vision Transformer）

把图像当"单词序列"处理的纯 Transformer 编码器。

- 流程：`图像 224×224×3 → 切 16×16 patch → 展平 Linear Proj → +位置编码 +[CLS] → Transformer Encoder(L=12) → 取 [CLS] 向量`
- **patch = 视觉单词**：16×16×3 = 768 值展平后线性投影，图像从 2D 网格变 1D 序列，直接套用 self-attention。
- **[CLS]**：特殊汇总 token，双向注意力下第 1 层即能看全图，最终向量作全局图像特征。
- **相对 CNN**：去掉局部性 / 平移不变等归纳偏置，靠大规模数据从零学空间关系（JFT-300M 上超 CNN）；小数据需 MAE 等强正则。

最小实现（核心几行）：
```python
patches = x.unfold(2,16,16).unfold(3,16,16)   # [B,3,14,14,16,16]
patches = patches.permute(0,2,3,1,4,5).reshape(B,196,-1)
tok = self.proj(patches)                      # [B,196,768]
tok = torch.cat([cls, tok], 1) + pos          # 前插 [CLS] + 位置
out = self.enc(tok)                           # TransformerEncoder
return self.head(out[:,0])                    # 取 [CLS] 分类
```

---

## 三、DiT（Diffusion Transformer）

用 Transformer 替代 U-Net 做扩散去噪。

- 流程：`z_T~N(0,I) + 时间步 t 嵌入 + 条件(y/文本) → DiT Block ×N(adaLN+Attn+MLP) → 预测噪声 ε → 调度器更新 z_{t-1} → 循环 T 次 → z_0 → VAE Decoder → 图像`
- **核心**：DiT 预测的是"噪声"不是"图像"：`ε_θ(z_t, t, y)`，调度器按 DDPM / DDIM / Flow Matching 规则更新。
- **adaLN-Zero**：条件 (t,y) 经 MLP 生成每层 LayerNorm 的 scale/shift 参数，贯穿每层调制激活（而非只在输入层注入）。
- **非自回归**：并行预测整张 latent 的噪声，不逐像素生成。
- 与经典 DDPM 的唯一区别是骨干网络（U-Net → Transformer），可扩展性更强。Sora / FLUX / SD3 均基于 DiT。

---

## 四、[CLS] token

CLS = **Classification**。BERT 发明的特殊占位 token，搬入 ViT。

- 自身无图像内容，是一个可学习的随机初始化向量（如 768 维）。
- 放在序列第 0 位，在**双向自注意力**下作为"纯接收方"吸收全部 patch 的信息。
- 取最终 [CLS] 向量作整图全局特征 / 接分类头。
- 优于平均池化：attention 学到的是**自适应加权**（按任务决定哪些 patch 重要），且架构统一便于下游复用（检索、检测直接拿 [CLS] 向量）。

---

## 五、AR（Autoregressive 自回归）

- **定义**：按顺序逐个生成 token，第 i 个 token 仅依赖前面 i−1 个：`P(x) = ∏ᵢ P(xᵢ | x_{<i})`。
- **实现**：因果自注意力（下三角 mask），每个位置只能看左侧，看不到未来。
- **与非 AR 对比**：
  - AR（LLM / Thinker / Talker）：串行生成，适合语言 / 推理链。
  - 非 AR（DiT 去噪、BERT 双向）：并行生成；BERT 双向理解、DiT 并行去噪。
- **在三范式里的位置**：Thinker / Talker / LLM 是 AR；Vocoder / Decoder / Gen. Expert 是非 AR（扩散 / 流匹配）。
- 一句话：**多模态模型 = ViT 编码输入 → AR 模块做推理规划 → 非 AR 模块生成像素/波形**。

最小实现（带因果 mask 的小 LM）：
```python
import torch, torch.nn as nn
class TinyAR(nn.Module):
    def __init__(self, vocab=1000, d=128, layers=2, heads=4, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(d, heads, batch_first=True)
            for _ in range(layers)])
        self.mask = nn.Transformer.generate_square_subsequent_mask(max_len)
        self.head = nn.Linear(d, vocab)
    def forward(self, x):                    # x: [B, L] token ids
        h = self.emb(x)                      # [B, L, d]
        for blk in self.blocks:
            h = blk(h, h, tgt_mask=self.mask) # 因果自注意力: 只看左边
        return self.head(h)                  # [B, L, vocab]
```

---

## 速记口诀
- **ViT** = "看图"的 Transformer（编码，理解已有图像）
- **DiT** = "画图"的 Transformer（解码，从噪声生成新图像）
- **[CLS]** = 全局信息汇总槽（海绵 token）
- **AR** = 自回归，串行生成语言/推理（因果 mask）
- 多模态 = ViT 编码 + AR 推理 + 非 AR 生成
