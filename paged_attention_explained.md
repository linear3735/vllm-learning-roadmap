# vLLM PagedAttention 代码讲解

> 基于 vLLM `main` 分支源码（2026-08 抓取）。
> 涉及文件：
> - vLLM V1 调度：`vllm/v1/core/sched/scheduler.py`
> - KV 缓存管理：`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/single_type_kv_cache_manager.py`
> - 缓存池与哈希表：`vllm/v1/core/block_pool.py`
> - 块哈希计算：`vllm/v1/core/kv_cache_utils.py`
>
> 配套文档：`prefix_caching_explained.md`（建在 PagedAttention 之上的 APC 上层优化）。

---

## 一、为什么需要 PagedAttention（KV 缓存的碎片化）

自回归推理时，每个生成的 token 都要缓存它的 Key / Value（KV Cache），供后续 attention 复用。问题在于 **KV 缓存的显存管理**：

- **传统连续预留**：为每条序列预留一段「最大可能长度」的连续显存。实际长度往往远小于预留值 → **内部碎片**（预留但用不满）。
- **批内碎片**：不同请求长短不一，预留块之间常塞不进新请求 → **外部碎片**。
- 论文指出传统做法可浪费 **60–80%** 显存，直接限制了 batch size 与吞吐。
- 另一个痛点：**跨请求无法共享相同前缀**（beam search、并行采样、投机解码都会重复计算同一段 prompt 前缀）。

PagedAttention 就是为这两个问题而生的显存管理算法（Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*, 2023）。

---

## 二、核心思想：把 OS 的「分页」借用到 KV 缓存

灵感来自操作系统的**虚拟内存分页**：

- 把每条序列的 KV 缓存切成固定大小的**块（block / page，含固定 token 数）**。
- 块在物理显存里**不需要连续**——靠一张「块表（block table）」像页表一样做 **逻辑块 → 物理块** 的映射。
- 按需分配：每多算一个块的 token，才分配一个物理块，绝不预留整段。

类比表：

| 概念 | 操作系统 | PagedAttention |
|---|---|---|
| 最小单位 | 内存页 (page) | KV 块 (block，固定 token 数) |
| 地址映射 | 页表 | 块表 (block table) |
| 逻辑 vs 物理 | 虚拟地址连续，物理分散 | 逻辑块连续，物理块分散 |
| 分配策略 | 按需调页 | 按 token 增量分配块 |
| 共享 | 写时复制 (CoW) | 前缀共享 + CoW |

---

## 三、块表（block table）是什么

块表就是一串物理块 ID，挂在请求上：

```python
# KVCacheManager.req_to_blocks[req_id] = [P7, P2, P9, P4, ...]
```

- `P7, P2, P9...` 是分散在显存各处的物理块；逻辑上它们按 0,1,2… 顺序排列。
- **块表是 append-only 的**：运行中请求每步只往里追加新块（`allocate_slots` 返回的 `new_blocks` extend 进 `req_to_blocks`），从不重排已有块——这正是「逻辑块序号固定、物理块可分散」的体现。
- 注意力后端（PagedAttention kernel）就靠块表把「逻辑连续的 token」**gather** 到「物理分散的 KV 块」上算 attention。

---

## 四、内存共享 + 引用计数 + 写时复制

因为物理块是独立寻址的，多个序列只要前缀相同，就能**指向同一批物理块**：

- beam search / 并行采样 / 投机解码：相同前缀只存一份，省显存。
- 用 **引用计数 `ref_cnt`** 记录有多少请求共享某块。
- 当某个请求要往「被共享的块」里写新 token 时，不能直接改共享块（会污染他人），于是用 **写时复制（CoW）** 切出私有副本。细节见 §七。

---

## 五、vLLM V1 调度器如何驱动块表

`Scheduler.schedule()` 每步（continuous batching 的核心）绕 `KVCacheManager` 和块表转：

1. **`new_step_starts()`** —— 重置管理器内部状态（本步 new_block_ids、pending CoW 等）。
2. **先跑 RUNNING**：对每个在跑请求算 `num_new_tokens = num_tokens - num_computed_tokens`，调 `allocate_slots(request, num_new_tokens)` 续分配新 KV 块；分配失败就 `_preempt_request`（释放块、退回 WAITING）。
3. **再接纳 WAITING**：若 `num_computed_tokens == 0`，先 `get_computed_blocks(request)` 查前缀缓存拿到可复用块 + 命中 token 数；然后 `allocate_slots(..., new_computed_blocks=...)` 把命中块挂上、再补新块。
4. **写 `SchedulerOutput`**：`req_to_new_blocks[req_id] = kv_cache_manager.get_blocks(req_id)`，把块 ID 抽成 `get_block_ids()` 塞进输出，交给 Model Runner。
5. **`update_from_output`**：消费 Model Runner 输出，`num_computed_tokens += num_scheduled_token`；finished/stopped 就 `_free_request` → `_free_request_blocks` 释放块。

> 注意（主干演进）：`Scheduler` 现在在 `vllm/v1/core/sched/scheduler.py`（旧 `core/scheduler.py` 已 404），分配入口叫 `allocate_slots`（不是旧的 `allocate_or_resize_blocks`）。

---

## 六、Prefix Caching 哈希表（内容寻址）

PagedAttention 提供「块级可共享、物理分散」的地基；前缀缓存（APC）要复用不同请求间相同前缀，需要一个「**内容 → 物理块**」的索引，这就是哈希表。

### 6.1 链式哈希怎么算

`vllm/v1/core/kv_cache_utils.py`：

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None):
    if not parent_block_hash:
        parent_block_hash = NONE_HASH          # 随机种子，防碰撞
    return BlockHash(hash_function((parent_block_hash, tuple(curr_block_token_ids), extra_keys)))
```

- 第 `i` 块哈希 = `hash(第 i-1 块哈希, 当前块 token ids, extra_keys)` → **链式**。
- 链式 ⇒ 前缀必须连续：某块不匹配，其后全不匹配，直接 `break`。
- `extra_keys`（LoRA / 多模态 / cache_salt）参与 ⇒ 上下文敏感，不同上下文不互污染。
- 计算后 `make_block_hash_with_group_id` 把 group id 以 4 字节大端追加到哈希后，保证不同 KV cache group 不串。
- 块哈希在 `Request` 创建/追加 token 时算好，存 `request.block_hashes`。

### 6.2 哈希表结构（BlockPool）

```python
self.cached_block_hash_to_block = BlockHashToBlockMap()        # 正向：hash → block
self.cached_block_hashes_by_block: dict[int, set[...]] = {}     # 反向：block_id → {hash}
```

- 正向 `cached_block_hash_to_block`：`{ BlockHashWithGroupId : KVCacheBlock | {block_id: KVCacheBlock} }`。通常一个 hash 对应一个 block；极少冲突时退化为 `{block_id: block}` 的 dict，**不做去重**以保证已分配的 block id 不变（块表 append-only）。
- 反向索引用于 `free` 时按块批量清哈希。

### 6.3 查表与写回

- **查（命中）**：`find_longest_cache_hit` 沿 `request.block_hashes` 逐个 `get_cached_block(block_hash, group_ids)`，返回 `computed_blocks + hit_length`。`max_length = num_tokens - 1`（留最后 1 token 重算拿 logits）；EAGLE 还退回一个块供给 draft head。
- **写回（缓存）**：每步结束 `cache_blocks` → `block_pool.cache_full_blocks` → `_insert_block_hash`：仅把**新产生的 full block**（且可达，非 SWA 窗口外）写进映射并给 block 打上 `block_hash`。

### 6.4 淘汰与 ref_cnt 守卫

`evict_blocks` / `_maybe_evict_cached_block` **只摘除哈希映射**，不删块；`ref_cnt > 0` 的块从池里**不会被回收**，只是失去「缓存身份」。`free_blocks` 时 `ref_cnt` 减到 0：无哈希的块优先回收，有哈希的块最后才驱逐（LRU 候选）。`get_new_blocks` 抢到带哈希的块时会先清掉它的哈希再分配。

---

## 七、Partial hit 的 CoW（两种实现）

触发条件——**部分命中（partial hit）**：前缀缓存命中结束在某个块**内部**而非块边界，即 `num_local_computed_tokens % block_size != 0`。这个尾块被多个请求共享（只读缓存），但当前请求要往里写新 token，**不能直接改共享块**。

### 7.1 标准路径 `_apply_cow`（全注意力）

1. `get_new_blocks(1)` 拿私有块 `B12`；
2. 把 `(B5, B12)` 入队 `_pending_cow_copies`，Worker 异步把 `B5` 的 KV 拷到 `B12`；
3. **块表重定向**：`req_blocks[block_idx] = B12`，请求指针从共享 `B5` 改指私有 `B12`；
4. `B12.ref_cnt += 1`；
5. `B12` 填满后 `cache_blocks` 把 `H3·G0 → B12` 也写进哈希表。

→ 哈希表现在同一 hash 下可能同时挂 `B5`（继续服务读者）和 `B12`（写者私有副本）；拷贝完成前两端都保引用，`B5` 不会同一步被 free。

### 7.2 Mamba 路径 `move_block_hashes(src, dst)`

`block_pool.py` 里还有个 CoW 分支：Mamba 这类运行请求的 worker 块表是 **append-only 的，不能改指针**。于是它**不重定向块表，而是把哈希条目本身挪给写者**：

```python
for block_hash in self._remove_cached_block_hashes(src_block):
    self._insert_block_hash(block_hash, dst_block, num_tokens=num_tokens)
```

即把 `H3·G0: B5 → B12` 重指向写者的块。**不拷 KV、不发事件**，缓存身份直接转移给写者块。这是「哈希表级别的 CoW」。

---

## 八、与 prefix_caching_explained.md 的关系

- **PagedAttention 是地基**：把 KV 缓存按固定块切分、按需分配、可共享（块表 + ref_cnt + CoW）。
- **prefix caching（APC）是楼上盖的房**：建在 PagedAttention 之上，用链式哈希（`BlockPool` + `find_longest_cache_hit`）在不同请求间复用相同前缀块。
- 本文 §六/§七 的哈希表与 CoW，正是 APC 得以工作的底层机制；`prefix_caching_explained.md` 讲解的是它的上层编排。

---

## 九、一句话调用链

```
请求到达
  → Request 计算 block_hashes（链式哈希）
  → Scheduler.schedule()
       → 处理 RUNNING：allocate_slots 续分配块
       → 处理 WAITING：get_computed_blocks
            → coordinator.find_longest_cache_hit
                 → 沿 block_hashes 逐块 get_cached_block（BlockPool 哈希表）
       → SchedulerOutput 带上块 ID 交给 Model Runner
  → update_from_output：更新 num_computed_tokens，结束则释放块
  → 每步结束 cache_blocks → block_pool.cache_full_blocks 写回哈希表
  → partial hit 时由 _apply_cow / move_block_hashes 保护共享块
```
