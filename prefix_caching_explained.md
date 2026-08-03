# vLLM V1 Prefix Caching（自动前缀缓存 / APC）代码讲解

> 基于 vLLM `main` 分支源码（2026-08 抓取）。
> 涉及文件：
> - `vllm/v1/core/kv_cache_manager.py` —— 顶层 `KVCacheManager`（调度入口）
> - `vllm/v1/core/kv_cache_coordinator.py` —— 多 attention group 协调
> - `vllm/v1/core/single_type_kv_cache_manager.py` —— 各注意力类型的缓存逻辑（核心）
> - `vllm/v1/core/block_pool.py` —— `BlockPool`（缓存块存储与淘汰）
> - `vllm/v1/core/kv_cache_utils.py` —— 块哈希计算

---

## 一、整体架构

vLLM V1 的 prefix caching 叫 **Automatic Prefix Caching (APC)**。一次请求进来后，KV 缓存的调度链路是：

```
KVCacheManager                     (kv_cache_manager.py)
   └─ KVCacheCoordinator          (kv_cache_coordinator.py)   按 KV cache group 分发
        └─ SingleTypeKVCacheManager 的子类（single_type_kv_cache_manager.py）
             ├─ FullAttentionManager      （全注意力，最常见）
             ├─ SlidingWindowManager      （滑动窗口 / SWA）
             ├─ ChunkedLocalAttentionManager
             ├─ MambaManager              （状态空间模型）
             └─ CrossAttentionManager     （编码器-解码器）
        └─ BlockPool                     (block_pool.py)  —— 真正的缓存池
```

要点：
- **同一个 `BlockPool` 被所有 group 共享**，缓存的块用「哈希 → 块」映射存放。
- `KVCacheManager` 只负责编排；真正的**前缀匹配**和**写缓存**都下沉到各 `SingleTypeKVCacheManager` 子类。
- 不同注意力类型复用同一套 `BlockPool`，但各自的命中/淘汰语义不同（例如 sliding window 只缓存窗口内可见的块）。

---

## 二、块哈希怎么算（链式哈希）

文件：`vllm/v1/core/kv_cache_utils.py`

```python
def hash_block_tokens(
    hash_function,
    parent_block_hash,                  # 父块（前一个 block）的哈希；首块为 None
    curr_block_token_ids,               # 当前 block 内的 token ids
    extra_keys=None,                    # LoRA / 多模态特征 / cache_salt 等
) -> BlockHash:
    if not parent_block_hash:
        parent_block_hash = NONE_HASH   # 随机种子，避免哈希碰撞 / 进程间不一致
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )
```

关键设计：

1. **链式（chained）哈希**：第 `i` 块的哈希 = `hash(第 i-1 块的哈希, 当前块 token ids, extra_keys)`。
   - 因此**前缀必须连续**：只要某个块哈希不匹配，它后面所有块的哈希都不可能匹配（因为依赖父哈希），直接 `break` 即可，无需继续探测。
2. **上下文敏感**：`extra_keys` 包含 LoRA id、多模态特征、cache_salt 等。相同 token 前缀 + 不同上下文（如不同 LoRA）不会互相命中，避免污染。
3. **多 group 区分**：哈希计算后，用 `make_block_hash_with_group_id` 把 group id 以 4 字节大端追加到哈希后面，保证不同 KV cache group 不会串：
   ```python
   def make_block_hash_with_group_id(block_hash, group_id):
       return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))
   ```
4. **缓存加速**：`hash_block_tokens` 用 LRU cache 缓存，避免对相同块内容重复计算。

块哈希在 `Request` 创建 / 追加 token 时即算好，存放在 `request.block_hashes`。

---

## 三、BlockPool：缓存的存储与淘汰

文件：`vllm/v1/core/block_pool.py`

### 3.1 核心数据结构

```python
class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching, hash_block_size, ...):
        self.blocks = [KVCacheBlock(i) for i in range(num_gpu_blocks)]
        # 空闲块按"淘汰顺序"维护的双向链表（LRU 顺序）
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        # 缓存映射：{ hash_key : 单个 block 或 {block_id: block} }
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        # null 占位块（block_id=0）
        self.null_block = self.free_block_queue.popleft()
```

`BlockHashToBlockMap` 是一个 `{hash_key: KVCacheBlock | dict[int, KVCacheBlock]}` 的字典：
- 通常一个 hash 对应一个 block；
- 若发生（极少）重复，退化为 `{block_id: block}` 的 dict。
- 注释强调**不做去重**：保证已分配的 block id 不变，使 block table 是 append-only。

### 3.2 查缓存（命中探测）

```python
def get_cached_block(self, block_hash, kv_cache_group_ids):
    cached_blocks = []
    for group_id in kv_cache_group_ids:
        key = make_block_hash_with_group_id(block_hash, group_id)
        block = self.cached_block_hash_to_block.get_one_block(key)
        if not block:
            return None          # 任一 group 未命中 → 整体 miss
        cached_blocks.append(block)
    return cached_blocks
```

### 3.3 写缓存

```python
def cache_full_blocks(self, request, blocks, num_cached_blocks,
                      num_full_blocks, block_size, kv_cache_group_id,
                      block_mask=None):
    if num_cached_blocks >= num_full_blocks:
        return
    new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
    block_hashes = resolve_block_hashes(request.block_hashes, self.hash_block_size, block_size)
    new_block_hashes = block_hashes[num_cached_blocks:]
    for i, blk in enumerate(new_full_blocks):
        if blk.is_null or (block_mask is not None and not block_mask[i]):
            continue            # 滑动窗口等稀疏注意力会跳过不可达块
        key = make_block_hash_with_group_id(new_block_hashes[i], kv_cache_group_id)
        self._insert_block_hash(key, blk, num_tokens=...)   # 写入映射 + 设置 blk.block_hash
```

### 3.4 引用计数与 LRU 淘汰

- **`touch(blocks)`**：前缀命中时调用，把命中块 `ref_cnt += 1` 并从 `free_block_queue` 移除（防止被淘汰）。
- **`free_blocks(ordered_blocks)`**：请求释放块时 `ref_cnt -= 1`；到 0 时：
  - **无哈希的块**（从未被 APC 命中）**先入队** → 优先被淘汰；
  - **有哈希的块**（曾命中 APC）后入队 → 作为 LRU 候选，最后才被驱逐。
- **`get_new_blocks(n)`**：从 `free_block_queue` 取 n 个块；若取到的块仍带哈希，在分配时调用 `_maybe_evict_cached_block` 清掉其哈希（即"抢"走这块做新用途）。
- **`reset_prefix_cache()`**：清空所有哈希映射与块哈希（例如 RLHF 权重更新后让旧缓存失效）。

---

## 四、核心算法：find_longest_cache_hit（前缀匹配）

文件：`single_type_kv_cache_manager.py`，`FullAttentionManager`（约 682 行）

这是 APC 的心脏——给定请求的 `block_hashes`，找出**最长可复用前缀**。

```python
@classmethod
def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids,
                           block_pool, kv_cache_spec, drop_eagle_block,
                           alignment_tokens, dcp_world_size=1, pcp_world_size=1):
    block_size = kv_cache_spec.block_size
    if dcp_world_size > 1:
        block_size *= dcp_world_size

    # 把 hash 列表规整到 block_size 粒度
    block_hashes = resolve_block_hashes(block_hashes, block_pool.hash_block_size, block_size, ...)

    computed_blocks = tuple([] for _ in range(len(kv_cache_group_ids)))

    # Phase 1: 从起始开始，逐个探测"整块"是否命中
    for block_hash in itertools.islice(full_block_hashes, max_length // block_size):
        cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
        if not cached_block:
            break                       # 链式哈希：一旦 miss，后面必 miss
        for computed, cached in zip(computed_blocks, cached_block):
            computed.append(cached)
    hit_length = len(computed_blocks[0]) * block_size

    # Phase 2 (fine-grained 模式，仅当 hash_block_size < block_size):
    # 在第一个未命中块"内部"，从高到低试探更细的 hash 边界，支持 sub-block 命中
    if fine_grained:
        scale_factor = block_size // alignment_tokens
        first_partial_idx = len(computed_blocks[0]) * scale_factor
        max_partial_idx = min(first_partial_idx + scale_factor - 1,
                              max_length // alignment_tokens, len(block_hashes))
        for fine_idx in range(max_partial_idx - 1, first_partial_idx - 1, -1):
            cached_tail = block_pool.get_cached_block(block_hashes[fine_idx], kv_cache_group_ids)
            if not cached_tail:
                continue
            for computed, cached in zip(computed_blocks, cached_tail):
                computed.append(cached)
            hit_length = (fine_idx + 1) * alignment_tokens
            break

    # EAGLE/MTP：需要重算最后一块以拿到 draft head 所需的 hidden states
    if drop_eagle_block and hit_length > 0:
        hit_length -= min(alignment_tokens, block_size)

    # 向下对齐到 alignment，并裁掉越界的块
    hit_length -= hit_length % alignment_tokens
    num_blocks = cdiv(hit_length, block_size)
    for computed in computed_blocks:
        del computed[num_blocks:]
    return computed_blocks, hit_length
```

要点：
- **Phase 1** 是块级前缀扫描，依赖链式哈希的"一 miss 全 miss"性质，O(前缀命中块数)。
- **Phase 2** 用于 fine-grained 哈希（当 `hash_block_size < block_size` 时），允许命中落在一个块的内部边界，进一步复用到更细粒度。
- **EAGLE 特例**：因为投机解码的 draft head 需要最后一块的 hidden states，所以命中长度要退回一个单位，强制重算。

---

## 五、上层调用流程

### 5.1 取已算块：`KVCacheManager.get_computed_blocks`

```python
def get_computed_blocks(self, request):
    if not self.prefix_cache_lookup_enabled(request):     # 关闭缓存 / 跳过读缓存
        return self.empty_kv_cache_blocks, 0, 0
    # 必须至少重算最后一个 token 以拿到 logits
    max_cache_hit_length = request.num_tokens - 1
    computed_blocks, num_new_computed_tokens, num_uncached = \
        self.coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)
    ...
    shared_prefix_boundary = (num_new_computed_tokens + num_uncached) if num_uncached else 0
    return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens, shared_prefix_boundary
```

返回的 `num_computed_tokens` 就是调度器可以直接跳过、不用重算的 token 数。

### 5.2 把命中的块挂到请求上

`SingleTypeKVCacheManager.add_local_computed_blocks`：
1. 按 sliding window 等跳过被窗口丢弃的块（用 null 块填充）；
2. `block_pool.touch(new_computed_blocks)` —— **提高引用计数，避免这些共享块被淘汰**；
3. 记录 `num_cached_block[request_id] = len(req_blocks)`，标记这部分已经"在缓存里"，后续 `cache_blocks` 不会再重复写。

### 5.3 解码阶段写回缓存：`cache_blocks`

每跑完一步（prefill 或 decode），调度器会调用 `KVCacheManager.cache_blocks(request, num_tokens)` → 各 manager 的 `cache_blocks` → `block_pool.cache_full_blocks(...)`，把**新产生的 full block** 写入哈希映射：

```python
def cache_blocks(self, request, num_tokens, retention_interval=None):
    num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
    num_full_blocks = num_tokens // self.block_size
    if num_cached_blocks >= num_full_blocks:
        return
    # block_mask：滑动窗口等稀疏注意力据此跳过不可达块
    block_mask = self.reachable_block_mask(
        start_block=num_cached_blocks, end_block=num_full_blocks, ...)
    self.block_pool.cache_full_blocks(
        request=request, blocks=self.req_to_blocks[request.request_id],
        num_cached_blocks=num_cached_blocks, num_full_blocks=num_full_blocks,
        block_size=self.block_size, kv_cache_group_id=self.kv_cache_group_id,
        block_mask=block_mask)
    self.num_cached_block[request.request_id] = num_full_blocks
```

`FullAttentionManager` 还会额外调用 `_cache_partial_tail_block`，把"结尾落在块内部"的提示尾注册为 partial 前缀缓存条目（fine-grained 命中）。

### 5.4 Partial-hit 的写时复制（CoW）

当命中落在某个块**内部**（而不是整块边界）时，请求如果继续往这块写，会污染被其他请求共享的缓存块。vLLM 用 **Copy-on-Write** 解决（`_apply_cow` / `allocate_new_blocks`）：
- 命中时请求的块表里该位置指向共享块；
- 一旦该请求要往这块写新 token，分配一个私有新块，把共享块的 KV 复制到新块，再把表里的指针重定向到私有块；
- 复制完成前，源块和目标块都保持引用，防止同一 step 内被回收。

---

## 六、关键设计点总结

| 设计点 | 说明 |
|---|---|
| **链式哈希** | 每块哈希依赖父块哈希，前缀连续 + 上下文敏感（LoRA/多模态/cache_salt），保证命中正确性 |
| **块级 + LRU** | 空闲块用双向链表按淘汰顺序排列；有哈希的块后淘汰，无哈希的块优先淘汰 |
| **ref_cnt 守卫** | 命中即 `touch`（ref_cnt+1）防止被抢；请求结束 `free` 时减回 |
| **fine-grained 哈希** | `hash_block_size < block_size` 时支持 sub-block 命中，进一步复用 |
| **CoW 保护共享块** | partial hit 时重定向到私有块，避免写脏共享缓存 |
| **多 group 协调** | 同一 `BlockPool` 服务全注意力 / SWA / Mamba / 交叉注意力，各自有命中与缓存语义 |
| **EAGLE 特例** | 命中长度退回一个单位，强制重算最后块以供给 draft head |
| **reset 接口** | `reset_prefix_cache()` 在权重更新等场景失效整个缓存 |

---

## 七、一句话调用链回顾

```
请求到达
  → Request 计算 block_hashes（链式哈希）
  → KVCacheManager.get_computed_blocks
       → coordinator.find_longest_cache_hit
            → FullAttentionManager.find_longest_cache_hit
                 → 沿 block_hashes 逐块 get_cached_block（BlockPool 哈希表）
       → 返回命中块 + 命中 token 数
  → add_local_computed_blocks：touch 命中的块 + 挂到请求
  → 调度器只跑未命中的 token
  → 每步结束 cache_blocks → block_pool.cache_full_blocks 把新 full block 写回哈希表
```
