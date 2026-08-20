# vLLM 单机部署实战 · AutoDL RTX 3090

> 来源: Bilibili「月球大叔」vLLM 教程方案
> 适用: AutoDL 实例已开机, 通过 JupyterLab 的 Terminal 执行
> 显存: 24GB (3090) — 可跑 Qwen2.5-7B FP16 / 14B INT4

---

## 第 0 步 · 开机后进 JupyterLab

1. AutoDL 控制台 → 实例 → 点 **「开机」**
2. 等 1-2 分钟, 状态变「运行中」
3. 点 **「JupyterLab」** 进入网页 IDE
4. Launcher 里点 **Terminal** 打开终端

---

## 第 1 步 · 验证 GPU 环境

```bash
nvidia-smi            # 应看到 RTX 3090, 24GB
nvcc --version        # CUDA 12.x
python --version      # 3.12
```

---

## 第 2 步 · 创建虚拟环境 + 安装 vLLM

```bash
# 创建 Python 虚拟环境
python -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 安装 vLLM(首次需要几分钟)
pip install vllm
```

> 看到 `Successfully installed vllm-...` 就说明装好了。

---

## 第 3 步 · 启动 vLLM 推理服务

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192
```

### 这步会发生什么:

| 阶段 | 现象 | 耗时 |
|---|---|---|
| 加载引擎 | 日志刷一堆 INFO | ~10 秒 |
| **下载模型**(首次) | 进度条从 HF 拉取 ~14GB | **3-10 分钟** |
| 加载到 GPU | 显存占用飙升到 ~14GB | ~30 秒 |
| 启动完成 | 出现 `Application startup complete.` | — |

> ⚠️ **这个终端不要关!** 服务运行在 `http://localhost:8000`。
>
> 如果模型下载太慢, 先 Ctrl+C 停掉, 配国内镜像:
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> # 再重新执行 vllm serve ...
> ```

---

## 第 4 步 · 测试 API(新开一个 Terminal)

在 JupyterLab 里点 **`+`** → 选 **Terminal**, 打开第二个终端:

```bash
source .venv/bin/activate    # 别忘了激活环境

# 健康检查 — 应返回模型列表 JSON
curl http://localhost:8000/v1/models

# 聊天补全测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "你好, 用一句话介绍 vLLM"}],
    "max_tokens": 100
  }'
```

看到返回 JSON 里有 `"content"` 字段包含文字回答 → **vLLM 跑通了!** ✓

---

## 第 5 步 · 部署 OpenWebUI 可视化聊天界面

在任意终端执行:

```bash
# 拉取 OpenWebUI 镜像
sudo docker pull ghcr.io/open-webui/open-webui:main

# 启动 OpenWebUI 容器
sudo docker run -d \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

### 访问 OpenWebUI:

1. 浏览器打开 **`http://<你的AutoDL公网IP>:3000`**
   - 公网 IP 在 AutoDL 实例详情页可以看到
2. **注册账号**(首次使用)
3. 点 **Settings(设置)** → **Connections(连接)**:
   - API Base URL 填: `http://host.docker.internal:8000`
4. 回到聊天页面, 模型选 **Qwen/Qwen2.5-7B-Instruct**
5. 开始聊天! 🎉

> `host.docker.internal` 是 Docker 特殊域名, 指向宿主机(也就是你跑 vLLM 的那台机器), 所以容器里的 OpenWebUI 能访问到宿主机的 8000 端口。

---

## 终端分工总览

| 终端 | 运行什么 | 能关吗? |
|---|---|---|
| **Terminal A** | `vllm serve ...`(推理服务) | ❌ 关了就断 |
| **Terminal B** | curl 测试 / 其他命令 | ✅ 随便关 |
| **Docker** | OpenWebUI(后台运行) | ❌ 关了前端打不开 |

---

## 常见问题

**Q: pip install vLLM 很慢/报错?**
A: 换 pip 源:
```bash
pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Q: 模型下载卡住/很慢?**
A: 配 HuggingFace 国内镜像:
```bash
export HF_ENDPOINT=https://hf-mirror.com
# 再重新执行 vllm serve ...
```
或用 ModelScope 预下载:
```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /root/autodl-tmp/Qwen2.5-7B
# 然后 vllm serve /root/autodl-tmp/Qwen2.5-7B (用本地路径)
```

**Q: 显存不够 OOM?**
A: 三种解法(任选):
1. 加 `--quantization awq`(用 AWQ 量化版模型, 显存减半)
2. 降 `--max-model-len 4096`(缩短上下文)
3. 加 `--gpu-memory-utilization 0.95`(压榨显存, 可能不稳定)

**Q: OpenWebUI 打不开?**
A: 检查:
1. `sudo docker ps` 看 open-webui 容器是否 Up
2. AutoDL 控制台确认 **自定义服务** 里 3000 端口已开放
3. 防火墙: AutoDL 一般默认全开, 但有些地区需手动放行

**Q: 从本机 MacBook 访问 AutoDL?**
A: AutoDL 实例详情页有 **「快捷指令」** 给 SSH 命令, 包含端口转发。或者用 AutoDL 的 **「自定义服务」** 功能直接绑定域名。

---

## 省钱提醒

- 用完点 **「停止」**(不是销毁), GPU 不计费, 数据盘保留
- 你的实例有 **14 天保留期**, 停止后 14 天内启动都不会释放
- 下次启动后模型还在 `/root/.cache/huggingface/`, 不用重新下载
- 建议: 充 ¥10, 够玩一整天 + 试错
