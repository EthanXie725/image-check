# CLIP + FAISS 相似图片检索服务

单机 Web 服务：用 CLIP 把图片编成向量，用 FAISS 做余弦相似度 Top-K 检索，供业务系统上传新图后在线查重/找相似图。

## 环境

- Python 3.10+
- 首次运行会下载 CLIP 权重（需联网）
- 有 NVIDIA GPU 时自动用 CUDA，否则用 CPU

```bash
cd image-check
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## 建库

把图库图片放到 `data/images/`（或指定目录），然后：

```bash
python scripts/build_index.py --image-dir data/images --batch-size 32 --rebuild
```

会生成：

- `data/index/faiss.index`
- `data/index/meta.db`

## 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

打开交互文档：<http://127.0.0.1:8000/docs>

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 模型/索引状态 |
| GET | `/api/stats` | 向量数、有效图片数等 |
| POST | `/api/search` | 上传图片检索相似图（`file`, `top_k`, `threshold`） |
| POST | `/api/index/add` | 上传并入库一张图 |
| DELETE | `/api/index/{image_id}` | 软删除并从 FAISS 移除 |

### 检索示例（curl）

```bash
curl -X POST "http://127.0.0.1:8000/api/search" ^
  -F "file=@path\to\query.jpg" ^
  -F "top_k=10" ^
  -F "threshold=0.85"
```

返回字段 `score` 为余弦相似度（约 -1~1，越接近 1 越相似）。业务侧可用 `threshold`（如 `0.85`）判断是否存在高度相似图。

### 入库示例

```bash
curl -X POST "http://127.0.0.1:8000/api/index/add" -F "file=@path\to\new.jpg"
```

## 说明

- 删除采用软删：SQLite 标记 `deleted=1`，同时从 FAISS 移除对应 id；大量删除后可用 `--rebuild` 全量重建。
- 向量维度 512（`ViT-B-32` / `openai`）。
- FAISS 使用 `IndexIDMap2(IndexFlatIP)`，适合约 50 万以内精确检索。
