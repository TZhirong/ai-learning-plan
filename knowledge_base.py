"""知识库模块：文档解析、切块、向量化、FAISS 检索与本地存储。

支持的文档格式：PDF / Word(.docx) / Markdown。
向量化：本地轻量特征哈希（字符 n-gram 签名 + L2 归一化），无需外部模型或 API。
检索：FAISS 内积索引（向量已归一化，等价于余弦相似度）。
"""
import hashlib
import io
import re

import faiss
import numpy as np

from database import get_conn

EMBED_DIM = 512      # 向量维度
CHUNK_SIZE = 600     # 文本块最大长度（字符）
TOP_K = 5            # 默认检索条数

# FAISS 索引内存缓存（模块级，进程内复用；DB 为唯一数据源，改动后重建）
_index = None
_id_to_chunk = []
_dirty = True


# ==================== 表结构 ====================
def init_kb():
    """初始化知识库表结构（幂等）。"""
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,               -- 文档名
                file_type   TEXT NOT NULL,               -- pdf / docx / md
                chunk_count INTEGER DEFAULT 0,           -- 文本块数
                created_at  TEXT DEFAULT (datetime('now', 'localtime'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id      INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text        TEXT NOT NULL,               -- 文本块内容
                source      TEXT,                        -- 来源标注（章节/页码）
                vector      BLOB NOT NULL,               -- float32 向量
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            )
            """
        )


# ==================== 文档解析 ====================
def _detect_chapter(text):
    """从文本中检测章节标题，未找到返回 None。"""
    patterns = [
        r"第[一二三四五六七八九十百\d]+[章节部分]",
        r"Chapter\s+\d+",
        r"^\d+(\.\d+)*\s+\S+",
    ]
    for line in text.splitlines()[:8]:
        line = line.strip()
        if line and len(line) < 40:
            for p in patterns:
                if re.search(p, line):
                    return line
    return None


def parse_pdf(data: bytes):
    """解析 PDF，返回 [(text, source)]，source 含章节与页码。"""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    paras = []
    current_chapter = None
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        ch = _detect_chapter(text)
        if ch:
            current_chapter = ch
        src = f"第{i}页"
        if current_chapter:
            src = f"{current_chapter} · {src}"
        paras.append((text, src))
    return paras


def parse_docx(data: bytes):
    """解析 Word，返回 [(text, source)]，source 用标题作为章节。"""
    from docx import Document

    doc = Document(io.BytesIO(data))
    paras = []
    current_chapter = ""
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style = (p.style.name or "") if p.style else ""
        if style and ("Heading" in style or "标题" in style or style == "Title"):
            current_chapter = text
            continue
        paras.append((text, current_chapter))
    return paras


def parse_markdown(data: bytes):
    """解析 Markdown，返回 [(text, source)]，source 用标题作为章节。"""
    text = data.decode("utf-8", errors="ignore")
    paras = []
    current_chapter = ""
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line.strip())
        if m:
            current_chapter = m.group(2).strip()
            continue
        if line.strip():
            paras.append((line.strip(), current_chapter))
    return paras


def parse_document(name: str, data: bytes):
    """根据扩展名分派解析器，返回 [(text, source)]。"""
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return parse_pdf(data)
    if ext == "docx":
        return parse_docx(data)
    if ext in ("md", "markdown"):
        return parse_markdown(data)
    raise ValueError("不支持的文档格式")


# ==================== 切块 ====================
def chunk_paras(paras, max_len=CHUNK_SIZE):
    """将 [(text, source)] 切成固定长度文本块，相同来源优先合并。"""
    chunks = []
    buf, buf_src = "", ""

    def flush():
        nonlocal buf, buf_src
        if buf:
            chunks.append({"text": buf, "source": buf_src})
        buf, buf_src = "", ""

    for text, source in paras:
        text = (text or "").strip()
        if not text:
            continue
        if source != buf_src:
            flush()
            buf_src = source
        while text:
            space = max_len - len(buf)
            if len(text) <= space:
                buf += text
                break
            buf += text[:space]
            flush()
            text = text[space:]
    flush()
    return chunks


# ==================== 向量化 ====================
def _ngrams(text):
    """提取中文单字/双字/三字 + 英文词 token。"""
    tokens = []
    chinese = re.findall(r"[一-鿿]", text or "")
    s = "".join(chinese)
    tokens.extend(s)
    tokens.extend(s[i : i + 2] for i in range(len(s) - 1))
    tokens.extend(s[i : i + 3] for i in range(len(s) - 2))
    tokens.extend(re.findall(r"[a-zA-Z0-9_]+", (text or "").lower()))
    return tokens


def embed_text(text, dim=EMBED_DIM):
    """特征哈希向量化 + L2 归一化，返回 float32 向量。"""
    vec = np.zeros(dim, dtype=np.float32)
    for tok in _ngrams(text):
        h = hashlib.md5(tok.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if h[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ==================== FAISS 索引与检索 ====================
def _ensure_index():
    """惰性构建 / 重建 FAISS 索引。"""
    global _index, _id_to_chunk, _dirty
    if _index is not None and not _dirty:
        return
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, text, source, vector FROM chunks ORDER BY id"
        ).fetchall()
    _id_to_chunk = []
    vectors = []
    for r in rows:
        _id_to_chunk.append({"id": r["id"], "text": r["text"], "source": r["source"]})
        vectors.append(np.frombuffer(r["vector"], dtype=np.float32))
    _index = faiss.IndexFlatIP(EMBED_DIM)
    if vectors:
        _index.add(np.vstack(vectors).astype(np.float32))
    _dirty = False


def kb_chunk_count() -> int:
    """知识库中的文本块总数。"""
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


def retrieve(query: str, top_k: int = TOP_K):
    """检索最相关的文本块。知识库为空时返回空列表。"""
    if kb_chunk_count() == 0:
        return []
    _ensure_index()
    qvec = embed_text(query).reshape(1, -1).astype(np.float32)
    k = min(top_k, _index.ntotal)
    scores, ids = _index.search(qvec, k)
    results = []
    for score, cid in zip(scores[0], ids[0]):
        if cid < 0:
            continue
        c = _id_to_chunk[cid]
        results.append({"text": c["text"], "source": c["source"], "score": float(score)})
    return results


# ==================== 文档管理 ====================
def add_document(name: str, data: bytes):
    """解析并入库文档，返回 (doc_id, chunk_count)。"""
    paras = parse_document(name, data)
    chunks = chunk_paras(paras)
    if not chunks:
        raise ValueError("未从文档中提取到有效文本")

    file_type = name.lower().rsplit(".", 1)[-1]
    global _dirty
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO documents (name, file_type, chunk_count) VALUES (?, ?, ?)",
            (name, file_type, len(chunks)),
        )
        doc_id = cur.lastrowid
        for i, c in enumerate(chunks):
            vec = embed_text(c["text"])
            conn.execute(
                "INSERT INTO chunks (doc_id, chunk_index, text, source, vector) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, i, c["text"], c["source"], vec.tobytes()),
            )
    _dirty = True
    return doc_id, len(chunks)


def list_documents():
    """返回全部文档（按创建时间倒序）。"""
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


def delete_document(doc_id: int):
    """删除文档及其全部文本块（级联）。"""
    global _dirty
    with get_conn() as conn:
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    _dirty = True
