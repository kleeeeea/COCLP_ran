from aiohttp import web
# from server import PromptServer
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
import json

# Try to import tqdm for a nice progress bar; fall back gracefully if it's missing
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        # Dummy replacement: just return the iterable unchanged
        return iterable if iterable is not None else []


class MarkdownInfoNode:
    """ComfyUI node: Scan a directory of Markdown files, match them with
    corresponding original files, and write the collected metadata to a JSONL
    file. The node returns the path to the generated JSONL file so it can be
    consumed by downstream nodes.

    Parameters (UI -> ``INPUT_TYPES``)
    ---------------------------------
    md_root : str
        Root folder containing the ``.md`` files to scan.
    original_root : str
        Root folder that may contain the corresponding original data (images,
        audio, etc.).  The node tries to find an asset with the same relative
        stem as each Markdown file.
    output_file : str
        Where to write the final JSONL file (default: ``./md_info.jsonl``).
    threads : int
        Thread‑pool size for parallel processing (default: ``8``).

    Output
    ------
    jsonl_path : str
        Absolute/relative path of the JSONL file just written.
    """

    def __init__(self):
        pass

    # ---------- UI schema ----------
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "md_root": (
                    "STRING",
                    {"default": "./markdown", "multiline": False},
                ),
                "original_root": (
                    "STRING",
                    {"default": "./source", "multiline": False},
                ),
                "output_file": (
                    "STRING",
                    {"default": "./md_info.jsonl", "multiline": False},
                ),
                "threads": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "display": "number",
                    },
                ),
            }
        }

    # ---------- ComfyUI meta ----------
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("jsonl_path",)
    OUTPUT_NODE = True
    FUNCTION = "execute"  # entry‑point method
    CATEGORY = "MY_NODES/MD_TO_JS"

    # ---------- helper functions ----------
    @staticmethod
    def _hash_path_to_id(file_path: str) -> str:
        """Stable SHA‑256 hash so each Markdown file gets a unique ID."""
        return hashlib.sha256(file_path.encode("utf-8")).hexdigest()

    @staticmethod
    def _human_readable_size(size_bytes: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"

    def _find_original_file(self, md_path: str, md_root: str, original_root: str):
        """Locate the companion asset that shares the stem of *md_path*."""
        rel_path = os.path.relpath(md_path, md_root)
        base_stem, _ = os.path.splitext(rel_path)
        asset_dir = os.path.join(original_root, os.path.dirname(base_stem))

        if not os.path.exists(asset_dir):
            return None  # no sibling directory – give up early

        target_stem = os.path.basename(base_stem)
        try:
            for entry in os.listdir(asset_dir):
                full_path = os.path.join(asset_dir, entry)
                if not os.path.isfile(full_path):
                    continue
                stem, ext = os.path.splitext(entry)
                if stem != target_stem:
                    continue

                size = os.path.getsize(full_path)
                return {
                    "path": full_path,
                    "format": ext.lstrip("."),
                    "size_bytes": size,
                    "size_human": self._human_readable_size(size),
                    "name": entry,
                }
        except Exception as e:
            print(f"[MarkdownInfoNode] ❌ scan failed: {asset_dir} ({e})")
        return None

    def _process_single_md(self, md_path: str, md_root: str, original_root: str):
        """Read one Markdown, compute its metadata, optionally match an asset."""
        try:
            with open(md_path, "r", encoding="utf-8") as fp:
                content = fp.read().strip()
            if not content:
                return None  # skip empty files

            item = {
                "id": self._hash_path_to_id(md_path),
                "md_path": md_path,
                "text": content,
            }
            asset_info = self._find_original_file(md_path, md_root, original_root)
            if asset_info:
                print(
                    f"[MarkdownInfoNode] 🔍 matched asset {asset_info['name']} ← {md_path}"
                )
                item.update(asset_info)
            else:
                print(f"[MarkdownInfoNode] ⚠ no asset for {md_path}")
            return item
        except Exception as e:
            print(f"[MarkdownInfoNode] ❌ failed at {md_path}: {e}")
            return None

    def _process_all(self, md_root: str, original_root: str, threads: int):
        """Kick off a thread‑pool to walk & process every Markdown file."""
        md_files = [
            os.path.join(dp, f)
            for dp, _, files in os.walk(md_root)
            for f in files
            if f.endswith(".md")
        ]

        results = []
        self._process_single_md( md_files[0], md_root, original_root)
        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = {
                pool.submit(self._process_single_md, p, md_root, original_root): p
                for p in md_files
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="📄 processing",
            ):
                r = fut.result()
                if r:
                    results.append(r)
        return results

    @staticmethod
    def _save_jsonl(data, output_file: str):
        """
        将 data 逐行写入 JSONL 文件。若目标目录或文件不存在，则自动创建。

        Args:
            data: 可迭代对象，每个元素是 dict。
            output_file: 输出 JSONL 文件路径（含文件名）。
        """
        # 1️⃣ 确保输出目录存在
        dir_name = os.path.dirname(os.path.abspath(output_file))
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)  # 递归创建目录

        # 2️⃣ 写入文件；文件不存在时 open(..., "w") 会自动创建
        with open(output_file, "w", encoding="utf-8") as fp:
            for row in data:
                if row.get("text"):  # 仅写入包含 "text" 字段的行
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---------- node entry‑point ----------
    def execute(self, md_root: str, original_root: str, output_file: str, threads: int):
        print(
            f"[MarkdownInfoNode] 🚀 start – md_root='{md_root}', original_root='{original_root}', threads={threads}"
        )
        data = self._process_all(md_root, original_root, threads)
        self._save_jsonl(data, output_file)
        print(
            f"[MarkdownInfoNode] ✅ done – {len(data)} markdown files → '{output_file}'"
        )
        return (original_root,)


# -------------------------- optional API route ---------------------------
# @PromptServer.instance.routes.get("/markdown_node/ping")
# async def markdown_node_ping(request):
#     """Simple health‑check route: GET /markdown_node/ping -> {"status":"ok"}"""
#     return web.json_response({"status": "ok"})


# ----------------------- ComfyUI registration ---------------------------




