#!/usr/bin/env python3
"""
下载 PGB (PubMed Graph Benchmark) 异质图数据。
Zenodo: https://zenodo.org/record/6406776
数据量：10 个分片，每个约 2GB，共约 20GB。
默认只下载 part1（约 2GB）用于快速测试；可指定 --all 下载全部。
"""
import argparse
import os
import subprocess
import zipfile
from pathlib import Path

ZENODO_RECORD = "6406776"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
PARTS = [f"pgb_part{i}.zip" for i in range(1, 11)]


def download_file(url: str, dest: Path) -> bool:
    """使用 wget 或 curl 下载"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for cmd in [
        ["wget", "-q", "--show-progress", "-O", str(dest), url],
        ["curl", "-L", "-o", str(dest), url],
    ]:
        try:
            subprocess.run(cmd, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/raw/PGB")
    parser.add_argument("--part", type=int, default=1, choices=range(1, 11),
                        help="下载第几个分片 (1-10)")
    parser.add_argument("--all", action="store_true", help="下载全部 10 个分片")
    parser.add_argument("--no-unzip", action="store_true", help="下载后不解压")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts_to_dl = list(range(1, 11)) if args.all else [args.part]
    for p in parts_to_dl:
        fname = f"pgb_part{p}.zip"
        url = f"{BASE_URL}/{fname}?download=1"
        dest = out_dir / fname
        if dest.exists():
            print(f"[SKIP] 已存在: {dest}")
        else:
            print(f"[DOWNLOAD] {fname} ...")
            if not download_file(url, dest):
                print(f"[ERROR] 下载失败，请手动访问: {url}")
                return 1
        if not args.no_unzip:
            extract_dir = out_dir / "extracted"
            extract_dir.mkdir(exist_ok=True)
            print(f"[UNZIP] {fname} -> {extract_dir}")
            with zipfile.ZipFile(dest, "r") as z:
                z.extractall(extract_dir)
    print("[DONE] 下载完成")
    return 0


if __name__ == "__main__":
    exit(main())
