# src/app/rag/index_cli.py
from __future__ import annotations
import argparse
from pathlib import Path

from src.app.rag.pipeline import index_pdfs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", type=str, default="data/pdfs")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    out = index_pdfs(Path(args.pdf_dir), rebuild=args.rebuild)
    print(out)

if __name__ == "__main__":
    main()
