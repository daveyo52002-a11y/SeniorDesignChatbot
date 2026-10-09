"""
One-time / repeatable CLI helper: ingest every document dropped into
backend/knowledge_base/ (the lab's project requirements, procedures,
equipment guides, troubleshooting notes, etc.) into the vector store.

Usage:
    python load_knowledge_base.py
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import rag  # noqa: E402

KB_DIR = Path(__file__).parent / "knowledge_base"


def main():
    if not KB_DIR.exists() or not any(KB_DIR.iterdir()):
        print(f"No files found in {KB_DIR}. Add your lab's .pdf/.txt/.md files there first.")
        return

    results = rag.ingest_directory(KB_DIR)
    if not results:
        print("No supported files (.pdf, .txt, .md) found.")
        return

    total = sum(results.values())
    print(f"Ingested {len(results)} file(s), {total} chunk(s) total:")
    for filename, count in results.items():
        print(f"  - {filename}: {count} chunks")


if __name__ == "__main__":
    main()
