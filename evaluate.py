"""Optional paid retrieval evaluation using sample questions and OpenAI embeddings."""
import csv
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from rag.core import Retriever, chunk_text

def main():
    load_dotenv()
    if not os.getenv('OPENAI_API_KEY'):
        raise SystemExit('Set OPENAI_API_KEY before running evaluation.')
    base = Path(__file__).resolve().parent
    chunks = []
    for path in sorted((base / 'sample_data').glob('*.txt')):
        chunks.extend(chunk_text(path.read_text(encoding='utf-8'), path.name))
    retriever = Retriever(chunks, OpenAI(api_key=os.environ['OPENAI_API_KEY'],base_url="https://generativelanguage.googleapis.com/v1beta/openai/"))
    with (base / 'sample_data' / 'evaluation_questions.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    passed = 0
    for row in rows:
        hits = retriever.search(row['question'], top_k=1, threshold=0.0)
        best = hits[0][0] if hits else None
        ok = bool(best and best.source == row['expected_source'] and row['expected_keyword'].lower() in best.text.lower())
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} | {row['question']} | {best.source if best else 'none'}")
    print(f'Top-1 matching source and keyword: {passed}/{len(rows)}' if rows else 'No evaluation questions')


if __name__ == '__main__':
    main()
