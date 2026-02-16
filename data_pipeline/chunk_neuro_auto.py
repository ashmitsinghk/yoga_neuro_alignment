import json
from transformers import AutoTokenizer

INPUT_FILE = "../data/processed/neuro_corpus_auto.json"
OUTPUT_FILE = "../data/processed/neuro_corpus_auto_chunked.json"

CHUNK_SIZE = 250
OVERLAP = 30

tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")

def chunk_text(text):
    tokens = tokenizer(
        text,
        truncation=False,
        add_special_tokens=False
    )["input_ids"]

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += CHUNK_SIZE - OVERLAP

    return chunks


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunked_data = []

    for entry in data:
        chunks = chunk_text(entry["text"])
        for chunk in chunks:
            chunked_data.append({
                "text": chunk,
                "domain": "neuro"
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunked_data, f, indent=2, ensure_ascii=False)

    print("Original abstracts:", len(data))
    print("Total neuro chunks:", len(chunked_data))


if __name__ == "__main__":
    main()
