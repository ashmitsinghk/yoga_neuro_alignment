import json
import random

YOGA_FILE = "../data/processed/yoga_corpus_chunked.json"
NEURO_FILE = "../data/processed/neuro_corpus_auto_chunked.json"
OUTPUT_FILE = "../data/processed/final_dapt_corpus.json"

def main():
    with open(YOGA_FILE, "r", encoding="utf-8") as f:
        yoga = json.load(f)

    with open(NEURO_FILE, "r", encoding="utf-8") as f:
        neuro = json.load(f)

    combined = yoga + neuro
    random.shuffle(combined)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print("Yoga chunks:", len(yoga))
    print("Neuro chunks:", len(neuro))
    print("Total training chunks:", len(combined))

if __name__ == "__main__":
    main()
