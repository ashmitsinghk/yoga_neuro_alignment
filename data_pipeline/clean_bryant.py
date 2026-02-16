import re
import json

INPUT_FILE = "../data/raw/bryant_2009.txt"
OUTPUT_FILE = "../data/processed/yoga_corpus.json"

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_core_text(text):
    """
    Extract sutras from the text.
    The Bryant text doesn't have clear chapter markers, so we'll search for sutra patterns directly.
    """
    # Find where sutras start (looking for I.1 pattern)
    # Sutras are formatted like: "I.16 tat-paraṁ puruṣa-khyāter guṇa-vaitṛṣṇyam"
    return text


def remove_noise(text):
    """
    Clean up the text but preserve sutra structure.
    We need to be careful not to remove numbers from sutra IDs.
    """
    # Don't remove all numbers - we need sutra IDs!
    # Just normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text

def segment_sutras(text):

    sutra_pattern = r"(?m)^([IV]{1,3}\.\d+)\s+(.+)$"
    
    matches = list(re.finditer(sutra_pattern, text))

    if not matches:
        print("No sutra matches found.")
        return []

    # Detect duplicate full pass
    restart_index = None
    for i in range(1, len(matches)):
        if matches[i].group(1) == "I.1":
            restart_index = i
            break

    if restart_index is not None:
        print("Duplicate sutra pass detected.")
        print("Keeping FIRST pass (with commentary).")
        matches = matches[:restart_index]

    corpus = []

    for idx, match in enumerate(matches):
        sutra_id = match.group(1)
        sanskrit_line = match.group(2).strip()

        start = match.end()

        if idx < len(matches) - 1:
            end = matches[idx + 1].start()
        else:
            end = len(text)

        block = text[start:end].strip()

        block = re.sub(r"\n{2,}", "\n", block)
        block = re.sub(r"[ \t]+", " ", block)

        chapter_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
        chapter_number = chapter_map[sutra_id.split(".")[0]]

        combined_text = sanskrit_line + " " + block

        corpus.append({
            "sutra_id": sutra_id,
            "chapter": chapter_number,
            "text": combined_text.strip(),
            "domain": "yoga"
        })

    return corpus


def main():
    text = load_text(INPUT_FILE)
    core_text = extract_core_text(text)
    cleaned_text = remove_noise(core_text)
    corpus = segment_sutras(cleaned_text)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    print("Extraction complete.")
    print("Total sutras extracted:", len(corpus))

if __name__ == "__main__":
    main()
