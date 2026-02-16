from Bio import Entrez
from Bio import Medline
import json
import time

# --------------------------------
# CONFIGURATION
# --------------------------------
Entrez.email = "ashmit.singh.k@gmail.com"

SEARCH_TERMS = [
    "executive control",
    "response inhibition",
    "default mode network",
    "attention network",
    "metacognition",
    "self-referential processing",
    "cognitive regulation",
    "habit formation neuroscience",
    "meditation neuroscience",
    "cognitive flexibility"
]

ABSTRACT_TARGET = 250  # number of abstracts to fetch
OUTPUT_FILE = "../data/processed/neuro_corpus_auto.json"

# --------------------------------
# FETCH ABSTRACTS
# --------------------------------
def fetch_abstracts():
    all_records = []
    
    for term in SEARCH_TERMS:
        print(f"Searching for: {term}")
        
        handle = Entrez.esearch(
            db="pubmed",
            term=term,
            retmax=ABSTRACT_TARGET // len(SEARCH_TERMS)
        )
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        
        if not id_list:
            continue
        
        handle = Entrez.efetch(
            db="pubmed",
            id=",".join(id_list),
            rettype="medline",
            retmode="text"
        )
        
        records = Medline.parse(handle)
        
        for r in records:
            if "AB" in r:
                all_records.append({
                    "paper_id": r.get("PMID", ""),
                    "text": r["AB"],
                    "domain": "neuro"
                })
        
        handle.close()
        time.sleep(1)  # avoid rate limits
    
    return all_records


def main():
    abstracts = fetch_abstracts()
    
    print("Total abstracts collected:", len(abstracts))
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(abstracts, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
