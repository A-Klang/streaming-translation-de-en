from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class Sentence:
    idx: int
    src_words: list[str]
    ref: str
    
## Split plan: devtest 0-299 for building the pipeline (already used in the pilot),
## dev for testing and tuning policies, devtest 300 onwards only for the final report runs.

def load(n: int, start: int = 0, src_lang: str = "deu_Latn", tgt_lang: str = "eng_Latn") -> list[Sentence]:
    src = load_dataset("openlanguagedata/flores_plus", src_lang, split="devtest")
    tgt = load_dataset("openlanguagedata/flores_plus", tgt_lang, split="devtest")
    ref_by_id = {r["id"]: r["text"] for r in tgt}
    
    sentences = []
    for i, r in enumerate(src):
        if i < start:
            continue
        if len(sentences) == n:
            break
        sentences.append(Sentence(idx=i, src_words=r["text"].split(), ref=ref_by_id[r["id"]]))
    return sentences

if __name__ == "__main__":
    for s in load(3):
        print(s.idx, len(s.src_words), s.src_words[:5], "|", s.ref[:60])