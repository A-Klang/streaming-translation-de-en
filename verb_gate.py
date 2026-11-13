"""Verb rule: never commit English that runs ahead of the German verbs.

An English lexical verb may be committed only if the German that has arrived
contains at least as many verbs; the same for numbers. The commit is cut just
before the first English word that breaks this, and that word and everything
after it wait for more German.

German verbs: full verbs in any form (finite, infinitive, participle,
zu-infinitive). An auxiliary or modal (haben, sein, werden, können ...) counts
only once its clause has closed without a full verb (copula or possessive use);
while it waits, it signals that the real verb is still to come.
English verbs: spaCy POS VERB, excluding participles used as adjectives,
compounds and prepositions (according to, including).

Needs de_core_news_lg and en_core_web_md:
  python -m spacy download de_core_news_lg
  python -m spacy download en_core_web_md
"""
import re

_de = _en = None

ORD = re.compile(r"^(erst|zweit|dritt|viert|fünft|sechst|siebt|acht|neunt|zehnt|elft|zwölft|"
                 r"\w+zehnt|\w+zigst|hundertst|tausendst)(e|er|en|es|em)$", re.I)
EN_SKIP_DEP = ("amod", "compound", "prep", "nmod", "npadvmod")


def _load():
    global _de, _en
    if _de is None:
        import spacy
        _de = spacy.load("de_core_news_lg", disable=["ner", "parser", "lemmatizer"])
        _en = spacy.load("en_core_web_md", disable=["ner", "lemmatizer"])


def german_counts(prefix):
    """(verbs, numbers) in the German that has arrived."""
    _load()
    doc = _de(prefix)

    def closed_alone(t):
        for u in doc[t.i + 1:]:
            if u.tag_.startswith("VV") or u.tag_ in ("VAPP", "VAINF", "VMINF"):
                return False
            if u.tag_ in ("$.", "$,", "KOUS", "KON"):
                return True
        return False

    verbs = sum(1 for t in doc if t.tag_.startswith("VV") or t.tag_ in ("VAPP", "VMPP")
                or (t.tag_ in ("VAFIN", "VAINF") and closed_alone(t)))
    nums = sum(1 for t in doc if t.tag_ == "CARD" or any(c.isdigit() for c in t.text)
               or ORD.match(t.text))
    return verbs, nums


def allowed(german_prefix, shown, new_shown):
    """Cut new_shown so the committed English never has more verbs or numbers
    than the German prefix. Returns (committed text, held text)."""
    if len(new_shown) <= len(shown):
        return new_shown, ""
    _load()
    dv, dn = german_counts(german_prefix)
    ev = en_n = 0
    start = len(shown)
    for t in _en(new_shown):
        is_v = t.pos_ == "VERB" and t.dep_ not in EN_SKIP_DEP
        is_n = any(c.isdigit() for c in t.text) or (t.like_num and t.pos_ == "NUM")
        ev += is_v
        en_n += is_n
        if t.idx < start:
            continue
        if (is_v and ev > dv) or (is_n and en_n > dn):
            cut = new_shown[:t.idx]
            if cut and not cut[-1].isspace():  # verb glued to a word: cut at the word start
                cut = cut[:cut.rfind(" ") + 1] if " " in cut else ""
            cut = cut.rstrip()
            if len(cut) < len(shown):          # never take back what is already shown
                cut = shown
            return cut, new_shown[len(cut):].strip()
    return new_shown, ""
