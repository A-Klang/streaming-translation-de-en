import re
import spacy

## spaCy models for english and german, loaded once
DE = spacy.load("de_core_news_lg", disable=["ner", "parser", "lemmatizer"])
EN = spacy.load("en_core_web_md", disable=["ner", "lemmatizer"])

## German ordinal numbers (erste, dritten, fünfzehnter, dreißigste ...):
## spaCy tags them as adjectives, so they are matched here to count as numbers
ORDINAL = re.compile(r"^(erst|zweit|dritt|viert|fünft|sechst|siebt|acht|neunt|zehnt|elft|zwölft|"
                     r"\w+zehnt|\w+[zß]igst|hundertst|tausendst)(e|er|en|es|em)$", re.I)

## English words tagged VERB that are not verbs here: adjectives ("the elected mayor"),
## compounds, and prepositions ("according to", "including")
NOT_A_VERB = ("amod", "compound", "prep", "nmod", "npadvmod")

def german_counts(seen: list[str]) -> tuple[int, int]:
    #verbs and numbers in the German that has arrived.
    doc = DE(" ".join(seen))
    
    def closed_alone(i: int) -> bool:
        # an auxiliary (haben, sein, werden) is the real verb only if its clause
        # ends before any full verb turns up: "Wir haben Mäuse," but not "Wir haben ... geheilt"
        for t in doc[i + 1:]:
            if t.tag_.startswith("VV") or t.tag_ in ("VAPP", "VAINF", "VMINF"):
                return False
            if t.tag_ in ("$.", "$,", "KOUS", "KON"):
                return True
        return False   # clause still open: wait
    
    verbs = 0
    numbers = 0
    for t in doc:
        if t.tag_.startswith("VV") or t.tag_ in ("VAPP", "VMPP"):
            verbs += 1
        elif t.tag_ in ("VAFIN", "VAINF") and closed_alone(t.i):
            verbs += 1
        if t.tag_ == "CARD" or any(c.isdigit() for c in t.text) or ORDINAL.match(t.text):
            numbers += 1
    return verbs, numbers

def hold_verbs(agreed: list[str], committed: list[str], seen: list[str]) -> list[str]:
    #Cut before the first new English verb or number the German hasn't delivered yet.
    de_verbs, de_numbers = german_counts(seen)
    text = " ".join(committed + agreed)
    en_verbs = 0
    en_numbers = 0
    for t in EN(text):
        word = text[:t.idx].count(" ")          # which of our words this token belongs to
        is_verb = t.pos_ == "VERB" and t.dep_ not in NOT_A_VERB
        is_number = any(c.isdigit() for c in t.text) or (t.like_num and t.pos_ == "NUM")
        en_verbs += is_verb
        en_numbers += is_number
        if word < len(committed):
            continue                            # already said, can't be held back
        if (is_verb and en_verbs > de_verbs) or (is_number and en_numbers > de_numbers):
            return agreed[:word - len(committed)]
    return agreed


if __name__ == "__main__":
    def words(s: str) -> list[str]:
        return s.split()

    # verbs: devtest 264, "allows" before the German verb has arrived
    seen = words("Der kalifornische Gouverneur Arnold Schwarzenegger unterzeichnete einen Gesetzentwurf, der den Verkauf")
    committed = words("The California Governor Arnold Schwarzenegger signed a bill that")
    print(german_counts(seen))                                      # expect (1, 0)
    print(hold_verbs(words("allows the sale"), committed, seen))    # expect []

    # numbers: devtest 255, "1,000" before the German number has arrived
    seen = words("Laut einigen Berichten beträgt die offizielle Zahl der")
    committed = words("According to some reports, the")
    print(hold_verbs(words("amount is 1,000"), committed, seen))    # expect ['amount', 'is']
