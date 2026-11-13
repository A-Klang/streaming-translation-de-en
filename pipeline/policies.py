import re
from backend import translate, Result
from verb_rule import hold_verbs

WORD_TOKENS = 8 #enough tokens to get one English word out
FINAL_TOKENS = 256 # last call of a sentence, room for a whole translation (the full-sentence run, --k 999)
PUNCT = '„“”"‘’\'«».,!?;:…()' # stripped from a word before comparing
MIN_AFTER = 2 # agreed words needed after a sentence end before committing

def join_words(words: list[str]) -> str:
    return re.sub(r" ([,.;:!?])", r"\1", " ".join(words))

def same_word(a: str, b: str) -> bool:
    # Two words agree if they match without surrounding punctuation and case.
    a_core = a.strip(PUNCT).lower()
    b_core = b.strip(PUNCT).lower()
    if not a_core or not b_core: # a word that is only punctuation, like "–", must match exactly
        return a == b
    return a_core == b_core

def ends_sentence(word: str) -> bool:
    return word.rstrip('”"’»').endswith((".", "!", "?", "…"))

def strip_end(word: str) -> str:
    # "old.”" -> "old", "…" -> ""
    return word.rstrip('”"’»').rstrip(".!?…")

def closes_quote(word: str) -> bool:
    return word.rstrip(".,;:!?").endswith(('”', '"', '»'))

def quote_open(seen: list[str]) -> bool:
    #True while the German has opened a quote („)
    src = " ".join(seen)
    return src.count("„") > src.count("“")

def hold_sentence_end(agreed: list[str]) -> list[str]:
    #Cut before a sentence end that has fewer than MIN_AFTER agreed words after it.
    for i, word in enumerate(agreed):
        if ends_sentence(word) and len(agreed) - 1 - i < MIN_AFTER:
            return agreed[:i]
    return agreed

def hold_closing_quote(agreed: list[str]) -> list[str]:
    #cut before the first word that closes a quote
    for i, word in enumerate(agreed):
        if closes_quote(word):
            return agreed[:i]
    return agreed


def strip_closing_quotes(words: list[str], glued: bool = False) -> list[str]:
    # 'added,”' -> 'added,'. Local agreement cuts before such a word instead; Grammar commits
    # a whole chunk at once, so it keeps the words and drops only the quote mark.
    # glued: the reply continues the last committed word, so a first word emptied here
    # still fills simulate.py's glue slot (as in hold_unearned_ends)
    kept = []
    for i, word in enumerate(words):
        if closes_quote(word):
            core = word.rstrip(".,;:!?")
            word = core.rstrip('”"»') + word[len(core):]
        if word or (i == 0 and glued):
            kept.append(word)
    return kept

def hold_unearned_ends(words: list[str], text: str, seen: list[str], committed: list[str]) -> list[str]:
    # An English sentence end only after the German has delivered one more than the English has.
    # Hy-MT2 ends cut-off input with "…" or ".", and once committed that stops every later reply.
    allowed = sum(map(ends_sentence, seen)) - sum(map(ends_sentence, committed))
    kept = []
    for i, word in enumerate(words):
        if ends_sentence(word):
            if allowed > 0:
                allowed -= 1
            else:
                word = strip_end(word)
        # a first word that was only an ending still fills simulate.py's glue slot, so it stays as ""
        glued = i == 0 and committed and not text[:1].isspace()
        if word or glued:
            kept.append(word)
    return kept


class WaitK:
    def __init__(self, arm: str, k: int, n_predict: int=64):
        self.arm = arm
        self.k = k
        self.n_predict = n_predict
    def reset(self):
        pass
    
    def step(self, seen: list[str], committed: list[str], source_done: bool) -> tuple[list[str], Result | None]:
        if len(seen) < self.k and not source_done:
            return [], None
        forced = join_words(committed)
        if source_done:
            res = translate(self.arm, seen, forced, FINAL_TOKENS)
            return res.text.split(), res
        # wait-k should have said len(seen) - k + 1 words by now, so ask for whatever is missing
        missing = len(seen) - self.k + 1 - len(committed)
        if missing <= 0:
            return [], None
        res = translate(self.arm, seen, forced, WORD_TOKENS * missing)
        return self.hold_sentence_ends(res.text.split()[:missing], res.text, seen, committed), res

    def hold_sentence_ends(self, words, text, seen, committed):
        return hold_unearned_ends(words, text, seen, committed)


    
class LocalAgreement:
    def __init__(self, arm: str, verb_rule: bool = False, stride: int = 1, n_predict: int = 64):
        self.arm = arm
        self.verb_rule = verb_rule
        self.stride = stride
        self.n_predict = n_predict
        self.prev = []
        
    def reset(self):
        self.prev = []
        
    def step(self, seen: list[str], committed: list[str], source_done: bool) -> tuple[list[str], Result | None]:
        # only call the model every "stride" German words (and always at the end of the sentence)
        if len(seen) % self.stride != 0 and not source_done:
            return [], None
        res = translate(self.arm, seen, join_words(committed), self.n_predict)

        words = res.text.split()
        # Hy-MT2 marks cut-off input with a trailing "…", which is not a real sentence end
        if not source_done and words:
            words[-1] = words[-1].rstrip("…")
            if not words[-1]:
                words.pop()
        if source_done:
            self.prev = []
            return words, res
        agreed = []
        for old, new in zip(self.prev, words):
            if not same_word(old, new):
                break
            agreed.append(new) #keep newer reply version
        agreed = hold_sentence_end(agreed)
        if quote_open(seen):
            agreed = hold_closing_quote(agreed)
        if self.verb_rule:
            agreed = hold_verbs(agreed, committed, seen)
        self.prev = words[len(agreed):]
        return agreed, res

class GrammarChunk:
    # German spaCy tags for conjugated verbs, infinitives, and participles.
    FINITE = {"VVFIN", "VAFIN", "VMFIN", "VVIMP", "VAIMP"}
    NONFINITE = {"VVINF", "VAINF", "VMINF", "VVIZU", "VVPP", "VAPP", "VMPP"}
    # Subordinating conjunctions (weil, dass) and relative pronouns (der, die).
    SUBORDINATE = {"KOUS", "PRELS", "PRELAT"}

    def __init__(self, arm: str, n_predict: int = 256, delay: int = 0):
        # Reuse the German tagger already loaded by the verb rule.
        from verb_rule import DE
        self.nlp = DE
        self.arm = arm
        self.n_predict = n_predict
        # Words of lookahead before a word is judged: with delay d, the rules decide about
        # the word d places behind the newest, tagged with the d words after it as context.
        # 0 is the original policy. On devtest 0 to 99 the newest word's tag differs from its
        # full-sentence tag for 8.5% of words, 0.8% one word later, 0.2% two words later.
        self.delay = delay
        self.reset()

    def reset(self):
        # Number of source words covered by previous translation calls.
        self.translated_until = 0

    def _boundary(self, seen: list[str]) -> int | None:
        # Tag the whole German prefix, so each word has its left context (the September
        # tagger test, tagger_prefix.py, found prefix tags reliable), then look only at
        # the tokens of the words not covered by an earlier translation.
        end = len(seen) - self.delay  # the word being judged is seen[end - 1]
        if end <= self.translated_until:
            return None
        doc_all = self.nlp(" ".join(seen))
        start = len(" ".join(seen[:self.translated_until])) + (1 if self.translated_until else 0)
        stop = len(" ".join(seen[:end]))  # the lookahead words are context only
        doc = [t for t in doc_all if start <= t.idx < stop]
        if not doc:
            return None
        last = doc[-1]

        # 1. Punctuation ends the chunk, even if it has no verb.
        if last.text in {",", ".", "!", "?", ";", ":", "…"}:
            return end

        has_verb = any(t.tag_ in self.FINITE for t in doc)

        # 2. After a verb, a conjunction starts the next chunk: keep "und/weil".
        if last.tag_ in {"KON", "KOUS"} and has_verb:
            return end - 1

        # 3. A final verb or separable particle can complete the chunk.
        # Examples: "weil er geht", "er hat gelesen", "er ruft mich an".
        if has_verb:
            subordinate_end = (doc[0].tag_ in self.SUBORDINATE
                               and last.tag_ in self.FINITE)
            verb_end = last.tag_ in self.NONFINITE | {"PTKVZ"}
            if subordinate_end or verb_end:
                return end
        return None  # No supported boundary yet: keep reading source words.

    def step(self, seen: list[str], committed: list[str], source_done: bool) -> tuple[list[str], Result | None]:
        # Avoid translating a source prefix that was already covered.
        if len(seen) <= self.translated_until:
            return [], None
        # Source end flushes any remaining fragment, even without a grammar boundary.
        boundary = len(seen) if source_done else self._boundary(seen)
        if boundary is None:
            return [], None

        # Keep the full source context up to the boundary. Force the English already
        # committed as the answer's start, so the model returns only its continuation.
        result = translate(self.arm, seen[:boundary], join_words(committed), self.n_predict)
        if result.stop_type == "limit" and not source_done:
            # A truncated reply must not mark the source chunk as fully translated: commit
            # nothing and leave the chunk open, so the next boundary translates it again with
            # more German. At the source end there is nothing to wait for, so the reply is kept
            # (as wait-k and local agreement do). The step record's "stop" field counts these.
            return [], result
        self.translated_until = boundary
        # Hold sentence ends the German has not delivered yet (shared with wait-k).
        words = result.text.split()
        if not source_done:
            words = hold_unearned_ends(words, result.text, seen[:boundary], committed)
            # No closing quote while the German quote is still open (as local agreement).
            if quote_open(seen[:boundary]):
                glued = bool(committed) and not result.text[:1].isspace()
                words = strip_closing_quotes(words, glued)
        return words, result



if __name__ == "__main__":
    from data import load
    arm = "gemma-e2b-q4"
    s = load(1)[0]
    for policy in [WaitK(arm, 3), LocalAgreement(arm)]:
        policy.reset()
        committed = []
        for t in range(1, len(s.src_words) + 1):
            done = t == len(s.src_words)
            new, _ = policy.step(s.src_words[:t], committed, done)
            committed += new
            print(t, s.src_words[t - 1], "->", " ".join(new))
        print(" ".join(committed), "\n")
        
