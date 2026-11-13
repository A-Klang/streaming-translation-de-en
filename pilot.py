#!/usr/bin/env python3
"""Full-sentence pilot for the streaming translation study (T-725-MALV).

Revision of 23 September 2026. Adds the HY-MT arms and a single prefix
forcing path for decoder-only models (Gemma and HY-MT share it). Opus-MT
and NLLB stay in as Arm B fallbacks and use the encoder-decoder path.

Nothing here streams. Every sentence is translated with the whole sentence
visible. Outputs, costs and a prefix forcing test are saved per arm and
language, and results/summary.json is merged across runs.

Examples
  python pilot.py --arms hymt15 --n 5 --no-comet --out smoke
  python pilot.py --arms hymt15 hymt2 opus nllb --n 100
  python pilot.py --arms gemma-e2b --n 100
  python pilot.py --score-only            # (re)score saved outputs, e.g. with COMET
  python pilot.py --src deu_Latn --langs eng_Latn --arms hymt15 opus nllb   # German to English
"""

import argparse
import gc
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------------
# Languages. German is the target; Mandarin is kept as an optional second
# language because every Arm B candidate covers it.
# ----------------------------------------------------------------------------

LANGS = {
    "eng_Latn": dict(name="English", zh_name="英语", nllb="eng_Latn", hymt=True, bleu_tok="13a"),
    "deu_Latn": dict(name="German", zh_name="德语", opus="Helsinki-NLP/opus-mt-en-de",
                     opus_tag="", nllb="deu_Latn", hymt=True, bleu_tok="13a"),
    "zho_Hans": dict(name="Chinese", zh_name="中文", opus="Helsinki-NLP/opus-mt-en-zh",
                     opus_tag=">>cmn_Hans<< ", nllb="zho_Hans", hymt=True, bleu_tok="zh"),
    # Kept only for Arm A / fallback comparisons. HY-MT does not cover them.
    "swe_Latn": dict(name="Swedish", zh_name="瑞典语", opus="Helsinki-NLP/opus-mt-en-sv",
                     opus_tag="", nllb="swe_Latn", hymt=False, bleu_tok="13a"),
    "isl_Latn": dict(name="Icelandic", zh_name="冰岛语", opus="Helsinki-NLP/opus-mt-en-is",
                     opus_tag="", nllb="isl_Latn", hymt=False, bleu_tok="13a"),
}

# ----------------------------------------------------------------------------
# Arms. "chat" = decoder-only model driven by a chat-template prompt.
# "seq2seq" = encoder-decoder dedicated translator.
# Prompts are taken from the model cards (checked 23 Sep 2026).
# ----------------------------------------------------------------------------

HYMT15_PROMPT = "Translate the following segment into {lang}, without additional explanation.\n\n{src}"
HYMT15_PROMPT_ZH = "将以下文本翻译为{zh_lang}，注意只需要输出翻译后的结果，不要额外解释：\n\n{src}"
HYMT2_PROMPT = ("Translate the following text into {lang}. Note that you should only output "
                "the translated result without any additional explanation:\n\n{src}")
# Incomplete-input instruction, identical wording for every arm that uses it.
PARTIAL_NOTE = ("The text may stop mid-sentence. Translate only what is there, "
                "and do not complete the sentence.")
HYMT2_PROMPT_PARTIAL = ("Translate the following text into {lang}. " + PARTIAL_NOTE + " Note that you should only output "
                        "the translated result without any additional explanation:\n\n{src}")
GEMMA_PROMPT_PARTIAL = ("Translate the following {src_lang} text into {lang}. " + PARTIAL_NOTE + " Output only the translation, "
                        "with no explanation or notes.\n\n{src}")
GEMMA_PROMPT = ("Translate the following {src_lang} text into {lang}. Output only the translation, "
                "with no explanation or notes.\n\n{src}")

ARMS = {
    "gemma-e2b": dict(role="A", kind="chat", repo="google/gemma-4-E2B-it", loader="multimodal",
                      prompt=GEMMA_PROMPT, needs_hymt_lang=False),
    "hymt15": dict(role="B", kind="chat", repo="tencent/HY-MT1.5-1.8B", loader="causal",
                   prompt=HYMT15_PROMPT, prompt_zh=HYMT15_PROMPT_ZH, needs_hymt_lang=True),
    "hymt2": dict(role="B-candidate", kind="chat", repo="tencent/Hy-MT2-1.8B", loader="causal",
                  trust_remote_code=True, prompt=HYMT2_PROMPT, needs_hymt_lang=True),
    # Quantized arms through llama.cpp (llama-server), the same runtime as the Pi.
    "gemma-e2b-q4": dict(role="A", kind="server", gguf="unsloth/gemma-4-E2B-it-GGUF:Q4_K_M",
                         prompt=GEMMA_PROMPT, needs_hymt_lang=False),
    "hymt15-q4": dict(role="B", kind="server", gguf="tencent/HY-MT1.5-1.8B-GGUF:Q4_K_M",
                      prompt=HYMT15_PROMPT, prompt_zh=HYMT15_PROMPT_ZH, needs_hymt_lang=True),
    "hymt2-q4": dict(role="B", kind="server", gguf="tencent/Hy-MT2-1.8B-GGUF:Q4_K_M",
                     prompt=HYMT2_PROMPT, needs_hymt_lang=True),
    # Same arms with the incomplete-input instruction (PARTIAL_NOTE).
    "gemma-e2b-q4-partial": dict(role="A", kind="server", gguf="unsloth/gemma-4-E2B-it-GGUF:Q4_K_M",
                                 prompt=GEMMA_PROMPT_PARTIAL, needs_hymt_lang=False),
    "hymt2-q4-partial": dict(role="B", kind="server", gguf="tencent/Hy-MT2-1.8B-GGUF:Q4_K_M",
                             prompt=HYMT2_PROMPT_PARTIAL, needs_hymt_lang=True),
    "opus": dict(role="B-fallback", kind="seq2seq", family="opus"),
    "nllb": dict(role="B-fallback", kind="seq2seq", family="nllb",
                 repo="facebook/nllb-200-distilled-600M"),
}

# Opus-MT is one model per direction: (src, tgt) -> (repo, target tag)
OPUS = {
    ("eng_Latn", "deu_Latn"): ("Helsinki-NLP/opus-mt-en-de", ""),
    ("deu_Latn", "eng_Latn"): ("Helsinki-NLP/opus-mt-de-en", ""),
    ("eng_Latn", "zho_Hans"): ("Helsinki-NLP/opus-mt-en-zh", ">>cmn_Hans<< "),
    ("zho_Hans", "eng_Latn"): ("Helsinki-NLP/opus-mt-zh-en", ""),
    ("eng_Latn", "swe_Latn"): ("Helsinki-NLP/opus-mt-en-sv", ""),
    ("eng_Latn", "isl_Latn"): ("Helsinki-NLP/opus-mt-en-is", ""),
}


def pair_key(sl, tl):
    """File and summary key. English-source runs keep the old bare target key."""
    return tl if sl == "eng_Latn" else f"{sl}-{tl}"


def parse_key(key):
    return tuple(key.split("-", 1)) if "-" in key else ("eng_Latn", key)


def split_src(text, sl):
    return list(text.replace(" ", "")) if sl == "zho_Hans" else text.split()


PREFIX_TEST_N = 20


# ----------------------------------------------------------------------------
# Device and memory helpers
# ----------------------------------------------------------------------------

def pick_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(torch, device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def device_mem_gb(torch, device):
    """Memory currently held by the accelerator (driver view on MPS)."""
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return 0.0


def process_peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # bytes on macOS, kilobytes on Linux
    return r / 1e9 if platform.system() == "Darwin" else r * 1024 / 1e9


def system_ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return float("nan")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_flores(sl, tl, n):
    from datasets import load_dataset
    src = load_dataset("openlanguagedata/flores_plus", sl, split="devtest")
    tgt = load_dataset("openlanguagedata/flores_plus", tl, split="devtest")
    if "id" in src.column_names and "id" in tgt.column_names:
        ref_by_id = {r["id"]: r["text"] for r in tgt}
        pairs = [(r["text"], ref_by_id[r["id"]]) for r in src if r["id"] in ref_by_id]
    else:
        pairs = list(zip(src["text"], tgt["text"]))
    return pairs[:n]


# ----------------------------------------------------------------------------
# Translators
# ----------------------------------------------------------------------------

class ChatTranslator:
    """Decoder-only model prompted through its chat template.

    Prefix forcing: the forced target words are appended as plain text after
    the generation prompt, so the model continues its own answer. This is the
    same mechanism for Gemma and HY-MT.
    """

    def __init__(self, arm, torch, device, max_new_tokens):
        import transformers as tf
        self.arm, self.torch, self.device = arm, torch, device
        self.max_new_tokens = max_new_tokens
        repo = arm["repo"]
        trc = arm.get("trust_remote_code", False)
        dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
        if arm["loader"] == "multimodal":
            cls = getattr(tf, "AutoModelForMultimodalLM", None) or tf.AutoModelForImageTextToText
            proc = tf.AutoProcessor.from_pretrained(repo)
            self.tok = getattr(proc, "tokenizer", proc)
            self.template_owner = proc
        else:
            cls = tf.AutoModelForCausalLM
            self.tok = tf.AutoTokenizer.from_pretrained(repo, trust_remote_code=trc)
            self.template_owner = self.tok
        self.model = cls.from_pretrained(repo, dtype=dtype, trust_remote_code=trc).to(device).eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self._template_cache = {}

    def _prompt_text(self, src, d):
        sl, tl = d
        L = LANGS[tl]
        # HY-MT's card uses the Chinese prompt whenever Chinese is on either side
        use_zh = "zho_Hans" in d and self.arm.get("prompt_zh")
        tmpl = self.arm["prompt_zh"] if use_zh else self.arm["prompt"]
        user = tmpl.format(lang=L["name"], zh_lang=L["zh_name"], src_lang=LANGS[sl]["name"], src=src)
        msgs = [{"role": "user", "content": user}]
        try:
            return self.template_owner.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            # processors may want structured content
            msgs = [{"role": "user", "content": [{"type": "text", "text": user}]}]
            return self.template_owner.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def template_tokens(self, d):
        """Prompt tokens that are not the source sentence (instruction overhead)."""
        if d not in self._template_cache:
            self._template_cache[d] = len(self._encode(self._prompt_text("", d)))
        return self._template_cache[d]

    def _encode(self, text):
        # the chat template already contains BOS and role markers
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def _generate(self, text):
        torch = self.torch
        ids = torch.tensor([self._encode(text)], device=self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=self.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        new = out[0, ids.shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True).strip(), ids.shape[1], int(new.shape[0])

    def translate(self, src, d):
        return self._generate(self._prompt_text(src, d))

    def continue_prefix(self, src, d, forced):
        text, _, _ = self._generate(self._prompt_text(src, d) + forced)
        return text


class ServerTranslator(ChatTranslator):
    """Quantized GGUF model served by llama.cpp's llama-server.

    Same prompts and the same prefix forcing as ChatTranslator: the chat
    template is applied by the server (from the GGUF), the forced words are
    appended as plain text, and the server completes the raw prompt.
    """

    PORT = 8089

    def __init__(self, arm, torch, device, max_new_tokens, log_dir=Path("results")):
        import shutil, subprocess
        self.arm, self.torch, self.device = arm, torch, "llama.cpp"
        self.max_new_tokens = max_new_tokens
        self._template_cache = {}
        exe = shutil.which("llama-server")
        if not exe:
            sys.exit("llama-server not found. Install llama.cpp first: brew install llama.cpp")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log = open(log_dir / "llama-server.log", "w")
        cmd = [exe, "-hf", arm["gguf"], "--port", str(self.PORT), "-c", "4096", "-np", "1",
               "-ngl", "99", "--jinja", "--no-mmproj"]
        print("starting: " + " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        self._wait_ready()
        meta = self._get("/v1/models")["data"][0].get("meta", {})
        self.n_params = meta.get("n_params", float("nan"))
        self.repo = arm["gguf"]
        props = self._get("/props")
        self.bos = props.get("bos_token", "") or ""

    # -- HTTP helpers --------------------------------------------------------
    def _url(self, path):
        return f"http://127.0.0.1:{self.PORT}{path}"

    def _get(self, path):
        import urllib.request
        with urllib.request.urlopen(self._url(path), timeout=30) as r:
            return json.loads(r.read())

    def _post(self, path, payload):
        import urllib.request
        req = urllib.request.Request(self._url(path), data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())

    def _wait_ready(self, timeout=1800):
        import urllib.error
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                sys.exit(f"llama-server exited early; see {self.log.name}")
            try:
                if self._get("/health").get("status") == "ok":
                    return
            except (urllib.error.URLError, ConnectionError, json.JSONDecodeError, OSError):
                pass
            time.sleep(2)
        sys.exit("llama-server did not become ready (download or load too slow?)")

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()
        self.log.close()

    def rss_gb(self):
        import subprocess
        try:
            kb = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(self.proc.pid)]).strip())
            return kb / 1e6
        except Exception:
            return float("nan")

    # -- translator interface -------------------------------------------------
    def _prompt_text(self, src, d):
        sl, tl = d
        L = LANGS[tl]
        use_zh = "zho_Hans" in d and self.arm.get("prompt_zh")
        tmpl = self.arm["prompt_zh"] if use_zh else self.arm["prompt"]
        user = tmpl.format(lang=L["name"], zh_lang=L["zh_name"], src_lang=LANGS[sl]["name"], src=src)
        text = self._post("/apply-template", {"messages": [{"role": "user", "content": user}],
                                              "chat_template_kwargs": {"enable_thinking": False}})["prompt"]
        # the server adds BOS itself when it tokenizes; avoid a double BOS
        if self.bos and text.startswith(self.bos):
            text = text[len(self.bos):]
        return text

    def _encode(self, text):
        return self._post("/tokenize", {"content": text, "add_special": False})["tokens"]

    def _generate(self, text):
        r = self._post("/completion", {"prompt": text, "n_predict": self.max_new_tokens,
                                       "temperature": 0.0, "top_k": 1, "cache_prompt": True})
        return r["content"].strip(), int(r.get("tokens_evaluated", 0)), int(r.get("tokens_predicted", 0))


def make_chat(arm, torch, device, max_new_tokens, log_dir=Path("results")):
    if arm["kind"] == "server":
        return ServerTranslator(arm, torch, device, max_new_tokens, log_dir)
    return ChatTranslator(arm, torch, device, max_new_tokens)


class Seq2SeqTranslator:
    """Encoder-decoder dedicated translator (Opus-MT or NLLB)."""

    def __init__(self, arm, d, torch, device, max_new_tokens):
        import transformers as tf
        sl, tl = d
        self.arm, self.torch, self.device = arm, torch, device
        self.max_new_tokens = max_new_tokens
        if arm["family"] == "opus":
            if d not in OPUS:
                raise ValueError(f"no Opus-MT model configured for {sl}->{tl}")
            (self.repo, self.src_tag), self.bos = OPUS[d], None
            self.tok = tf.AutoTokenizer.from_pretrained(self.repo)
        else:
            self.repo, self.src_tag = arm["repo"], ""
            self.tok = tf.AutoTokenizer.from_pretrained(self.repo, src_lang=LANGS[sl]["nllb"])
            self.bos = self.tok.convert_tokens_to_ids(LANGS[tl]["nllb"])
        self.model = tf.AutoModelForSeq2SeqLM.from_pretrained(self.repo).to(device).eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def template_tokens(self, d):
        return len(self.tok(self.src_tag)["input_ids"]) - 1 if self.src_tag else 0

    def _decoder_start(self):
        start = [self.model.config.decoder_start_token_id]
        if self.bos is not None:
            start.append(self.bos)
        return start

    def _run(self, src, forced=None):
        torch = self.torch
        enc = self.tok(self.src_tag + src, return_tensors="pt").to(self.device)
        dec = self._decoder_start()
        if forced:
            dec += self.tok(text_target=forced, add_special_tokens=False)["input_ids"]
        dec_ids = torch.tensor([dec], device=self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, decoder_input_ids=dec_ids,
                                      max_new_tokens=self.max_new_tokens,
                                      do_sample=False, num_beams=1)
        new = out[0, dec_ids.shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True).strip(), enc["input_ids"].shape[1], int(new.shape[0])

    def translate(self, src, d):
        return self._run(src)

    def continue_prefix(self, src, d, forced):
        text, _, _ = self._run(src, forced)
        return text


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------

def score_rows(rows, lang):
    import sacrebleu
    hyps = [r["hyp"] for r in rows]
    refs = [r["ref"] for r in rows]
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=LANGS[lang]["bleu_tok"]).score
    return dict(chrF=chrf, BLEU=bleu)


def comet_scores(jobs, batch_size=8):
    """jobs: list of (key, rows). Loads COMET once and scores all."""
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    out = {}
    for key, rows in jobs:
        data = [dict(src=r["src"], mt=r["hyp"], ref=r["ref"]) for r in rows]
        res = model.predict(data, batch_size=batch_size, gpus=0, num_workers=1, progress_bar=False)
        out[key] = float(res.system_score)
    return out


# ----------------------------------------------------------------------------
# Prefix forcing test
# ----------------------------------------------------------------------------

def prefix_case(src, full_hyp, sl="eng_Latn"):
    """Cut the source at its midpoint and force the first quarter of the
    model's own full-sentence translation (at least one word)."""
    sw = split_src(src, sl)
    joiner = "" if sl == "zho_Hans" else " "
    half_src = joiner.join(sw[: max(1, (len(sw) + 1) // 2)])
    hw = full_hyp.split()
    if not hw:
        return half_src, ""
    k = max(1, len(hw) // 4)
    return half_src, " ".join(hw[:k])


def restarted(forced, continuation):
    """Heuristic: the continuation starts by repeating the forced words."""
    f = forced.strip().lower()
    c = continuation.strip().lower()
    return bool(f) and c.startswith(f)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def run_arm(arm_name, langs, args, torch, device):
    sl = args.src
    arm = ARMS[arm_name]
    results = {}
    shared = None
    for tl in langs:
        d, lang = (sl, tl), pair_key(sl, tl)
        if arm.get("needs_hymt_lang") and not (LANGS[sl]["hymt"] and LANGS[tl]["hymt"]):
            print(f"[{arm_name}] skipping {lang}: not covered by HY-MT")
            continue
        pairs = load_flores(sl, tl, args.n)

        if arm["kind"] in ("chat", "server"):
            if shared is None:
                shared = make_chat(arm, torch, device, args.max_new_tokens, Path(args.out) / arm_name)
                print(f"loaded {arm.get('repo') or arm.get('gguf')}: {shared.n_params/1e6:.0f}M parameters (stored, incl. embeddings)")
            tr = shared
        else:
            tr = Seq2SeqTranslator(arm, d, torch, device, args.max_new_tokens)
            print(f"loaded {tr.repo}: {tr.n_params/1e6:.0f}M parameters")

        # warm-up, excluded from timing
        tr.translate(pairs[0][0], d)
        sync(torch, device)

        outdir = Path(args.out) / arm_name
        outdir.mkdir(parents=True, exist_ok=True)
        rows, peak_dev = [], 0.0
        tmpl_tok = tr.template_tokens(d)
        with open(outdir / f"{lang}.jsonl", "w", encoding="utf-8") as f:
            for i, (src, ref) in enumerate(pairs):
                sync(torch, device)
                t0 = time.perf_counter()
                hyp, n_in, n_out = tr.translate(src, d)
                sync(torch, device)
                dt = time.perf_counter() - t0
                peak_dev = max(peak_dev, tr.rss_gb() if arm["kind"] == "server" else device_mem_gb(torch, device))
                row = dict(i=i, src=src, ref=ref, hyp=hyp, sec=dt, in_tokens=n_in,
                           out_tokens=n_out, template_tokens=tmpl_tok, src_words=len(split_src(src, sl)))
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[{arm_name} {lang} {i}] {dt:.2f}s  {hyp[:100]}")

        # prefix forcing test (all arms, so Arm A and Arm B are checked the same way)
        n_empty = n_restart = 0
        with open(outdir / f"{lang}_prefix.jsonl", "w", encoding="utf-8") as f:
            for r in rows[:PREFIX_TEST_N]:
                half_src, forced = prefix_case(r["src"], r["hyp"], sl)
                cont = tr.continue_prefix(half_src, d, forced) if forced else ""
                # an empty full-sentence output leaves nothing to force; that is not a forcing failure
                empty, rs = (bool(forced) and not cont.strip()), restarted(forced, cont)
                n_empty += empty
                n_restart += rs
                f.write(json.dumps(dict(i=r["i"], half_src=half_src, forced=forced, continuation=cont,
                                        full_hyp=r["hyp"], empty=empty, restart=rs),
                                   ensure_ascii=False) + "\n")

        src_words = sum(r["src_words"] for r in rows)
        s = score_rows(rows, tl)
        s.update(
            n=len(rows),
            sec_per_sentence=sum(r["sec"] for r in rows) / len(rows),
            out_tokens_per_sec=sum(r["out_tokens"] for r in rows) / sum(r["sec"] for r in rows),
            tokens_per_src_word=sum(r["in_tokens"] + r["out_tokens"] for r in rows) / src_words,
            template_tokens_per_src_word=sum(r["template_tokens"] for r in rows) / src_words,
            peak_mem_GB=peak_dev if (device != "cpu" or arm["kind"] == "server") else process_peak_rss_gb(),
            process_peak_rss_GB=process_peak_rss_gb(),
            params_M=tr.n_params / 1e6,
            prefix_test_n=min(PREFIX_TEST_N, len(rows)),
            prefix_test_empty_continuations=n_empty,
            prefix_test_restarts=n_restart,
            device="llama.cpp-metal" if arm["kind"] == "server" else device, role=arm["role"],
        )
        results[lang] = s
        print(json.dumps({arm_name: {lang: s}}, indent=1))

        if arm["kind"] == "seq2seq":
            del tr
            free(torch, device)
    if shared is not None:
        if hasattr(shared, "close"):
            shared.close()
        del shared
        free(torch, device)
    return results


def free(torch, device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def merge_summary(path, new, meta):
    data = {"results": {}, "meta": {}}
    if path.exists():
        data = json.loads(path.read_text())
    for arm, per in new.items():
        data["results"].setdefault(arm, {})
        for lang, s in per.items():
            old = data["results"][arm].get(lang, {})
            old.update(s)
            data["results"][arm][lang] = old
    data["meta"].update(meta)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))


def score_only(args):
    """Rescore every saved <arm>/<lang>.jsonl, including COMET unless --no-comet."""
    out = Path(args.out)
    jobs, new = [], {}
    for f in sorted(out.glob("*/*.jsonl")):
        if f.stem.endswith("_prefix"):
            continue
        arm, lang = f.parent.name, f.stem
        if args.arms and arm not in args.arms:
            continue
        rows = [json.loads(l) for l in f.open(encoding="utf-8")]
        new.setdefault(arm, {})[lang] = score_rows(rows, parse_key(lang)[1])
        jobs.append(((arm, lang), rows))
    if not args.no_comet and jobs:
        for (arm, lang), c in comet_scores(jobs).items():
            new[arm][lang]["COMET"] = c
    merge_summary(out / "summary.json", new, {})
    print(json.dumps(new, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", choices=list(ARMS), default=None)
    p.add_argument("--src", choices=list(LANGS), default="eng_Latn",
                   help="source language (default English, the 23 Sep setup)")
    p.add_argument("--langs", nargs="+", choices=list(LANGS), default=["deu_Latn"],
                   help="target language(s)")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", default="results")
    p.add_argument("--no-comet", action="store_true")
    p.add_argument("--score-only", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = p.parse_args()

    if args.score_only:
        return score_only(args)
    if not args.arms:
        p.error("--arms is required unless --score-only is given")
    if args.src in args.langs:
        p.error("--src must differ from every --langs target")

    import torch
    device = pick_device(torch) if args.device == "auto" else args.device
    ram = system_ram_gb()
    print(f"device={device}  system RAM={ram:.1f} GB  torch={torch.__version__}")
    if "gemma-e2b" in args.arms and ram < 15:
        print("WARNING: Gemma 4 E2B stores about 5.1B parameters (about 10 GB in bf16). "
              "With under 16 GB of RAM it will likely not fit; see the manual's troubleshooting section.")

    new = {}
    for a in args.arms:
        new[a] = run_arm(a, args.langs, args, torch, device)

    if not args.no_comet:
        jobs = []
        for a, per in new.items():
            for lang in per:
                rows = [json.loads(l) for l in (Path(args.out) / a / f"{lang}.jsonl").open(encoding="utf-8")]  # lang is the pair key
                jobs.append(((a, lang), rows))
        try:
            for (a, lang), c in comet_scores(jobs).items():
                new[a][lang]["COMET"] = c
        except Exception as e:  # keep the MT results even if COMET fails
            print(f"COMET failed ({e!r}); rerun later with --score-only")

    meta = dict(machine=platform.platform(), python=sys.version.split()[0],
                torch=torch.__version__, system_ram_GB=ram)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    merge_summary(Path(args.out) / "summary.json", new, meta)

    cols = ["chrF", "BLEU", "COMET", "sec_per_sentence", "tokens_per_src_word", "peak_mem_GB"]
    print("arm\tlang\t" + "\t".join(cols))
    for a, per in new.items():
        for lang, s in per.items():
            print(a, lang, *[f"{s.get(c, float('nan')):.2f}" for c in cols], sep="\t")


if __name__ == "__main__":
    main()
