from dataclasses import dataclass
import time
import requests

## Settings
SERVER_URL = "http://localhost:8080"
CACHE_PROMPT = True # reuse the unchanged start of the previous prompt (llama.cpp prompt cache)
RETRIES = 4 # attempts per request
RETRY_WAIT = [5, 15, 45] # seconds to wait before each retry


## Prompts
### Hy-MT2 prompt
HYMT2_TEMPLATE = (
    "<｜hy_User｜>"
    "Translate the following text into English. Note that you should only output "
    "the translated result without any additional explanation:\n\n{src}"
    "<｜hy_Assistant｜>"
)

### Gemma E2B prompt
GEMMA_TEMPLATE = (
    "<|turn>user\n"
    "Translate the following German text into English. "
    "Output only the translation, with no explanation or notes.\n\n{src}"
    "<turn|>\n"
    "<|turn>model\n"
)

### Which template each arm uses
TEMPLATES = {
    "hymt2-q4": HYMT2_TEMPLATE,
    "gemma-e2b-q4": GEMMA_TEMPLATE,
}

### Which model file each arm runs (as given to llama-server -hf)
MODELS = {
    "hymt2-q4": "tencent/Hy-MT2-1.8B-GGUF:Q4_K_M",
    "gemma-e2b-q4": "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M",
}


@dataclass
class Result:
    text: str
    stop_type: str
    prompt_tokens: int
    output_tokens: int
    cached_tokens: int
    prompt_ms: float
    decode_ms: float
    
def build_prompt(arm: str, source_words: list[str], forced: str = "") -> str:
    src = " ".join(source_words)
    return TEMPLATES[arm].format(src=src) + forced
    
def translate(arm: str, source_words: list[str], forced: str = "", n_predict: int = 64) -> Result:
    payload = {
        "prompt": build_prompt(arm, source_words, forced),
        "n_predict": n_predict,
        "temperature": 0,
        "top_k": 1,
        "repeat_penalty": 1.0,
        "cache_prompt": CACHE_PROMPT,
    }
    # retry a failed request a few times before giving up, so one hiccup doesn't end an overnight run
    for attempt in range(RETRIES):
        try:
            r = requests.post(f"{SERVER_URL}/completion", json=payload, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == RETRIES - 1:
                raise
            print(f"request failed ({e}), retrying in {RETRY_WAIT[attempt]} s", flush=True)
            time.sleep(RETRY_WAIT[attempt])
    data = r.json()
    t = data["timings"]
    return Result(
        text=data["content"],
        stop_type=data["stop_type"],
        prompt_tokens=t["prompt_n"],
        output_tokens=t["predicted_n"],
        cached_tokens=t.get("cache_n", 0),
        prompt_ms=t["prompt_ms"],
        decode_ms=t["predicted_ms"],
    )

def loaded_model() -> str:
    r = requests.get(f"{SERVER_URL}/props", timeout=10)
    r.raise_for_status()
    return r.json()["model_alias"]
    
if __name__ == "__main__":
    arm = "gemma-e2b-q4"
    src = str("„Wir haben jetzt 4 Monate alte Mäuse, die Diabetes hatten und jetzt keinen mehr haben“, fügte er hinzu.").split()
    half = src[: len(src) // 2]

    print(translate(arm, src))                      # full sentence
    print(translate(arm, half))                     # half sentence
    print(translate(arm, half, forced="We now"))    # half, with forced start

