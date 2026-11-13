# Commands

Run from the repository folder (`~/simulmt-pilot`) unless noted.

## Environment

```bash
source .venv/bin/activate
```
Activates the main environment. Needed in every new terminal.

## Model server

```bash
llama-server -hf tencent/Hy-MT2-1.8B-GGUF:Q4_K_M --port 8080 -c 4096 -np 1 -ngl 99
```
Starts Hy-MT2 (Arm B) on port 8080. Leave it running in its own terminal; stop with Ctrl+C.

```bash
llama-server -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_M --port 8080 -c 4096 -np 1 -ngl 99 --swa-full --no-mmproj
```
Starts Gemma 4 E2B (Arm A) instead. Only one model at a time on the Air.
`--swa-full` keeps the full prompt cache for Gemma's sliding-window layers. Without it only about a fifth of each prompt is reused instead of two thirds (computed 56 vs 24 tokens per source word), which would make Gemma look slower than it is.
`--no-mmproj` skips Gemma's image encoder, which `-hf` downloads and loads by default. It is not used for text and takes about 1.1 GiB of GPU memory.


```bash
curl -s localhost:8080/health
```
Checks the server is up and the model loaded.

## Pipeline

```bash
python pipeline/backend.py
```
Runs the backend test: full sentence, half sentence, half sentence with forced start.

```bash
python pipeline/simulate.py --arm hymt2-q4 --policy la --n 10
python pipeline/simulate.py --arm hymt2-q4 --policy waitk --k 3 --n 10
```
Streams the first 10 sentences through local agreement or wait-k with Hy-MT2 and saves a new file in `results/pipeline/hymt2-q4_<policy>/` (see below). Needs the Hy-MT2 server running.

```bash
python pipeline/simulate.py --arm gemma-e2b-q4 --policy la --n 10
python pipeline/simulate.py --arm gemma-e2b-q4 --policy waitk --k 3 --n 10
```
Same with Gemma, saved in `results/pipeline/gemma-e2b-q4_<policy>/`. Needs the Gemma server running. If `--arm` doesn't match the model the server has loaded, the script stops before translating anything.

- `--arm` picks the model: `hymt2-q4` or `gemma-e2b-q4`.
- `--verb-rule` adds the verb rule to local agreement: no English verb or number is committed before the German has delivered it.
- `--stride` calls the model only every s German words in local agreement (default 1). Higher means less compute and more lag. Saved under `<arm>_la-s<stride>[-verb]`.
- `--policy` is `la` (local agreement) or `waitk`; `--k` sets the lag for wait-k (default 3).
- `--n` is the number of sentences, `--start` the first sentence index (default 0).
- `--live` prints the word-by-word table while it runs.
- `--replies`, with `--live`, also prints the model's raw reply at each call as a dimmed `↳` line under the row.
- `--resume FILE` continues an interrupted run in that file: sentences already in it are skipped, the rest appended. The file must be from the same setting.

Each setting has its own folder, `results/pipeline/<arm>_<policy>/`, and each run adds a new file to it named `<arm>_<policy>_<YYYYMMDD-HHMMSS>.jsonl`. A run never overwrites an earlier one. Files from before 3 Nov were moved into these folders with their last-modified time as the timestamp.


```bash
jq -r '"[\(.idx)]  \(.calls) calls, \(.prompt_tokens)+\(.output_tokens) tokens, \(.compute_ms/1000|floor) s\n  DE : \(.src)\n  HYP: \(.hyp)\n  REF: \(.ref)\n"' results/pipeline/hymt2-q4_la-s1/hymt2-q4_la-s1_20261031-212943.jsonl
```
Prints a results file readably: German, committed English and reference per sentence. Add `| less` to scroll.

## Overnight runs

```bash
pipeline/night.sh hymt2-q4
pipeline/night.sh gemma-e2b-q4
```
Runs all 9 report settings (wait-k 1, 3, 5; local agreement at stride 1, 2, 3, with and without the verb rule) for one arm on devtest 300 to 1011. Starts and stops the arm's llama-server itself, so no server may be running on port 8080 beforehand. A setting that crashes is resumed in the same file, up to 3 times. Progress in `logs/night_<arm>_<time>/night.log`, one log per setting next to it. `START=300 N=5 pipeline/night.sh hymt2-q4` runs a shorter range. `pipeline/night.sh hymt2-q4 grammar` runs the Grammar delays 0, 1, 2 instead of the 9 settings (logs in `logs/night_<arm>_grammar_<time>/`).

## Raw requests

```bash
curl -s localhost:8080/apply-template -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hallo Welt"}]}'
```
Shows the prompt string the model's chat template produces.

```bash
curl -s localhost:8080/completion -H "Content-Type: application/json" \
  -d '{"prompt":"<PROMPT>","n_predict":64,"temperature":0,"top_k":1,"repeat_penalty":1.0,"cache_prompt":false}'
```
One completion with the pipeline's settings. Replace `<PROMPT>` with the full template text.

```bash
curl -s localhost:8080/tokenize -H "Content-Type: application/json" \
  -d '{"content":"Hallo Welt","with_pieces":true}'
```
Shows how a string is split into tokens.

## Significance

```
python pipeline/bootstrap.py > results/bootstrap.txt
```

Paired bootstrap over all report runs (needs `score.py --comet` run first, for the `.comet.json` sentence scores, and the Haiku judge files). About a minute.
