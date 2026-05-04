"""
Robustness v2: 3 conditions × 6 SD × 11 stakes × 5 reps, both models.

SD grid (subset of main v2): [1, 2, 10, 20, 50, 100]
Stake grid (subset of main v2, ~log-spaced): [1, 10, 27, 95, 328, 1135, 3926, 13578, 46962, 162379, 250000]
N_rep: 5

Conditions: low temp (0.3), high temp (1.5), alt prompt (default temp)

Output:
  out/results_v2_robust_deepseek_temp_low.csv
  out/results_v2_robust_deepseek_temp_high.csv
  out/results_v2_robust_deepseek_altprompt.csv
  out/results_v2_robust_grok_temp_low.csv
  out/results_v2_robust_grok_temp_high.csv
  out/results_v2_robust_grok_altprompt.csv
"""
from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
import requests

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "out"

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_DEEPSEEK = "deepseek/deepseek-v3.2"
MODEL_GROK = "x-ai/grok-4.1-fast"
GAME = "dictator"

# Subset of main v2 SD grid, includes human comparison points
ROBUST_SD = [1, 2, 10, 20, 50, 100]

# Subset of main v2 stake grid (~every 3rd value), includes $10
ROBUST_STAKES = [1, 10, 27, 95, 328, 1135, 3926, 13578, 46962, 162379, 250000]

N_TARGET = 3
MAX_ATTEMPTS = 8
MAX_WORKERS = 6

PROMPT_TMPL = (
    "You will be asked to make a decision regarding how much\n"
    "money you might offer another person under a situation\n"
    "that will be explained shortly. There are no correct or incorrect\n"
    "answers, and the money is hypothetical – that is, no one will "
    "receive\n"
    "the actual money. Nonetheless, we want you to make your decision\n"
    "as if the amount and situation were real.\n"
    "Before describing the situation, we want you to imagine that you\n"
    "have made a list of the 100 people closest to you in the world,\n"
    "ranging from your dearest friend or relative at position #1 to a\n"
    "mere acquaintance at #100. The person at number one would be\n"
    "someone you know well and is your closest friend or relative. The\n"
    "person at #100 might be someone you recognize and encounter but\n"
    "perhaps you may not even know their name. You do not have to\n"
    "create the list – just imagine that you have done so.\n"
    "Imagine you have been given the amount of money: {m}\n"
    "You are to divide the amount of money between\n"
    "yourself and another person who is in place {x} at that list.\n"
    "You are free to give as much or as little\n"
    "of the amount of money as you wish, and you will receive what is\n"
    "left. Please respond with the amount you wish to offer the other\n"
    "person. Follow the exact answer template: \"I share "
    "[decided amount] dollars\"."
)

ALT_PROMPT_TMPL = (
    "You are participating in a resource allocation study. You have "
    "received {m} and must decide how much to give to another person. "
    "The other person is someone you know—they are ranked #{x} in "
    "social closeness to you, where #1 is your closest friend or family "
    "member and #100 is a distant acquaintance whose name you might not "
    "know. You may give any amount from $0 to {m}, and you keep the "
    "rest. Please reply with the exact template: "
    "\"I share [decided amount] dollars\"."
)

_SHARE_RE = re.compile(
    r"\bi\s*share\s*([$€£]?\s*[0-9][0-9,]*(?:\.[0-9]+)?)\s*dollar(?:s)?\b",
    re.IGNORECASE,
)

CONDITIONS = [
    {"name": "temp_low",   "temperature": 0.3,  "alt_prompt": False},
    {"name": "temp_high",  "temperature": 1.5,  "alt_prompt": False},
    {"name": "altprompt",  "temperature": None, "alt_prompt": True},
]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = val.strip().strip('"').strip("'")


def parse_share(text: str) -> float | None:
    m = _SHARE_RE.search(text or "")
    if not m:
        return None
    raw = m.group(1).replace(",", "").replace("$", "").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def api_call(
    api_key: str, model_id: str, prompt: str, temperature: float | None = None
) -> tuple[str, float]:
    payload: dict = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    site = os.getenv("OPENROUTER_SITE_URL", "").strip()
    if site:
        headers["HTTP-Referer"] = site

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            t0 = time.time()
            resp = requests.post(
                OPENROUTER_API_URL, headers=headers, json=payload, timeout=90
            )
            dt = time.time() - t0
            if resp.status_code == 403:
                raise RuntimeError(f"API key error (403): {resp.text[:200]}")
            if resp.status_code == 402:
                raise RuntimeError(f"Insufficient credits (402): {resp.text[:200]}")
            if resp.status_code >= 500:
                time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "\n".join(
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in content
                )
            return str(content), dt
        except RuntimeError:
            raise
        except Exception as exc:
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("API call failed after retries") from last_exc


def run_cell(
    x: int,
    stake: int,
    model_id: str,
    api_key: str,
    temperature: float | None = None,
    alt_prompt: bool = False,
    condition: str = "main",
) -> list[dict]:
    if alt_prompt:
        prompt = ALT_PROMPT_TMPL.format(m=f"${stake:,}", x=x)
    else:
        prompt = PROMPT_TMPL.format(m=f"${stake:,}", x=x)

    rows: list[dict] = []
    rep = 0
    attempts = 0

    while rep < N_TARGET and attempts < MAX_ATTEMPTS:
        attempts += 1
        try:
            text, latency = api_call(api_key, model_id, prompt, temperature)
            share = parse_share(text)
        except RuntimeError:
            raise
        except Exception:
            text, share, latency = "", None, float("nan")

        if share is not None:
            rep += 1
            share_pct = float(share) / float(stake)
            rows.append({
                "condition": condition,
                "model": model_id,
                "game": GAME,
                "temperature": temperature,
                "social_distance": x,
                "money": stake,
                "rep": rep,
                "raw_response": text,
                "share": share,
                "share_pct": share_pct,
                "latency_s": latency,
            })

    return rows


PROGRESS_FILE = OUT_DIR / "progress_v2.txt"


def write_progress(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)


def run_condition_grid(
    model_id: str,
    api_key: str,
    condition: dict,
    out_csv: Path,
) -> None:
    if out_csv.exists():
        write_progress(f"{out_csv.name} already exists — skipping.")
        return

    cond_name = condition["name"]
    temperature = condition["temperature"]
    alt_prompt = condition["alt_prompt"]
    model_short = "DeepSeek" if "deepseek" in model_id else "Grok"

    cells = [(x, s) for x in ROBUST_SD for s in ROBUST_STAKES]
    total = len(cells)
    write_progress(f"START {model_short}/{cond_name}: {total} cells × {N_TARGET} reps")

    all_rows: list[dict] = []
    lock = Lock()
    done = [0]

    def run_and_collect(args: tuple[int, int]) -> int:
        x, stake = args
        try:
            rows = run_cell(x, stake, model_id, api_key, temperature, alt_prompt, cond_name)
        except RuntimeError as e:
            write_progress(f"FATAL: {e}")
            raise
        with lock:
            all_rows.extend(rows)
            done[0] += 1
            if done[0] % 20 == 0 or done[0] == total:
                write_progress(f"  {model_short}/{cond_name}: {done[0]}/{total} cells | {len(all_rows)} rows")
        return len(rows)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_and_collect, c): c for c in cells}
        for f in as_completed(futures):
            try:
                f.result()
            except RuntimeError:
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    write_progress(f"DONE {model_short}/{cond_name}: saved {len(df)} rows to {out_csv.name}")


def main() -> None:
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env or environment.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Robust SD: {ROBUST_SD}")
    print(f"Robust stakes: {ROBUST_STAKES}")
    print(f"N_rep: {N_TARGET}, conditions: {[c['name'] for c in CONDITIONS]}")

    for model_id in [MODEL_DEEPSEEK, MODEL_GROK]:
        model_short = "deepseek" if "deepseek" in model_id else "grok"
        print(f"\n{'='*60}\nRunning robustness for {model_short}")
        for cond in CONDITIONS:
            out_csv = OUT_DIR / f"results_v2_robust_{model_short}_{cond['name']}.csv"
            run_condition_grid(model_id, api_key, cond, out_csv)

    print("\nAll done.")


if __name__ == "__main__":
    main()
