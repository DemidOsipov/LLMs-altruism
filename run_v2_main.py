"""
Experiment v2 — main grid: both DeepSeek V3.2 and Grok 4.1 Fast.

Changes from v1:
  - Social distance: [1, 2, 4, 6, ..., 100]  (51 levels, step 2 — 2x denser)
  - Stakes: 31 log-spaced values $1–$250,000  (uniform on log scale)
  - N_rep: 5 (up from 3)

Output:
  out/results_v2_deepseek.csv
  out/results_v2_grok.csv

Estimated cost/time: ~$1–2, ~60–90 min (two models run sequentially,
6 threads each).
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

# 51 SD values: step=2, includes SD=1,2,4,...,100
# Includes key comparison points: 1, 2, 20, 100
SOCIAL_DISTANCES = [1] + list(range(2, 102, 2))

# 31 log-spaced stakes from $1 to $250,000 (includes $10 for human comparisons)
# Generated via np.logspace(0, log10(250000), 31), rounded, deduplicated, $10 added
STAKES = [
    1, 2, 3, 5, 8, 10, 12, 18, 27, 41, 63, 95, 144, 217, 328, 496,
    751, 1135, 1717, 2596, 3926, 5938, 8979, 13578, 20531, 31054,
    46962, 71000, 107392, 162379, 250000
]

N_TARGET = 3
MAX_ATTEMPTS = 8
MAX_WORKERS = 6  # per model

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

_SHARE_RE = re.compile(
    r"\bi\s*share\s*([$€£]?\s*[0-9][0-9,]*(?:\.[0-9]+)?)\s*dollar(?:s)?\b",
    re.IGNORECASE,
)


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


def run_cell(x: int, stake: int, model_id: str, api_key: str) -> list[dict]:
    prompt = PROMPT_TMPL.format(m=f"${stake:,}", x=x)
    rows: list[dict] = []
    rep = 0
    attempts = 0

    while rep < N_TARGET and attempts < MAX_ATTEMPTS:
        attempts += 1
        try:
            text, latency = api_call(api_key, model_id, prompt)
            share = parse_share(text)
        except RuntimeError:
            raise
        except Exception:
            text, share, latency = "", None, float("nan")

        if share is not None:
            rep += 1
            share_pct = float(share) / float(stake)
            rows.append({
                "model": model_id,
                "game": GAME,
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
    """Append timestamped line to progress file (visible externally)."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)


def run_model_grid(model_id: str, api_key: str, out_csv: Path) -> None:
    if out_csv.exists():
        write_progress(f"{out_csv.name} already exists — skipping.")
        return

    conditions = [(x, s) for x in SOCIAL_DISTANCES for s in STAKES]
    total = len(conditions)
    model_short = "DeepSeek" if "deepseek" in model_id else "Grok"

    # Resume from partial CSV if it exists
    partial_csv = out_csv.with_suffix(".partial.csv")
    csv_lock = Lock()
    all_rows: list[dict] = []
    done_cells: set[tuple[int, int]] = set()

    if partial_csv.exists():
        try:
            df_partial = pd.read_csv(partial_csv)
            # Only resume cells with full N_TARGET reps
            cell_counts = df_partial.groupby(["social_distance", "money"]).size()
            complete = cell_counts[cell_counts >= N_TARGET].index
            done_cells = {(int(sd), int(m)) for sd, m in complete}
            all_rows = df_partial[
                df_partial.apply(lambda r: (int(r.social_distance), int(r.money)) in done_cells, axis=1)
            ].to_dict("records")
            write_progress(f"RESUME {model_short}: {len(done_cells)}/{total} cells already complete")
        except Exception as e:
            write_progress(f"WARNING: could not load partial CSV ({e}), starting fresh")
            partial_csv.unlink(missing_ok=True)

    conditions = [(x, s) for x in SOCIAL_DISTANCES for s in STAKES if (x, s) not in done_cells]
    remaining = len(conditions)
    write_progress(f"START {model_short}: {remaining} remaining cells × {N_TARGET} reps, {MAX_WORKERS} threads")

    header_written = [partial_csv.exists() and partial_csv.stat().st_size > 0]

    lock = Lock()
    done = [0]
    errors = [0]

    def run_and_collect(args: tuple[int, int]) -> int:
        x, stake = args
        try:
            rows = run_cell(x, stake, model_id, api_key)
        except RuntimeError as e:
            msg = str(e)
            write_progress(f"FATAL: {msg}")
            raise
        with lock:
            all_rows.extend(rows)
            if not rows:
                errors[0] += 1
            done[0] += 1
            n_done = done[0]
            n_rows = len(all_rows)
        # Append rows to partial CSV incrementally
        if rows:
            with csv_lock:
                df_chunk = pd.DataFrame(rows)
                df_chunk.to_csv(
                    partial_csv,
                    mode="a",
                    index=False,
                    header=not header_written[0],
                )
                header_written[0] = True
        # Log progress every 50 cells or at end
        if n_done % 50 == 0 or n_done == remaining:
            write_progress(
                f"{model_short}: {n_done}/{remaining} new cells | {n_rows} rows | {errors[0]} empty cells"
            )
        return len(rows)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_and_collect, c): c for c in conditions}
        for f in as_completed(futures):
            try:
                f.result()
            except RuntimeError:
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    # all_rows already contains both resumed rows and newly collected rows
    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    if partial_csv.exists():
        partial_csv.unlink()
    write_progress(f"DONE {model_short}: saved {len(df)} rows to {out_csv.name}")
    by_sd = df.groupby("social_distance")["share_pct"].agg(["mean", "std"])
    print(by_sd.to_string(float_format="{:.3f}".format))


def main() -> None:
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env or environment.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"SD grid ({len(SOCIAL_DISTANCES)} values): {SOCIAL_DISTANCES[:5]}...{SOCIAL_DISTANCES[-3:]}")
    print(f"Stake grid ({len(STAKES)} values): ${STAKES[0]}–${STAKES[-1]:,}")
    print(f"N_rep: {N_TARGET}")

    run_model_grid(MODEL_DEEPSEEK, api_key, OUT_DIR / "results_v2_deepseek.csv")
    run_model_grid(MODEL_GROK, api_key, OUT_DIR / "results_v2_grok.csv")
    print("\nAll done.")


if __name__ == "__main__":
    main()
