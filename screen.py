"""
ML-assisted abstract screening pipeline for scoping and systematic reviews.

Reads a CSV of titles and abstracts (e.g., exported from Covidence), sends each
to an OpenAI model using the Abstract ScreenPrompt methodology, and writes
screening decisions to an output CSV.

Methodology based on:
  Cao et al. (2025). "Development of Prompt Templates for Large Language
  Model-Driven Screening in Systematic Reviews." Ann Intern Med.
  doi:10.7326/ANNALS-24-02189

Usage:
  python screen.py --csv abstracts.csv
  python screen.py --csv abstracts.csv --limit 25       # process 25 at a time
  python screen.py --csv abstracts.csv --out_csv out.csv # custom output path
"""

import os
import re
import sys
import time
import argparse

import yaml
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

from prompt_templates import build_abstract_prompt

load_dotenv()

DECISION_LINE = re.compile(r"^\s*(YYY|XXX)\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def extract_reasoning(text: str) -> str:
    """Return only the 'Reasoning:' block, stripping any 'Decision:' lines."""
    if not text:
        return ""
    lines = [ln.rstrip() for ln in text.splitlines()]
    out = []
    in_reasoning = False
    for ln in lines:
        if ln.strip().lower().startswith("decision:"):
            break
        if ln.strip().lower().startswith("reasoning:"):
            in_reasoning = True
            continue
        if in_reasoning:
            out.append(ln)
    # Fallback: if no explicit "Reasoning:" header, take everything before "Decision:"
    if not out:
        out2 = []
        for ln in lines:
            if ln.strip().lower().startswith("decision:"):
                break
            out2.append(ln)
        out = out2
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out).strip()


def parse_last_line_decision(text: str) -> str:
    """Parse the model output for a YYY (include) or XXX (exclude) decision.

    Checks for an explicit terminal YYY/XXX token, then falls back to scanning
    'Decision:' lines, common synonyms, and heuristic signals in the reasoning.

    When the output is empty or ambiguous, defaults to YYY (include) to minimize
    false exclusions, consistent with Cao et al.'s recommendation to prioritize
    sensitivity in abstract screening.
    """
    if not text:
        return "YYY"
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]

    # Direct last-line check
    if lines:
        m = DECISION_LINE.match(lines[-1])
        if m:
            return m.group(1).upper()

    # Scan for explicit decision token anywhere (prefer last occurrence)
    for ln in reversed(lines):
        m = DECISION_LINE.match(ln)
        if m:
            return m.group(1).upper()

    # Look for "Decision:" then next non-empty line
    for i, ln in enumerate(lines):
        if ln.lower().startswith("decision:"):
            for j in range(i + 1, len(lines)):
                nxt = lines[j].strip()
                if not nxt:
                    continue
                m = DECISION_LINE.match(nxt)
                if m:
                    return m.group(1).upper()
                low = nxt.lower()
                if "exclude" in low and "include" not in low:
                    return "XXX"
                if "include" in low and "exclude" not in low:
                    return "YYY"
                break

    # Heuristic fallback from reasoning text
    low = text.lower()
    negative_signals = [
        "should be excluded",
        "therefore, exclude",
        "exclusion applies",
        "fails key population",
        "fails key concept",
        "no patient/public input",
        "purely technical",
        "does not meet inclusion",
    ]
    positive_signals = [
        "include/uncertain",
        "meets inclusion",
        "eligible for full-text",
        "keep for full text",
        "retain for full-text",
    ]
    if any(s in low for s in negative_signals) and not any(
        s in low for s in positive_signals
    ):
        return "XXX"

    # Default to include to avoid false exclusions
    return "YYY"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_record_id(row, idx):
    """Extract a record identifier from common Covidence CSV column names."""
    return (
        row.get("record_id")
        or row.get("Covidence #")
        or row.get("Accession Number")
        or idx + 1
    )


def load_config(path: str) -> dict:
    """Load and validate the YAML configuration file."""
    if not os.path.exists(path):
        sys.exit(f"Error: Config file '{path}' not found. Copy config.yaml and fill in your criteria.")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("review_objectives", "inclusion_criteria", "exclusion_criteria"):
        if not cfg.get(key):
            sys.exit(f"Error: '{key}' is missing or empty in {path}.")
    return cfg


def load_existing_results(path: str) -> set:
    """Load record IDs already processed from a previous run."""
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
        return set(df["record_id"].astype(str))
    except Exception:
        return set()


def append_result(path: str, row: dict, write_header: bool):
    """Append a single result row to the output CSV."""
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", index=False, header=write_header)


def screen_abstract(client, model: str, prompt: str, temperature: float,
                    max_tokens: int, max_retries: int = 3) -> str:
    """Send a screening prompt to the OpenAI API with retry logic."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"\n  API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n  API error after {max_retries} attempts: {e}. Skipping.")
                return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="ML-assisted abstract screening for scoping/systematic reviews."
    )
    ap.add_argument("--csv", required=True,
                    help="Input CSV with Title and Abstract columns (e.g., Covidence export)")
    ap.add_argument("--out_csv", default="decisions.csv",
                    help="Output CSV path (default: decisions.csv)")
    ap.add_argument("--config", default="config.yaml",
                    help="Path to YAML config file (default: config.yaml)")
    ap.add_argument("--model",
                    help="Override the model set in config.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max number of NEW records to process in this run (for batching)")
    args = ap.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    model = args.model or cfg.get("model", "gpt-4o")
    temperature = cfg.get("temperature", 0)
    max_tokens = cfg.get("max_output_tokens", 300)

    # Validate API key
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Error: OPENAI_API_KEY not found. Copy .env.example to .env and add your key.")
    client = OpenAI()

    # Load input CSV
    df = pd.read_csv(args.csv)
    title_col = next((c for c in df.columns if c.lower() == "title"), None)
    abstract_col = next((c for c in df.columns if c.lower() == "abstract"), None)
    if not title_col or not abstract_col:
        sys.exit("Error: CSV must have columns named 'Title' and 'Abstract' (case-insensitive).")
    if len(df) == 0:
        sys.exit("Error: CSV has no data rows.")

    # Check for already-processed records (enables resume)
    done_ids = load_existing_results(args.out_csv)
    write_header = len(done_ids) == 0

    if done_ids:
        print(f"Found {len(done_ids)} already-screened records in {args.out_csv}. Resuming.\n")

    # Determine which records still need screening
    pending = []
    for idx, row in df.iterrows():
        rid = str(get_record_id(row, idx))
        if rid not in done_ids:
            pending.append((idx, row, rid))

    if not pending:
        print("All records have already been screened. Nothing to do.")
        return

    # Apply batch limit
    if args.limit and args.limit < len(pending):
        pending = pending[: args.limit]
        print(f"Batch mode: processing {args.limit} of {len(pending) + (len(df) - len(pending) - len(done_ids))} remaining records.\n")

    # Screen abstracts
    included = 0
    excluded = 0

    for idx, row, rid in tqdm(pending, desc="Screening"):
        title_val = str(row.get(title_col) or "")
        abstract_val = str(row.get(abstract_col) or "")

        prompt = build_abstract_prompt(
            title=title_val,
            abstract=abstract_val,
            review_objectives=cfg["review_objectives"],
            inclusion=cfg["inclusion_criteria"],
            exclusion=cfg["exclusion_criteria"],
        )

        out_text = screen_abstract(client, model, prompt, temperature, max_tokens)
        decision = parse_last_line_decision(out_text)
        rationale = extract_reasoning(out_text)

        result = {
            "record_id": rid,
            "decision": decision,
            "rationale": rationale[:2000],
        }

        append_result(args.out_csv, result, write_header)
        write_header = False

        if decision == "YYY":
            included += 1
        else:
            excluded += 1

    # Summary
    total = included + excluded
    print(f"\nDone. Screened {total} abstracts.")
    print(f"  Included (YYY): {included}")
    print(f"  Excluded (XXX): {excluded}")
    print(f"  Results saved to: {args.out_csv}")

    if done_ids:
        print(f"  Previously screened: {len(done_ids)}")
        print(f"  Total in output file: {len(done_ids) + total}")


if __name__ == "__main__":
    main()
