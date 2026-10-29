import os, time, re, math
from datetime import datetime
import requests
import pandas as pd
from dotenv import load_dotenv
from utils.decisions import DECISION_SPECS, decision_block
import time
import csv
from openai import OpenAI

# ==================== ENV ====================
load_dotenv()
API_URL_ENV = os.getenv("API_URL", "")
API_KEY_ENV = os.getenv("API_KEY", "")

LOG_PATH = "../results/logs/log.txt"
CSV_PATH = "../results/results.csv"

# ==================== HELPERS ====================
def _as_str(x) -> str:
    if x is None: return ""
    if isinstance(x, float):
        try:
            if math.isnan(x): return ""
        except Exception:
            pass
    return str(x)

def ensure_dirs():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

def log_response(project_id: str, entry_id: str, model: str, temperature: float):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp","project_id","entry_id","model","temperature"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), project_id, entry_id, model, temperature])

def save_csv_row(entry_id, decision_value, reason, response_time, model, decision_scheme, examples_count, project_id=""):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    row = {
        "project_id": project_id,
        "id": entry_id,
        "llm_decision": decision_value,
        "reason": reason,
        "response_time_sec": round(response_time, 3),
        "model": model,
        "decision_scheme": decision_scheme,
        "examples_count": (examples_count if examples_count is not None else "")
    }
    df = pd.DataFrame([row])
    if not os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, index=False)
    else:
        df.to_csv(CSV_PATH, mode="a", header=False, index=False)

# ---------- optional codebook preview ----------
def read_codebook_csv(codebook_csv_path: str) -> str:
    if not codebook_csv_path or not os.path.exists(codebook_csv_path):
        return ""
    try:
        df = pd.read_csv(codebook_csv_path, keep_default_na=False)
        df_small = df.iloc[:10, : min(6, df.shape[1])]
        head = ["| " + " | ".join(map(str, df_small.columns)) + " |",
                "| " + " | ".join(["---"] * df_small.shape[1]) + " |"]
        body = []
        for _, row in df_small.iterrows():
            body.append("| " + " | ".join(map(lambda x: str(x)[:200], row.astype(str))) + " |")
        return "CODEBOOK (excerpt):\n" + "\n".join(head + body)
    except Exception as e:
        return f"CODEBOOK (could not read): {e}"

import csv
from pandas.errors import ParserError

def _read_csv_flexible(path: str):
    if not path or not os.path.exists(path):
        return None
    seps = [";", ",", "\t", "|"]
    encs = ["utf-8-sig", "utf-8"]
    last_err = None
    for enc in encs:
        for sep in seps:
            try:
                return pd.read_csv(
                    path,
                    sep=sep,
                    engine="python",
                    encoding=enc,
                    quoting=csv.QUOTE_MINIMAL,
                    keep_default_na=False,
                    on_bad_lines="error"  # ggf. "skip"
                )
            except Exception as e:
                last_err = e
                continue
    # Fallback: auto-sep
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig",
                           keep_default_na=False, on_bad_lines="error")
    except Exception as e:
        last_err = e
        raise last_err

def read_examples_csv(path: str, max_rows: int = None):
    if not path or not os.path.exists(path):
        return "", None

    df = _read_csv_flexible(path)
    if df is None or df.empty:
        return "", 0

    total_count = len(df)

    if max_rows is not None:
        df = df.head(max_rows)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep    = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    body   = ["| " + " | ".join(row.astype(str).tolist()) + " |" for _, row in df.iterrows()]
    table  = "EXAMPLES:\n" + "\n".join([header, sep] + body)

    return table, total_count



# ---------- parsing----------
def extract_decision(raw_text: str) -> str:
    m = re.search(r"\bDecision:\s*([A-Za-z0-9]+)", raw_text, re.IGNORECASE)
    if m: return m.group(1).strip()
    m = re.search(r"\bCategory:\s*([A-Za-z0-9]+)", raw_text, re.IGNORECASE)  # fallback
    return m.group(1).strip() if m else ""

def extract_reason(raw_text: str) -> str:
    m = re.search(r"\bReason:\s*(.*?)(?:\n[A-Z][A-Za-z ]*:\s*|$)", raw_text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""

def validate_decision_exact(value: str, scheme: str) -> bool:
    if not isinstance(value, str) or scheme not in DECISION_SPECS:
        return False
    allowed = DECISION_SPECS[scheme]["allowed"]
    # digits (likert) vs strings
    if value.isdigit():
        return value in allowed
    return value.lower() in {a.lower() for a in allowed}

# ---------- API call with retries ----------
def _build_headers(api_key: str, api_url: str):
    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # OpenAI optional: Organization-Header
    if "api.openai.com" in api_url:
        org = os.getenv("OPENAI_ORG", "").strip()
        if org:
            h["OpenAI-Organization"] = org
    return h

def _post_with_retry(url, json, headers, timeout=120, retries=3, backoff=1.5):
    for i in range(retries):
        r = requests.post(url, json=json, headers=headers, timeout=timeout)
        if r.status_code not in (429, 500, 502, 503, 504):
            return r
        # simple backoff
        time.sleep(backoff ** (i+1))
    return r

# ---------- prompt loading (uses decision_block from decisions.py) ----------
# ==================== CACHES ====================
SYSTEM_PROMPT = None
USER_TEMPLATE = None
CODEBOOK_TEXT_CACHED = None
EXAMPLES_TABLE_CACHED = None
EXAMPLES_COUNT_CACHED = None

def init_prompts(system_prompt_path: str, user_prompt_path: str):
    """Einmalig System- und User-Prompt laden und cachen."""
    global SYSTEM_PROMPT, USER_TEMPLATE
    if SYSTEM_PROMPT is None:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            SYSTEM_PROMPT = f.read()
    if USER_TEMPLATE is None:
        with open(user_prompt_path, "r", encoding="utf-8") as f:
            USER_TEMPLATE = f.read()

def init_context(codebook_csv: str = "", examples_csv: str = "", max_example_rows: int = 25):
    """Einmalig Codebook/Examples lesen, kürzen und cachen."""
    global CODEBOOK_TEXT_CACHED, EXAMPLES_TABLE_CACHED, EXAMPLES_COUNT_CACHED
    if CODEBOOK_TEXT_CACHED is None:
        CODEBOOK_TEXT_CACHED = read_codebook_csv(codebook_csv)  # dein read_codebook_csv zeigt ohnehin nur einen Auszug
    if EXAMPLES_TABLE_CACHED is None:
        table, cnt = read_examples_csv(examples_csv, max_rows=max_example_rows)
        print(f"[examples] total_rows={cnt} used_rows={min(cnt or 0, max_example_rows)}")
        if table:
            lines = table.splitlines()
            header = lines[:2]          # Kopf + Separator
            body   = lines[2:2+max_example_rows]  # erste N Beispiele
            table  = "\n".join(header + body)
        EXAMPLES_TABLE_CACHED = table or ""
        EXAMPLES_COUNT_CACHED = cnt

def load_prompt_from_strings(system_prompt, user_template, codebook_text, *,
                             title="", abstract="", decision_block_text="",
                             examples_text="", topic="", domain=""):
    user_prompt = (
        _as_str(user_template)
        .replace("{{codebook}}", _as_str(codebook_text))
        .replace("{{examples}}", _as_str(examples_text))
        .replace("{{decision}}", _as_str(decision_block_text))
        .replace("{{topic}}", _as_str(topic))
        .replace("{{domain}}", _as_str(domain))
        .replace("{{title}}", _as_str(title))
        .replace("{{abstract}}", _as_str(abstract))
    )
    return system_prompt, user_prompt


# ==================== MAIN ====================
def classify_one_entry(
    entry: dict,
    system_prompt_path: str,
    user_prompt_path: str,
    *,
    model: str = "gpt-5",
    temperature: float = 0.0,
    decision: str = "binary",
    codebook_csv: str = "",
    examples_csv: str = "",
    project_id: str = "",
    topic: str = "",
    domain: str = "",
    api_url: str = None,
    api_key: str = None,
    prompt_version: str = "v1"
):
    """
    Klassifiziert einen Eintrag (title/abstract) anhand der vorgegebenen Decision-Spec.
    Nutzt gecachte Prompts/Context (init_prompts / init_context), um I/O im Hot Path zu vermeiden.
    """
    # --- Decision validieren ---
    scheme = decision.lower().strip()
    assert scheme in DECISION_SPECS, f"decision must be one of: {list(DECISION_SPECS.keys())}"

    # --- Einmalige Initialisierung (idempotent) ---
    init_prompts(system_prompt_path, user_prompt_path)
    init_context(codebook_csv, examples_csv, max_example_rows=10)

    # --- API-Konfiguration ---
    api_url = (api_url or API_URL_ENV).rstrip("/")
    api_key = api_key or API_KEY_ENV
    assert api_key, "Missing API_KEY."
    chat_url = f"{api_url}/chat/completions"
    headers = _build_headers(api_key, api_url)

    # --- Prompt-Inhalte rendern ---
    decision_block_text = decision_block(scheme)
    title = entry.get("title", "")
    abstract = entry.get("abstract", "")

    system_prompt, user_prompt = load_prompt_from_strings(
        SYSTEM_PROMPT,                   # aus init_prompts gecacht
        USER_TEMPLATE,                   # aus init_prompts gecacht
        codebook_text=CODEBOOK_TEXT_CACHED,   # aus init_context gecacht
        title=title,
        abstract=abstract,
        decision_block_text=decision_block_text,
        examples_text=EXAMPLES_TABLE_CACHED,  # aus init_context gecacht (gekappt)
        topic=topic,
        domain=domain
    )

    # --- API-Call ---
    t0 = time.perf_counter()
    try:
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 128  # schlank: reicht für Decision + Reason
        }

        # Kürzeres Timeout; _post_with_retry nutzt Exponential Backoff
        resp = _post_with_retry(chat_url, json=payload, headers=headers, timeout=45)
        dt = time.perf_counter() - t0

        if resp.ok:
            j = resp.json()
            content = j.get("choices", [{}])[0].get("message", {}).get("content", "")
            result = (content or "").strip()

            dec_value = extract_decision(result)
            reason = extract_reason(result)

            # Validierung gegen DECISION_SPECS
            if not validate_decision_exact(dec_value, scheme):
                dec_value = ""  # mark invalid

            # Logging & Persistenz
            log_response(
                project_id=project_id,
                entry_id=str(entry.get("id", "NA")),
                model=model,
                temperature=temperature
            )
            save_csv_row(
                entry.get("id", "NA"),
                dec_value,
                reason,
                dt,
                model,
                scheme,
                EXAMPLES_COUNT_CACHED,
                project_id=project_id
            )

            return {
                "project_id": project_id,
                "topic": topic,
                "domain": domain,
                "id": entry.get("id", "NA"),
                "model": model,
                "llm_decision": dec_value,
                "reason": reason,
                "response_time_sec": round(dt, 3),
                "decision_scheme": scheme,
                "examples_count": EXAMPLES_COUNT_CACHED if EXAMPLES_COUNT_CACHED is not None else ""
            }
        else:
            # Fehlerfall (HTTP != 2xx)
            save_csv_row(
                entry.get("id", "NA"),
                "ERROR",
                "",
                dt,
                model,
                scheme,
                EXAMPLES_COUNT_CACHED,
                project_id=project_id
            )
            return {
                "project_id": project_id,
                "topic": topic,
                "domain": domain,
                "id": entry.get("id", "NA"),
                "model": model,
                "llm_decision": "ERROR",
                "reason": resp.text,
                "response_time_sec": round(dt, 3),
                "decision_scheme": scheme,
                "examples_count": EXAMPLES_COUNT_CACHED if EXAMPLES_COUNT_CACHED is not None else ""
            }

    except Exception as e:
        # Unerwartete Exceptions
        dt = time.perf_counter() - t0
        save_csv_row(
            entry.get("id", "NA"),
            "ERROR",
            str(e),
            dt,
            model,
            scheme,
            EXAMPLES_COUNT_CACHED,
            project_id=project_id
        )
        return {
            "project_id": project_id,
            "topic": topic,
            "domain": domain,
            "id": entry.get("id", "NA"),
            "model": model,
            "llm_decision": "ERROR",
            "reason": str(e),
            "response_time_sec": round(dt, 3),
            "decision_scheme": scheme,
            "examples_count": EXAMPLES_COUNT_CACHED if EXAMPLES_COUNT_CACHED is not None else ""
        }
