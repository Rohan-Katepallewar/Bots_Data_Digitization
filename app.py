import os, io, json, base64, time, re
from typing import Any, Dict, List, Tuple

import streamlit as st
import pandas as pd
from PIL import Image

# Optional Sheets
try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_SHEETS = True
except Exception:
    HAS_SHEETS = False

# ---------- App config ----------
st.set_page_config(page_title="English Form Digitizer", page_icon="📝", layout="centered")
st.title("📝 English Form Digitizer (JPEG → JSON/CSV/Google Sheets)")

st.markdown("""
- Upload up to **10 JPEGs** of the fixed English tutoring form.
- Extracts **header** (Teacher, School), each student (name, caregiver, phone, grade), **baseline/endline ops**, **3 lessons**, and the **footer totals**.
- Processes **sequentially**; shows per-file success/failure.
""")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", help="Only stored in this session.")
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=0)

    # Defaults to your sheet + tab
    default_sheet = "16fnfCvrUBysEXtf_OAWLW1aVQSAegqtxehxwkWF3q_0"
    spreadsheet_id_input = st.text_input("Spreadsheet ID or URL", value=default_sheet)
    worksheet_name = st.text_input("Worksheet/Tab for rows", value="sheet1")

    write_rows = st.checkbox("Append student-session rows to Google Sheet", value=False)
    write_footer = st.checkbox("Append footer totals to 'summary' tab", value=False,
                               help="Creates 'summary' tab if not present")

uploaded = st.file_uploader("Upload JPEG files (max 10)", type=["jpg","jpeg"], accept_multiple_files=True)

# ---------- Prompt ----------
EXTRACTION_PROMPT = """
You are an OCR + information extraction agent for English handwriting on a fixed whiteboard-style tutoring form (the layout does not change).

Return ONLY a single JSON object with these keys exactly:
{
  "teacherName": "", "schoolName": "",
  "students": [
    {
      "name": "", "caregiverName": "", "phone": "", "grade": "",
      "baselineOp": "",
      "sessions": [
        {"session":"Lesson 1","datetime":"","currentTopic":"","checkpointCorrect":"","nextTopic":""},
        {"session":"Lesson 2","datetime":"","currentTopic":"","checkpointCorrect":"","nextTopic":""},
        {"session":"Lesson 3","datetime":"","currentTopic":"","checkpointCorrect":"","nextTopic":""}
      ],
      "endlineOp": ""
    }
  ],
  "footer": {
    "lesson1":{"totalStudents":"","successfullyReached":""},
    "lesson2":{"totalStudents":"","successfullyReached":""},
    "lesson3":{"totalStudents":"","successfullyReached":""}
  }
}

Rules:
- Teacher/School: read from the header area.
- Extract EACH visible student block: name, caregiver (if present), phone (digits-only), grade.
- BaselineOp & EndlineOp: choose the option that is CIRCLED or TICK-MARKED (ignore adjacent handwriting).
  Valid values ONLY: "None", "Addition", "Subtraction", "Multiplication", "Division". If nothing clear, use "".
- Sessions (3 rows): For each row:
  - session: "Lesson 1" .. "Lesson 3" (row order).
  - datetime: "DD/MM/YY HH:mm" (24h). If only date or time is present, include the available part.
  - currentTopic: copy what’s written (e.g., D or M).
  - checkpointCorrect: "Yes" if explicitly marked correct, "No" if marked incorrect, else "".
  - nextTopic: copy what’s written.
- Phones: digits only (strip spaces, dashes, country codes). No fixed length.
- Footer table: extract the numbers for Lesson 1–3 (total students, successfully reached).
- If a field is empty or illegible, output "".
- Output must be VALID JSON and contain ONLY the JSON object.
"""

# ---------- Helpers ----------
def sanitize_spreadsheet_id(text: str) -> str:
    # Accept raw ID or full URL; extract between /d/ and /edit if present
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", text)
    return m.group(1) if m else text.split("?")[0].split("#")[0].strip()

def make_client(key: str):
    from openai import OpenAI
    return OpenAI(api_key=key)

def to_data_url_jpeg(b: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")

def ensure_jpeg(file) -> bytes:
    img = Image.open(file).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

def call_vision_json(client, model: str, prompt: str, jpeg_bytes: bytes) -> str:
    data_url = to_data_url_jpeg(jpeg_bytes)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role":"system","content":"You are a precise JSON-only extractor. Output valid JSON only."},
            {"role":"user","content":[
                {"type":"text","text": prompt},
                {"type":"image_url","image_url":{"url": data_url}}
            ]}
        ]
    )
    return resp.choices[0].message.content

def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    if "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}")+1]
    return json.loads(text)

ROW_COLUMNS = [
    "fileName","teacherName","schoolName",
    "studentName","caregiverName","phone","grade",
    "baselineOp","session","datetime","currentTopic","checkpointCorrect","nextTopic","endlineOp"
]

def flatten_rows(obj: Dict[str, Any], file_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    teacher = obj.get("teacherName","") or ""
    school  = obj.get("schoolName","") or ""
    students = obj.get("students",[]) or []
    for s in students:
        base = {
            "fileName": file_name,
            "teacherName": teacher,
            "schoolName": school,
            "studentName": s.get("name","") or "",
            "caregiverName": s.get("caregiverName","") or "",
            "phone": "".join(ch for ch in (s.get("phone","") or "") if ch.isdigit()),
            "grade": s.get("grade","") or "",
            "baselineOp": s.get("baselineOp","") or "",
            "endlineOp": s.get("endlineOp","") or "",
        }
        sessions = s.get("sessions",[]) or []
        wrote = False
        for idx, sess in enumerate(sessions, start=1):
            if not any((sess.get("datetime"), sess.get("currentTopic"), sess.get("checkpointCorrect"), sess.get("nextTopic"))):
                continue
            row = {**base,
                   "session": f"Lesson {idx}",
                   "datetime": sess.get("datetime","") or "",
                   "currentTopic": sess.get("currentTopic","") or "",
                   "checkpointCorrect": sess.get("checkpointCorrect","") or "",
                   "nextTopic": sess.get("nextTopic","") or ""}
            rows.append(row)
            wrote = True
        if not wrote:
            row = {**base,"session":"","datetime":"","currentTopic":"","checkpointCorrect":"","nextTopic":""}
            rows.append(row)
    # order columns
    for r in rows:
        for col in ROW_COLUMNS:
            r.setdefault(col,"")
    return rows

def footer_to_df(obj: Dict[str, Any], file_name: str) -> pd.DataFrame:
    f = obj.get("footer",{}) or {}
    records = []
    for i, key in enumerate(["lesson1","lesson2","lesson3"], start=1):
        v = f.get(key, {}) or {}
        records.append({
            "fileName": file_name,
            "lesson": f"Lesson {i}",
            "totalStudents": str(v.get("totalStudents","") or ""),
            "successfullyReached": str(v.get("successfullyReached","") or "")
        })
    return pd.DataFrame(records)

def get_sheets_client():
    if not HAS_SHEETS:
        raise RuntimeError("gspread not installed (see requirements).")
    if "gcp_service_account" not in st.secrets:
        raise RuntimeError("Service account JSON missing in secrets as 'gcp_service_account'.")
    info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"],
    )
    return gspread.authorize(creds)

def append_df(ws, df: pd.DataFrame):
    values = df.astype(str).fillna("").values.tolist()
    if ws.row_count < len(values) + 1:
        ws.add_rows(len(values) + 10)
    ws.append_rows(values, value_input_option="USER_ENTERED")

def write_rows_to_sheets(df_rows: pd.DataFrame, spreadsheet_id: str, tab: str):
    gc = get_sheets_client()
    sh = gc.open_by_key(spreadsheet_id)
    try:
        ws = sh.worksheet(tab)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab, rows=1000, cols=len(df_rows.columns)+2)
        ws.append_row(df_rows.columns.tolist())
    append_df(ws, df_rows)

def write_footer_to_summary(df_footer: pd.DataFrame, spreadsheet_id: str):
    gc = get_sheets_client()
    sh = gc.open_by_key(spreadsheet_id)
    tab = "summary"
    cols = ["fileName","lesson","totalStudents","successfullyReached"]
    try:
        ws = sh.worksheet(tab)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab, rows=1000, cols=len(cols)+2)
        ws.append_row(cols)
    # ensure columns order
    if list(df_footer.columns) != cols:
        df_footer = df_footer[cols]
    append_df(ws, df_footer)

# ---------- Main ----------
if uploaded:
    if len(uploaded) > 10:
        st.error("Please upload 10 or fewer JPEG files.")
        st.stop()
    if not api_key:
        st.warning("Enter your OpenAI API key to start.")
        st.stop()

    client = make_client(api_key)
    prog = st.progress(0)
    status = st.empty()

    all_rows: List[Dict[str, Any]] = []
    all_footer: List[pd.DataFrame] = []
    failures: List[str] = []

    for i, f in enumerate(uploaded, start=1):
        status.info(f"Processing {f.name} ({i}/{len(uploaded)}) ...")
        try:
            jpeg = ensure_jpeg(f)
            raw = call_vision_json(client, model=model, prompt=EXTRACTION_PROMPT, jpeg_bytes=jpeg)
            obj = safe_json_loads(raw)
            rows = flatten_rows(obj, f.name)
            all_rows.extend(rows)
            all_footer.append(footer_to_df(obj, f.name))
        except Exception as e:
            failures.append(f"{f.name}: {str(e)[:200]}")
        finally:
            prog.progress(int(i * 100 / len(uploaded)))
            time.sleep(0.05)

    status.empty()

    if failures:
        st.error("Some files failed to digitize:")
        st.write("\n".join(failures))

    if all_rows:
        df_rows = pd.DataFrame(all_rows)[ROW_COLUMNS]
        st.success(f"Digitized {len(df_rows)} row(s) from {len(uploaded) - len(failures)} file(s).")
        st.dataframe(df_rows, use_container_width=True)

        st.download_button("Download CSV (rows)", df_rows.to_csv(index=False).encode("utf-8"),
                           "digitized_rows.csv", "text/csv")

        # Footer
        if all_footer:
            df_footer = pd.concat(all_footer, ignore_index=True)
            with st.expander("Footer totals (from bottom table)"):
                st.dataframe(df_footer, use_container_width=True)
            st.download_button("Download CSV (footer summary)",
                               df_footer.to_csv(index=False).encode("utf-8"),
                               "footer_summary.csv", "text/csv")
        else:
            df_footer = pd.DataFrame(columns=["fileName","lesson","totalStudents","successfullyReached"])

        # Sheets writing
        if write_rows or write_footer:
            try:
                sid = sanitize_spreadsheet_id(spreadsheet_id_input)
                if write_rows:
                    write_rows_to_sheets(df_rows, spreadsheet_id=sid, tab=worksheet_name)
                    st.success(f"Appended {len(df_rows)} rows to '{worksheet_name}'.")
                if write_footer and not df_footer.empty:
                    write_footer_to_summary(df_footer, spreadsheet_id=sid)
                    st.success("Appended footer totals to 'summary' tab.")
            except Exception as e:
                st.error(f"Google Sheets append failed: {e}")
    else:
        st.warning("No rows extracted. Check image quality and ensure the form layout matches the sample.")
else:
    st.info("Upload JPEG files to begin.")
