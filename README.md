# Rappi Store Monitor

An interactive analytics dashboard and AI-powered chatbot for exploring the `synthetic_monitoring_visible_stores` metric — the number of Rappi stores that are online and visible to users on the platform at any given moment.

Built with Python, Streamlit, Plotly, and the Google Gemini API.

---

## What it does

The app has two tabs:

**Dashboard** — seven interactive Plotly charts covering the full Feb 1–6 2026 dataset (33,000+ readings at 10-second intervals):

| Chart | What it answers |
|---|---|
| Time series with range selector | Overall availability trend; quick zoom to 1 h / 6 h / 1 d |
| Hour × Day heatmap | Which hours on which days had the most / least stores online |
| Daily bar chart (peak / avg / min) | Day-by-day performance comparison |
| Hourly pattern ± std deviation | Typical intraday shape and how much it varies |
| Anomaly detection | Sudden drops or spikes using a rolling z-score |
| Weekday vs weekend distribution | Histogram overlay comparing the two populations |
| Average by day of week | Which day of the week had the highest availability |

**Asistente IA** — a conversational chatbot backed by Gemini 2.5 Flash that answers questions in Spanish or English using the actual dataset statistics. Responses stream token by token directly into the interface.

---

## Project structure

```
Rappi_Entregable/
├── app.py                        # Main Streamlit application
├── data_loader.py                # CSV ingestion, parsing, feature engineering
├── chatbot.py                    # Gemini API client and streaming logic
├── requirements.txt              # Python dependencies
├── rappi_company_icon.png        # Brand logo
├── AVAILABILITY-data (1).csv     # Raw data files (1–100)
│   ...
├── AVAILABILITY-data (100).csv
└── .streamlit/
    ├── config.toml               # Forces light theme + Rappi orange as primary color
    ├── secrets.toml              # API key — NOT committed (see .gitignore)
    └── secrets.toml.example      # Template showing the required key name
```

---

## How to run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure the API key**

Copy the example secrets file and fill in your Gemini API key:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Then edit `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```
Get a free key at [aistudio.google.com](https://aistudio.google.com).

**3. Launch**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Key technical decisions

### Pure Python web app with Streamlit

Streamlit was chosen because it turns a Python script into a full interactive web app with zero HTML/JS. It provides built-in chat primitives (`st.chat_input`, `st.write_stream`) that made integrating the streaming chatbot trivial. A Flask or FastAPI approach would have required a separate frontend.

### Dynamic CSV loading — no pre-merging

The 100 CSV files are loaded and parsed at runtime by `data_loader.load_all_data()` instead of merging them into a single file beforehand. This keeps the repository clean (raw data stays raw) and makes it easy to add more files later without any pre-processing step. `@st.cache_data` ensures this only runs once per session.

### Vectorized timestamp parsing

Each CSV column header contains a full JavaScript `Date.toString()` string like:
```
Sun Feb 01 2026 06:59:40 GMT-0500 (hora estándar de Colombia)
```
We extract the parseable portion with a single vectorized regex call (`Series.str.extract`) and parse with `pd.to_datetime` using an explicit format string. This is approximately 50× faster than calling `dateutil.parser.parse` row by row across 33,000 readings.

### Rolling z-score for anomaly detection

Anomalies are flagged when a reading deviates more than N standard deviations from its **local** 5-minute rolling mean, not the global mean. This means a normal night-time low doesn't trigger alerts just because it's below the daily average — only genuinely sudden changes are flagged. The user controls the threshold via a sidebar slider.

### Gemini 2.5 Flash

`gemini-2.5-flash` was selected over `gemini-2.0-flash` because the 2.0 model had `limit: 0` on the free tier at the time of development, causing immediate quota errors. The `google-genai >= 1.0.0` SDK (not the deprecated `google-generativeai` package) is required for this model.

### Seeded chat history

The dataset summary (~3 KB of structured text) is injected once as the very first exchange in the API history (`role: user` → `role: model`) rather than prepended to every user message. This means the model has full data context throughout the entire conversation without re-sending the summary on every turn, reducing latency and token cost.

### Secrets management

The Gemini API key is stored in `.streamlit/secrets.toml`, which is excluded from version control via `.gitignore`. Streamlit loads this file automatically — the app reads it with `st.secrets.get("GEMINI_API_KEY")` and, if found, shows a green "IA Conectada" badge in the sidebar instead of exposing an input field. If the key is not configured, the input field appears as a fallback.

### UI/UX design

The interface uses Rappi's brand palette (orange `#FF441F`, dark navy `#1A1A2E`, white) with a set of CSS overrides injected via `st.markdown`:

- **Dark navy sidebar** with an orange right-border stripe separating it from the main content
- **Pill-style tab switcher** — a rounded gray container where the active tab turns white with orange text
- **KPI cards** with an orange top border and a hover lift effect
- **Orange badge section labels** to visually group chart sections
- **Outline buttons** that fill with orange on hover
- A single `chart_style()` helper function ensures all seven Plotly charts share identical typography, grid colors, and margins

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit >= 1.30` | Web framework, UI components, chat primitives |
| `plotly >= 5.18` | Interactive charts |
| `pandas >= 2.0` | Data loading, resampling, aggregation |
| `numpy >= 1.24` | Z-score computation |
| `Pillow >= 9.0` | Logo image handling for page icon |
| `google-genai >= 1.0` | Gemini API client (new SDK, not deprecated generativeai) |
| `python-dateutil >= 2.8` | Date utilities |