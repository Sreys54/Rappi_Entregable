# Rappi Store Monitor

An interactive analytics dashboard and AI-powered chatbot for exploring `synthetic_monitoring_visible_stores` — the count of Rappi stores that are online and visible to users on the platform at every moment.

Built with **Python · Streamlit · Plotly · Google Gemini 2.5 Flash**.

---

## Table of contents

1. [What it does](#what-it-does)
2. [AI strategy](#ai-strategy)
3. [UX and creativity](#ux-and-creativity)
4. [Code quality](#code-quality)
5. [Project structure](#project-structure)
6. [How to run](#how-to-run)
7. [Technical decisions](#technical-decisions)
8. [Dependencies](#dependencies)

---

## What it does

The app has two tabs that work together:

### Dashboard tab
Seven interactive Plotly charts built on 33,000+ readings at 10-second intervals (Feb 1–6, 2026):

| Chart | Business question answered |
|---|---|
| Time series + range selector | What is the overall availability trend? (zoom to 1 h / 6 h / 1 d) |
| Hour × Day heatmap | Which hours on which specific days had the most / fewest stores online? |
| Daily statistics (peak / avg / min) | How did each day perform overall? |
| Hourly pattern ± std deviation | What is the typical intraday shape and how much does it vary? |
| Anomaly detection | When did sudden drops or spikes occur? |
| Weekday vs weekend distribution | Is availability structurally different on weekends? |
| Average by day of week | Which day of the week has the best store availability? |

All charts respond to sidebar filters: date range, hour of day, and time granularity (10 s → 1 h). Changes propagate instantly to every chart simultaneously.

### Asistente IA tab
A conversational chatbot powered by Gemini 2.5 Flash that answers natural-language questions about the dataset in Spanish or English. Responses stream token by token into the interface. The chatbot knows the full dataset statistics (peaks, minimums, patterns, anomalies) and cites specific numbers rather than generic answers.

---

## AI strategy

This section covers both how AI was integrated into the product and how AI tools were used strategically during development.

### AI in the product: designed for accuracy, not just presence

A naive chatbot approach would send raw data to the model or re-describe the dataset in every message. Instead, we designed a two-layer strategy:

**Layer 1 — Structured data summary as ground truth**

Before the first conversation turn, `data_loader.compute_summary()` generates a ~3 KB structured text document containing:
- Global statistics (peak, minimum, average, coefficient of variation)
- Hourly patterns across all days
- Per-day breakdown (avg, max, min)
- Day-of-week averages
- Top 5 sudden drops and spikes (>5% change in 10 seconds)

This is the only "data" the model ever sees. It cannot hallucinate values it was never given.

**Layer 2 — Seeded conversation history**

The summary is injected once as the very first exchange in the Gemini API history (user turn → model acknowledgment), not prepended to every user message. This means:
- The model has the full context across the entire conversation
- The ~3 KB summary is sent only once (not on every turn), saving latency and tokens
- All subsequent turns are pure question/answer, keeping the API calls lightweight

**System prompt design**

The system prompt instructs the model to:
- Answer concisely with specific figures
- Format large numbers with thousand separators
- Proactively surface interesting patterns when relevant
- Respond in whichever language the user writes in (Spanish or English)
- Explicitly admit when a question cannot be answered from the available data (prevents hallucination)

**Streaming UX**

Responses are streamed token by token using `chat.send_message_stream()` and rendered live with Streamlit's `st.write_stream()`. This gives immediate feedback and makes the chatbot feel responsive even on slower connections.

### AI in development: strategic use of Claude Code

The solution was built iteratively with Claude Code as an active collaborator, not a code generator. Each decision was reasoned through before implementation:

- **Architecture first**: the three-file split (`app.py`, `data_loader.py`, `chatbot.py`) was discussed and justified before writing any code, not added as an afterthought
- **Debugging together**: errors like the `gemini-2.0-flash` quota exhaustion (limit=0 on free tier) and the `google-generativeai` deprecation were diagnosed by analyzing API responses, not by trial and error
- **Iterative design**: the UI/UX went through two full redesigns based on specific feedback, with each change argued on UX grounds (e.g. why dark sidebar + orange divider creates better visual hierarchy than a white sidebar with colored accents)
- **Problem-specific solutions**: the vectorized timestamp regex was chosen after analyzing the actual column header format in the CSV files — it wasn't a generic approach

---

## UX and creativity

Several design choices go beyond a standard Streamlit dashboard:

### Visual design system
A consistent design system was defined upfront with four color tokens (`ORANGE`, `ORANGE_MID`, `ORANGE_PALE`, `DARK`) and a `chart_style()` helper that applies identical typography, grid colors, and margins to all seven Plotly charts. This is why all charts look like they belong together rather than like separate widgets.

### Dark sidebar with orange divider
The sidebar uses Rappi's dark navy (`#1A1A2E`) with a 4 px orange right border that visually separates the control panel from the data area. This creates a natural two-zone layout — controls on the left, content on the right — without relying on Streamlit's default gray sidebar.

### Pill-style tab switcher
The tab bar is styled as a rounded pill container (gray background, white active pill, orange active text) instead of Streamlit's default underline tabs. This matches the visual language of modern analytics products.

### KPI cards with hover interaction
The five KPI cards have an orange top border and lift on hover (`translateY(-3px)` + shadow). The hover is subtle but signals interactivity and adds depth to what would otherwise be flat number boxes.

### Adaptive rolling average window
The rolling average window in the time series chart scales automatically with the number of visible data points (`max(12, n // 40)`), so it smooths approximately 2.5% of the visible range at any granularity. At 10 s granularity it smooths over 4 minutes; at 1 h granularity it smooths over ~1.7 days — always the right amount.

### User-controlled anomaly threshold
The anomaly detection threshold is a sidebar slider, not a hardcoded value. The user can tighten it to see minor fluctuations or loosen it to see only major events. This turns a static analysis into an interactive exploration tool.

### Transparent secrets management
When the API key is configured in `.streamlit/secrets.toml`, the sidebar shows a green "IA Conectada" badge instead of a text input. This avoids exposing the key in the UI and gives users clear feedback that the AI feature is ready.

---

## Code quality

### Separation of concerns

The project is split into three focused modules:

| File | Responsibility |
|---|---|
| `data_loader.py` | All I/O and data transformation. No UI code. |
| `chatbot.py` | All Gemini API communication. No data loading or UI code. |
| `app.py` | Layout, state management, chart rendering. Imports from the other two. |

This means each module can be read, tested, or replaced independently.

### Caching strategy

`@st.cache_data` is applied to both `load_all_data()` and `compute_summary()`. The first call parses 100 CSV files and builds the summary; every subsequent rerun (triggered by filter changes) skips both operations entirely. Charts update instantly because only the filter mask and resample are re-executed.

### No silent failures in data loading

Each CSV file is parsed inside a `try/except` block that skips the file on any error and continues with the rest. If all 100 files fail, the app returns an empty DataFrame with the correct schema and shows a user-friendly warning — it never crashes.

### Type annotations and explicit contracts

Public functions use Python type annotations (`-> pd.DataFrame`, `-> str`, `-> Generator[str, None, None]`). This makes the data flow between modules explicit and allows static analysis tools to catch errors at development time.

### Secrets never in code or git

The API key lives only in `.streamlit/secrets.toml`, which is excluded by `.gitignore`. A `.streamlit/secrets.toml.example` template is committed so collaborators know exactly what to configure. The app reads the key with `st.secrets.get()` and falls back to a manual input field — it never hardcodes credentials.

---

## Project structure

```
Rappi_Entregable/
├── app.py                        # Main Streamlit application (UI, charts, state)
├── data_loader.py                # CSV ingestion, parsing, feature engineering
├── chatbot.py                    # Gemini API client and streaming response logic
├── requirements.txt              # Python dependencies
├── rappi_company_icon.png        # Brand logo
├── .gitignore                    # Excludes secrets, __pycache__, IDE files
├── AVAILABILITY-data (1).csv     # Raw monitoring data files (1–100)
│   ...
├── AVAILABILITY-data (100).csv
└── .streamlit/
    ├── config.toml               # Forces light theme + Rappi orange as primary color
    ├── secrets.toml              # API key — NOT committed (excluded by .gitignore)
    └── secrets.toml.example      # Template showing required key name
```

---

## How to run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure the API key**

Copy the example and add your Gemini key (free at [aistudio.google.com](https://aistudio.google.com)):
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Edit `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```

**3. Launch**
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Data loads and caches on the first run; subsequent filter changes respond instantly.

---

## Technical decisions

### Vectorized timestamp parsing (~50× faster than row-by-row)

Each CSV column header is a JavaScript `Date.toString()` string:
```
Sun Feb 01 2026 06:59:40 GMT-0500 (hora estándar de Colombia)
```
We extract the parseable portion with one vectorized `Series.str.extract()` call and parse with `pd.to_datetime` using an explicit format string. Applying `dateutil.parser.parse` row-by-row across 33,000 readings would take several seconds; the vectorized approach is near-instant.

### Dynamic CSV loading — no pre-merging

The 100 files are loaded at runtime sorted by their numeric suffix (not alphabetically — `"data (2)"` sorts before `"data (10)"` alphabetically, which would scramble the timeline). `@st.cache_data` ensures this only runs once per session.

### Local rolling z-score for anomaly detection

Anomalies are flagged when a reading deviates more than N standard deviations from its **local** 5-minute rolling mean, not the global daily mean. This prevents normal overnight lows from being flagged as anomalies — only sudden changes relative to the immediate context are detected.

### Gemini 2.5 Flash on the free tier

`gemini-2.5-flash` was selected because `gemini-2.0-flash` showed `limit: 0` on the free tier, causing immediate 429 quota errors. The `google-genai >= 1.0.0` SDK is required (the older `google-generativeai` package was deprecated and returned 404 for current model IDs).

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | >= 1.30 | Web framework, UI components, chat primitives, secrets |
| `plotly` | >= 5.18 | Interactive charts |
| `pandas` | >= 2.0 | CSV loading, resampling, pivot tables, aggregation |
| `numpy` | >= 1.24 | Z-score computation |
| `Pillow` | >= 9.0 | Logo image for browser tab icon |
| `google-genai` | >= 1.0 | Gemini API client (new SDK — not deprecated `generativeai`) |
| `python-dateutil` | >= 2.8 | Date arithmetic utilities |
