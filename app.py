import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from chatbot import init_client, stream_response
from data_loader import compute_summary, load_all_data

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
LOGO_PATH = Path(__file__).parent / "rappi_company_icon.png"
logo = Image.open(LOGO_PATH)

st.set_page_config(
    page_title="Rappi Store Monitor",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Brand palette
ORANGE     = "#FF441F"
ORANGE_MID = "#FF8C00"
ORANGE_PALE = "rgba(255,68,31,0.12)"
DARK       = "#1A1A2E"
GRID_COLOR = "#F2F2F2"

# Base64 logo for HTML embedding
with open(LOGO_PATH, "rb") as _f:
    _LOGO_B64 = base64.b64encode(_f.read()).decode()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Global ─────────────────────────────────────── */
  .block-container {{
      padding-top: 0 !important;
      padding-bottom: 2rem;
  }}

  /* ── Sidebar ────────────────────────────────────── */
  [data-testid="stSidebar"] {{
      background: #FFFFFF;
      border-right: 4px solid {ORANGE};
  }}
  [data-testid="stSidebar"] h2 {{
      color: {ORANGE};
      font-size: 1.05rem;
      font-weight: 800;
      letter-spacing: 0.02em;
  }}
  [data-testid="stSidebar"] h3 {{
      color: {DARK};
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 0.4rem;
  }}

  /* ── KPI Cards ──────────────────────────────────── */
  div[data-testid="metric-container"] {{
      background: #FFFFFF;
      border-radius: 10px;
      padding: 1.1rem 1.3rem;
      border-left: 5px solid {ORANGE};
      box-shadow: 0 2px 10px rgba(0,0,0,0.07);
      transition: transform 0.15s ease, box-shadow 0.15s ease;
  }}
  div[data-testid="metric-container"]:hover {{
      transform: translateY(-2px);
      box-shadow: 0 5px 18px rgba(255,68,31,0.18);
  }}
  [data-testid="stMetricValue"] {{
      font-size: 1.75rem !important;
      font-weight: 800 !important;
      color: {ORANGE} !important;
  }}
  [data-testid="stMetricLabel"] {{
      font-size: 0.72rem !important;
      font-weight: 700 !important;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: #888 !important;
  }}

  /* ── Tabs ───────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {{
      gap: 0.4rem;
      border-bottom: 2px solid {ORANGE};
      padding-bottom: 0;
  }}
  .stTabs [data-baseweb="tab"] {{
      border-radius: 8px 8px 0 0;
      padding: 0.6rem 2.2rem;
      font-weight: 600;
      font-size: 0.95rem;
      background: #F5F5F5;
      color: #999;
      border: none;
  }}
  .stTabs [aria-selected="true"] {{
      background: {ORANGE} !important;
      color: #FFFFFF !important;
  }}

  /* ── Buttons ────────────────────────────────────── */
  .stButton > button {{
      background-color: {ORANGE};
      color: #FFFFFF;
      border: none;
      border-radius: 8px;
      font-weight: 600;
      padding: 0.45rem 1.4rem;
      transition: background-color 0.2s, transform 0.1s;
  }}
  .stButton > button:hover {{
      background-color: #d93a19;
      color: #FFFFFF;
      border: none;
      transform: translateY(-1px);
  }}
  .stButton > button:active {{
      transform: translateY(0);
  }}
  .stButton > button:focus {{
      outline: none;
      box-shadow: 0 0 0 3px rgba(255,68,31,0.3);
  }}

  /* ── Section labels ─────────────────────────────── */
  .section-label {{
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: {ORANGE};
      margin: 1rem 0 0.5rem;
      padding-bottom: 0.35rem;
      border-bottom: 2px solid rgba(255,68,31,0.2);
  }}

  /* ── Dividers ───────────────────────────────────── */
  hr {{
      border-top: 1px solid rgba(255,68,31,0.15) !important;
      margin: 1rem 0 !important;
  }}

  /* ── Alerts / info boxes ────────────────────────── */
  div[data-baseweb="notification"] {{
      border-radius: 8px !important;
      border-left: 4px solid {ORANGE} !important;
  }}

  /* ── Chat messages ──────────────────────────────── */
  [data-testid="stChatMessageContent"] {{
      border-radius: 10px;
  }}
</style>
""", unsafe_allow_html=True)

# ── DATA LOAD ─────────────────────────────────────────────────────────────────
df_all = load_all_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(logo, width=110)
    st.markdown("## Rappi Store Monitor")
    st.divider()

    st.markdown("### Configuracion")
    try:
        default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        default_key = ""
    api_key = st.text_input("Gemini API Key", value=default_key, type="password")

    st.divider()
    st.markdown("### Filtros")

    if not df_all.empty:
        dates = sorted(df_all["date"].unique())
        d_min, d_max = dates[0], dates[-1]
        date_range = st.date_input(
            "Rango de fechas",
            value=(d_min, d_max),
            min_value=d_min,
            max_value=d_max,
        )
        hour_range    = st.slider("Horas del dia", 0, 23, (0, 23))
        granularity   = st.select_slider(
            "Granularidad del grafico",
            options=["10s", "1min", "5min", "15min", "1h"],
            value="5min",
        )
        show_rolling  = st.checkbox("Mostrar media movil", value=True)
        anomaly_sigma = st.slider("Umbral de anomalias (s)", 1.5, 4.0, 2.5, 0.1)
    else:
        date_range    = None
        hour_range    = (0, 23)
        granularity   = "5min"
        show_rolling  = True
        anomaly_sigma = 2.5

    st.divider()
    st.caption(
        "**Fuente:** synthetic_monitoring_visible_stores  \n"
        "**Periodo:** Feb 1-6, 2026  \n"
        "**Frecuencia:** cada 10 segundos"
    )

# ── FILTER DATA ───────────────────────────────────────────────────────────────
if (
    not df_all.empty
    and isinstance(date_range, (list, tuple))
    and len(date_range) == 2
):
    mask = (
        (df_all["date"] >= date_range[0])
        & (df_all["date"] <= date_range[1])
        & (df_all["hour"] >= hour_range[0])
        & (df_all["hour"] <= hour_range[1])
    )
    df = df_all[mask].copy()
else:
    df = df_all.copy()

if df.empty:
    st.warning("No hay datos para los filtros seleccionados.")
    st.stop()

df_res = (
    df.set_index("timestamp")
    .resample(granularity)["stores"]
    .agg(["mean", "min", "max"])
    .reset_index()
    .rename(columns={"mean": "avg", "timestamp": "time"})
    .dropna()
)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, {ORANGE} 0%, {ORANGE_MID} 100%);
    padding: 1.4rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 1.4rem;
">
    <img
        src="data:image/png;base64,{_LOGO_B64}"
        style="height:56px; border-radius:8px; background:white; padding:5px;"
    />
    <div>
        <h1 style="margin:0; color:white; font-size:1.85rem; font-weight:800; line-height:1.2;">
            Rappi Store Monitor
        </h1>
        <p style="margin:0.3rem 0 0; color:rgba(255,255,255,0.88); font-size:0.88rem;">
            Dashboard de Disponibilidad de Tiendas &nbsp;&middot;&nbsp;
            <code style="
                background: rgba(255,255,255,0.2);
                padding: 0.1rem 0.5rem;
                border-radius: 4px;
                font-size: 0.82rem;
                color: white;
            ">synthetic_monitoring_visible_stores</code>
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_dash, tab_chat = st.tabs(["Dashboard", "Asistente IA"])


# ── Chart style helper ────────────────────────────────────────────────────────
def chart_style(**overrides) -> dict:
    """Returns a base Plotly layout dict with consistent Rappi branding."""
    base = dict(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color=DARK),
        title_font=dict(size=14, color=DARK),
        title_x=0,
        xaxis=dict(
            showgrid=True, gridcolor=GRID_COLOR,
            linecolor="#E0E0E0", zerolinecolor=GRID_COLOR,
        ),
        yaxis=dict(
            showgrid=True, gridcolor=GRID_COLOR,
            linecolor="#E0E0E0", zerolinecolor=GRID_COLOR,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E8E8E8",
            borderwidth=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter, sans-serif",
            bordercolor="#E0E0E0",
        ),
        margin=dict(t=50, b=20, l=10, r=10),
    )
    base.update(overrides)
    return base


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:

    # ── KPI CARDS ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Metricas Clave</p>', unsafe_allow_html=True)

    latest  = df["stores"].iloc[-1]
    peak    = df["stores"].max()
    avg     = df["stores"].mean()
    minimum = df["stores"].min()
    cv      = df["stores"].std() / avg * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Tiendas (actual)", f"{latest:,.0f}")
    k2.metric("Pico maximo",      f"{peak:,.0f}")
    k3.metric("Promedio",         f"{avg:,.0f}")
    k4.metric("Minimo",           f"{minimum:,.0f}")
    k5.metric(
        "Estabilidad (CV)",
        f"{cv:.1f}%",
        help="Coeficiente de variacion: menor % = mayor estabilidad operacional",
    )

    # ── 1. TIME SERIES ────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Tendencia General</p>', unsafe_allow_html=True)

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_res["time"], y=df_res["avg"],
        mode="lines", name="Tiendas online",
        line=dict(color=ORANGE, width=1.5),
        fill="tozeroy", fillcolor=ORANGE_PALE,
    ))
    if show_rolling and len(df_res) > 10:
        window = max(12, len(df_res) // 40)
        fig_ts.add_trace(go.Scatter(
            x=df_res["time"],
            y=df_res["avg"].rolling(window, center=True).mean(),
            mode="lines",
            name=f"Media movil ({window} puntos)",
            line=dict(color=DARK, width=2, dash="dot"),
        ))
    fig_ts.update_layout(**chart_style(
        title="Disponibilidad de Tiendas en el Tiempo",
        xaxis_title="Fecha / Hora",
        yaxis_title="Tiendas Online",
        hovermode="x unified",
        height=360,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)", bordercolor="#E8E8E8", borderwidth=1),
        xaxis=dict(
            showgrid=True, gridcolor=GRID_COLOR,
            linecolor="#E0E0E0", zerolinecolor=GRID_COLOR,
            rangeselector=dict(
                bgcolor="#F5F5F5",
                activecolor=ORANGE,
                buttons=[
                    dict(count=1,  label="1h",  step="hour", stepmode="backward"),
                    dict(count=6,  label="6h",  step="hour", stepmode="backward"),
                    dict(count=1,  label="1d",  step="day",  stepmode="backward"),
                    dict(step="all", label="Todo"),
                ],
            ),
            rangeslider=dict(visible=True, thickness=0.05, bgcolor="#FFF3F0"),
            type="date",
        ),
        margin=dict(t=50, b=10, l=10, r=10),
    ))
    st.plotly_chart(fig_ts, use_container_width=True)

    # ── 2. HEATMAP + DAILY BAR ────────────────────────────────────────────────
    st.markdown('<p class="section-label">Distribucion Temporal</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        pivot = df.pivot_table(index="hour", columns="date", values="stores", aggfunc="mean")
        fig_hm = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=pivot.index,
            colorscale=[[0,"#FFF3E0"],[0.4,ORANGE_MID],[0.75,ORANGE],[1,"#B71C1C"]],
            hovertemplate="%{y}:00 del %{x}<br>Promedio: %{z:,.0f} tiendas<extra></extra>",
        ))
        fig_hm.update_layout(**chart_style(
            title="Mapa de Calor: Hora x Dia",
            yaxis=dict(
                title="Hora del dia", tickmode="linear", dtick=2,
                autorange="reversed", showgrid=True, gridcolor=GRID_COLOR,
            ),
            xaxis=dict(title="Fecha", showgrid=True, gridcolor=GRID_COLOR),
            height=340,
        ))
        st.plotly_chart(fig_hm, use_container_width=True)

    with col_b:
        daily = df.groupby("date")["stores"].agg(["mean","max","min"]).reset_index()
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(
            x=daily["date"].astype(str), y=daily["max"],
            name="Pico", marker_color=ORANGE_MID, opacity=0.85,
        ))
        fig_daily.add_trace(go.Bar(
            x=daily["date"].astype(str), y=daily["mean"],
            name="Promedio", marker_color=ORANGE,
        ))
        fig_daily.add_trace(go.Bar(
            x=daily["date"].astype(str), y=daily["min"],
            name="Minimo", marker_color="#B71C1C", opacity=0.8,
        ))
        fig_daily.update_layout(**chart_style(
            title="Estadisticas Diarias",
            barmode="group",
            xaxis_title="Fecha", yaxis_title="Tiendas",
            height=340,
            legend=dict(orientation="h", y=1.05, bgcolor="rgba(255,255,255,0.9)", bordercolor="#E8E8E8", borderwidth=1),
        ))
        st.plotly_chart(fig_daily, use_container_width=True)

    # ── 3. HOURLY PATTERN + ANOMALY DETECTION ────────────────────────────────
    st.markdown('<p class="section-label">Analisis de Patrones</p>', unsafe_allow_html=True)
    col_c, col_d = st.columns(2)

    with col_c:
        hg = df.groupby("hour")["stores"].agg(["mean","std"]).reset_index().fillna(0)
        hrs, mu, sigma = hg["hour"], hg["mean"], hg["std"]

        fig_hp = go.Figure()
        fig_hp.add_trace(go.Scatter(
            x=hrs, y=mu + sigma,
            line=dict(width=0), showlegend=False, mode="lines",
        ))
        fig_hp.add_trace(go.Scatter(
            x=hrs, y=mu - sigma,
            fill="tonexty", mode="lines",
            line=dict(width=0),
            name="+/- 1 Desv. estandar",
            fillcolor=ORANGE_PALE,
        ))
        fig_hp.add_trace(go.Scatter(
            x=hrs, y=mu,
            mode="lines+markers", name="Promedio",
            line=dict(color=ORANGE, width=2.5),
            marker=dict(size=6, color=ORANGE, line=dict(color="white", width=1.5)),
        ))
        fig_hp.update_layout(**chart_style(
            title="Patron por Hora del Dia",
            xaxis=dict(title="Hora", tickmode="linear", dtick=2, showgrid=True, gridcolor=GRID_COLOR),
            yaxis_title="Tiendas Online",
            height=310,
        ))
        st.plotly_chart(fig_hp, use_container_width=True)

    with col_d:
        df_ano = (
            df.set_index("timestamp").resample("5min")["stores"]
            .mean().reset_index()
            .rename(columns={"timestamp": "time"}).dropna()
        )
        roll_m = df_ano["stores"].rolling(12, center=True, min_periods=3).mean()
        roll_s = df_ano["stores"].rolling(12, center=True, min_periods=3).std()
        df_ano["z"] = (df_ano["stores"] - roll_m) / roll_s.replace(0, np.nan)
        anomalies = df_ano[df_ano["z"].abs() > anomaly_sigma]

        fig_ano = go.Figure()
        fig_ano.add_trace(go.Scatter(
            x=df_ano["time"], y=df_ano["stores"],
            mode="lines", name="Tiendas (5 min)",
            line=dict(color=ORANGE, width=1.5),
        ))
        if not anomalies.empty:
            fig_ano.add_trace(go.Scatter(
                x=anomalies["time"], y=anomalies["stores"],
                mode="markers", name=f"Anomalia (|z|>{anomaly_sigma})",
                marker=dict(symbol="x", size=10, color="#D50000", line=dict(width=2.5)),
            ))
        fig_ano.update_layout(**chart_style(
            title=f"Deteccion de Anomalias (umbral s={anomaly_sigma})",
            xaxis_title="Fecha / Hora", yaxis_title="Tiendas",
            height=310,
        ))
        st.plotly_chart(fig_ano, use_container_width=True)

    # ── 4. DISTRIBUTION + DAY OF WEEK ────────────────────────────────────────
    st.markdown('<p class="section-label">Distribucion y Comparativa Semanal</p>', unsafe_allow_html=True)
    col_e, col_f = st.columns(2)

    with col_e:
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df[~df["is_weekend"]]["stores"],
            name="Lunes-Viernes",
            marker_color=ORANGE, opacity=0.75, nbinsx=50,
        ))
        fig_dist.add_trace(go.Histogram(
            x=df[df["is_weekend"]]["stores"],
            name="Fin de semana",
            marker_color=DARK, opacity=0.75, nbinsx=50,
        ))
        fig_dist.update_layout(**chart_style(
            title="Distribucion: Semana vs Fin de Semana",
            barmode="overlay",
            xaxis_title="Tiendas Online", yaxis_title="Frecuencia",
            height=300,
            legend=dict(orientation="h", y=1.05, bgcolor="rgba(255,255,255,0.9)", bordercolor="#E8E8E8", borderwidth=1),
        ))
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_f:
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_avg = df.groupby("day_name")["stores"].mean().reindex(dow_order).dropna()
        bar_colors = [ORANGE if d not in ("Saturday","Sunday") else DARK for d in dow_avg.index]

        fig_dow = go.Figure(go.Bar(
            x=dow_avg.index, y=dow_avg.values,
            marker_color=bar_colors,
            text=[f"{v:,.0f}" for v in dow_avg.values],
            textposition="outside",
            textfont=dict(size=10, color=DARK),
        ))
        fig_dow.update_layout(**chart_style(
            title="Promedio por Dia de la Semana",
            xaxis_title="Dia", yaxis_title="Tiendas",
            height=300,
            margin=dict(t=50, b=30, l=10, r=10),
        ))
        st.plotly_chart(fig_dow, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · AI CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if not api_key:
        st.warning("Ingresa tu Gemini API Key en la barra lateral para activar el asistente.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "data_summary" not in st.session_state:
        st.session_state.data_summary = compute_summary(df_all)

    # Chat header
    st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.5rem;
        background: white;
        border-radius: 10px;
        border-left: 5px solid {ORANGE};
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 1.2rem;
    ">
        <img
            src="data:image/png;base64,{_LOGO_B64}"
            style="height:40px; border-radius:6px;"
        />
        <div>
            <p style="margin:0; font-weight:800; font-size:1.05rem; color:{DARK};">
                DataBot Rappi
            </p>
            <p style="margin:0; font-size:0.8rem; color:#888;">
                Asistente de datos de disponibilidad de tiendas
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Limpiar conversacion"):
        st.session_state.messages = []
        st.rerun()

    if not st.session_state.messages:
        st.info(
            "Hola, soy **DataBot Rappi**. Puedes preguntarme:\n\n"
            "- Cual fue el pico maximo de tiendas?\n"
            "- A que hora hay mas tiendas activas?\n"
            "- Hubo caidas significativas en los datos?\n"
            "- Que dia de la semana tuvo mejor disponibilidad?"
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Pregunta sobre los datos de Rappi..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                client = init_client(api_key)
                response_text = st.write_stream(
                    stream_response(
                        client,
                        st.session_state.messages,
                        st.session_state.data_summary,
                    )
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )
            except Exception as e:
                err = f"Error conectando con Gemini: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})