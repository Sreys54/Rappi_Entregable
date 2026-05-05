import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent


@st.cache_data(show_spinner="Cargando datos...")
def load_all_data() -> pd.DataFrame:
    # Busca los 100 archivos CSV por patrón de nombre
    pattern = str(DATA_DIR / "AVAILABILITY-data (*).csv")

    # Ordenamos por el número entre paréntesis del nombre, no alfabéticamente,
    # porque "data (2)" viene antes que "data (10)" en orden alfabético.
    files = sorted(
        glob.glob(pattern),
        key=lambda x: int(re.search(r"\((\d+)\)", x).group(1)),
    )

    chunks = []
    for fp in files:
        try:
            # dtype=str previene que pandas interprete los encabezados de
            # timestamp como fechas automáticamente antes de que los procesemos.
            raw = pd.read_csv(fp, header=0, dtype=str)
            if raw.shape[0] < 1:
                continue

            # El archivo tiene una sola fila de datos; las columnas 0-3 son
            # metadatos (nombre de métrica, etc.) y desde la columna 4 en
            # adelante cada columna es una lectura con su timestamp en el nombre.
            data_row = raw.iloc[0]
            ts_cols = raw.columns[4:].tolist()
            vals = pd.to_numeric(pd.Series(data_row.iloc[4:].values), errors="coerce")

            # Los encabezados tienen formato JS Date.toString():
            # "Sun Feb 01 2026 06:59:40 GMT-0500 (hora estándar de Colombia)"
            # Extraemos solo la parte parseable con regex vectorizado, que es
            # ~50x más rápido que aplicar dateutil.parser fila a fila.
            ts_series = pd.Series(ts_cols)
            extracted = ts_series.str.extract(
                r"(\w{3} \w{3} \d{2} \d{4} \d{2}:\d{2}:\d{2})"
            )[0]
            timestamps = pd.to_datetime(extracted, format="%a %b %d %Y %H:%M:%S")

            chunks.append(
                pd.DataFrame({"timestamp": timestamps.values, "stores": vals.values})
            )
        except Exception:
            continue

    if not chunks:
        return pd.DataFrame(
            columns=["timestamp", "stores", "date", "hour", "day_name", "is_weekend", "zscore"]
        )

    df = pd.concat(chunks, ignore_index=True)
    df = (
        df.dropna()
        .sort_values("timestamp")
        # Archivos de horas consecutivas pueden solapar 1-2 lecturas en el
        # borde; eliminamos duplicados para evitar distorsión en los promedios.
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    df["date"]       = df["timestamp"].dt.date
    df["hour"]       = df["timestamp"].dt.hour
    df["day_name"]   = df["timestamp"].dt.day_name()
    df["day_num"]    = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_num"] >= 5
    df["delta"]      = df["stores"].diff()
    df["pct_change"] = df["stores"].pct_change() * 100

    # Z-score con ventana de 30 puntos = 5 minutos (intervalo de 10 s × 30).
    # Usamos la media/desv. local en lugar de la global para detectar caídas
    # repentinas sin que el umbral cambie con el nivel promedio del día.
    roll_mean = df["stores"].rolling(30, center=True, min_periods=5).mean()
    roll_std  = df["stores"].rolling(30, center=True, min_periods=5).std()
    df["zscore"] = (df["stores"] - roll_mean) / roll_std.replace(0, np.nan)

    return df


@st.cache_data
def compute_summary(df: pd.DataFrame) -> str:
    # Genera un resumen textual estructurado para inyectarlo como contexto
    # al chatbot. Al pre-calcularlo una sola vez con @st.cache_data evitamos
    # recorrer los ~33k registros en cada turno de conversación.
    if df.empty:
        return "No hay datos disponibles."

    peak_row = df.loc[df["stores"].idxmax()]
    min_row  = df.loc[df["stores"].idxmin()]
    g_avg    = df["stores"].mean()
    g_std    = df["stores"].std()
    cv       = g_std / g_avg * 100

    hourly_avg = df.groupby("hour")["stores"].mean()
    peak_h     = int(hourly_avg.idxmax())
    low_h      = int(hourly_avg.idxmin())

    daily  = df.groupby("date")["stores"].agg(["mean", "max", "min"])
    dow    = df.groupby("day_name")["stores"].mean().sort_values(ascending=False)
    drops  = df[df["pct_change"] < -5].nsmallest(5, "pct_change")
    spikes = df[df["pct_change"] > 5].nlargest(5, "pct_change")

    lines = [
        "=== RAPPI STORE AVAILABILITY DATA SUMMARY ===",
        "Metric: synthetic_monitoring_visible_stores",
        "Definition: number of Rappi stores visible/online on the platform at each moment",
        f"Period: {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} — "
        f"{df['timestamp'].max().strftime('%Y-%m-%d %H:%M')} (Colombia, UTC-5)",
        f"Total readings: {len(df):,} (one every 10 seconds)",
        "",
        "GLOBAL STATISTICS:",
        f"  Peak:    {peak_row['stores']:,.0f} stores at {peak_row['timestamp'].strftime('%Y-%m-%d %H:%M')}",
        f"  Minimum: {min_row['stores']:,.0f} stores at {min_row['timestamp'].strftime('%Y-%m-%d %H:%M')}",
        f"  Average: {g_avg:,.0f}  |  Std: {g_std:,.0f}  |  CV: {cv:.1f}%",
        "",
        "HOURLY PATTERNS (avg across all days):",
        f"  Busiest hour:  {peak_h:02d}:00 — avg {hourly_avg[peak_h]:,.0f} stores",
        f"  Quietest hour: {low_h:02d}:00 — avg {hourly_avg[low_h]:,.0f} stores",
        "",
        "DAILY BREAKDOWN:",
    ]
    for d, row in daily.iterrows():
        lines.append(
            f"  {d}: avg={row['mean']:,.0f}, max={row['max']:,.0f}, min={row['min']:,.0f}"
        )

    lines += ["", "DAY-OF-WEEK AVERAGES:"]
    for day, avg in dow.items():
        lines.append(f"  {day}: {avg:,.0f}")

    if not drops.empty:
        lines += ["", "NOTABLE DROPS (>5% in 10 seconds):"]
        for _, r in drops.iterrows():
            lines.append(
                f"  {r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}: {r['pct_change']:.1f}%"
            )

    if not spikes.empty:
        lines += ["", "NOTABLE SPIKES (>5% in 10 seconds):"]
        for _, r in spikes.iterrows():
            lines.append(
                f"  {r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}: +{r['pct_change']:.1f}%"
            )

    return "\n".join(lines)