"""Streamlit Sail Log app with date slider, map, and daily statistics."""
from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import numpy as np
import pydeck as pdk
import streamlit as st

DATA_FILES: Iterable[Path] = (
    Path(__file__).parent / "data" / "wadkrabber.csv",
)

def _prepare_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Shared cleanup applied to any dataset variant."""
    df = df.copy()
    cleaned_columns = []
    drop_columns = []
    for col in df.columns:
        if not isinstance(col, str):
            cleaned_columns.append(col)
            continue
        stripped = col.strip()
        if not stripped or stripped.lower().startswith("unnamed"):
            drop_columns.append(col)
        else:
            cleaned_columns.append(stripped)
    if drop_columns:
        df = df.drop(columns=drop_columns)
    df.columns = cleaned_columns
    return df


def _prepare_standard_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Handle datasets that already follow the 'timestamp' schema."""
    if "timestamp" not in df.columns:
        raise ValueError("Dataset is missing required 'timestamp' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["local_dt"] = df["timestamp"].dt.tz_convert(None)
    df["date"] = df["local_dt"].dt.date
    df["timestamp_label"] = df["local_dt"].dt.strftime("%Y-%m-%d %H:%M")

    required_columns = {"latitude", "longitude"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    return df.sort_values("timestamp").reset_index(drop=True)


def _select_first(series: pd.Series) -> Optional[str]:
    """Return the first non-empty string from a series, if any."""
    if series.empty:
        return None
    cleaned = (
        series.astype(str)
        .str.strip()
        .replace({"nan": "", "None": ""}, regex=False)
    )
    cleaned = cleaned[cleaned.astype(bool)]
    if cleaned.empty:
        return None
    return str(cleaned.iloc[0])


def _select_last(series: pd.Series) -> Optional[str]:
    """Return the last non-empty string from a series, if any."""
    if series.empty:
        return None
    cleaned = (
        series.astype(str)
        .str.strip()
        .replace({"nan": "", "None": ""}, regex=False)
    )
    cleaned = cleaned[cleaned.astype(bool)]
    if cleaned.empty:
        return None
    return str(cleaned.iloc[-1])


def _prepare_wadkrabber_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Transform Wadkrabber exports to the app's working schema."""
    if "Time" not in df.columns:
        raise ValueError("Wadkrabber dataset is missing required 'Time' column.")

    df["timestamp"] = pd.to_datetime(df["Time"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df = df.dropna(subset=["timestamp", "date", "latitude", "longitude"]).copy()

    df["timestamp_label"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

    speed_sog_knots = pd.to_numeric(df.get("SOG"), errors="coerce")
    speed_stw_knots = pd.to_numeric(df.get("STW"), errors="coerce")
    df["speed_knots"] = speed_sog_knots
    df["speed_knots"].replace(0,np.nan, inplace=True)
    df["speed_stw_knots"] = speed_stw_knots
    df["speed_stw_knots"].replace(0, np.nan, inplace=True)
    df["current_speed_knots"] = df['speed_knots'] - df['speed_stw_knots']

    wind_angle_awa = pd.to_numeric(df.get("AWA"), errors="coerce")
    wind_angle_twa = pd.to_numeric(df.get("TWA"), errors="coerce")
    df["wind_angle_awa"] = wind_angle_awa
    df["wind_angle_twa"] = wind_angle_twa

    df["wind_speed_knots"] = pd.to_numeric(df.get("TWS"), errors="coerce")
    df["leg_distance_nm"] = pd.to_numeric(df.get("distance"), errors="coerce")
    df["depth_m"] = pd.to_numeric(df.get("Depth"), errors="coerce")

    preferred_location = (
        df.get("havens")
        .fillna("")
        .astype(str)
        .str.strip()
    )
    fallback_location = (
        df.get("vertrekhaven")
        .fillna("")
        .astype(str)
        .str.strip()
    )
    df["location"] = preferred_location.where(preferred_location.astype(bool), fallback_location)

    if "vertrekhaven" in df.columns:
        df["start_harbour"] = df.groupby("date")["vertrekhaven"].transform(_select_first)
    if "aankomsthaven" in df.columns:
        df["end_harbour"] = df.groupby("date")["aankomsthaven"].transform(_select_last)
    df["notes"] = (
        df.get("log_entry")
        .fillna("")
        .astype(str)
        .str.replace(".md", "", regex=False)
        .str.replace("_", " ")
        .str.strip()
    )
    df["notes"] = df["notes"].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    empty_location = ~df["location"].astype(bool)
    df.loc[empty_location, "location"] = df.loc[empty_location, "notes"]
    df.loc[df["location"].astype(str).str.strip() == "", "location"] = "Track position"
    df.loc[df["notes"].astype(str).str.len() == 0, "notes"] = pd.NA

    return df.sort_values("timestamp").reset_index(drop=True)


def _sparkline_frame(day_df: pd.DataFrame, column: str, *, cumulative: bool = False) -> Optional[pd.DataFrame]:
    """Return a tidy dataframe for sparkline visualisations."""
    if column not in day_df.columns:
        return None

    data = day_df.loc[:, ["timestamp", column]].dropna(subset=[column]).copy()
    if data.empty:
        return None

    data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=[column])
    if data.empty:
        return None

    data = data.sort_values("timestamp")
    if cumulative:
        data[column] = data[column].cumsum()

    timestamps = data["timestamp"]
    if pd.api.types.is_datetime64tz_dtype(timestamps):
        timestamps = timestamps.dt.tz_convert(None)

    result = data.copy()
    result["timestamp"] = timestamps
    result = result.set_index("timestamp")[[column]].rename(columns={column: "value"})
    result.index.name = "timestamp"
    return result


def _load_dataset(path: Path) -> pd.DataFrame:
    """Load and preprocess whichever dataset is provided."""
    df = pd.read_csv(path)
    df = _prepare_common_columns(df)

    if "timestamp" in df.columns:
        return _prepare_standard_dataset(df)
    if "Time" in df.columns:
        return _prepare_wadkrabber_dataset(df)

    raise ValueError(
        "Unsupported dataset format. Expected columns like 'timestamp' or 'Time'."
    )


@st.cache_data(show_spinner=False)
def load_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Cached loader that wraps preprocessing with error handling."""
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find sail log data at '{path}'.")
        return _load_dataset(path)

    for candidate in DATA_FILES:
        if candidate.exists():
            return _load_dataset(candidate)

    search_hint = " or ".join(str(p) for p in DATA_FILES)
    raise FileNotFoundError(
        f"Could not find sail log data. Place a CSV at {search_hint}."
    )


def daily_summary(day_df: pd.DataFrame) -> Dict[str, float | str]:
    """Compute daily stats for the selected day."""
    stats: Dict[str, float | str] = {}

    if "leg_distance_nm" in day_df.columns:
        distance = day_df["leg_distance_nm"].dropna()
        if not distance.empty:
            stats["distance_nm"] = float(distance.sum())
    elif "cumulative_distance_nm" in day_df.columns:
        total = day_df["cumulative_distance_nm"].dropna()
        if not total.empty:
            stats["distance_nm"] = float(total.iloc[-1])

    if "speed_knots" in day_df.columns:
        speed = day_df["speed_knots"].dropna()
        if not speed.empty:
            stats["avg_speed_knots"] = float(speed.mean())
            stats["max_speed_knots"] = float(speed.max())

    if "wind_speed_knots" in day_df.columns:
        wind = day_df["wind_speed_knots"].dropna()
        if not wind.empty:
            stats["avg_wind_knots"] = float(wind.mean())

    if "sea_state" in day_df.columns and day_df["sea_state"].notna().any():
        stats["sea_state"] = day_df["sea_state"].dropna().iloc[-1]

    if "current_speed_knots" in day_df.columns:
        current = day_df["current_speed_knots"].dropna()
        if not current.empty:
            stats["avg_current_speed_knots"] = float(current.mean())
            stats["max_current_speed_knots"] = float(current.max())
            stats["min_current_speed_knots"] = float(current.min())

    if "notes" in day_df.columns:
        note_value = _select_last(day_df["notes"])
        if note_value:
            stats["notes"] = note_value

    start = day_df.iloc[0]
    end = day_df.iloc[-1]

    start_series = day_df["start_harbour"] if "start_harbour" in day_df.columns else day_df.get("vertrekhaven")
    end_series = day_df["end_harbour"] if "end_harbour" in day_df.columns else day_df.get("aankomsthaven")

    start_label = _select_first(start_series) if start_series is not None else None
    end_label = _select_last(end_series) if end_series is not None else None

    stats["start_position"] = (
        f"{start_label or start.get('location', 'Start')} "
        f"({start['latitude']:.3f}, {start['longitude']:.3f})"
    )
    stats["end_position"] = (
        f"{end_label or end.get('location', 'Finish')} "
        f"({end['latitude']:.3f}, {end['longitude']:.3f})"
    )

    return stats


def route_layers(day_df: pd.DataFrame) -> List[pdk.Layer]:
    """Create pydeck layers representing the sailing route."""
    path_data = day_df[["longitude", "latitude"]].values.tolist()
    route_data = [{"path": path_data}]

    route_layer = pdk.Layer(
        "PathLayer",
        data=route_data,
        get_path="path",
        get_color=[16, 127, 201],
        width_scale=10,
        width_min_pixels=3,
        pickable=False,
    )

    points_layer = pdk.Layer(
        "ScatterplotLayer",
        data=day_df[['longitude', 'latitude']],
        get_position="[longitude, latitude]",
        get_color="[16, 127, 201]",
        get_radius=20,
        pickable=True,
        get_fill_color="[16, 127, 201]",
    )

    return [route_layer, points_layer]


def center_view(day_df: pd.DataFrame) -> pdk.ViewState:
    """Compute a viewport that centers on the daily track."""
    lat = float(day_df["latitude"].mean())
    lon = float(day_df["longitude"].mean())

    zoom = 11
    if "zoom_hint" in day_df.columns and day_df["zoom_hint"].notna().any():
        zoom = float(day_df["zoom_hint"].dropna().iloc[-1])

    return pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=45)


def render_map(day_df: pd.DataFrame) -> None:
    """Render the day's route on a map."""
    layers = route_layers(day_df)
    view_state = center_view(day_df)
    tooltip = {
        "text": "{timestamp_label}\n{location}\nSpeed: {speed_knots} kn",
    }

    st.pydeck_chart(
        pdk.Deck(
            map_style="light",
            layers=layers,
            initial_view_state=view_state,
            tooltip=tooltip,
        )
    )


def render_stats(stats: Dict[str, float | str], day_df: pd.DataFrame) -> None:
    """Display daily statistics in columns."""
    if not stats:
        st.info("No statistics available for this day.")
        return

    cols = st.columns(2)
    items = list(stats.items())
    sparkline_sources: Dict[str, Dict[str, object]] = {}
    if "leg_distance_nm" in day_df.columns:
        sparkline_sources["distance_nm"] = {"column": "leg_distance_nm", "cumulative": True}
    elif "cumulative_distance_nm" in day_df.columns:
        sparkline_sources["distance_nm"] = {"column": "cumulative_distance_nm", "cumulative": False}

    if "speed_knots" in day_df.columns:
        for key in ("avg_speed_knots", "max_speed_knots"):
            sparkline_sources.setdefault(key, {"column": "speed_knots", "cumulative": False})

    if "wind_speed_knots" in day_df.columns:
        sparkline_sources.setdefault("avg_wind_knots", {"column": "wind_speed_knots", "cumulative": False})

    for idx, (label, value) in enumerate(items):
        col = cols[idx % 2]
        pretty_label = label.replace("_", " ").title()
        if isinstance(value, float):
            if label.endswith("knots"):
                col.metric(pretty_label, f"{value:.1f} kn")
            elif label.endswith("nm"):
                col.metric(pretty_label, f"{value:.1f} nm")
            else:
                col.metric(pretty_label, f"{value:.2f}")
        else:
            col.write(f"**{pretty_label}:** {value}")

        source_details = sparkline_sources.get(label)
        if source_details:
            sparkline_df = _sparkline_frame(
                day_df,
                source_details["column"],
                cumulative=bool(source_details.get("cumulative")),
            )
            if sparkline_df is not None and not sparkline_df.empty:
                col.line_chart(sparkline_df, height=90, width=200)


def _format_metric_value(value: float, suffix: str) -> str:
    return f"{value:.1f} {suffix}" if value is not None else "—"


def compute_overall_stats(df: pd.DataFrame) -> Dict[str, object]:
    """Aggregate dataset-wide statistics for the overview page."""
    metrics: Dict[str, object] = {}

    depth_col = df.get("depth_m")
    if depth_col is not None:
        depth_values = depth_col.dropna()
        if not depth_values.empty:
            idx = depth_values.idxmax()
            record = df.loc[idx]
            metrics["deepest_water"] = {
                "value": float(depth_values.loc[idx]),
                "context": record.get("location")
                or record.get("start_harbour")
                or record.get("vertrekhaven")
                or "Unknown location",
                "timestamp": record.get("timestamp_label", ""),
            }

    wind_col = df.get("wind_speed_knots")
    if wind_col is not None:
        wind_values = wind_col.dropna()
        if not wind_values.empty:
            idx = wind_values.idxmax()
            record = df.loc[idx]
            metrics["hardest_wind"] = {
                "value": float(wind_values.loc[idx]),
                "context": record.get("location")
                or record.get("end_harbour")
                or record.get("aankomsthaven")
                or "At sea",
                "timestamp": record.get("timestamp_label", ""),
            }

    speed_col = df.get("speed_knots")
    if speed_col is not None:
        speed_values = speed_col.dropna()
        if not speed_values.empty:
            idx = speed_values.idxmax()
            record = df.loc[idx]
            metrics["fastest_speed"] = {
                "value": float(speed_values.loc[idx]),
                "context": record.get("location")
                or record.get("start_harbour")
                or record.get("vertrekhaven")
                or "Underway",
                "timestamp": record.get("timestamp_label", ""),
            }

    harbour_counter: Counter[str] = Counter()
    trip_rows: List[Dict[str, object]] = []

    for sailing_date, day_df in df.groupby("date"):
        if day_df.empty:
            continue

        summary = daily_summary(day_df)
        distance_nm = summary.get("distance_nm")
        start_series = (
            day_df["start_harbour"]
            if "start_harbour" in day_df.columns
            else day_df.get("vertrekhaven")
        )
        end_series = (
            day_df["end_harbour"]
            if "end_harbour" in day_df.columns
            else day_df.get("aankomsthaven")
        )
        start_label = _select_first(start_series) if start_series is not None else None
        end_label = _select_last(end_series) if end_series is not None else None

        if start_label:
            harbour_counter[start_label] += 1
        if end_label:
            harbour_counter[end_label] += 1

        if distance_nm is not None:
            trip_rows.append(
                {
                    "date": sailing_date,
                    "distance_nm": float(distance_nm),
                    "start_harbour": start_label or "—",
                    "end_harbour": end_label or "—",
                }
            )

    trips_df = pd.DataFrame(trip_rows)
    if not trips_df.empty:
        longest_trip = trips_df.sort_values("distance_nm", ascending=False).iloc[0]
        metrics["longest_trip"] = longest_trip
        metrics["trip_table"] = trips_df.sort_values("date", ascending=True)

    if harbour_counter:
        harbour_df = pd.DataFrame(
            [
                {"harbour": harbour, "visits": visits}
                for harbour, visits in harbour_counter.most_common(10)
            ]
        )
        metrics["harbour_counts"] = harbour_df

    return metrics


def render_overall_statistics(df: pd.DataFrame) -> None:
    """Render the overall statistics page."""
    stats = compute_overall_stats(df)

    headline_cols = st.columns(3)
    deepest = stats.get("deepest_water")
    if deepest:
        headline_cols[0].metric(
            "Deepest Water",
            _format_metric_value(deepest["value"], "m"),
            deepest.get("context", ""),
        )
        if deepest.get("timestamp"):
            headline_cols[0].caption(deepest["timestamp"])

    hardest = stats.get("hardest_wind")
    if hardest:
        headline_cols[1].metric(
            "Hardest Wind",
            _format_metric_value(hardest["value"], "kn"),
            hardest.get("context", ""),
        )
        if hardest.get("timestamp"):
            headline_cols[1].caption(hardest["timestamp"])

    fastest = stats.get("fastest_speed")
    if fastest:
        headline_cols[2].metric(
            "Fastest Boat Speed",
            _format_metric_value(fastest["value"], "kn"),
            fastest.get("context", ""),
        )
        if fastest.get("timestamp"):
            headline_cols[2].caption(fastest["timestamp"])

    longest_trip = stats.get("longest_trip")
    if longest_trip is not None:
        st.subheader("Longest Trip")
        st.write(
            f"{longest_trip['date']} — {longest_trip['start_harbour']} → {longest_trip['end_harbour']}"
        )
        st.metric("Distance", f"{longest_trip['distance_nm']:.1f} nm")

    trip_table = stats.get("trip_table")
    if isinstance(trip_table, pd.DataFrame) and not trip_table.empty:
        display_df = trip_table.assign(
            date=lambda d: d["date"].astype(str),
            distance_nm=lambda d: d["distance_nm"].map(lambda val: f"{val:.1f}"),
        )
        st.subheader("Trips By Day")
        st.dataframe(display_df, width='content', hide_index=True)

    harbour_counts = stats.get("harbour_counts")
    if isinstance(harbour_counts, pd.DataFrame) and not harbour_counts.empty:
        st.subheader("Most Visited Harbours")
        st.bar_chart(
            harbour_counts.set_index("harbour"),
            width='content',
        )
        st.dataframe(harbour_counts, hide_index=True, width='content')

    st.scatter_chart(
            df,
            x="AWA",
            y="speed_knots",
            color="AWS",
#            size="col3",
        )
    st.scatter_chart(
            df,
            x="AWS",
            y="speed_knots",
            color="AWS",
#            size="col3",
        )


def main() -> None:
    st.set_page_config(page_title="Sail Log", layout="wide")
    st.title("Wadkrabber Sail Log")
    st.caption("Explore voyages day by day, visualize tracks, and review daily stats.")

    try:
        df = load_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except ValueError as exc:
        st.error(f"Failed to parse sail log data: {exc}")
        st.stop()

    view = st.sidebar.radio(
        "Navigation",
        ("Daily Explorer", "Overall Statistics"),
    )

    if view == "Overall Statistics":
        st.header("Overall Statistics")
        render_overall_statistics(df)
        return

    available_dates = sorted(df["date"].unique())
    
    if not available_dates:
        st.warning("No sail log entries available yet.")
        st.stop()

    max_date: date = available_dates[-1]

    selected_date = st.select_slider(
        "Select a sailing day",
        options=available_dates,
        value=max_date,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
    )

    day_df = df[df["date"] == selected_date]
    if day_df.empty:
        st.warning("No data for the selected day.")
        st.stop()

    day_label = selected_date.strftime("%B %d, %Y")
    start_haven = day_df.vertrekhaven.iloc[0]
    aankomst_haven = day_df.aankomsthaven.iloc[0]
    st.subheader(f"Van {start_haven} naar {aankomst_haven} op {day_label}")

    render_map(day_df)
    st.subheader("Daily Statistics")
    stats = daily_summary(day_df)
    render_stats(stats, day_df)

    with st.expander("Show raw log entries"):
        display_columns = [
            "timestamp_label",
            "location",
            "latitude",
            "longitude",
            "speed_knots",
            "wind_speed_knots",
            "leg_distance_nm",
            "notes",
        ]
        existing_columns = [c for c in display_columns if c in day_df.columns]
        st.dataframe(day_df[existing_columns])

    with st.expander("Show raw ALL log entries"):
        st.dataframe(day_df)

if __name__ == "__main__":
    main()
