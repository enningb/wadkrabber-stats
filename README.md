# Wadkrabber Sail Log

Interactive Streamlit app to explore sailing routes day by day. Use the date slider to switch between voyages, view each day's track on an interactive map, and review daily statistics (now with inline sparklines for speed, wind, and distance trends) pulled from your logbook.

## Getting started

1. Create a virtual environment (optional but recommended) and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Launch Streamlit:
   ```bash
   streamlit run app.py
   ```

The app now reads from `data/wadkrabber.csv` by default. A sample `data/sail_log.csv` is still bundled as a fallback example; drop in your own CSV with the same headers to replace either file.

## Data format

Two CSV layouts are supported:

- **Wadkrabber export** (the default): requires `Time`, `latitude`, `longitude`, and `date` columns. Optional numeric columns such as `SOG` (used for boat speed), `TWS` (wind speed), and `distance` (per-leg nautical miles) are automatically folded into the daily statistics. Harbour metadata (`vertrekhaven`, `aankomsthaven`, `havens`) and `log_entry` values feed the labels and notes.
- **Generic sail log**: expects `timestamp`, `latitude`, and `longitude` columns, with optional extras (`speed_knots`, `wind_speed_knots`, `leg_distance_nm`, `sea_state`, `notes`, etc.) as documented below. This is the schema in `data/sail_log.csv`.

In both cases, the CSV should contain one row per position fix (or event). Any additional columns are shown in the "Show raw log entries" expander so you can inspect the original data.

## Customising

- Drop your own logbook export in `data/wadkrabber.csv` (or `data/sail_log.csv` if you prefer the generic schema); keep headers identical.
- Adjust the `daily_summary` function in `app.py` if you track different metrics.
- To change the map styling or layers, edit `render_map` in `app.py`.

Happy sailing!
