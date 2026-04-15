"""
Data fetching from all external sources.

Each function returns a pandas Series or DataFrame with a DatetimeIndex
in LST (UTC-8, no DST) at hourly or finer resolution.

CO-OPS API notes:
  - Observed products: 31-day max per request → fetch monthly chunks
  - Predictions: 1-year max per request → fetch annually
  - API returns empty strings for missing values; use pd.to_numeric(..., errors='coerce')
  - All timestamps in LST (UTC-8, no DST)

HFR fetch notes:
  - Uses multiprocessing (not threads) — netcdf4 C library is not thread-safe
  - Cached to parquet after first fetch (~45 min for 2015-2025)
"""

import logging
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import dataretrieval.nwis as nwis
import numpy as np
import pandas as pd
import requests
import xarray as xr

from src.config import (
    BEGIN_ISO,
    CDEC_API,
    CDEC_OUTFLOW_SENSOR,
    CDEC_OUTFLOW_STA,
    CFS_TO_CMS,
    COOPS_API,
    COOPS_FORTPOINT,
    COOPS_RICHMOND,
    END_ISO,
    HFR_CACHE_PATH,
    HFR_LAT,
    HFR_LON,
    HFR_SEARCH_RADIUS_DEG,
    OPENMETEO_ARCHIVE_URL,
    PFEL_URL,
    USGS_SAC_SITE,
    WIND_LAT,
    WIND_LON,
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _coops_chunk(begin: str, end: str, product: str, station: str,
                 datum: str = "MLLW", units: str = "metric",
                 time_zone: str = "lst", interval: str = None,
                 retries: int = 3) -> pd.Series | None:
    """Fetch a single date range from the CO-OPS API. Returns a Series or None on failure."""
    params = dict(
        begin_date=begin, end_date=end,
        product=product, station=station,
        datum=datum, units=units,
        time_zone=time_zone, format="json",
        application="richmond-current-ml",
    )
    if interval:
        params["interval"] = interval
    # CO-OPS returns data under product-specific keys:
    # "water_level", "air_pressure", "water_temperature" → "data"
    # "predictions" → "predictions"
    data_key = "predictions" if product == "predictions" else "data"
    for attempt in range(retries):
        try:
            resp = requests.get(COOPS_API, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if data_key not in data:
                logger.warning("CO-OPS %s %s %s: no '%s' key — %s",
                               station, product, begin, data_key,
                               data.get("error", {}).get("message", ""))
                return None
            df = pd.DataFrame(data[data_key])
            df.index = pd.to_datetime(df["t"])
            df.index.name = None
            return pd.to_numeric(df["v"], errors="coerce")
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error("CO-OPS fetch failed: %s %s %s — %s", station, product, begin, exc)
                return None


def _monthly_chunks(begin_iso: str, end_iso: str):
    """Yield (begin_str, end_str) pairs in YYYYMMDD format, one month at a time."""
    current = datetime.strptime(begin_iso, "%Y-%m-%d")
    end     = datetime.strptime(end_iso,   "%Y-%m-%d")
    while current <= end:
        month_end = (current.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        chunk_end = min(month_end, end)
        yield current.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")
        current = chunk_end + timedelta(days=1)


def _annual_chunks(begin_iso: str, end_iso: str):
    """Yield (begin_str, end_str) pairs in YYYYMMDD format, one year at a time."""
    start_year = int(begin_iso[:4])
    end_year   = int(end_iso[:4])
    for year in range(start_year, end_year + 1):
        yield f"{year}0101", f"{year}1231"


# ── Water level & tide predictions ────────────────────────────────────────────

def fetch_water_level(station: str = COOPS_RICHMOND,
                      begin_iso: str = BEGIN_ISO,
                      end_iso: str = END_ISO) -> pd.Series:
    """Observed water level (MLLW, metric, LST) from CO-OPS."""
    logger.info("Fetching observed water level — station %s", station)
    chunks = [
        _coops_chunk(b, e, "water_level", station)
        for b, e in _monthly_chunks(begin_iso, end_iso)
    ]
    series = pd.concat([c for c in chunks if c is not None])
    series.name = f"obs_wl_{station}"
    return series.sort_index()


def fetch_tide_predictions(station: str = COOPS_RICHMOND,
                           begin_iso: str = BEGIN_ISO,
                           end_iso: str = END_ISO) -> pd.Series:
    """Harmonic tide predictions (MLLW, metric, LST) from CO-OPS."""
    logger.info("Fetching tide predictions — station %s", station)
    chunks = [
        _coops_chunk(b, e, "predictions", station, interval="h")
        for b, e in _annual_chunks(begin_iso, end_iso)
    ]
    series = pd.concat([c for c in chunks if c is not None])
    series.name = f"pred_tide_{station}"
    return series.sort_index()


def fetch_pressure(station: str = COOPS_FORTPOINT,
                   begin_iso: str = BEGIN_ISO,
                   end_iso: str = END_ISO) -> pd.Series:
    """Atmospheric pressure (mb) from CO-OPS."""
    logger.info("Fetching pressure — station %s", station)
    chunks = [
        _coops_chunk(b, e, "air_pressure", station)
        for b, e in _monthly_chunks(begin_iso, end_iso)
    ]
    series = pd.concat([c for c in chunks if c is not None])
    series.name = "pressure_hpa"
    return series.sort_index()


def fetch_sst(station: str = COOPS_RICHMOND,
              begin_iso: str = BEGIN_ISO,
              end_iso: str = END_ISO) -> pd.Series:
    """Sea surface temperature (°C) from CO-OPS Richmond.

    Fort Point SST is intentionally excluded — Bay interior and ocean-entrance
    temperatures diverge significantly in summer, making Fort Point a poor proxy
    for Richmond Reach stratification.

    Gaps > 48h are filled with monthly climatology computed from the same station.
    """
    logger.info("Fetching SST — station %s", station)
    chunks = [
        _coops_chunk(b, e, "water_temperature", station)
        for b, e in _monthly_chunks(begin_iso, end_iso)
    ]
    series = pd.concat([c for c in chunks if c is not None])
    series.name = "sst_c"
    series = series.sort_index()

    # Fill gaps > 48h with monthly climatology
    monthly_clim = series.groupby(series.index.month).transform("median")
    gap_mask    = series.isna()
    gap_lengths = gap_mask.groupby((~gap_mask).cumsum()).transform("sum")
    long_gap    = gap_mask & (gap_lengths > 48)
    series[long_gap] = monthly_clim[long_gap]

    return series


# ── Discharge ─────────────────────────────────────────────────────────────────

def fetch_discharge(site: str = USGS_SAC_SITE,
                    begin_iso: str = BEGIN_ISO,
                    end_iso: str = END_ISO) -> pd.Series:
    """Sacramento River discharge at Freeport (m³/s), resampled to hourly.

    Uses df['00060'] explicitly — df.iloc[:,0] returns site_no (constant ~324160).
    """
    logger.info("Fetching USGS discharge — site %s", site)
    df, _ = nwis.get_iv(sites=site, parameterCd="00060",
                        start=begin_iso, end=end_iso)
    discharge = df["00060"].astype(float) * CFS_TO_CMS
    discharge = discharge.resample("h").mean()
    discharge.name = "discharge_cms"
    # USGS returns UTC; convert to LST (UTC-8)
    discharge.index = discharge.index.tz_localize(None) - pd.Timedelta(hours=8)
    return discharge.sort_index()


# ── Delta outflow ─────────────────────────────────────────────────────────────

def fetch_delta_outflow(begin_iso: str = BEGIN_ISO,
                        end_iso: str = END_ISO) -> pd.Series:
    """CDEC DTO net Delta outflow (m³/s), daily → forward-filled to hourly.

    BAN/TRP CDEC endpoints return empty data; DTO pre-calculates net outflow.
    """
    logger.info("Fetching CDEC delta outflow (DTO)")
    url = (
        f"{CDEC_API}?Stations={CDEC_OUTFLOW_STA}"
        f"&SensorNums={CDEC_OUTFLOW_SENSOR}&dur_code=D"
        f"&Start={begin_iso}&End={end_iso}"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(
        __import__("io").StringIO(resp.text),
        skiprows=1,
        names=["station_id", "duration", "sensor_num", "sensor_type",
               "datetime", "obs_date", "value", "data_flag", "units"],
    )
    df = df.dropna(subset=["datetime", "value"])
    df.index = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna()
    outflow = pd.to_numeric(df["value"], errors="coerce") * CFS_TO_CMS
    outflow.name = "outflow_cms"
    # Resample daily → hourly via forward fill
    outflow = outflow.resample("h").ffill()
    return outflow.sort_index()


# ── Wind ──────────────────────────────────────────────────────────────────────

def fetch_wind(begin_iso: str = BEGIN_ISO,
               end_iso: str = END_ISO) -> pd.DataFrame:
    """ERA5 wind via Open-Meteo archive (no API key).

    Returns DataFrame with columns [wind_u, wind_v, wind_spd_max_d] in LST.
    wind_spd_max_d: daily maximum wind speed for the day of each timestamp
    (captures peak sea breeze regardless of forecast hour).
    """
    logger.info("Fetching ERA5 wind via Open-Meteo archive")
    params = dict(
        latitude=WIND_LAT, longitude=WIND_LON,
        start_date=begin_iso, end_date=end_iso,
        hourly="wind_speed_10m,wind_direction_10m",
        wind_speed_unit="ms", timezone="UTC",
    )
    resp = requests.get(OPENMETEO_ARCHIVE_URL, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df["time"]) - pd.Timedelta(hours=8)  # UTC → LST
    df.index.name = None

    spd = df["wind_speed_10m"].astype(float)
    dir_rad = np.deg2rad(df["wind_direction_10m"].astype(float))
    df["wind_u"] = -(spd * np.sin(dir_rad))
    df["wind_v"] = -(spd * np.cos(dir_rad))
    df["wind_spd_max_d"] = spd.resample("D").max().reindex(df.index, method="ffill")

    return df[["wind_u", "wind_v", "wind_spd_max_d"]].sort_index()


# ── Upwelling index ───────────────────────────────────────────────────────────

def fetch_upwelling(begin_iso: str = BEGIN_ISO,
                    end_iso: str = END_ISO) -> pd.Series:
    """NOAA PFEL coastal upwelling index at 36°N 122°W (6-hourly → daily mean).

    Positive = upwelling-favorable (NW winds). ~1 day lag in source data.
    """
    logger.info("Fetching NOAA PFEL upwelling index")
    url = PFEL_URL.format(begin=begin_iso, end=end_iso)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    df = pd.read_csv(__import__("io").StringIO(resp.text), skiprows=2,
                     names=["time", "upwelling_index"])
    df.index = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None) - pd.Timedelta(hours=8)
    df.index.name = None
    upwelling = pd.to_numeric(df["upwelling_index"], errors="coerce")
    upwelling = upwelling.resample("D").mean()
    upwelling.name = "upwelling_idx"
    return upwelling.sort_index()


# ── HFR surface currents ──────────────────────────────────────────────────────

def _ncei_url(dt: datetime) -> str:
    return (
        f"https://www.ncei.noaa.gov/data/oceans/ioos/hfradar/rtv/"
        f"{dt.strftime('%Y')}/{dt.strftime('%Y%m')}/USWC/"
        f"{dt.strftime('%Y%m%d%H%M')}_hfr_uswc_6km_rtv_uwls_NDBC.nc"
    )


def _extract_uv(ds: xr.Dataset, target_lat: float, target_lon: float,
                radius: float) -> tuple[float, float]:
    """Extract (u, v) from nearest valid HFR pixel within radius degrees."""
    lats = ds["lat"].values
    lons = ds["lon"].values
    u_arr = ds["u"].values.squeeze()
    v_arr = ds["v"].values.squeeze()

    lat_idx = np.where(np.abs(lats - target_lat) <= radius)[0]
    lon_idx = np.where(np.abs(lons - target_lon) <= radius)[0]

    best_u, best_v, best_dist = np.nan, np.nan, np.inf
    for i in lat_idx:
        for j in lon_idx:
            if not (np.isnan(u_arr[i, j]) or np.isnan(v_arr[i, j])):
                dist = (lats[i] - target_lat) ** 2 + (lons[j] - target_lon) ** 2
                if dist < best_dist:
                    best_u, best_v, best_dist = float(u_arr[i, j]), float(v_arr[i, j]), dist
    return best_u, best_v


def _fetch_single_hfr(dt: datetime) -> tuple[datetime, float, float]:
    """Fetch one HFR netCDF file and extract the Richmond Reach pixel. Process-safe."""
    url = _ncei_url(dt)
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return dt, np.nan, np.nan
        fd, fpath = tempfile.mkstemp(suffix=".nc")
        try:
            os.write(fd, resp.content)
            os.close(fd)
            with xr.open_dataset(fpath, engine="netcdf4") as ds:
                u, v = _extract_uv(ds, HFR_LAT, HFR_LON, HFR_SEARCH_RADIUS_DEG)
        finally:
            os.unlink(fpath)
        return dt, u, v
    except Exception:
        return dt, np.nan, np.nan


def fetch_hfr(begin_iso: str = BEGIN_ISO,
              end_iso: str = END_ISO,
              cache_path: str = HFR_CACHE_PATH,
              workers: int = 4) -> pd.DataFrame:
    """Fetch HFR USWC 6km surface currents for Richmond Reach.

    Loads from parquet cache if available. Otherwise fetches from NCEI archive
    using a ProcessPoolExecutor (NOT threads — netcdf4 C library is not thread-safe).

    Returns DataFrame with columns [curr_u, curr_v], hourly LST index.
    """
    if os.path.exists(cache_path):
        logger.info("Loading HFR from cache: %s", cache_path)
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        return df

    logger.info("Fetching HFR from NCEI archive (this takes ~45 min) ...")
    timestamps = pd.date_range(begin_iso, end_iso, freq="h")
    results = {}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_single_hfr, dt.to_pydatetime()): dt for dt in timestamps}
        for i, fut in enumerate(as_completed(futures)):
            dt, u, v = fut.result()
            results[dt] = {"curr_u": u, "curr_v": v}
            if (i + 1) % 1000 == 0:
                logger.info("  HFR: %d / %d fetched", i + 1, len(timestamps))

    df = pd.DataFrame.from_dict(results, orient="index").sort_index()
    df.index.name = None
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path)
    logger.info("HFR cache saved to %s", cache_path)
    return df
