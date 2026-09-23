"""Construct daily EHCI and prepare STPM/E-SVD predictand matrices.

EHCI is constructed directly from ``Tmax_5Day_filt.nc`` using the full
available period.  The Y-matrix climatology, standard deviation and optional
linear trend are still fitted using 1962--2006 only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from climkit.masked import masked


# ------------------------------ User configuration -------------------------
TRAIN_START, TRAIN_END = 1962, 2006
TEST_START, TEST_END = 2007, 2022
DETREND = False  # Main experiment; True is a training-only detrending sensitivity test.
PYFILE = Path("/Volumes/TiPlus7100/PyFile")
DATA_DIR = Path("/Volumes/TiPlus7100/data")
THRESHOLD_QUANTILE = 0.90
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "predictand_output"
INPUT_VARIABLE = "EHCI"
OUTPUT_NPZ_NAME = "predictand_daily_heat_area.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyfile", type=Path, default=PYFILE,
                        help="Root containing p2/data and map directories.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Root containing CN05.1 data.")
    parser.add_argument("--threshold-quantile", type=float,
                        default=THRESHOLD_QUANTILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--detrend", action="store_true",
                        help="Fit and remove trends using training years only.")
    return parser.parse_args()


def build_daily_ehci(pyfile: Path, data_dir: Path,
                     threshold_quantile: float = 0.90) -> xr.DataArray:
    """按既有定义用完整 Tmax 时段构造逐日 EHCI。

    这里严格采用既有 EHCI 定义：温度阈值 t95 使用 Tmax 文件中的
    全部年份计算。之后的 Y 气候态、标准差和可选趋势仍在主流程中
    只使用 1962--2006 年计算。
    """
    if not 0 < threshold_quantile < 1:
        raise ValueError("threshold_quantile 必须位于 (0, 1) 区间。")

    tmax_path = pyfile / "p2/data/Tmax_5Day_filt.nc"
    threshold_shp = (
        pyfile / "map/地图边界数据/长江区1：25万界线数据集（2002年）/长江区.shp"
    )
    yangtze_shp = pyfile / "map/self/长江_TP/长江_tp.shp"
    cn05_path = data_dir / "CN05.1/2022/CN05.1_Tmax_2022_daily_025x025.nc"
    for required in (tmax_path, threshold_shp, yangtze_shp, cn05_path):
        if not required.exists():
            raise FileNotFoundError(f"缺少构造 EHCI 所需文件：{required}")

    with xr.open_dataarray(tmax_path) as source:
        tmax_5day = source.load()
    if not {"year", "day", "lat", "lon"}.issubset(tmax_5day.dims):
        raise ValueError(f"Tmax 数据维度不符合预期：{tmax_5day.dims}")

    # year 坐标可能是字符串；这里保留 Tmax 文件中的完整年份序列。
    years = pd.to_numeric(tmax_5day.year.values).astype(int)

    # 按既有定义，t95 使用完整 Tmax 文件的所有年份和目标日计算。
    t95 = (
        masked(tmax_5day.sel(day=slice("1", "88")), str(threshold_shp))
        .mean(dim=["year", "day"])
        .quantile(threshold_quantile)
        .item()
    )
    if not np.isfinite(t95):
        raise ValueError(f"全期温度阈值异常：{t95}")

    ehd_binary = masked((tmax_5day - t95 > 0).astype(int), str(yangtze_shp))
    with xr.open_dataset(cn05_path) as cn05:
        if "tmax" not in cn05:
            raise ValueError(f"CN05.1 文件中没有 tmax 变量：{list(cn05.data_vars)}")
        cn05_tmax = cn05["tmax"]
        cn05_grid = xr.ones_like(cn05_tmax)
        zone_stations = masked(
            cn05_grid.sel(time="2022-01-01"),
            str(yangtze_shp),
        ).sum().item()
    if not np.isfinite(zone_stations) or zone_stations <= 0:
        raise ValueError(f"长江流域 CN05.1 格点数异常：{zone_stations}")

    ehci_year_day = (
        ehd_binary.sum(dim=["lat", "lon"], skipna=True) / zone_stations
    ).transpose("year", "day")
    dates = np.concatenate([
        pd.date_range(f"{year}-06-04", periods=ehci_year_day.sizes["day"],
                      freq="D").values
        for year in years
    ])
    ehci = xr.DataArray(
        np.asarray(ehci_year_day.values, dtype=float).reshape(-1),
        coords={"time": dates}, dims="time", name=INPUT_VARIABLE,
        attrs={
            "long_name": "Yangtze River basin extreme heat coverage index",
            "threshold_quantile": threshold_quantile,
            "threshold_period": "all_available_years",
            "temperature_threshold": t95,
            "zone_grid_count": zone_stations,
        },
    )
    return ehci


def ehci_to_frame(ehci: xr.DataArray) -> pd.DataFrame:
    """Convert the constructed EHCI series to a validated daily table."""
    frame = pd.DataFrame({
        "date": pd.to_datetime(ehci.time.values),
        INPUT_VARIABLE: np.asarray(ehci.values, dtype=float),
    }).sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("构造的 EHCI 存在重复日期。")
    if not np.isfinite(frame[INPUT_VARIABLE]).all():
        raise ValueError("构造的 EHCI 包含 NaN 或无穷值。")
    if (frame[INPUT_VARIABLE] < 0).any():
        raise ValueError("构造的 EHCI 包含负值。")
    return frame


def build_matrix(frame: pd.DataFrame, start_year: int, end_year: int) -> tuple[np.ndarray, np.ndarray]:
    """Build year x 62-day matrix and require every target date."""
    labels = pd.date_range("2000-07-01", "2000-08-31", freq="D").strftime("%m-%d").to_numpy()
    years = np.arange(start_year, end_year + 1, dtype=int)
    indexed = frame.set_index("date")[INPUT_VARIABLE]
    matrix = np.empty((len(years), len(labels)), dtype=float)
    for row, year in enumerate(years):
        for col, month_day in enumerate(labels):
            date = pd.Timestamp(f"{year}-{month_day}")
            if date not in indexed.index:
                raise ValueError(f"缺少目标日期：{year}-{month_day}")
            matrix[row, col] = float(indexed.loc[date])
    return years, matrix


def fit_training_trend(years_train: np.ndarray, y_train: np.ndarray,
                       years_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit one trend per forecast day on training years, then evaluate all years."""
    center = float(years_train.mean())
    x_train = years_train.astype(float) - center
    x_all = years_all.astype(float) - center
    coefficients = np.array([np.polyfit(x_train, y_train[:, d], 1)
                             for d in range(y_train.shape[1])])
    slopes, intercepts = coefficients[:, 0], coefficients[:, 1]
    # Remove only the changing (slope) component.  Centering on the training
    # mean year preserves the training climatological level.  The intercept is
    # retained in coefficients for reproducibility but is not subtracted.
    trend = x_all[:, None] * slopes[None, :]
    return trend, coefficients


def matrix_csv(path: Path, years: np.ndarray, matrix: np.ndarray, labels: np.ndarray) -> None:
    frame = pd.DataFrame(matrix, index=years, columns=labels)
    frame.index.name = "year"
    frame.to_csv(path, encoding="utf-8-sig")


def save_plots(output_dir: Path, labels: np.ndarray, climatology: np.ndarray,
               train_anom: np.ndarray, test_anom: np.ndarray) -> None:
    """Save the requested climatology line and two anomaly heatmaps."""
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, climatology, color="#1f4e79", linewidth=2)
    ax.set(title="Training-period daily climatology (1962–2006)",
           xlabel="Forecast day", ylabel="EHCI")
    ax.set_xticks(x[::5])
    ax.set_xticklabels(labels[::5], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "01_training_climatology.png", dpi=220)
    plt.close(fig)

    vmax = max(float(np.max(np.abs(np.r_[train_anom.ravel(), test_anom.ravel()]))), 1e-12)
    for filename, data, title, first_year in [
        ("02_training_anomaly_heatmap.png", train_anom, "Training anomaly (1962–2006)", 1962),
        ("03_test_anomaly_heatmap.png", test_anom, "Independent test anomaly (2007–2022)", 2007),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        image = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          interpolation="nearest")
        ax.set(title=title, xlabel="Forecast day", ylabel="Year")
        ax.set_xticks(x[::5])
        ax.set_xticklabels(labels[::5], rotation=45, ha="right")
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_yticklabels(np.arange(first_year, first_year + data.shape[0]))
        fig.colorbar(image, ax=ax, label="Daily EHCI anomaly")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    detrend = bool(DETREND or args.detrend)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ehci = build_daily_ehci(
        args.pyfile.expanduser().resolve(),
        args.data_dir.expanduser().resolve(),
        args.threshold_quantile,
    )
    frame = ehci_to_frame(ehci)
    frame.to_csv(output_dir / "EHCI_daily_constructed.csv", index=False,
                 encoding="utf-8-sig")
    train_years, y_train_raw = build_matrix(frame, TRAIN_START, TRAIN_END)
    test_years, y_test_raw = build_matrix(frame, TEST_START, TEST_END)
    all_years = np.r_[train_years, test_years]

    # Raw values are retained.  If enabled, the trend is fitted on training data only.
    if detrend:
        trend, trend_parameters = fit_training_trend(train_years, y_train_raw, all_years)
        analysis = np.vstack([y_train_raw, y_test_raw]) - trend
        y_train_analysis, y_test_analysis = analysis[:45], analysis[45:]
    else:
        trend_parameters = np.full((62, 2), np.nan)
        y_train_analysis, y_test_analysis = y_train_raw.copy(), y_test_raw.copy()

    # Leakage barrier: climatology and std are calculated from training only.
    y_climatology = y_train_analysis.mean(axis=0)
    y_std = y_train_analysis.std(axis=0, ddof=1)
    if not np.isfinite(y_std).all() or np.any(y_std <= 0):
        raise ValueError("训练期至少一个 forecast_day 的标准差无效或为零。")
    y_train_anom = y_train_analysis - y_climatology
    y_test_anom = y_test_analysis - y_climatology
    y_train_std = y_train_anom / y_std
    y_test_std = y_test_anom / y_std

    if y_train_raw.shape != (45, 62) or y_test_raw.shape != (16, 62):
        raise AssertionError(f"Y_train={y_train_raw.shape}, Y_test={y_test_raw.shape}")
    labels = pd.date_range("2000-07-01", "2000-08-31", freq="D").strftime("%m-%d").to_numpy()
    matrices = {"Y_train_raw": y_train_raw, "Y_train_anom": y_train_anom,
                "Y_train_std": y_train_std, "Y_test_raw": y_test_raw,
                "Y_test_anom": y_test_anom, "Y_test_std": y_test_std}
    np.savez(output_dir / OUTPUT_NPZ_NAME, **matrices, y_climatology=y_climatology,
             y_std=y_std, train_years=train_years, test_years=test_years,
             forecast_day=labels, detrend=np.array(detrend),
             trend_parameters=trend_parameters,
             threshold_quantile=np.array(args.threshold_quantile),
             temperature_threshold=np.array(ehci.attrs["temperature_threshold"]),
             zone_grid_count=np.array(ehci.attrs["zone_grid_count"]))
    for name, data in matrices.items():
        years = train_years if data.shape[0] == 45 else test_years
        matrix_csv(output_dir / f"{name}.csv", years, data, labels)
    pd.DataFrame({"forecast_day": labels, "y_climatology": y_climatology}).to_csv(
        output_dir / "y_climatology.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"forecast_day": labels, "y_std": y_std}).to_csv(
        output_dir / "y_std.csv", index=False, encoding="utf-8-sig")
    save_plots(output_dir, labels, y_climatology, y_train_anom, y_test_anom)

    print(f"Tmax input: {(args.pyfile / 'p2/data/Tmax_5Day_filt.nc').resolve()}")
    print(f"EHCI temperature threshold (full Tmax period): {ehci.attrs['temperature_threshold']:.6f}")
    print(f"Yangtze grid count: {ehci.attrs['zone_grid_count']}")
    print(f"Output directory: {output_dir}")
    print(f"DETREND: {detrend}")
    print("All climatology/std/trend parameters use 1962–2006 only.")
    for name, data in matrices.items():
        print(f"{name}.shape = {data.shape}")
    print(f"y_climatology.shape = {y_climatology.shape}")
    print(f"y_std.shape = {y_std.shape}")
    print(f"Saved: {output_dir / OUTPUT_NPZ_NAME}")


if __name__ == "__main__":
    main()