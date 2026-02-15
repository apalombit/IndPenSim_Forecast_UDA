"""Data loading utilities for IndPenSim fermentation dataset."""

import zipfile
from pathlib import Path

import pandas as pd

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Column name mappings for cleaner access
COLUMN_MAP = {
    "Time (h)": "time",
    "Aeration rate(Fg:L/h)": "Fg",
    "Agitator RPM(RPM:RPM)": "RPM",
    "Sugar feed rate(Fs:L/h)": "Fs",
    "Acid flow rate(Fa:L/h)": "Fa",
    "Base flow rate(Fb:L/h)": "Fb",
    "Heating/cooling water flow rate(Fc:L/h)": "Fc",
    "Heating water flow rate(Fh:L/h)": "Fh",
    "Water for injection/dilution(Fw:L/h)": "Fw",
    "Air head pressure(pressure:bar)": "pressure",
    "Dumped broth flow(Fremoved:L/h)": "Fremoved",
    "Substrate concentration(S:g/L)": "S",
    "Dissolved oxygen concentration(DO2:mg/L)": "DO2",
    "Penicillin concentration(P:g/L)": "P",
    "Vessel Volume(V:L)": "V",
    "Vessel Weight(Wt:Kg)": "Wt",
    "pH(pH:pH)": "pH",
    "Temperature(T:K)": "T",
    "Generated heat(Q:kJ)": "Q",
    "carbon dioxide percent in off-gas(CO2outgas:%)": "CO2outgas",
    "PAA flow(Fpaa:PAA flow (L/h))": "Fpaa",
    "PAA concentration offline(PAA_offline:PAA (g L^{-1}))": "PAA_offline",
    "Oil flow(Foil:L/hr)": "Foil",
    "NH_3 concentration off-line(NH3_offline:NH3 (g L^{-1}))": "NH3_offline",
    "Oxygen Uptake Rate(OUR:(g min^{-1}))": "OUR",
    "Oxygen in percent in off-gas(O2:O2  (%))": "O2",
    "Offline Penicillin concentration(P_offline:P(g L^{-1}))": "P_offline",
    "Offline Biomass concentratio(X_offline:X(g L^{-1}))": "X_offline",
    "Carbon evolution rate(CER:g/h)": "CER",
    "Ammonia shots(NH3_shots:kgs)": "NH3_shots",
    "Viscosity(Viscosity_offline:centPoise)": "Viscosity_offline",
    "Fault reference(Fault_ref:Fault ref)": "fault_ref",
    "0 - Recipe driven 1 - Operator controlled(Control_ref:Control ref)": "control_ref",
    "Batch reference(Batch_ref:Batch ref)": "batch_id",
}

# Default data paths (relative to project root)
DEFAULT_ZIP_PATH = PROJECT_ROOT / "data/100_Batches_IndPenSim.zip"
DEFAULT_CSV_PATH = PROJECT_ROOT / "data/Mendeley_data/100_Batches_IndPenSim_V3.csv"
DEFAULT_STATS_PATH = PROJECT_ROOT / "data/Mendeley_data/100_Batches_IndPenSim_Statistics.csv"


def extract_data(zip_path: Path | str = DEFAULT_ZIP_PATH, extract_to: Path | str | None = None) -> None:
    """Extract IndPenSim data from zip archive."""
    zip_path = Path(zip_path)
    if extract_to is None:
        extract_to = PROJECT_ROOT / "data"
    else:
        extract_to = Path(extract_to)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def load_raw_data(csv_path: Path | str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load raw CSV data without processing."""
    return pd.read_csv(csv_path)


def load_process_data(csv_path: Path | str = DEFAULT_CSV_PATH, rename_columns: bool = True) -> pd.DataFrame:
    """Load process variables only (exclude Raman spectra).

    Args:
        csv_path: Path to the CSV file.
        rename_columns: If True, rename columns to short names.

    Returns:
        DataFrame with process variables for all batches.
    """
    df = pd.read_csv(csv_path)

    # Find split point (column "2400" marks start of Raman spectra)
    if "2400" in df.columns:
        split_idx = df.columns.get_loc("2400")
        df = df.iloc[:, :split_idx]

    # Fix column swap issue from original data
    if "2-PAT control(PAT_ref:PAT ref)" in df.columns and "Batch reference(Batch_ref:Batch ref)" in df.columns:
        df = df.rename(columns={
            "2-PAT control(PAT_ref:PAT ref)": "Batch reference(Batch_ref:Batch ref)",
            "Batch reference(Batch_ref:Batch ref)": "2-PAT control(PAT_ref:PAT ref)",
        })

    if rename_columns:
        df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    return df


def load_batches(csv_path: Path | str = DEFAULT_CSV_PATH) -> dict[int, pd.DataFrame]:
    """Load data split by batch into a dictionary.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Dictionary mapping batch_id (1-100) to DataFrame.
    """
    df = load_process_data(csv_path)
    batches = {}
    for batch_id, group in df.groupby("batch_id"):
        batches[int(batch_id)] = group.reset_index(drop=True)
    return batches


def load_statistics(stats_path: Path | str = DEFAULT_STATS_PATH) -> pd.DataFrame:
    """Load batch statistics summary."""
    df = pd.read_csv(stats_path)
    df = df.rename(columns={
        "Batch ref": "batch_id",
        "Penicllin_harvested_during_batch(kg)": "P_harvested_during",
        "Penicllin_harvested_end_of_batch (kg)": "P_harvested_end",
        "Penicllin_yield_total (kg)": "P_yield_total",
        "Fault ref(0-NoFault 1-Fault)": "is_fault",
    })
    return df


def get_batch_info(batches: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Get summary info for each batch.

    Args:
        batches: Dictionary from load_batches().

    Returns:
        DataFrame with batch_id, length, duration, control_mode, is_fault.
    """
    info = []
    for batch_id, df in batches.items():
        length = len(df)
        duration = df["time"].max() - df["time"].min()
        control_ref = df["control_ref"].iloc[0] if "control_ref" in df.columns else None
        is_fault = df["fault_ref"].max() > 0 if "fault_ref" in df.columns else False
        p_conc = df["P"].values[-1] if "P" in df.columns else 0.0

        # Determine control mode based on batch_id
        if batch_id <= 30:
            control_mode = "recipe"
        elif batch_id <= 60:
            control_mode = "operator"
        elif batch_id <= 90:
            control_mode = "apc"
        else:
            control_mode = "fault"

        info.append({
            "batch_id": batch_id,
            "length": length,
            "duration_h": duration,
            "control_mode": control_mode,
            "is_fault": is_fault,
            "p_conc": p_conc
        })

    return pd.DataFrame(info).sort_values("batch_id").reset_index(drop=True)


def get_final_penicillin(batches: dict[int, pd.DataFrame], column: str = "P") -> pd.DataFrame:
    """Get final penicillin concentration for each batch.

    Args:
        batches: Dictionary from load_batches().
        column: Column name for penicillin ('P' or 'P_offline').

    Returns:
        DataFrame with batch_id and final_P.
    """
    results = []
    for batch_id, df in batches.items():
        if column in df.columns:
            final_p = df[column].iloc[-1]
        else:
            final_p = None
        results.append({"batch_id": batch_id, "final_P": final_p})

    return pd.DataFrame(results).sort_values("batch_id").reset_index(drop=True)
