"""
Módulo de carga de datos.
Carga, normaliza y combina datasets CSV/TSV desde data/raw.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    LABEL_COLUMN,
    LABEL_COLUMN_ALIASES,
    LABEL_MAPPING,
    PROCESSED_DATA_DIR,
    PROCESSED_DATASET_NAME,
    RAW_DATA_DIR,
    TEXT_COLUMN,
    TEXT_COLUMN_ALIASES,
)

_SUPPORTED_EXTENSIONS = ("*.csv", "*.tsv", "*.txt")

logger = logging.getLogger(__name__)


def load_csv(filepath: Path) -> pd.DataFrame:
    """Carga un archivo CSV/TSV manejando distintas codificaciones y archivos sin encabezado."""
    sep = "\t" if filepath.suffix.lower() in (".tsv", ".txt") else ","
    for encoding in ("utf-8", "latin-1", "utf-8-sig"):
        try:
            df = pd.read_csv(filepath, encoding=encoding, sep=sep)
            if detect_text_column(df) is None and detect_label_column(df) is None:
                df_nh = pd.read_csv(filepath, encoding=encoding, sep=sep, header=None)
                col0 = df_nh[0].astype(str).str.lower().str.strip().unique()
                if any(v in LABEL_MAPPING for v in col0):
                    df_nh.columns = (
                        ["label", "message"] if len(df_nh.columns) == 2
                        else [str(i) for i in df_nh.columns]
                    )
                    df = df_nh
            logger.info(f"Cargado {len(df)} filas desde {filepath.name} ({encoding})")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo leer {filepath} con ninguna codificación conocida.")


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    cols_lower = {col.lower().strip(): col for col in df.columns}
    for alias in TEXT_COLUMN_ALIASES:
        if alias.lower() in cols_lower:
            return cols_lower[alias.lower()]
    return None


def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    cols_lower = {col.lower().strip(): col for col in df.columns}
    for alias in LABEL_COLUMN_ALIASES:
        if alias.lower() in cols_lower:
            return cols_lower[alias.lower()]
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    text_col = detect_text_column(df)
    label_col = detect_label_column(df)

    if text_col is None:
        raise ValueError(
            f"No se encontró columna de texto. "
            f"Alias esperados: {TEXT_COLUMN_ALIASES}. "
            f"Columnas disponibles: {list(df.columns)}"
        )
    if label_col is None:
        raise ValueError(
            f"No se encontró columna de etiqueta. "
            f"Alias esperados: {LABEL_COLUMN_ALIASES}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    rename_map: dict[str, str] = {}
    if text_col != TEXT_COLUMN:
        rename_map[text_col] = TEXT_COLUMN
    if label_col != LABEL_COLUMN:
        rename_map[label_col] = LABEL_COLUMN

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"Columnas renombradas: {rename_map}")

    return df[[TEXT_COLUMN, LABEL_COLUMN]].copy()


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    original_labels = df[LABEL_COLUMN].dropna().unique().tolist()
    df = df.copy()
    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN]
        .astype(str)
        .str.lower()
        .str.strip()
        .map(LABEL_MAPPING)
    )
    n_unmapped = df[LABEL_COLUMN].isna().sum()
    if n_unmapped > 0:
        logger.warning(
            f"{n_unmapped} filas tienen etiquetas no reconocidas y serán descartadas. "
            f"Etiquetas originales: {original_labels}"
        )
        df = df.dropna(subset=[LABEL_COLUMN])
    dist = df[LABEL_COLUMN].value_counts().to_dict()
    logger.info(f"Distribución de etiquetas normalizadas: {dist}")
    return df


def validate_dataset(df: pd.DataFrame) -> bool:
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Falta columna requerida: '{TEXT_COLUMN}'")
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Falta columna requerida: '{LABEL_COLUMN}'")
    df_clean = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    if df_clean.empty:
        raise ValueError("El dataset está vacío después del procesamiento.")
    logger.info(f"Dataset válido: {len(df_clean)} filas.")
    return True


def load_and_normalize(filepath: Path) -> pd.DataFrame:
    df = load_csv(filepath)
    df = normalize_columns(df)
    df = normalize_labels(df)
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    df = df[df[TEXT_COLUMN].astype(str).str.strip() != ""]
    return df


def load_multiple_datasets(directory: Optional[Path] = None) -> pd.DataFrame:
    if directory is None:
        directory = RAW_DATA_DIR

    csv_files = sorted(
        f for ext in _SUPPORTED_EXTENSIONS for f in directory.glob(ext)
        if f.name != ".gitkeep"
    )
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV/TSV en {directory}")

    dfs: list[pd.DataFrame] = []
    for csv_file in csv_files:
        try:
            df = load_and_normalize(csv_file)
            df["source"] = csv_file.stem
            dfs.append(df)
        except Exception as exc:
            logger.error(f"No se pudo cargar {csv_file.name}: {exc}")

    if not dfs:
        raise ValueError("Ningún dataset pudo cargarse correctamente.")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(
        f"Dataset combinado: {len(combined)} filas de {len(dfs)} archivo(s). "
        f"Distribución: {combined[LABEL_COLUMN].value_counts().to_dict()}"
    )
    return combined


def save_processed(df: pd.DataFrame, filename: Optional[str] = None) -> Path:
    if filename is None:
        filename = PROCESSED_DATASET_NAME
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset procesado guardado en {output_path}")
    return output_path


def prepare_dataset(output_filename: Optional[str] = None) -> pd.DataFrame:
    df = load_multiple_datasets()
    validate_dataset(df)
    save_processed(df, output_filename)
    return df
