from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pandas.api import types as ptypes


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


def summarize_dataset(
    df: pd.DataFrame,
    example_values_per_column: int = 3,
) -> DatasetSummary:
    """
    Полный обзор датасета по колонкам:
    - количество строк/столбцов;
    - типы;
    - пропуски;
    - количество уникальных;
    - несколько примерных значений;
    - базовые числовые статистики (для numeric).
    """
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))

        # Примерные значения выводим как строки
        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None

        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица пропусков по колонкам: count/share.
    """
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    total = df.isna().sum()
    share = total / len(df)
    result = (
        pd.DataFrame(
            {
                "missing_count": total,
                "missing_share": share,
            }
        )
        .sort_values("missing_share", ascending=False)
    )
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Корреляция Пирсона для числовых колонок.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Для категориальных/строковых колонок считает top-k значений.
    Возвращает словарь: колонка -> DataFrame со столбцами value/count/share.
    """
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []

    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)

    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": share.values,
            }
        )
        result[name] = table

    return result


def compute_quality_flags(
    summary: DatasetSummary,
    missing_df: pd.DataFrame,
    df: Optional[pd.DataFrame] = None,
    *,
    high_cardinality_threshold: int = 50,
    zero_share_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Набор эвристик «качества» данных.

    Базовые эвристики (были в проекте):
    - too_few_rows
    - too_many_columns
    - max_missing_share / too_many_missing

    Новые эвристики (HW03):
    - has_constant_columns: есть признаки с (почти) константными значениями
    - has_high_cardinality_categoricals: есть категориальные признаки с высокой кардинальностью
    - has_suspicious_id_duplicates: в идентификаторах (колонки вида *_id / id / user_id) есть дубликаты
    - has_many_zero_values: в числовых колонках слишком большая доля нулей
    """
    flags: Dict[str, Any] = {}

    # --- базовые эвристики ---
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5

    # --- эвристика 1: константные колонки ---
    constant_cols: List[str] = []
    for col in summary.columns:
        # summary.unique считается по dropna=True, поэтому:
        # - уникальных 0 -> колонка полностью пустая (считаем деградацией);
        # - уникальных 1 -> по сути константа (даже если есть пропуски).
        if col.unique <= 1:
            constant_cols.append(col.name)

    flags["has_constant_columns"] = len(constant_cols) > 0
    flags["constant_columns"] = constant_cols

    # --- эвристика 2: высокая кардинальность категориальных ---
    high_cardinality: List[Dict[str, Any]] = []
    for col in summary.columns:
        dtype = col.dtype.lower()
        is_cat = ("object" in dtype) or ("category" in dtype)
        if is_cat and col.unique >= high_cardinality_threshold:
            high_cardinality.append({"column": col.name, "unique": col.unique})

    flags["has_high_cardinality_categoricals"] = len(high_cardinality) > 0
    flags["high_cardinality_categoricals"] = high_cardinality
    flags["high_cardinality_threshold"] = high_cardinality_threshold

    # --- эвристика 3: подозрительные дубликаты в ID ---
    suspicious_id_dups: List[Dict[str, Any]] = []
    if df is not None and not df.empty:
        def _is_id_column(name: str) -> bool:
            n = name.lower()
            return n == "id" or n == "user_id" or n.endswith("_id") or n.endswith("id")

        for name in df.columns:
            if not _is_id_column(name):
                continue
            s = df[name]
            non_null = int(s.notna().sum())
            uniq = int(s.nunique(dropna=True))
            if non_null > 0 and uniq < non_null:
                suspicious_id_dups.append(
                    {"column": name, "non_null": non_null, "unique": uniq, "duplicates": non_null - uniq}
                )

    flags["has_suspicious_id_duplicates"] = len(suspicious_id_dups) > 0
    flags["suspicious_id_duplicates"] = suspicious_id_dups

    # --- эвристика 4: слишком много нулей в числовых ---
    zero_heavy: List[Dict[str, Any]] = []
    if df is not None and not df.empty:
        numeric_df = df.select_dtypes(include="number")
        for name in numeric_df.columns:
            s = numeric_df[name]
            denom = int(s.notna().sum())
            if denom == 0:
                continue
            zero_share = float((s == 0).sum() / denom)
            if zero_share >= zero_share_threshold:
                zero_heavy.append({"column": name, "zero_share": zero_share})

    flags["has_many_zero_values"] = len(zero_heavy) > 0
    flags["zero_heavy_numeric"] = zero_heavy
    flags["zero_share_threshold"] = zero_share_threshold

    # --- интегральный score (0..1) ---
    # базовая логика проекта
    score = 1.0
    score -= max_missing_share  # чем больше пропусков, тем хуже
    if flags["too_few_rows"]:
        score -= 0.2
    if flags["too_many_columns"]:
        score -= 0.1

    # дополнительные штрафы за новые эвристики
    if flags["has_constant_columns"]:
        score -= 0.05
    if flags["has_high_cardinality_categoricals"]:
        score -= 0.05
    if flags["has_suspicious_id_duplicates"]:
        score -= 0.05
    if flags["has_many_zero_values"]:
        score -= 0.05

    score = max(0.0, min(1.0, score))
    flags["quality_score"] = score

    return flags


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    """
    Превращает DatasetSummary в табличку для более удобного вывода.
    """
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )
    return pd.DataFrame(rows)
