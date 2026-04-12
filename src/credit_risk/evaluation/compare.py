from __future__ import annotations

import pandas as pd


def build_model_comparison(results: dict[str, dict]) -> pd.DataFrame:
    """Buuild a comparison tabler from multiple model metric dictionaries.

    Parameters
    ----------
    results: dict[str,dict]
        Dictionary where keys are model names and values are metric dictionaries.

    Returns
    -------
    pd.DataFrame table sorted by roc_auc and f1_fedault descending.
    """

    comparison_df = pd.DataFrame(results).T.reset_index()
    comparison_df = comparison_df.rename(columns={"index": "model"})

    sort_cols = [col for col in ["roc_auc", "f1_default"] if col in comparison_df.columns]
    if sort_cols:
        comparison_df = comparison_df.sort_values(by=sort_cols, ascending=False)
    return comparison_df.reset_index(drop=True)


def add_model_notes(
    comparison_df: pd.DataFrame,
    notes: dict[str, str],
) -> pd.DataFrame:
    """
     Add business or technical notes to the comparison table.

     Parameters
     ----------
     comparison_df : pd.DataFrame
         Comparison table with a 'model' column.
    notes : dict[str, str]
         Dictionary mapping model names to notes.
    Returns
    -------
    pd.DataFrame
         Comparison table with a 'notes' column.
    """

    comparison_df = comparison_df.copy()
    comparison_df["notes"] = comparison_df["model"].map(notes)
    return comparison_df
