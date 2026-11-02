"""Diagnostics déterministes de qualité des données.

Examine structure, valeurs manquantes, doublons, types et signaux statistiques.
Les contrôles actuariels optionnels utilisent un mapping explicite ; aucune
donnée n’est corrigée et aucun modèle IA n’est appelé dans ce module.
"""


import pandas as pd
import json
import numpy as np

def check_structure(df: pd.DataFrame) -> dict:
    """Décrit les dimensions, types et noms de colonnes, y compris leurs doublons."""

    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "duplicate_column_names": (
            df.columns[df.columns.duplicated()].tolist()
        ),
        "empty_dataset": df.empty,
    }

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne les effectifs et pourcentages de manquants, triés par proportion décroissante."""

    result = pd.DataFrame({
        "variable": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_pct": (
            df.isna().mean().values * 100
        ).round(2),
        "non_missing_count": df.notna().sum().values
    })

    return result.sort_values(
        by="missing_pct",
        ascending=False
    ).reset_index(drop=True)


def check_duplicates(
    df: pd.DataFrame,
    key_columns: list[str] | None = None
) -> dict:
    """Compte les lignes répétées au-delà de la première occurrence.

    Le contrôle sur clé est facultatif et exige des colonnes existantes.
    Aucune ligne n’est supprimée.
    """


    exact_duplicates = int(df.duplicated().sum())

    result = {
        "exact_duplicates": exact_duplicates,
        "exact_duplicates_pct": round(
            exact_duplicates / len(df) * 100, 2
        ) if len(df) > 0 else 0,
    }


    if key_columns is not None:

        missing_columns = [
            col for col in key_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Colonnes inexistantes : {missing_columns}"
            )

        key_duplicates = int(
            df.duplicated(subset=key_columns).sum()
        )

        result["key_columns"] = key_columns
        result["key_duplicates"] = key_duplicates
        result["key_duplicates_pct"] = round(
            key_duplicates / len(df) * 100, 2
        ) if len(df) > 0 else 0

    return result

def check_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Signale les colonnes vides et les textes partiellement ou entièrement numériques.

    Ignore les valeurs manquantes et accepte la virgule décimale pour le diagnostic ;
    la conversion testée ne modifie pas les colonnes originales.
    """

    results = []

    for col in df.columns:

        series = df[col].dropna()
        dtype = str(df[col].dtype)

        issue = "none"

        if series.empty:
            issue = "all_missing"

        elif pd.api.types.is_object_dtype(df[col]) or \
             pd.api.types.is_string_dtype(df[col]):


            numeric_series = pd.to_numeric(
                series.astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )

            numeric_ratio = numeric_series.notna().mean()

            if numeric_ratio == 1:
                issue = "numeric_stored_as_text"

            elif numeric_ratio > 0:
                issue = "mixed_numeric_text"

        results.append({
            "variable": col,
            "dtype": dtype,
            "non_missing_count": len(series),
            "issue": issue
        })

    return pd.DataFrame(results)


def check_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Repère les valeurs numériques hors des bornes à 1,5 fois l’écart interquartile.

    Ignore les valeurs non finies et les colonnes à IQR nul. Les pourcentages
    portent sur les valeurs finies renseignées ; une atypie n’est pas une erreur métier.
    """

    results = []

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for col in numeric_columns:

        series = df[col].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        # Si IQR nul, la méthode n'est pas informative
        if iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (
            (series < lower_bound) |
            (series > upper_bound)
        )

        results.append({
            "variable": col,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": int(outliers.sum()),
            "outlier_pct": round(
                outliers.mean() * 100, 2
            )
        })

    return pd.DataFrame(
        results,
        columns=[
            "variable",
            "lower_bound",
            "upper_bound",
            "outlier_count",
            "outlier_pct"
        ]
    )


def check_actuarial_consistency(
    df: pd.DataFrame,
    column_mapping: dict
) -> pd.DataFrame:
    """Signale les valeurs négatives des colonnes désignées par le mapping.

    Chaque catégorie (exposure, claim_count, premium, incurred) reçoit une liste
    de colonnes. Signale aussi les colonnes absentes ou non numériques, sans
    correction ; ce mapping diffère de celui du moteur descriptif.
    """

    if not isinstance(column_mapping, dict):
        raise ValueError(
            "column_mapping doit être un dictionnaire."
        )

    allowed_categories = {
        "exposure", "claim_count", "premium", "incurred"
    }

    for category, columns in column_mapping.items():

        if category not in allowed_categories:
            raise ValueError(
                f"Catégorie actuarielle inconnue : {category}"
            )

        if (
            not isinstance(columns, list)
            or not columns
            or any(
                not isinstance(col, str) or not col.strip()
                for col in columns
            )
        ):
            raise ValueError(
                f"Le mapping '{category}' doit contenir "
                "une liste non vide de noms de colonnes."
            )

    results = []

    descriptions = {
        "exposure": "Exposition négative",
        "claim_count": "Nombre de sinistres négatif",
        "premium": "Prime négative à examiner",
        "incurred": "Coût de sinistre négatif à examiner"
    }


    for category, description in descriptions.items():

        for col in column_mapping.get(category, []):

            status = "checked"
            issue_count = 0

            if col not in df.columns:
                status = "missing_column"

            elif not pd.api.types.is_numeric_dtype(df[col]):
                status = "non_numeric_column"

            else:
                issue_count = int((df[col] < 0).sum())

            results.append({
                "check": f"negative_{category}",
                "variable": col,
                "status": status,
                "issue_count": (
                    issue_count if status == "checked" else None
                ),
                "issue_pct": (
                    round(issue_count / len(df) * 100, 2)
                    if status == "checked" and len(df) > 0
                    else None
                ),
                "description": description
            })

    return pd.DataFrame(
        results,
        columns=[
            "check",
            "variable",
            "status",
            "issue_count",
            "issue_pct",
            "description"
        ]
    )

def check_statistical_issues(
    df: pd.DataFrame,
    dominant_threshold: float = 0.95,
    rare_threshold: float = 0.01,
    cardinality_threshold: int = 50
) -> pd.DataFrame:
    """Détecte constance, dominance, forte cardinalité et modalités rares.

    Les fréquences et seuils portent sur les valeurs renseignées de chaque
    colonne. Les signaux retournés nécessitent une interprétation métier.
    """

    results = []

    for col in df.columns:

        series = df[col].dropna()

        if series.empty:
            continue


        n_unique = series.nunique()


        frequencies = series.value_counts(normalize=True)


        dominant_pct = frequencies.iloc[0] * 100


        rare_count = int(
            (frequencies < rare_threshold).sum()
        )


        issues = []

        if n_unique == 1:
            issues.append("constant_variable")

        elif dominant_pct >= dominant_threshold * 100:
            issues.append("near_constant_variable")

        if n_unique > cardinality_threshold:
            issues.append("high_cardinality")

        if rare_count > 0:
            issues.append("rare_categories")

        results.append({
            "variable": col,
            "n_unique": int(n_unique),
            "dominant_pct": round(dominant_pct, 2),
            "rare_count": rare_count,
            "issues": issues
        })

    return pd.DataFrame(results)


def generate_quality_report(
    df: pd.DataFrame,
    key_columns: list[str] | None = None,
    column_mapping: dict | None = None
) -> dict:
    """Regroupe les diagnostics de qualité dans un dictionnaire sérialisable.

    La clé de doublons et le mapping actuariel sont facultatifs. Ne modifie
    pas le DataFrame ; ajoute les contrôles actuariels seulement si demandés.
    """

    def to_records(dataframe):
        """Convertit un tableau de diagnostic en enregistrements compatibles JSON."""
        return json.loads(
            dataframe.to_json(orient="records")
        )

    report = {
        "structure": check_structure(df),

        "missing_values": to_records(
            check_missing_values(df)
        ),

        "duplicates": check_duplicates(
            df,
            key_columns=key_columns
        ),

        "data_types": to_records(
            check_data_types(df)
        ),

        "outliers": to_records(
            check_outliers(df)
        ),

        "statistical_issues": to_records(
            check_statistical_issues(df)
        )
    }


    if column_mapping is not None:
        report["actuarial_consistency"] = to_records(
            check_actuarial_consistency(
                df,
                column_mapping
            )
        )

    return report
