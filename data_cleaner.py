"""Nettoyage contrôlé à partir du diagnostic de qualité.

Distingue les corrections proposées des alertes à examiner. Applique uniquement
les choix validés sur une copie du DataFrame et produit un journal des actions.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


_KEEP = "keep"


def _records(value: Any) -> list[dict[str, Any]]:
    """Normalise une section du rapport qui contient des enregistrements."""
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, list):
        return [x if isinstance(x, dict) else {"detail": str(x)} for x in value]
    if isinstance(value, dict):
        return [value]
    return []


def _numeric_conversion(series: pd.Series) -> tuple[pd.Series, int]:
    """Teste la conversion numérique et compte les valeurs renseignées non convertibles.

    Accepte la virgule décimale et les espaces périphériques ; renvoie la série convertie
    et le nombre de pertes potentielles pour décider si une action est admissible.
    """
    raw = series.astype("string").str.strip()
    ambiguous = raw.str.contains(r"[.,].*[.,]", regex=True, na=False)
    normalized = raw.str.replace(",", ".", regex=False)
    result = pd.to_numeric(normalized, errors="coerce")
    bad = series.notna() & (result.isna() | ambiguous | result.isin([float("inf"), -float("inf")]))
    return result, int(bad.sum())


def _id(kind: str, column: Any) -> str:
    """Construit un identifiant d’action distinguant le type et la valeur du nom de colonne."""
    return f"{kind}::{type(column).__name__}:{column!r}"


def suggest_cleaning_actions(
    df: pd.DataFrame,
    quality_report: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Construit deux listes : 'proposals' et 'to_review'.

    Ne modifie jamais df. Les propositions contiennent un identifiant stable,
    un diagnostic et une liste d'options. L'option 'keep' est toujours possible.
    Les éléments 'to_review' ne comportent aucune option de correction.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un DataFrame pandas")
    if not isinstance(quality_report, Mapping):
        raise TypeError("quality_report doit être un dictionnaire")

    proposals: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    unique_columns = df.columns.is_unique

    def propose(pid: str, issue: str, options: list[dict[str, str]], column: Any = None) -> None:
        """Ajoute une correction proposée avec l’option de conservation par défaut."""
        proposals.append({
            "id": pid,
            "column": column,
            "issue": issue,
            "options": [{"id": _KEEP, "label": "Ne rien modifier"}] + options,
        })

    def examine(section: str, issue: str, details: Any = None) -> None:
        """Ajoute un signal informatif sans proposer de correction automatique."""
        review.append({"section": section, "issue": issue, "details": details})

    # Les noms de colonnes en doublon rendent les opérations ciblées ambiguës.
    if not unique_columns:
        examine("structure", "Noms de colonnes dupliqués : intervention manuelle nécessaire")

    if unique_columns and len(df.columns) and all(isinstance(col, str) for col in df.columns):
        stripped = [col.strip() for col in df.columns]
        if stripped != list(df.columns):
            if len(set(stripped)) == len(stripped):
                propose("columns::trim", "Espaces au début ou à la fin des noms de colonnes", [
                    {"id": "trim_column_names", "label": "Supprimer les espaces superflus"}
                ])
            else:
                examine("structure", "La suppression des espaces créerait des noms de colonnes dupliqués")

    # Les diagnostics structurels ne sont pas tous des erreurs réparables.
    structure = quality_report.get("structure", {})
    if isinstance(structure, dict) and structure.get("empty_dataset"):
        examine("structure", "Base de données vide")

    # Contrôles des types : ne proposer la conversion QUE si elle est sans perte.
    # 'mixed_numeric_text' peut comporter du vrai texte : information uniquement.
    types = _records(quality_report.get("data_types"))
    numeric_candidates: set[Any] = set()
    if unique_columns:
        for item in types:
            col, issue = item.get("variable"), item.get("issue")
            if col not in df.columns or issue not in {"numeric_stored_as_text", "mixed_numeric_text"}:
                continue
            converted, invalid = _numeric_conversion(df[col])
            if invalid == 0 and int(converted.notna().sum()) > 0:
                numeric_candidates.add(col)
                propose(_id("type", col), f"{col} : nombres enregistrés comme texte", [
                    {"id": "convert_numeric", "label": "Convertir en numérique (vérifier les identifiants et codes)"}
                ], col)
            else:
                examine("data_types", f"{col} : conversion numérique potentiellement destructive", {
                    "invalid_values": invalid, "diagnostic": issue
                })

    # Les valeurs manquantes peuvent être corrigées de plusieurs façons,
    # mais jamais sans sélection de l'utilisateur.
    if unique_columns:
        for item in _records(quality_report.get("missing_values")):
            col = item.get("variable")
            if col not in df.columns:
                continue
            missing_count = int(df[col].isna().sum())
            if missing_count <= 0:
                continue
            all_missing = missing_count == len(df)
            options: list[dict[str, str]] = []
            if not all_missing:
                options.append({"id": "drop_missing_rows", "label": "Supprimer les lignes où cette valeur manque"})
            else:
                options.append({"id": "drop_column", "label": "Supprimer cette colonne entièrement vide"})

            col_series = df[col]
            if pd.api.types.is_numeric_dtype(col_series) and not pd.api.types.is_bool_dtype(col_series):
                if col_series.notna().any():
                    options.append({"id": "fill_median", "label": "Remplacer les manquants par la médiane"})
            elif (pd.api.types.is_object_dtype(col_series)
                  or isinstance(col_series.dtype, pd.StringDtype)) and col not in numeric_candidates:
                options.append({"id": "fill_unknown", "label": "Remplacer les manquants par « Inconnu »"})

            propose(_id("missing", col), f"{col} : {missing_count} valeur(s) manquante(s)", options, col)

    duplicates = quality_report.get("duplicates", {})
    if isinstance(duplicates, dict):
        n_exact = int(duplicates.get("exact_duplicates", 0) or 0)
        n_key = int(duplicates.get("key_duplicates", 0) or 0)
        if n_exact > 0 and unique_columns:
            propose("duplicates::exact", f"{n_exact} doublon(s) exact(s)", [
                {"id": "drop_exact_duplicates", "label": "Conserver une seule copie de chaque ligne identique"}
            ])
        if n_key > 0:
            examine("duplicates", f"{n_key} doublon(s) sur la clé métier : aucune suppression automatique")

    # Ces alertes restent strictement informatives, mais seules les anomalies
    # effectivement constatées sont montrées (pas toutes les colonnes analysées).
    for item in _records(quality_report.get("outliers")):
        count = item.get("outlier_count", item.get("n_outliers"))
        if count is None:
            # Compatibilité avec un rapport plus ancien ne fournissant qu'un diagnostic.
            if not item.get("issue"):
                continue
            label = str(item.get("variable") or item["issue"])
        else:
            if pd.isna(count) or count <= 0:
                continue
            label = f"{item.get('variable', 'Variable')} : {int(count)} valeur(s) atypique(s)"
        examine("outliers", label, item)

    for item in _records(quality_report.get("statistical_issues")):
        # Rapport réel : une ligne par variable, y compris lorsqu'issues == [].
        issues = item.get("issues")
        if issues is not None:
            if not issues:
                continue
        elif not item.get("issue"):
            continue
        label = item.get("variable") or item.get("issue") or "Alerte statistique"
        examine("statistical_issues", str(label), item)

    for item in _records(quality_report.get("actuarial_consistency")):
        # Ne pas faire remonter une entrée explicitement déclarée sans anomalie.
        if "issues" in item and not item["issues"]:
            continue
        if "issue" in item and not item["issue"]:
            continue
        label = item.get("variable") or item.get("issue") or item.get("detail") or "Contrôle actuariel"
        examine("actuarial_consistency", str(label), item)

    return {"proposals": proposals, "to_review": review}


def apply_cleaning_actions(
    df: pd.DataFrame,
    selections: Mapping[str, str],
    suggestions: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Applique les choix validés sur une copie et retourne (base, journal).

    selections : {id_proposition: id_option}. Absence de choix = aucune action.
    suggestions : résultat de suggest_cleaning_actions(df, quality_report).

    Les choix obsolètes/inconnus sont rejetés. En cas d'erreur, aucune base
    partiellement modifiée n'est retournée à l'appelant.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un DataFrame pandas")
    if not isinstance(selections, Mapping):
        raise TypeError("selections doit être un dictionnaire")
    if not isinstance(suggestions, Mapping):
        raise TypeError("suggestions doit être un dictionnaire")

    proposals = {p["id"]: p for p in suggestions.get("proposals", [])}
    for pid, action in selections.items():
        if pid not in proposals:
            raise ValueError(f"Proposition inconnue : {pid}")
        valid_options = {opt["id"] for opt in proposals[pid]["options"]}
        if action not in valid_options:
            raise ValueError(f"Option invalide pour {pid} : {action}")

    # Les actions sont exécutées dans un ordre fixe, indépendamment de l'UI.
    priority = {"columns": 0, "type": 1, "missing": 2, "duplicates": 3}
    to_apply = [(pid, action) for pid, action in selections.items() if action != _KEEP]
    to_apply.sort(key=lambda pair: priority.get(pair[0].split("::", 1)[0], 99))

    cleaned = df.copy(deep=True)
    log: list[dict[str, Any]] = []
    renames: dict[Any, Any] = {}

    for pid, action in to_apply:
        proposal = proposals[pid]
        col = renames.get(proposal["column"], proposal["column"])
        before = len(cleaned)
        affected = 0
        unit = "values"
        details: dict[str, Any] = {}

        if action == "trim_column_names":
            new_names = [name.strip() for name in cleaned.columns]
            if len(set(new_names)) != len(new_names):
                raise ValueError("Collision de colonnes après renommage")
            renames = dict(zip(cleaned.columns, new_names))
            affected = sum(a != b for a, b in renames.items())
            cleaned.columns = new_names
            unit = "columns"
            details = {"renamed_columns": {old: new for old, new in renames.items() if old != new}}
        elif action == "convert_numeric":
            if col not in cleaned.columns:
                raise ValueError(f"Colonne absente : {col}")
            converted, invalid = _numeric_conversion(cleaned[col])
            if invalid:
                raise ValueError(f"Conversion refusée pour {col} : {invalid} valeurs non convertibles")
            cleaned[col] = converted
            affected = int(converted.notna().sum())
        elif action in {"drop_missing_rows", "drop_column", "fill_median", "fill_unknown"}:
            if col not in cleaned.columns:
                raise ValueError(f"Colonne absente : {col}")
            affected = int(cleaned[col].isna().sum())
            if action == "drop_missing_rows":
                cleaned = cleaned.loc[cleaned[col].notna()].copy()
                unit = "rows"
                if before and cleaned.empty:
                    raise ValueError(f"Suppression refusée : {col} viderait toute la base")
            elif action == "drop_column":
                if not cleaned[col].isna().all():
                    raise ValueError(f"Suppression refusée : {col} n'est pas entièrement vide")
                cleaned = cleaned.drop(columns=[col])
                affected = 1
                unit = "columns"
            elif action == "fill_median":
                if not pd.api.types.is_numeric_dtype(cleaned[col]):
                    raise ValueError(f"Imputation médiane impossible : {col} n'est pas numérique")
                median = cleaned[col].median()
                if pd.isna(median):
                    raise ValueError(f"Imputation médiane impossible : {col} est vide")
                cleaned[col] = cleaned[col].fillna(median)
                details = {"fill_value": float(median)}
            elif action == "fill_unknown":
                if not (pd.api.types.is_object_dtype(cleaned[col])
                        or isinstance(cleaned[col].dtype, pd.StringDtype)):
                    raise ValueError(f"Remplacement par « Inconnu » incompatible avec le type de {col}")
                cleaned[col] = cleaned[col].fillna("Inconnu")
                details = {"fill_value": "Inconnu"}
        elif action == "drop_exact_duplicates":
            cleaned = cleaned.drop_duplicates().copy()
            affected = before - len(cleaned)
            unit = "rows"
        else:
            raise ValueError(f"Opération non prise en charge : {action}")

        log.append({
            "proposal_id": pid,
            "action": action,
            "column": col,
            "affected": affected,
            "unit": unit,
            "details": details,
            "rows_before": before,
            "rows_after": len(cleaned),
        })

    return cleaned, log


def generate_cleaning_report(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    log: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bilan simple et sérialisable en JSON des modifications appliquées."""
    return {
        "original_rows": int(len(original_df)),
        "cleaned_rows": int(len(cleaned_df)),
        "removed_rows": int(len(original_df) - len(cleaned_df)),
        "original_columns": int(len(original_df.columns)),
        "cleaned_columns": int(len(cleaned_df.columns)),
        "operations_applied": len(log),
        "operations": log,
    }
