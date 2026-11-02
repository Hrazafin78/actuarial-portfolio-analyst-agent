"""Outils autorisés de consultation du rapport descriptif.

Déclare les schémas de tool calling et contrôle la sélection, le tri et la
pagination des agrégats. N’expose ni lignes individuelles ni exécution de code.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from descriptive_analyzer import build_llm_context


def _clean(value: Any) -> Any:
    """Transforme les agrégats pandas/numpy en objets JSON compatibles."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if value is None or isinstance(value, (str, bool)):
        return value
    if pd.isna(value):
        return None
    return str(value)


def _rows(table: pd.DataFrame, *, limit: int = 100, offset: int = 0) -> dict:
    """Retourne une page d’agrégats avec colonnes, effectif total et position suivante.

    Valide une taille comprise entre une et cent lignes et un décalage positif ou nul.
    """
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit doit être un entier entre 1 et 100")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset doit être un entier positif ou nul")
    view = table.iloc[offset:offset + limit]
    return {
        "rows": _clean(view.to_dict(orient="records")),
        "returned_rows": len(view),
        "total_rows": len(table),
        "truncated": offset + len(view) < len(table),
        "offset": offset,
        "next_offset": offset + len(view) if offset + len(view) < len(table) else None,
        "columns": list(table.columns),
    }


TOOLS_V2 = [
    {
        "type": "function", "name": "get_portfolio_overview",
        "description": "Retourne une synthèse générale du portefeuille et ses indicateurs actuariels disponibles.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function", "name": "get_variable_statistics",
        "description": "Statistiques descriptives et, si disponibles, répartition des modalités d'une variable précise.",
        "parameters": {"type": "object", "properties": {
            "variable": {"type": "string", "description": "Nom exact d'une colonne du portefeuille."}
        }, "required": ["variable"], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function", "name": "get_segment_analysis",
        "description": "Indicateurs et parts du portefeuille par segment d'une variable explicative, pré-calculés en Python.",
        "parameters": {"type": "object", "properties": {
            "variable": {"type": "string", "description": "Nom exact de la variable de segmentation."}
        }, "required": ["variable"], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function", "name": "get_associations",
        "description": "Renvoie le classement des associations statistiques calculées (Pearson, V de Cramér, eta²).",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function", "name": "get_temporal_analysis",
        "description": "Renvoie les statistiques annuelles disponibles pour une variable temporelle précise.",
        "parameters": {"type": "object", "properties": {
            "variable": {"type": "string", "description": "Nom exact de la variable temporelle."}
        }, "required": ["variable"], "additionalProperties": False},
        "strict": True,
    },
]


def _extend_tools() -> None:
    """Enrichit les schémas stricts avec paramètres nullables et outil de catalogue."""
    page = {
        "limit": {"type": ["integer", "null"], "description": "Taille de page de 1 à 100 ; null = 100."},
        "offset": {"type": ["integer", "null"], "description": "Position de départ ; null = 0."},
    }
    for tool in TOOLS_V2:
        if tool["name"] == "get_portfolio_overview":
            continue
        props = tool["parameters"]["properties"]
        props.update(page)
        if tool["name"] == "get_associations":
            props.update({
                "variable": {"type": ["string", "null"], "description": "Filtrer les associations impliquant cette variable ; null = toutes."},
                "other_variable": {"type": ["string", "null"], "description": "Seconde variable pour une paire exacte, dans les deux sens ; null = toutes."},
                "method": {"type": ["string", "null"], "enum": [None, "Pearson", "V de Cramér corrigé", "Eta²"]},
            })
        if tool["name"] in {"get_segment_analysis", "get_temporal_analysis"}:
            props.update({
                "segments": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Libellés exacts à comparer ; null = tous. Aucun filtrage des données brutes."},
                "sort_by": {"type": ["string", "null"], "description": "Colonne numérique de tri ; consulter list_available_analyses. null = ordre existant."},
                "descending": {"type": ["boolean", "null"], "description": "Tri décroissant ; null = true."},
                "min_observations": {"type": ["integer", "null"], "description": "Effectif minimum ; null = aucun seuil."},
                "min_exposure": {"type": ["number", "null"], "description": "Exposition minimum ; null = aucun seuil."},
            })
        tool["parameters"]["required"] = list(props)
    TOOLS_V2.append({
        "type": "function", "name": "list_available_analyses",
        "description": "Inventaire complet : noms et rôles des variables, colonnes des segmentations et analyses temporelles, méthodes, mapping et limites. Consulter avant une exploration précise.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    })


_extend_tools()


def _page(args: dict) -> dict:
    """Remplace les paramètres de pagination absents ou nuls par leurs valeurs par défaut."""
    return {"limit": 100 if args.get("limit") is None else args["limit"],
            "offset": 0 if args.get("offset") is None else args["offset"]}


def _select_groups(table: pd.DataFrame, args: dict) -> pd.DataFrame:
    """Sélectionne les libellés et applique seuils et tri avant pagination.

    Travaille sur les groupes existants ; ne filtre pas les contrats et ne
    recalcule pas les parts du portefeuille. Refuse les paramètres incompatibles.
    """
    selected = table
    labels = args.get("segments")
    if labels is not None:
        if not isinstance(labels, list) or any(not isinstance(x, str) for x in labels):
            raise ValueError("segments doit être une liste de libellés exacts")
        missing = set(labels) - set(table["segment"].astype(str))
        if missing:
            raise ValueError(f"Segments inconnus : {sorted(missing)}")
        selected = selected.loc[selected["segment"].astype(str).isin(labels)]
    for arg, column in (("min_observations", "observations"), ("min_exposure", "total_exposure")):
        threshold = args.get(arg)
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
                raise ValueError(f"{arg} doit être un nombre fini positif ou nul")
            if arg == "min_observations" and int(threshold) != threshold:
                raise ValueError("min_observations doit être entier")
            if column not in selected:
                raise ValueError(f"Seuil indisponible : {column} absent")
            selected = selected.loc[selected[column] >= threshold]
    sort_by = args.get("sort_by")
    if sort_by is not None:
        if sort_by not in selected or not pd.api.types.is_numeric_dtype(selected[sort_by]):
            raise ValueError(f"Tri numérique indisponible : {sort_by}")
        descending = args.get("descending")
        if descending is not None and type(descending) is not bool:
            raise ValueError("descending doit être booléen")
        selected = selected.sort_values(sort_by, ascending=descending is False, na_position="last", kind="stable")
    return selected


def execute_tool_v2(report: dict, name: str, args: dict) -> dict:
    """Routeur en liste blanche : accès aux seuls agrégats du rapport V2."""
    if not isinstance(args, dict):
        raise ValueError("Les arguments doivent être un objet JSON")
    if name == "list_available_analyses":
        return _clean({
            "variables": report["roles"], "actuarial_mapping": report["actuarial_mapping"],
            "segments": {v: list(t.columns) for v, t in report["segments"].items()},
            "temporal": {v: list(t.columns) for v, t in report["temporal"].items()},
            "warnings": report["warnings"],
            "association_skipped": report["associations"]["skipped"],
            "limits": ["Agrégats du rapport courant uniquement ; aucun filtre sur les lignes brutes ni nouvelle segmentation.",
                       "Les distributions catégorielles sont limitées au top configuré ; Autres peut regrouper les modalités restantes.",
                       "Les parts sont celles du portefeuille complet, même après filtrage des segments.",
                       "Les associations excluent les variables du mapping actuariel et ne démontrent pas de causalité."],
        })
    if name == "get_portfolio_overview":
        return _clean(build_llm_context(report, max_items=10))

    if name == "get_associations":
        table = report["associations"]["strongest"]
        for key in ("variable", "other_variable"):
            variable = args.get(key)
            if variable is not None:
                if variable not in report["roles"]:
                    raise ValueError(f"Variable inconnue : {variable}")
                table = table.loc[(table["variable_1"] == variable) | (table["variable_2"] == variable)]
        if args.get("other_variable") is not None and args.get("variable") is None:
            raise ValueError("other_variable nécessite variable")
        if args.get("other_variable") == args.get("variable") and args.get("variable") is not None:
            raise ValueError("Choisir deux variables distinctes")
        method = args.get("method")
        if method is not None:
            if method not in {"Pearson", "V de Cramér corrigé", "Eta²"}:
                raise ValueError("Méthode inconnue")
            table = table.loc[table["method"] == method]
        return {**_rows(table, **_page(args)), "skipped": report["associations"]["skipped"],
                "note": "Aucune ligne signifie association indisponible, pas absence d'association."}

    if name not in {"get_variable_statistics", "get_segment_analysis", "get_temporal_analysis"}:
        raise ValueError(f"Outil non autorisé : {name}")

    variable = args.get("variable")
    if not isinstance(variable, str) or variable not in report["roles"]:
        return {"error": "Variable inconnue", "available_variables": list(report["roles"])}

    if name == "get_variable_statistics":
        role = report["roles"][variable]
        if role == "numeric":
            table = report["numeric_stats"]
        elif role in {"categorical", "identifier"}:
            table = report["categorical_stats"]
        else:
            table = report["overview"]["columns"]
        stats = table.loc[table["variable"] == variable] if "variable" in table.columns else table.head(0)
        result = {"variable": variable, "role": role, "statistics": _rows(stats)}
        if variable in report["category_distributions"]:
            result["distribution"] = _rows(report["category_distributions"][variable], **_page(args))
            result["distribution_note"] = "Top des modalités du rapport ; les modalités restantes peuvent être regroupées dans Autres. La pagination ne récupère pas leurs détails."
        return result

    source = report["segments"] if name == "get_segment_analysis" else report["temporal"]
    if variable not in source:
        return {"error": "Analyse indisponible pour cette variable", "available_variables": list(source)}
    table = source[variable]
    selected = _select_groups(table, args)
    return {"variable": variable, **_rows(selected, **_page(args)), "source_rows": len(table),
            "selection": _clean(args),
            "note": "Parts calculées sur le portefeuille complet ; taux non définis renvoyés à null. Les tranches numériques sont celles du rapport."}
