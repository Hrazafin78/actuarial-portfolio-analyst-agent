"""Interface de préparation des portefeuilles IARD.

Orchestre l’import CSV/XLSX, les diagnostics de qualité, le nettoyage choisi
sur une copie et l’export. Conserve les résultats dépendants en session.
"""
from __future__ import annotations

import csv
import io
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from data_cleaner import (
    apply_cleaning_actions,
    generate_cleaning_report,
    suggest_cleaning_actions,
)
from quality_checker import generate_quality_report


_CLEAN_KEYS = (
    "v2_cleaned_df", "v2_cleaning_log", "v2_cleaning_report",
    "v2_quality_after", "v2_suggestions_after", "v2_csv_bytes",
)
_QUALITY_KEYS = ("v2_quality_report", "v2_suggestions",) + _CLEAN_KEYS

# L'analyse descriptive et le chat dépendent des données nettoyées/importées.
# Les invalider ensemble empêche tout dialogue sur un portefeuille obsolète.
_DESCRIPTIVE_KEYS = (
    "v2d_report", "v2d_signature", "v2d_pair_detail",
    "v2_agent_messages", "v2_agent_previous_response_id", "v2_agent_signature",
)


def _forget(*keys: str) -> None:
    """Retire de la session les clés indiquées si elles existent."""
    for key in keys:
        st.session_state.pop(key, None)


def _reset_cleaning() -> None:
    """Invalide le nettoyage et les résultats descriptifs ou IA qui en dépendent."""
    _forget(*_CLEAN_KEYS)
    _forget(*_DESCRIPTIVE_KEYS)
    _reset_report()


def _reset_report() -> None:
    """Retire les brouillons de rapport, exports et signature de configuration courante."""
    for key in list(st.session_state):
        if key.startswith('report_ai_'):
            st.session_state.pop(key, None)
    st.session_state.pop('v2d_current_signature', None)


def _reset_quality() -> None:
    """Invalide le diagnostic, le nettoyage et les résultats dépendants de la base."""
    _forget(*_QUALITY_KEYS)
    _forget(*_DESCRIPTIVE_KEYS)
    _reset_report()
    for key in list(st.session_state):
        if key.startswith("v2_action_") or key.startswith("v2_alert_detail_"):
            st.session_state.pop(key, None)


def _reset_import() -> None:
    """Réinitialise la source importée et ses résultats dépendants."""
    _reset_quality()
    _forget("v2_original_df", "v2_file_name", "v2_key_columns")


def _detect_separator(sample: str) -> str:
    """Détecte un séparateur courant, avec repli sur la virgule."""
    lines = "\n".join(sample.splitlines()[:12])
    try:
        return csv.Sniffer().sniff(lines, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _read_file(uploaded, delimiter_option: str) -> pd.DataFrame:
    """Lit un CSV ou XLSX en mémoire, sans sauvegarder le fichier importé."""
    raw = uploaded.getvalue()
    if not raw:
        raise ValueError("Le fichier est vide.")
    extension = Path(uploaded.name).suffix.lower()

    if extension == ".xlsx":
        # Pandas a besoin de openpyxl pour lire les fichiers .xlsx.
        data = pd.read_excel(io.BytesIO(raw))
    elif extension == ".csv":
        # Vérifier le décodage du FICHIER ENTIER : l'échantillon seul peut
        # être en ASCII, alors que des accents apparaissent plus loin.
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                sample = raw[:65536].decode(encoding)
                separator = (
                    _detect_separator(sample)
                    if delimiter_option == "Automatique"
                    else {"Point-virgule (;)": ";", "Virgule (,)": ",",
                          "Tabulation": "\t", "Barre verticale (|)": "|"}[delimiter_option]
                )
                # low_memory=False stabilise la déduction des types.
                data = pd.read_csv(
                    io.BytesIO(raw), sep=separator, encoding=encoding,
                    low_memory=False,
                )
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Impossible de décoder le fichier CSV.")
    else:
        raise ValueError("Format non pris en charge (CSV ou XLSX uniquement).")

    if data.empty and not len(data.columns):
        raise ValueError("Le fichier ne contient aucune donnée exploitable.")
    return data


_ACTION_LABELS = {
    "fill_unknown": "Remplacer les valeurs manquantes par « Inconnu »",
    "fill_median": "Remplacer les valeurs manquantes par la médiane",
    "drop_missing_rows": "Supprimer les lignes concernées par les valeurs manquantes",
    "drop_column": "Supprimer la colonne entièrement vide",
    "drop_exact_duplicates": "Supprimer les doublons exacts",
    "convert_numeric": "Convertir les valeurs en numérique",
    "trim_column_names": "Supprimer les espaces des noms de colonnes",
}
_UNIT_LABELS = {"values": "valeurs", "rows": "lignes", "columns": "colonnes"}
_SIGNAL_LABELS = {
    "rare_categories": "Modalités rares",
    "high_cardinality": "Nombre élevé de modalités",
    "constant": "Variable constante",
    "constant_variable": "Variable constante",
    "near_constant": "Variable quasi constante",
    "quasi_constant": "Variable quasi constante",
    "high_correlation": "Corrélation élevée",
    "high_correlations": "Corrélations élevées",
}
_SECTION_LABELS = {
    "outliers": "Valeurs atypiques",
    "statistical_issues": "Points d'attention statistiques",
    "actuarial_consistency": "Contrôles actuariels",
    "duplicates": "Doublons sur clé métier",
    "structure": "Structure",
    "data_types": "Types de données",
}


def _friendly_label(value: object) -> str:
    """Traduit un signal connu ou rend son identifiant lisible en solution de repli."""
    raw = str(value)
    return _SIGNAL_LABELS.get(raw, raw.replace("_", " ").strip().capitalize())


def _format_french_number(value: object) -> str:
    """Formate les effectifs et valeurs de remplissage sans conversion en pourcentage."""
    if isinstance(value, int):
        return f"{value:,}".replace(",", " ")
    if isinstance(value, float):
        # Les valeurs de remplissage sont des nombres observés, pas des pourcentages.
        return f"{value:,.6g}".replace(",", " ").replace(".", ",")
    return str(value)


def _cleaning_log_table(log: list[dict]) -> pd.DataFrame:
    """Transforme le journal technique en tableau compréhensible par l'utilisateur."""
    rows = []
    for op in log:
        details = op.get("details") or {}
        extra = ""
        if "fill_value" in details:
            extra = f"Valeur utilisée : {_format_french_number(details['fill_value'])}"
        elif "renamed_columns" in details:
            extra = f"{len(details['renamed_columns'])} colonne(s) renommée(s)"
        rows.append({
            "Variable": op.get("column") or "Toutes les colonnes",
            "Correction appliquée": _ACTION_LABELS.get(op["action"], _friendly_label(op["action"])),
            "Éléments affectés": op["affected"],
            "Unité": _UNIT_LABELS.get(op["unit"], op["unit"]),
            "Précision": extra,
        })
    return pd.DataFrame(rows)


def _review_table(records: list[dict]) -> pd.DataFrame:
    """Synthèse d'alertes réelles : elles ne représentent pas toutes des erreurs."""
    rows = []
    for item in records:
        details = item.get("details") or {}
        diagnostic = "À interpréter selon le contexte du portefeuille"
        if isinstance(details, dict):
            if "outlier_count" in details:
                count = _format_french_number(int(details["outlier_count"]))
                pct = details.get("outlier_pct")
                if pct is None:
                    suffix = ""
                elif pct == 0 and int(details["outlier_count"]) > 0:
                    suffix = " (< 0,01 %)"  # pourcentage arrondi à zéro par le diagnostic qualité
                else:
                    suffix = f" ({str(pct).replace('.', ',')} %)"
                diagnostic = f"{count} valeur(s) atypique(s){suffix} — critère IQR"
            elif isinstance(details.get("issues"), list) and details["issues"]:
                flags = [_friendly_label(x) for x in details["issues"]]
                diagnostic = ", ".join(flags)
                if "rare_categories" in details["issues"] and "rare_count" in details:
                    n = details["rare_count"]
                    diagnostic += f" ({n} modalité(s) rare(s))"
            elif "invalid_values" in details:
                diagnostic = f"{details['invalid_values']} valeur(s) non convertible(s) sans perte"
        name = (details.get("variable") or item["issue"]) if isinstance(details, dict) else item["issue"]
        rows.append({"Variable / signal": name, "Diagnostic": diagnostic})
    return pd.DataFrame(rows)


def _review_signatures(records: list[dict]) -> set[tuple[str, str, str]]:
    """Compare les types de signaux, sans confondre leur nombre et leur identité."""
    signatures = set()
    for item in records:
        details = item.get("details") or {}
        variable = (
            str(details.get("variable") or item["issue"])
            if isinstance(details, dict) else str(item["issue"])
        )
        flags = details.get("issues") if isinstance(details, dict) else None
        for flag in flags if isinstance(flags, list) and flags else [""]:
            signatures.add((item["section"], variable, str(flag)))
    return signatures


def render_v2_page() -> None:
    """Orchestre l’import, le diagnostic et le nettoyage choisi depuis Streamlit.

    Stocke les résultats uniquement après succès et conserve la source
    originale lorsque des corrections sont appliquées sur une copie.
    """
    st.title("Qualité & Nettoyage")
    st.write(
        "Importez un portefeuille IARD, identifiez les problèmes de qualité "
        "et choisissez les corrections à appliquer."
    )
    st.caption(
        "Les alertes « À examiner » sont informatives : "
        "aucune correction n'est proposée pour ces alertes."
    )

    with st.expander('Importer ou remplacer un portefeuille', expanded=st.session_state.get('v2_original_df') is None):
        st.subheader("1. Importer une base")
        uploaded = st.file_uploader(
            "Fichier CSV ou Excel (.xlsx)",
            type=["csv", "xlsx"],
            key="v2_upload",
            on_change=_reset_import,
        )
        delimiter = st.selectbox(
            "Séparateur CSV", options=[
                "Automatique", "Point-virgule (;)", "Virgule (,)",
                "Tabulation", "Barre verticale (|)",
            ], key="v2_delimiter", on_change=_reset_import,
            help="Sans effet pour les fichiers Excel. Modifiez-le si les colonnes du CSV sont mal séparées.",
        )

        if uploaded is not None and st.button("Charger la base", type="primary"):
            try:
                with st.spinner("Lecture du fichier..."):
                    imported = _read_file(uploaded, delimiter)
                _reset_import()
                st.session_state["v2_original_df"] = imported
                st.session_state["v2_file_name"] = uploaded.name
                st.success("Fichier chargé. La base originale est conservée.")
            except Exception as exc:
                st.error(f"Import impossible : {exc}")

    df = st.session_state.get("v2_original_df")
    if df is None:
        st.info("Chargez un fichier pour commencer le contrôle qualité.")
        return

    st.caption(f"Fichier actif : {st.session_state['v2_file_name']}")
    a, b, c = st.columns(3)
    a.metric("Observations", _format_french_number(len(df)))
    b.metric("Variables", len(df.columns))
    c.metric("Valeurs manquantes", _format_french_number(int(df.isna().sum().sum())))
    with st.expander("Aperçu de la base importée"):
        st.dataframe(df.head(20), use_container_width=True)

    st.subheader("2. Contrôle qualité")
    key_columns = st.multiselect(
        "Clé métier (facultative)",
        options=list(df.columns),
        key="v2_key_columns",
        on_change=_reset_quality,
        help="Exemple pour la base espagnole : insured_id et year. "
             "Laissez vide si la clé métier n'est pas connue.",
    )
    if st.button("Contrôler la qualité"):
        try:
            with st.spinner("Contrôle qualité en cours..."):
                report = generate_quality_report(
                    df, key_columns=key_columns or None,
                )
                suggestions = suggest_cleaning_actions(df, report)
            _reset_quality()
            st.session_state["v2_quality_report"] = report
            st.session_state["v2_suggestions"] = suggestions
        except Exception as exc:
            st.error(f"Contrôle qualité impossible : {exc}")

    suggestions = st.session_state.get("v2_suggestions")
    if suggestions is None:
        st.info("Lancez le contrôle qualité pour obtenir les propositions et alertes.")
        return

    proposals = suggestions["proposals"]
    reviews = suggestions["to_review"]
    m1, m2 = st.columns(2)
    m1.metric("Corrections proposées", len(proposals))
    m2.metric("Points d’attention à examiner", len(reviews))

    st.subheader("3. Corrections proposées")
    if not proposals:
        st.success("Aucune correction proposée pour ce contrôle qualité.")
    selections = {}
    for item in proposals:
        choices = {opt["id"]: opt["label"] for opt in item["options"]}
        option = st.selectbox(
            label=item["issue"],
            options=list(choices),
            format_func=lambda x, labels=choices: labels[x],
            key="v2_action_" + item["id"],
            on_change=_reset_cleaning,
        )
        selections[item["id"]] = option
    n_selected = sum(opt != "keep" for opt in selections.values())
    st.caption(f"{n_selected} correction(s) sélectionnée(s) — aucune n'est appliquée automatiquement.")

    st.subheader("4. À examiner")
    st.info(
        "Ces points d’attention sont des diagnostics à interpréter, pas forcément "
        "des erreurs de données. Aucune correction n'est proposée ni appliquée."
    )
    if not reviews:
        st.success("Aucun point d’attention à afficher.")
    groups = Counter(item["section"] for item in reviews)
    for section, count in groups.items():
        items = [item for item in reviews if item["section"] == section]
        with st.expander(f"{_SECTION_LABELS.get(section, section)} ({count})"):
            if section == "outliers":
                st.caption("Une valeur atypique selon l’IQR peut être légitime, "
                           "notamment en présence de sinistres importants.")
            st.dataframe(_review_table(items), hide_index=True, use_container_width=True)
            index = st.selectbox(
                "Afficher le détail d'un point d’attention",
                options=list(range(len(items))),
                format_func=lambda i, group=items: group[i]["issue"],
                key=f"v2_alert_detail_{section}",
            )
            st.write(items[index].get("details") or "Aucun détail supplémentaire.")
            st.caption("Diagnostic informatif : aucune correction proposée.")

    st.subheader("5. Appliquer les corrections")
    if st.button(
        "Appliquer les corrections sélectionnées",
        type="primary", disabled=(n_selected == 0),
    ):
        _reset_cleaning()  # Aucun ancien résultat ne doit survivre à un échec.
        try:
            with st.spinner("Nettoyage et nouveau contrôle qualité..."):
                cleaned, log = apply_cleaning_actions(df, selections, suggestions)
                report = generate_cleaning_report(df, cleaned, log)

                # Si les noms de colonnes ont changé, adapter la clé métier.
                keys_after = [
                    col.strip() if selections.get("columns::trim") == "trim_column_names"
                    and isinstance(col, str) else col
                    for col in key_columns
                ]
                keys_after = keys_after if all(k in cleaned.columns for k in keys_after) else None
                quality_after = generate_quality_report(
                    cleaned, key_columns=keys_after or None,
                )
                suggestions_after = suggest_cleaning_actions(cleaned, quality_after)

            # N'enregistrer le résultat qu'après la réussite de toutes les étapes.
            st.session_state["v2_cleaned_df"] = cleaned
            st.session_state["v2_cleaning_log"] = log
            st.session_state["v2_cleaning_report"] = report
            st.session_state["v2_quality_after"] = quality_after
            st.session_state["v2_suggestions_after"] = suggestions_after
            st.session_state.pop("v2_csv_bytes", None)
            st.success("Nettoyage effectué sur une copie ; la base importée est inchangée.")
        except Exception as exc:
            st.error(f"Nettoyage ou recontrôle impossible : {exc}")

    cleaned = st.session_state.get("v2_cleaned_df")
    if cleaned is None:
        return

    final_suggestions = st.session_state["v2_suggestions_after"]
    report = st.session_state["v2_cleaning_report"]
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Lignes avant", _format_french_number(report["original_rows"]))
    r2.metric("Lignes après", _format_french_number(report["cleaned_rows"]))
    r3.metric("Corrections restantes", len(final_suggestions["proposals"]))
    r4.metric("Points à examiner", len(final_suggestions["to_review"]))

    n_missing_before = int(df.isna().sum().sum())
    n_missing_after = int(cleaned.isna().sum().sum())
    st.caption(
        f"Valeurs manquantes : {_format_french_number(n_missing_before)} avant "
        f"→ {_format_french_number(n_missing_after)} après nettoyage. "
        "Une correction des manquants n'implique pas nécessairement "
        "la correction de tous les points d’attention."
    )

    st.write("**Journal des modifications**")
    log = st.session_state["v2_cleaning_log"]
    st.dataframe(_cleaning_log_table(log), hide_index=True, use_container_width=True)

    before_alerts = len(reviews)
    after_alerts = len(final_suggestions["to_review"])
    appeared = _review_signatures(final_suggestions["to_review"]) - _review_signatures(reviews)
    st.caption(
        f"Points d’attention : {before_alerts} avant → {after_alerts} après. "
        "Les seuils et distributions statistiques peuvent évoluer après un nettoyage. "
        f"{len(appeared)} nouveau(x) type(s) de signal(s) détecté(s). "
        "Aucune correction n'est appliquée à ces signaux."
    )

    # Prépare une seule fois l'export, sans modifier df ni cleaned.
    if "v2_csv_bytes" not in st.session_state:
        st.session_state["v2_csv_bytes"] = cleaned.to_csv(
            index=False, sep=";",
        ).encode("utf-8-sig")
    st.download_button(
        "Télécharger la base nettoyée (CSV)",
        data=st.session_state["v2_csv_bytes"],
        file_name="portefeuille_nettoye.csv",
        mime="text/csv",
    )
