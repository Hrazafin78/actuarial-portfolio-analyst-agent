"""Interface d’exploration statistique et actuarielle.

Configure la source, les rôles et les options ; affiche la vue de résultats
sélectionnée et les exports. Le chat et la rédaction disposent d’espaces séparés.
"""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent_page import render_agent_page, reset_agent_chat

from descriptive_analyzer import (
    ACTUARIAL_FIELDS, AnalysisOptions, ROLES,
    analyze_variable_pair, build_llm_context, detect_variable_roles, generate_descriptive_report,
)

ROLE_LABELS = {
    "numeric": "Numérique", "categorical": "Catégorielle",
    "temporal": "Temporelle", "identifier": "Identifiant", "exclude": "Exclure",
}
ACTUARIAL_LABELS = {
    "claim_count": "Nombre de sinistres", "exposure": "Exposition",
    "premium": "Primes", "claim_amount": "Charge de sinistres",
    "year": "Année / date", "policy_id": "Identifiant de contrat / assuré",
}
METRIC_LABELS = {
    "claim_frequency": "Fréquence (sinistres / exposition)",
    "average_severity": "Coût moyen (charge / nb de sinistres)",
    "pure_premium": "Prime pure (charge / exposition)",
    "loss_ratio": "Ratio S/P (charge / primes)",
    "premium_per_exposure": "Prime / exposition",
    "average_premium_per_row": "Prime / observation",
    "observations": "Nombre d'observations",
    "total_claim_count": "Nombre de sinistres total",
    "total_exposure": "Exposition totale",
    "total_premium": "Primes totales",
    "total_claim_amount": "Charge de sinistres totale",
    "unique_entity_ids": "Identifiants distincts",
}


def _csv_export_bundle(report: dict) -> bytes:
    """Assemble les tableaux descriptifs et avertissements dans une archive ZIP de CSV."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        tables = {
            "structure.csv": report["overview"]["columns"],
            "statistiques_numeriques.csv": report["numeric_stats"],
            "statistiques_categorielles.csv": report["categorical_stats"],
            "associations_pearson.csv": report["associations"]["pearson"].reset_index().rename(columns={"index":"variable"}),
            "associations_cramer.csv": report["associations"]["cramers_v"].reset_index().rename(columns={"index":"variable"}),
            "associations_eta2.csv": report["associations"]["eta_squared"].reset_index().rename(columns={"index":"variable"}),
            "associations_classement.csv": report["associations"]["strongest"],
        }
        for name, tab in report["category_distributions"].items():
            # Les fichiers à l'intérieur de l'archive ont un numéro, pas un nom de colonne arbitraire.
            tables[f"distribution_{list(report['category_distributions']).index(name)+1:03d}.csv"] = tab.assign(variable=name)
        for i, (name, tab) in enumerate(report["segments"].items(), 1):
            tables[f"segments_{i:03d}.csv"] = tab
        for i, (name, tab) in enumerate(report["temporal"].items(), 1):
            tables[f"temps_{i:03d}.csv"] = tab
        for file, table in tables.items():
            z.writestr(file, table.to_csv(index=False, sep=";").encode("utf-8-sig"))
        z.writestr("avertissements.txt", "\n".join(report["warnings"]).encode("utf-8"))
    return buffer.getvalue()


def _matrix_chart(matrix: pd.DataFrame, title: str, signed: bool) -> None:
    """Affiche une matrice d’associations sur une échelle signée ou positive."""
    if matrix.empty:
        st.info(f"{title} : pas assez de variables éligibles.")
        return
    values = matrix.to_numpy(dtype=float)
    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=list(matrix.columns), y=list(matrix.index),
        zmin=-1 if signed else 0, zmax=1,
        colorscale="RdBu" if signed else "Blues",
        hovertemplate="%{y} × %{x} : %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(title=title, height=min(800, max(340, 30 * len(matrix) + 130)),
                      margin=dict(t=60, b=80, l=90, r=40))
    st.plotly_chart(fig, width="stretch")


def _view_univariate(report: dict, df: pd.DataFrame) -> None:
    """Une seule fiche de variable : statistiques adaptées puis visualisation."""
    roles = report["roles"]
    variables = list(df.columns)
    if not variables:
        st.info("Aucune variable à analyser.")
        return

    selected = st.selectbox("Variable à explorer", variables, key="v2d_variable_plot")
    role = roles.get(selected, "exclude")
    st.caption(f"Nature statistique : {ROLE_LABELS.get(role, role)}")

    if role == "numeric":
        stats = report["numeric_stats"]
        row = stats.loc[stats["variable"] == selected] if "variable" in stats.columns else pd.DataFrame()
    elif role in {"categorical", "identifier"}:
        stats = report["categorical_stats"]
        row = stats.loc[stats["variable"] == selected] if "variable" in stats.columns else pd.DataFrame()
    else:
        structure = report["overview"]["columns"]
        row = structure.loc[structure["variable"] == selected]

    st.markdown("#### Statistiques de la variable")
    if row.empty:
        st.info("Aucune statistique disponible pour cette variable.")
    else:
        st.dataframe(row, hide_index=True, use_container_width=True)

    st.markdown("#### Distribution / répartition")
    if role == "numeric":
        x = pd.to_numeric(df[selected], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        x = x[np.isfinite(x)]
        if not x.size:
            st.info("Aucune valeur numérique exploitable pour cette variable.")
            return
        counts, borders = np.histogram(x, bins=min(40, max(1, int(np.sqrt(x.size)))))
        centers = (borders[1:] + borders[:-1]) / 2
        fig = go.Figure(data=go.Bar(x=centers, y=counts, width=np.diff(borders)))
        fig.update_layout(title=f"Distribution : {selected}",
                          xaxis_title=selected, yaxis_title="Nombre d'observations")
        st.plotly_chart(fig, width="stretch")
    elif role == "categorical":
        dist = report["category_distributions"].get(selected)
        if dist is None or dist.empty:
            st.info("Aucune répartition disponible pour cette variable.")
            return
        fig = go.Figure(data=go.Bar(x=dist["modalite"], y=dist["part_pct"],
                                    customdata=dist["effectif"],
                                    hovertemplate="%{x}<br>Part : %{y:.2f} %<br>Effectif : %{customdata:,}<extra></extra>"))
        fig.update_layout(title=f"Répartition : {selected}", xaxis_title=selected,
                          yaxis_title="Part des observations (%)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Les modalités au-delà du Top N sont regroupées sous « Autres modalités ».")
    elif role == "temporal":
        # Une représentation des effectifs par période, sans interpréter les dates.
        counts = df[selected].value_counts(dropna=False)
        try:
            counts = counts.sort_index()
        except TypeError:
            # Certaines colonnes mélangent des périodes et des libellés texte.
            pass
        if counts.empty:
            st.info("Aucune période disponible.")
            return
        if len(counts) > 100:
            st.info("Plus de 100 périodes distinctes : graphique masqué pour rester lisible.")
            return
        labels = ["(Manquant)" if pd.isna(v) else str(v) for v in counts.index]
        fig = go.Figure(data=go.Bar(x=labels, y=counts.to_numpy()))
        fig.update_layout(title=f"Répartition temporelle : {selected}",
                          xaxis_title=selected, yaxis_title="Nombre d'observations")
        st.plotly_chart(fig, width="stretch")
    elif role == "identifier":
        st.info("Identifiant : aucune distribution individuelle affichée (cardinalité potentiellement élevée).")
    else:
        st.info("Variable exclue de l'analyse : modifiez son rôle dans la section 2 pour l'explorer.")


def _view_associations(report: dict, df: pd.DataFrame) -> None:
    """Présente les associations calculées et l’exploration locale d’une paire choisie."""
    st.caption("Toutes les paires de variables explicatives éligibles sont examinées. "
               "Les identifiants, variables temporelles et variables actuarielles désignées sont exclus. "
               "Les associations descriptives n'établissent pas de causalité.")
    matrices = report["associations"]
    _matrix_chart(matrices["pearson"], "Pearson — numérique × numérique", signed=True)
    _matrix_chart(matrices["cramers_v"], "V de Cramér corrigé — catégorielle × catégorielle", signed=False)
    _matrix_chart(matrices["eta_squared"], "η² — catégorielle × numérique", signed=False)
    st.markdown("#### Classement des associations")
    rank = matrices["strongest"]
    st.dataframe(rank, hide_index=True, use_container_width=True)
    eligible = [c for c, role in report["roles"].items()
                if role in {"numeric", "categorical"} and c not in report["actuarial_mapping"].values()]
    if len(eligible) >= 2:
        st.markdown("#### Explorer un couple de variables")
        x = st.selectbox("Première variable", eligible, key="v2d_pair_x")
        y = st.selectbox("Deuxième variable", [c for c in eligible if c != x], key="v2d_pair_y")
        if st.button("Calculer le détail de ce couple"):
            try:
                detail = analyze_variable_pair(df, x, y, report["roles"])
                st.session_state["v2d_pair_detail"] = (x, y, detail)
            except Exception as e:
                st.session_state.pop("v2d_pair_detail", None)
                st.error(str(e))
        detail_saved = st.session_state.get("v2d_pair_detail")
        if detail_saved is not None and detail_saved[:2] == (x,y):
            detail = detail_saved[2]
            if detail["type"] == "numeric_numeric":
                points = detail["plot_data"]
                fig = go.Figure(data=go.Scattergl(x=points[x], y=points[y], mode="markers",
                                                 marker={"size": 5, "opacity": .5}))
                fig.update_layout(title=f"Nuage de points : {x} et {y}", xaxis_title=x, yaxis_title=y)
                st.plotly_chart(fig, width="stretch")
                st.caption(f"Pearson : {detail['correlation']:.3f} ; "
                           f"{detail['plot_sample_size']:,} points affichés sur {detail['observations']:,} paires valides. "
                           "Seul le graphique est échantillonné.")
            elif detail["type"] == "categorical_categorical":
                st.write("Effectifs croisés")
                st.dataframe(detail["counts"], use_container_width=True)
                st.write("Pourcentages par ligne")
                st.dataframe(detail["row_pct"], use_container_width=True)
            else:
                st.dataframe(detail["statistics"], hide_index=True, use_container_width=True)
                plot = detail["plot_data"]
                fig = go.Figure()
                for modalite, rows in plot.groupby("modalite", dropna=False):
                    fig.add_trace(go.Box(y=rows["value"], name=str(modalite), boxpoints=False))
                fig.update_layout(title="Distribution numérique par modalité (échantillon graphique)",
                                  yaxis_title="Valeur")
                st.plotly_chart(fig, width="stretch")
                st.caption(f"Statistiques calculées sur {detail['observations']:,} observations ; "
                           f"boxplots affichés sur un échantillon de {detail['plot_sample_size']:,} observations.")
    if matrices["skipped"]:
        with st.expander(f"Variables non retenues pour certaines associations ({len(matrices['skipped'])})"):
            for line in matrices["skipped"]:
                st.write("•", line)


def _metric_options(tab: pd.DataFrame) -> list[str]:
    """Liste les indicateurs connus qui sont présents dans le tableau choisi."""
    return [k for k in METRIC_LABELS if k in tab.columns]


def _view_actuarial(report: dict) -> None:
    """Affiche les indicateurs disponibles et les comparaisons entre segments existants."""
    overall = report["actuarial_overview"]
    metrics = {k: v for k, v in overall.items() if v is not None and k != "observations"}
    if not metrics:
        st.info("Aucune mesure actuarielle sélectionnée. Les répartitions par segments restent consultables.")
    else:
        headline = [k for k in ("total_exposure", "total_claim_count", "claim_frequency", "average_severity", "loss_ratio") if k in metrics]
        for i in range(0, len(headline), 3):
            columns = st.columns(min(3, len(headline)-i))
            for box, name in zip(columns, headline[i:i+3]):
                value = metrics[name]
                display = f"{value:.2%}" if name in {"loss_ratio", "claim_frequency"} else f"{value:,.3f}"
                box.metric(METRIC_LABELS[name], display)
        with st.expander("Tous les indicateurs actuariels"):
            st.dataframe(pd.DataFrame([{"indicateur": METRIC_LABELS.get(k,k), "valeur":v}
                                       for k,v in overall.items()]), hide_index=True)
    segments = report["segments"]
    if not segments:
        st.info("Aucune variable explicative éligible à segmenter.")
        return
    col = st.selectbox("Variable de segmentation", list(segments), key="v2d_segment_var")
    table = segments[col]
    if table.empty:
        st.info("La table ne contient aucun segment.")
        return
    metrics = _metric_options(table)
    if not metrics:
        st.dataframe(table, hide_index=True, use_container_width=True)
        return
    metric = st.selectbox("Indicateur à comparer", metrics,
                          format_func=lambda k: METRIC_LABELS.get(k,k), key="v2d_segment_metric")
    minimum = st.number_input("Exposition minimale pour l'affichage (facultatif)",
                              min_value=0.0, value=0.0, step=10.0,
                              key="v2d_min_expo", disabled="total_exposure" not in table)
    visible = table
    if "total_exposure" in table and minimum > 0:
        visible = visible.loc[visible["total_exposure"] >= minimum]
    if visible.empty:
        st.info("Aucun segment au-dessus du seuil d'exposition choisi.")
        return
    if len(visible) > 40:
        st.caption("Les 40 segments les plus représentés sont affichés ; le tableau conserve toutes les lignes.")
        visible_graph = visible.nlargest(40, "observations")
    else:
        visible_graph = visible
    is_percent = metric in {"claim_frequency", "loss_ratio"}
    y_values = visible_graph[metric] * 100 if is_percent else visible_graph[metric]
    axis_label = METRIC_LABELS.get(metric, metric) + (" (%)" if is_percent else "")
    fig = go.Figure(data=go.Scatter(
        x=visible_graph["segment"].astype(str), y=y_values,
        mode="lines+markers", line={"width": 2}, marker={"size": 8},
        connectgaps=False,
        hovertemplate="Segment : %{x}<br>Valeur : %{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(title=f"{METRIC_LABELS.get(metric,metric)} selon {col}",
                      xaxis_title=col, yaxis_title=axis_label,
                      xaxis={"type": "category"})
    st.plotly_chart(fig, width="stretch")
    share = "share_exposure_pct" if "share_exposure_pct" in visible_graph else "share_observations_pct"
    fig2 = go.Figure(data=go.Bar(x=visible_graph["segment"], y=visible_graph[share]))
    fig2.update_layout(title=f"Poids du segment — {col}", xaxis_title=col,
                       yaxis_title="Part (%)")
    st.plotly_chart(fig2, width="stretch")
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Les petits segments peuvent produire des taux instables. "
               "Les parts sont calculées sur toute la base ; le filtre n'altère pas les calculs.")


def _view_time(report: dict) -> None:
    """Présente les agrégats annuels disponibles pour la variable temporelle choisie."""
    frames = report["temporal"]
    if not frames:
        st.info("Aucune colonne temporelle exploitable. Vous pouvez modifier son rôle avant l'analyse.")
        return
    var = st.selectbox("Variable temporelle", list(frames), key="v2d_time_var")
    frame = frames[var]
    options = _metric_options(frame)
    if not options:
        st.dataframe(frame, hide_index=True)
        return
    metric = st.selectbox("Indicateur", options, format_func=lambda k: METRIC_LABELS.get(k,k),
                          key="v2d_time_metric")
    graph = frame[frame["segment"] != "(Manquant)"]
    is_percent = metric in {"claim_frequency", "loss_ratio"}
    y_values = graph[metric] * 100 if is_percent else graph[metric]
    axis_label = METRIC_LABELS.get(metric, metric) + (" (%)" if is_percent else "")
    fig = go.Figure(data=go.Scatter(x=graph["segment"], y=y_values, mode="lines+markers"))
    fig.update_layout(title=f"Évolution annuelle — {METRIC_LABELS.get(metric,metric)}",
                      xaxis_title="Année", yaxis_title=axis_label)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(frame, hide_index=True, use_container_width=True)


def render_descriptive_page() -> None:
    """Configure et lance l’analyse, puis affiche la vue de résultats sélectionnée.

    Conserve la source, les paramètres et signatures en session pour détecter
    les résultats obsolètes et partager l’analyse avec les espaces IA.
    """
    st.title("Analyse descriptive")
    st.write("Statistiques complètes, associations, répartition du portefeuille et sinistralité par segment.")
    st.caption("Configurez l’analyse, puis retrouvez ses résultats dans les espaces Assistant IA et Rapports IA.")

    sources = {}
    if st.session_state.get("v2_original_df") is not None:
        sources["Base importée (non nettoyée)"] = st.session_state["v2_original_df"]
    if st.session_state.get("v2_cleaned_df") is not None:
        sources["Base nettoyée"] = st.session_state["v2_cleaned_df"]
    if not sources:
        st.info("Importez un portefeuille dans l’espace Qualité & Nettoyage pour commencer.")
        return
    selected = st.radio("Base à analyser", list(sources), index=len(sources)-1,
                        horizontal=True, key="v2d_source")
    df = sources[selected]
    st.caption(f"{len(df):,} lignes · {len(df.columns)} variables · Source : {selected}")
    token = str(id(df))

    with st.expander('Configuration de l’analyse', expanded=st.session_state.get('v2d_report') is None):
        st.subheader("1. Identification manuelle des variables actuarielles")
        st.caption("Sélectionnez uniquement les colonnes dont vous connaissez la définition. "
                   "L'application ne devine pas leur rôle actuariel.")
        mapping = {}
        mapping_columns = st.columns(3)
        for index, key in enumerate(ACTUARIAL_FIELDS):
            with mapping_columns[index % 3]:
                value = st.selectbox(ACTUARIAL_LABELS[key], options=[None]+list(df.columns),
                                     format_func=lambda v: "Non renseigné" if v is None else str(v),
                                     key=f"v2d_map_{token}_{key}")
                if value is not None:
                    mapping[key] = value

        st.subheader("2. Nature statistique des variables")
        detected = detect_variable_roles(df)
        with st.expander("Vérifier/modifier les rôles (utile pour les codes numériques et les identifiants)"):
            editor_df = pd.DataFrame({"variable": list(detected), "role": list(detected.values())})
            editor = st.data_editor(
                editor_df,
                key=f"v2d_roles_{token}", hide_index=True, use_container_width=True,
                disabled=["variable"],
                column_config={"role": st.column_config.SelectboxColumn(
                    "Rôle statistique", options=list(ROLES), required=True)},
            )
            st.caption("numeric = mesure, categorical = modalité, temporal = temps, identifier = clé, exclude = ignorée.")
        roles = dict(zip(editor["variable"], editor["role"])) if isinstance(editor, pd.DataFrame) else detected

        with st.expander("Paramètres de calcul (valeurs recommandées)"):
            a, b, c = st.columns(3)
            with a:
                top_n = st.number_input("Top modalités par variable", min_value=2, max_value=100,
                                        value=20, key="v2d_top_categories")
            with b:
                max_categories = st.number_input("Maximum de modalités pour associations", min_value=2,
                                                 max_value=500, value=80, key="v2d_max_categories")
            with c:
                bins = st.number_input("Tranches numériques", min_value=2, max_value=30,
                                       value=8, key="v2d_num_bins")

    options = AnalysisOptions(top_categories=int(top_n), max_categories_association=int(max_categories),
                              max_categories_segment=int(max_categories), numerical_bins=int(bins))
    signature = (token, tuple(sorted(mapping.items())), tuple(sorted(roles.items())), options)
    st.session_state['v2d_current_signature'] = signature
    if st.button("Lancer l'analyse descriptive complète", type="primary"):
        try:
            with st.spinner("Calcul des statistiques et associations en cours..."):
                report = generate_descriptive_report(
                    df, roles=roles, actuarial_mapping=mapping, options=options,
                )
            # Même si les paramètres sont identiques, une nouvelle exécution
            # peut produire de nouvelles statistiques : repartir d'un chat vierge.
            reset_agent_chat()
            st.session_state.pop("v2d_pair_detail", None)
            st.session_state["v2d_report"] = report
            st.session_state["v2d_signature"] = signature
            st.success("Analyse calculée. Vous pouvez consulter tous les résultats ci-dessous.")
        except Exception as exc:
            st.session_state.pop("v2d_report", None)
            st.session_state.pop("v2d_signature", None)
            st.error(f"Analyse impossible : {exc}")
    report = st.session_state.get("v2d_report")
    if report is None or st.session_state.get("v2d_signature") != signature:
        st.info("Cliquez sur « Lancer l'analyse descriptive complète » pour calculer les résultats de la configuration actuelle.")
        return

    st.subheader("3. Résultats")
    o = report["overview"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Observations", f"{o['n_rows']:,}")
    m2.metric("Variables", o["n_columns"])
    m3.metric("Valeurs manquantes", f"{o['total_missing']:,}")
    view = st.segmented_control('Explorer les résultats', ['Variables', 'Associations', 'Sinistralité & segments', 'Temps', 'Exports & limites'], default='Variables', key='v2d_result_view', selection_mode='single')
    if view == 'Variables':
        _view_univariate(report, df)
    if view == 'Associations':
        _view_associations(report, df)
    if view == 'Sinistralité & segments':
        _view_actuarial(report)
    if view == 'Temps':
        _view_time(report)
    if view == 'Exports & limites':
        st.markdown("#### Avertissements et limites")
        if report["warnings"]:
            for warning in report["warnings"]:
                st.write("•", warning)
        else:
            st.success("Aucun avertissement particulier.")
        st.markdown("#### Exporter les tableaux et agrégats")
        st.download_button(
            "Télécharger les résultats (ZIP de CSV)",
            data=_csv_export_bundle(report),
            file_name="analyse_descriptive_resultats.zip",
            mime="application/zip",
        )
        st.download_button(
            "Télécharger la synthèse structurée pour l’agent IA (JSON)",
            data=json.dumps(build_llm_context(report), ensure_ascii=False, indent=2, allow_nan=False),
            file_name="analyse_descriptive_synthese.json", mime="application/json",
        )
        st.caption("Cette synthèse contient les agrégats utilisables par l’IA, pas un rapport rédigé. "
                   "Aucune donnée individuelle brute n'est envoyée à un LLM.")
