"""Interface de génération, révision et export des rapports IA.

Gère les options et brouillons de la session, présente les preuves chiffrées
et prépare le PDF du texte courant après validation des références.
"""
import copy
import streamlit as st
import pandas as pd
from report_generator import SECTIONS, generate_report, render_text, export_pdf, evidence_rows


def render_report_page(report, signature):
    """Gère les options, l’édition et le téléchargement du rapport lié à l’analyse courante.

    Réinitialise l’état lors d’un changement de signature et ne propose le PDF
    préparé que s’il correspond encore au texte courant validé.
    """
    st.subheader('Créer et consulter un rapport')
    st.caption('Les chiffres proviennent de l’analyse Python. Relisez les interprétations avant diffusion. Les longueurs sont indicatives et dépendent des sections et résultats disponibles.')
    if st.session_state.get('report_ai_signature') != signature:
        for key in list(st.session_state):
            if key.startswith('report_ai_'):
                del st.session_state[key]
        st.session_state['report_ai_signature'] = signature
    with st.expander('Configurer le rapport', expanded=st.session_state.get('report_ai_document') is None):
        length = st.radio('Format du rapport', ['Synthétique (2–3 pages)', 'Complet (5–10 pages)'], key='report_ai_length')
        sections = st.multiselect('Sections à inclure', list(SECTIONS), default=list(SECTIONS), format_func=SECTIONS.get, key='report_ai_sections')
        charts = st.checkbox('Inclure les graphiques disponibles', value=True, key='report_ai_charts')
    configuration = (length, tuple(sections), charts)
    if st.button('Générer le rapport IA', disabled=not sections, key='report_ai_generate', type='primary'):
        try:
            with st.spinner('Rédaction et validation du rapport — correction automatique si nécessaire…'):
                doc = generate_report(report, sections, complete=length.startswith('Complet'))
            st.session_state['report_ai_document'] = doc
            st.session_state['report_ai_configuration'] = configuration
            st.session_state.pop('report_ai_pdf', None)
            for section, text in doc['texts'].items():
                st.session_state['report_ai_edit_' + section] = text
        except Exception as exc:
            st.error(f'Génération impossible : {exc}')
    doc = st.session_state.get('report_ai_document')
    if doc is None:
        return
    if st.session_state.get('report_ai_configuration') != configuration:
        st.info('Les options ont changé : régénérez le rapport pour les appliquer.')
        return
    edited = copy.deepcopy(doc)
    st.caption('Les références renvoient aux tableaux de résultats. Ouvrez « Modifier le texte » pour réviser une section en conservant ses références.')
    valid = True
    for section in doc['texts']:
        st.markdown('#### ' + SECTIONS[section])
        with st.expander('Modifier le texte — ' + SECTIONS[section]):
            text = st.text_area('Texte de la section', key='report_ai_edit_' + section, height=220)
        edited['texts'][section] = text
        try:
            st.markdown(render_text(text, doc['evidence']))
            rows = evidence_rows(text, doc['evidence'], section, doc['complete'])
            if rows:
                st.caption('Repères chiffrés et sources — résultats calculés en Python')
                st.dataframe(pd.DataFrame(rows, columns=['Source', 'Indicateur', 'Résultat']), hide_index=True, use_container_width=True)
        except ValueError as exc:
            st.error(str(exc)); valid = False
    with st.expander('Sources Python et limites de couverture'):
        st.json(doc['evidence'])
    st.warning('Les contrôles automatiques ne garantissent pas la justesse de toutes les interprétations. Une relecture actuarielle reste nécessaire.')
    if st.button('Préparer le PDF du texte actuel', disabled=not valid, key='report_ai_prepare'):
        try:
            st.session_state['report_ai_pdf'] = (edited, export_pdf(edited, charts), 'layout_v2')
        except Exception as exc:
            st.error(f'Export impossible : {exc}')
    saved = st.session_state.get('report_ai_pdf')
    if saved and len(saved) == 3 and saved[2] == 'layout_v2' and saved[0] == edited and valid:
        st.download_button('Télécharger le rapport PDF', saved[1], 'rapport_actuariel_v2.pdf', 'application/pdf')
