"""Point d’entrée de l’application Actuarial Portfolio Analyst V2.

Configure Streamlit et oriente vers les quatre espaces de travail, sans
effectuer de calcul métier ni charger automatiquement un portefeuille.
"""
import streamlit as st
from ui import apply_style, preserve_controls, sidebar, active_analysis, analysis_required

st.set_page_config(page_title='Actuarial Portfolio Analyst', layout='wide', initial_sidebar_state='expanded')
preserve_controls()
apply_style()
page = sidebar()

if page == 'Qualité & Nettoyage':
    from v2_page import render_v2_page
    render_v2_page()
elif page == 'Analyse descriptive':
    from descriptive_page import render_descriptive_page
    render_descriptive_page()
else:
    report, signature = active_analysis()
    if page == 'Assistant IA':
        st.title('Assistant IA')
        st.caption('Interrogez les résultats calculés et approfondissez votre lecture du portefeuille.')
        if report is None:
            analysis_required()
        else:
            from agent_page import render_agent_page
            render_agent_page(report, signature)
    else:
        st.title('Rapports IA')
        st.caption('Transformez les résultats du portefeuille en un rapport structuré, révisable et exportable.')
        if report is None:
            analysis_required()
        else:
            from report_page import render_report_page
            render_report_page(report, (signature, id(report)))
