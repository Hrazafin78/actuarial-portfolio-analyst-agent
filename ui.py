"""Présentation et coordination des espaces Streamlit.

Centralise le thème, la navigation et la conservation des contrôles. Vérifie
la validité de l’analyse courante avant l’accès aux fonctionnalités IA.
"""
import html
import streamlit as st
import plotly.io as pio

PAGES = ['Qualité & Nettoyage', 'Analyse descriptive', 'Assistant IA', 'Rapports IA']


def preserve_controls():
    # Streamlit supprime les widgets absents de la page suivante. Réassocier
    # leur valeur à la session préserve les choix sans relancer de calcul.
    """Conserve les valeurs des contrôles absents lors de la navigation entre espaces."""
    prefixes = ('v2d_', 'report_ai_', 'v2_action_', 'v2_alert_detail_')
    for key in list(st.session_state):
        if key.startswith(prefixes) and key not in {'report_ai_generate', 'report_ai_prepare'}:
            st.session_state[key] = st.session_state[key]
    if 'v2_key_columns' in st.session_state:
        st.session_state['v2_key_columns'] = st.session_state['v2_key_columns']


def active_analysis():
    """Renvoie le rapport et sa signature seulement si source et configuration correspondent.

    Une analyse absente ou obsolète renvoie (None, None) ; les identités de
    source sont locales à la session, sans empreinte persistante du portefeuille.
    """
    report = st.session_state.get('v2d_report')
    signature = st.session_state.get('v2d_signature')
    current = st.session_state.get('v2d_current_signature')
    if report is None or signature is None or current != signature:
        return None, None
    sources = [st.session_state.get(k) for k in ('v2_original_df', 'v2_cleaned_df')]
    if not any(df is not None and str(id(df)) == signature[0] for df in sources):
        return None, None
    return report, signature


def apply_style():
    """Installe le thème Plotly et les styles communs de l’interface Streamlit."""
    pio.templates['actuarial'] = pio.templates['plotly_white']
    pio.templates['actuarial'].layout.colorway = ['#245b78', '#409b99', '#8ca6be', '#b68b57', '#7b7198']
    pio.templates['actuarial'].layout.font.family = 'Arial, sans-serif'
    pio.templates.default = 'actuarial'
    st.markdown('''<style>
    .stApp { background: #f7f9fc; color: #1e293b; }
    .block-container { max-width: 1440px; padding-top: 2.5rem; padding-bottom: 4rem; }
    h1 { font-size: 2rem !important; font-weight: 650 !important; letter-spacing: -.04em; }
    h2 { font-size: 1.45rem !important; letter-spacing: -.025em; }
    h3 { font-size: 1.12rem !important; }
    [data-testid="stSidebar"] { background: #edf2f7; border-right: 1px solid #dce4ed; }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: 6px; }
    [data-testid="stSidebar"] [data-baseweb="radio"] { padding: 10px 14px; border-radius: 8px; margin: 0; }
    [data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) { background: #dbe7ef; font-weight: 600; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e0e7ef; border-radius: 12px; padding: 18px 22px; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #193f56; }
    [data-testid="stExpander"], [data-testid="stDataFrame"] { background: white; border-radius: 10px; }
    [data-testid="stChatMessage"] { background: white; border: 1px solid #e0e7ef; border-radius: 12px; padding: 1.3rem; }
    [data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { display: none; }
    [data-baseweb="tab-list"] { gap: 18px; }
    button[kind="primary"] { background: #245b78; border-color: #245b78; }
    .portfolio-strip { padding: 14px 18px; border: 1px solid #dce4ed; background: #fff; border-radius: 10px; margin-bottom: 26px; font-size: .9rem; color: #526379; }
    .portfolio-strip strong { color: #193f56; }
    .brand-label { font-size: .72rem; letter-spacing: .16em; font-weight: 700; color: #597083; margin-bottom: 8px; }
    @media(max-width: 768px) { .block-container { padding: 1.5rem 1rem; } h1 { font-size: 1.6rem !important; } [data-testid="stMetric"] { padding: 12px; } }
    </style>''', unsafe_allow_html=True)


def navigate(page):
    """Change l’espace sélectionné par un rappel de bouton Streamlit."""
    st.session_state['workspace_page'] = page


def sidebar():
    """Affiche navigation, portefeuille et avancement, puis renvoie l’espace choisi."""
    with st.sidebar:
        st.markdown('<div class="brand-label">PORTFOLIO ANALYTICS</div>', unsafe_allow_html=True)
        st.markdown('### Actuarial Analyst')
        st.caption('Espace de travail actuariel')
        page = st.radio('Navigation', PAGES, key='workspace_page', label_visibility='collapsed')
        st.divider()
        st.markdown('**Portefeuille actif**')
        name = st.session_state.get('v2_file_name')
        st.write(name or 'Aucun fichier importé')
        source = st.session_state.get('v2_cleaned_df')
        if source is None:
            source = st.session_state.get('v2_original_df')
        if source is not None:
            st.caption(f'{len(source):,} observations · {len(source.columns)} variables'.replace(',', ' '))
        st.markdown('**Avancement**')
        analysis, signature = active_analysis()
        states = [('Import', source is not None), ('Contrôle qualité', st.session_state.get('v2_quality_report') is not None),
                  ('Base nettoyée', st.session_state.get('v2_cleaned_df') is not None), ('Analyse', analysis is not None),
                  ('Rapport rédigé', analysis is not None and st.session_state.get('report_ai_document') is not None
                   and st.session_state.get('report_ai_signature') == (signature, id(analysis)))]
        for label, ready in states:
            st.caption(f'{label} : {"disponible" if ready else "à préparer"}')
        st.divider()
        st.caption('Les résultats restent disponibles pendant votre session. Les appels IA sont lancés uniquement à votre demande.')
    name = html.escape(str(st.session_state.get('v2_file_name', 'Aucun portefeuille chargé')))
    report, _ = active_analysis()
    status = 'Analyse disponible' if report is not None else 'Préparation du portefeuille'
    st.markdown(f'<div class="portfolio-strip"><strong>{name}</strong> &nbsp; / &nbsp; {status}</div>', unsafe_allow_html=True)
    return page


def analysis_required():
    """Explique le prérequis d’analyse et propose un accès à sa configuration."""
    st.info('Préparez une analyse descriptive pour accéder aux résultats du portefeuille dans cet espace.')
    st.button('Ouvrir l’analyse descriptive', on_click=navigate, args=('Analyse descriptive',))
