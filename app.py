import streamlit as st

from agent import ask_agent

from tools import (
    load_and_validate_data,
    portfolio_summary,
    plot_frequency
)


# --------------------------------------------------
# Configuration de la page
# --------------------------------------------------

st.set_page_config(
    page_title="Actuarial Portfolio Analyst",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Chargement des données
# --------------------------------------------------

df = load_and_validate_data(
    "data/sample_portfolio.csv"
)

summary = portfolio_summary(df)


# --------------------------------------------------
# En-tête
# --------------------------------------------------

st.title("📊 Actuarial Portfolio Analyst")

st.write(
    "Analysez un portefeuille d'assurance automobile "
    "en langage naturel grâce à un agent IA."
)


# --------------------------------------------------
# Indicateurs principaux
# --------------------------------------------------

st.subheader("Vue d'ensemble du portefeuille")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Contrats",
        f"{summary['number_of_policies']:,}"
    )

with col2:
    st.metric(
        "Exposition",
        f"{summary['total_exposure']:,.1f}"
    )

with col3:
    st.metric(
        "Sinistres",
        f"{summary['total_claims']:,}"
    )

with col4:
    st.metric(
        "Fréquence",
        f"{summary['claim_frequency']:.2%}"
    )


# --------------------------------------------------
# Visualisation interactive
# --------------------------------------------------

st.subheader("Analyse des fréquences par segment")

segment_options = {
    "Région": "region",
    "Type de carburant": "vehicle_fuel",
    "Marque du véhicule": "vehicle_brand",
    "Zone": "area",
    "Puissance du véhicule": "vehicle_power"
}

selected_label = st.selectbox(
    "Variable de segmentation",
    options=list(segment_options.keys())
)

min_exposure = st.number_input(
    "Exposition minimale par segment",
    min_value=0.0,
    value=100.0,
    step=50.0
)

selected_variable = segment_options[selected_label]


fig = plot_frequency(
    df,
    variable=selected_variable,
    min_exposure=min_exposure
)

st.plotly_chart(
    fig,
    width="stretch"
)


st.divider()


# --------------------------------------------------
# Agent IA
# --------------------------------------------------

st.subheader("💬 Assistant actuariel")

st.caption(
    "Posez une question en langage naturel sur le portefeuille."
)


# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []


# Affichage de l'historique
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Zone de saisie
question = st.chat_input(
    "Posez une question sur le portefeuille..."
)


if question:

    # Message utilisateur
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Réponse de l'agent
    with st.chat_message("assistant"):

        with st.spinner("Analyse du portefeuille..."):

            try:

                answer = ask_agent(question)

                st.markdown(answer)

            except Exception as error:

                answer = (
                    "Une erreur est survenue pendant l'analyse : "
                    f"{error}"
                )

                st.error(answer)

    # Sauvegarde de la réponse
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )