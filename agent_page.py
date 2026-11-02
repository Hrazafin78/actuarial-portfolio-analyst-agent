"""Interface de conversation sur l’analyse descriptive courante.

Affiche les échanges et gère leur état local. Réinitialise la chaîne de
conversation lorsque la signature d’analyse change ; aucun calcul n’est réalisé ici.
"""
from __future__ import annotations

import os

import streamlit as st

from agent_v2 import ask_agent_v2


CHAT_KEYS = (
    "v2_agent_messages",
    "v2_agent_previous_response_id",
    "v2_agent_signature",
)


def reset_agent_chat() -> None:
    """Oublie l'historique local et la chaîne de réponses API de la session."""
    for key in CHAT_KEYS:
        st.session_state.pop(key, None)


def render_agent_page(report: dict | None = None, signature: object = None) -> None:
    """Affiche l'assistant IA sur la configuration descriptive courante.

    Une analyse doit avoir été validée et calculée dans descriptive_page.py.
    Le token de configuration évite de réutiliser le chat d'une ancienne analyse.
    """
    st.subheader("Conversation sur le portefeuille")
    st.caption(
        "Posez vos questions en langage naturel sur les résultats de l'analyse. "
        "Les calculs viennent du moteur Python ; l'IA interprète des résultats statistiques."
    )
    with st.expander("Exemples de questions d'exploration"):
        st.markdown(
            "- Quelles variables et analyses sont disponibles ?\n"
            "- Quelles associations sont disponibles pour une variable donnée ?\n"
            "- Quelle est l'association entre deux variables précises ?\n"
            "- Quels sont les cinq segments à la fréquence la plus élevée, "
            "avec au moins 100 unités d'exposition ?\n"
            "- Compare deux modalités d'une variable sur les indicateurs disponibles.\n"
            "- Compare les résultats des années disponibles."
        )
        st.caption(
            "Utilisez les noms de vos variables. L'assistant consulte les résultats statistiques "
            "de l’analyse courante : les tranches existantes et les limites de l’analyse "
            "s'appliquent aussi à ses réponses."
        )

    if report is None:
        report = st.session_state.get("v2d_report")
    if signature is None:
        signature = st.session_state.get("v2d_signature")

    if report is None or signature is None:
        st.info("Lancez d'abord l'analyse descriptive pour pouvoir interroger l'agent.")
        return

    # La signature change si la source, le mapping, les rôles ou les paramètres
    # de l'analyse changent. L'historique distant ne doit pas être réutilisé.
    if st.session_state.get("v2_agent_signature") != signature:
        reset_agent_chat()
        st.session_state["v2_agent_signature"] = signature

    if "v2_agent_messages" not in st.session_state:
        st.session_state["v2_agent_messages"] = []

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("L’assistant n’est pas encore connecté au service IA. Configurez la connexion pour démarrer une conversation.")
        return

    if st.button("Effacer la conversation", key="v2_agent_clear_chat"):
        reset_agent_chat()
        st.session_state["v2_agent_signature"] = signature
        st.rerun()

    for message in st.session_state["v2_agent_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Ex. : Quels sont les points marquants de mon portefeuille ?",
        key="v2_agent_question",
    )
    if not prompt or not prompt.strip():
        return

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Analyse des résultats du portefeuille..."):
                answer, response_id = ask_agent_v2(
                    question=prompt,
                    report=report,
                    previous_response_id=st.session_state.get("v2_agent_previous_response_id"),
                )
            if not answer or not answer.strip():
                raise RuntimeError("Le modèle n'a renvoyé aucune réponse textuelle.")
        except Exception as exc:
            st.error(f"Impossible d'obtenir la réponse de l'agent : {exc}")
            # Ne conserver localement que les échanges terminés avec succès.
            return
        st.markdown(answer)

    st.session_state["v2_agent_messages"].extend([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ])
    st.session_state["v2_agent_previous_response_id"] = response_id
