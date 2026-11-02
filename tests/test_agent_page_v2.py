"""Tests de l’état local du chat Streamlit.

Simule Streamlit et l’agent pour vérifier qu’un changement de portefeuille
supprime l’ancien historique et son identifiant de réponse, sans serveur.
"""
import importlib
import sys
import types
from contextlib import nullcontext


def test_agent_chat_clears_previous_portfolio(monkeypatch):
    state = {
        "v2_agent_signature": "ancienne_base",
        "v2_agent_previous_response_id": "remote_old",
        "v2_agent_messages": [{"role": "assistant", "content": "Réponse ancienne"}],
    }

    fake_st = types.ModuleType("streamlit")
    fake_st.session_state = state
    fake_st.subheader = lambda *_a, **_k: None
    fake_st.caption = lambda *_a, **_k: None
    fake_st.warning = lambda *_a, **_k: None
    fake_st.info = lambda *_a, **_k: None
    fake_st.button = lambda *_a, **_k: False
    fake_st.chat_input = lambda *_a, **_k: None
    fake_st.chat_message = lambda *_a, **_k: nullcontext()
    fake_st.markdown = lambda *_a, **_k: None
    fake_st.spinner = lambda *_a, **_k: nullcontext()
    fake_st.expander = lambda *_a, **_k: nullcontext()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    # Empêcher toute requête réseau ; le test ne vise que la gestion de session.
    import agent_v2
    monkeypatch.setattr(agent_v2, "ask_agent_v2", lambda **_kwargs: ("Réponse", "new_id"))
    monkeypatch.setenv("OPENAI_API_KEY", "test_key_fake")
    import agent_page
    importlib.reload(agent_page)

    agent_page.render_agent_page(report={"overview": {}}, signature="nouvelle_base")

    assert state["v2_agent_signature"] == "nouvelle_base"
    assert state["v2_agent_messages"] == []
    assert "v2_agent_previous_response_id" not in state

    agent_page.reset_agent_chat()
    assert "v2_agent_signature" not in state
