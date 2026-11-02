"""Tests du parcours Streamlit avec AppTest, sans serveur navigateur.

Vérifie la navigation, la conservation des paramètres et textes édités,
l’invalidation des résultats et l’absence de recalcul complet à la navigation.
"""
from pathlib import Path
import importlib
import pandas as pd
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'app.py'


def application():
    # Un autre test utilise un faux module Streamlit pour vérifier le chat.
    """Initialise AppTest et restaure le module du chat après les simulations Streamlit."""
    import agent_page
    importlib.reload(agent_page)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    return app


def navigate(app, page):
    """Change de page et vérifie l’absence d’exception dans le parcours testé."""
    app.sidebar.radio(key='workspace_page').set_value(page).run(timeout=30)
    assert not app.exception


def test_empty_navigation():
    app = application()
    assert list(app.sidebar.radio(key='workspace_page').options) == [
        'Qualité & Nettoyage', 'Analyse descriptive', 'Assistant IA', 'Rapports IA']
    for page in ['Analyse descriptive', 'Assistant IA', 'Rapports IA', 'Qualité & Nettoyage']:
        navigate(app, page)
    assert 'v2d_report' not in app.session_state


def test_analysis_persists_across_pages_and_configuration_invalidates(monkeypatch):
    import descriptive_page
    original = descriptive_page.generate_descriptive_report
    calls = []
    def calculate(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(descriptive_page, 'generate_descriptive_report', calculate)
    app = application()
    app.session_state['v2_original_df'] = pd.DataFrame({'age':[20,30,40,50,60,70], 'fuel':['A','B','A','B','A','B'], 'claims':[0,1,0,1,0,2]})
    app.session_state['v2_file_name'] = 'test_portfolio.csv'
    navigate(app, 'Analyse descriptive')
    next(b for b in app.button if b.label == "Lancer l'analyse descriptive complète").click().run(timeout=30)
    assert not app.exception
    assert len(calls) == 1
    signature = app.session_state['v2d_signature']
    for page in ['Assistant IA', 'Rapports IA', 'Qualité & Nettoyage', 'Analyse descriptive']:
        navigate(app, page)
        assert app.session_state['v2d_signature'] == signature
    assert len(calls) == 1
    assert app.session_state['v2d_current_signature'] == signature
    app.number_input(key='v2d_num_bins').set_value(5).run(timeout=30)
    assert not app.exception
    assert app.session_state['v2d_current_signature'] != signature
    navigate(app, 'Assistant IA')
    assert any('Préparez une analyse' in message.value for message in app.info)
    assert len(calls) == 1


def test_reset_clears_reports_and_chat():
    import streamlit as st
    from unittest.mock import patch
    from v2_page import _reset_import
    state = {'report_ai_document': {'texts':{}}, 'report_ai_edit_executive':'draft',
             'v2_agent_messages':[{'role':'user','content':'old'}], 'v2d_report':{},
             'v2d_current_signature':'old', 'v2_original_df':pd.DataFrame({'x':[1]})}
    with patch.object(st, 'session_state', state):
        _reset_import()
    assert not any(key.startswith('report_ai_') for key in state)
    assert 'v2d_report' not in state and 'v2_agent_messages' not in state

def test_report_edits_survive_navigation(monkeypatch):
    import report_page
    from report_generator import build_evidence, fingerprint
    def generate(report, sections, complete=False):
        evidence = build_evidence(report)
        return {'texts':{s:'Constat disponible [[E1]].' for s in sections}, 'evidence':evidence,
                'complete':complete, 'fingerprint':fingerprint(evidence)}
    monkeypatch.setattr(report_page, 'generate_report', generate)
    app = application()
    app.session_state['v2_original_df'] = pd.DataFrame({'x':[1,2,3,4]})
    app.session_state['v2_file_name'] = 'test.csv'
    navigate(app, 'Analyse descriptive')
    next(b for b in app.button if b.label == "Lancer l'analyse descriptive complète").click().run(timeout=30)
    navigate(app, 'Rapports IA')
    app.button(key='report_ai_generate').click().run(timeout=30)
    assert not app.exception
    draft = 'Texte revu par un actuaire [[E1]].'
    app.text_area(key='report_ai_edit_executive').set_value(draft).run(timeout=30)
    assert not app.exception
    navigate(app, 'Assistant IA')
    navigate(app, 'Rapports IA')
    assert app.text_area(key='report_ai_edit_executive').value == draft
    assert not app.exception
