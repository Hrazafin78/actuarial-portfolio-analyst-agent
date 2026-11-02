"""Tests de la rédaction référencée et de l’export PDF.

Simule le modèle pour vérifier les preuves, les corrections bornées et les
contrôles numériques. Examine le texte et la pagination de PDF de test.
"""

import io
import json
from types import SimpleNamespace
import pandas as pd
import pytest
from pypdf import PdfReader
from descriptive_analyzer import generate_descriptive_report
from report_generator import build_evidence, generate_report, validate_text, export_pdf, SECTIONS


def sample():
    """Construit une analyse contrôlée incluant des identifiants à exclure du contexte IA."""
    return generate_descriptive_report(pd.DataFrame({'id':['SECRET_A','SECRET_B','SECRET_C','SECRET_D'], 'fuel':['A','B','A','B'], 'age':[20,30,40,50], 'expo':[1.,1.,1.,1.], 'claims':[0,1,0,2]}), roles={'id':'identifier','fuel':'categorical','age':'numeric'}, actuarial_mapping={'policy_id':'id','exposure':'expo','claim_count':'claims'})


def fake_client(sections):
    """Simule une rédaction JSON référencée et vérifie le contexte transmis, sans appel réseau."""
    def create(**kwargs):
        assert 'SECRET_' not in kwargs['input']
        assert kwargs['text']['format']['strict']
        return SimpleNamespace(output_text=json.dumps({s:'Constat fondé sur [[E1]]. Limites à examiner et investigations complémentaires proposées.' for s in sections}))
    return SimpleNamespace(responses=SimpleNamespace(create=create))


def test_generation_and_pdf():
    doc = generate_report(sample(), list(SECTIONS), client=fake_client(SECTIONS))
    pdf = export_pdf(doc)
    reader = PdfReader(io.BytesIO(pdf))
    assert len(reader.pages) == 3
    text = ''.join(p.extract_text() for p in reader.pages)
    assert 'SECRET_' not in text and '[[E' not in text
    assert 'Synthèse exécutive' in text
    full = generate_report(sample(), list(SECTIONS), complete=True, client=fake_client(SECTIONS))
    assert 5 <= len(PdfReader(io.BytesIO(export_pdf(full))).pages) <= 10


@pytest.mark.parametrize('text', ['Fréquence : 12 %', 'Résultat [[E99999]]', 'Résultat [[E1]', 'Ce facteur provoque les sinistres.'])
def test_reject_unverified_text(text):
    with pytest.raises(ValueError):
        validate_text(text, build_evidence(sample())['facts'])


def test_missing_mapping_and_selected_sections():
    report = generate_descriptive_report(pd.DataFrame({'x':[1,2,3]}))
    sections = ['actuarial']
    doc = generate_report(report, sections, client=fake_client(sections))
    assert list(doc['texts']) == sections
    assert 'claim_frequency' not in report['actuarial_overview']
    assert export_pdf(doc, charts=False).startswith(b'%PDF')


def test_bad_model_output_rejected():
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: SimpleNamespace(output_text='{"executive":"Fréquence de 99 %"}')))
    with pytest.raises(ValueError):
        generate_report(sample(), ['executive'], client=client)


def test_evidence_changes_and_no_raw_identifiers():
    from report_generator import fingerprint
    report = sample()
    first = build_evidence(report)
    assert 'SECRET_' not in json.dumps(first)
    report['actuarial_overview']['total_exposure'] += 1
    assert fingerprint(first) != fingerprint(build_evidence(report))


def test_citations_never_expand_into_raw_records():
    from report_generator import render_text, evidence_rows, format_value
    evidence = {'facts': {
        'E1': {'section':'actuarial', 'label':'Segment fuel', 'value':{'variable':'fuel', 'segment':'A', 'observations':354140, 'loss_ratio':0.702941, 'total_premium':105080000.0}},
        'E2': {'section':'actuarial', 'label':'total_claim_count', 'value':12345},
    }}
    text = 'Des écarts sont observés [[E1]][[E2]]. Ils restent descriptifs.'
    rendered = render_text(text, evidence)
    assert '[E1, E2]. Ils' in rendered
    assert 'loss_ratio' not in rendered and '354140' not in rendered
    rows = evidence_rows(text, evidence)
    assert ('E1', 'Segment fuel / A — Ratio S/P', '70,29 %') in rows
    assert any(row[2] == '354 140' for row in rows)
    assert format_value('total_premium', 105080000.0) == '105 080 000,00'


def test_generation_repairs_invalid_draft_and_keeps_valid_section():
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            texts = {'executive': 'Un ratio de 99 %.', 'conclusions': 'Investigation proposée [[E1]].'}
        else:
            assert 'validation_errors' in kwargs['input']
            texts = {'executive': 'Constat disponible [E1, E2].', 'conclusions': 'Modification non nécessaire.'}
        return SimpleNamespace(output_text=json.dumps(texts))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    doc = generate_report(sample(), ['executive', 'conclusions'], client=client)
    assert len(calls) == 2
    assert doc['texts']['executive'] == 'Constat disponible [[E1]][[E2]].'
    assert doc['texts']['conclusions'] == 'Investigation proposée [[E1]].'


def test_invalid_draft_retries_are_bounded():
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text='{"executive":"Ratio inventé 99 %"}')
    with pytest.raises(ValueError, match='deux corrections'):
        generate_report(sample(), ['executive'], client=SimpleNamespace(responses=SimpleNamespace(create=create)))
    assert len(calls) == 3
