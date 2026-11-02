"""Validation manuelle du moteur qualité sur le portefeuille local.

Lit le CSV et la clé métier configurés dans ce script, vérifie les sections
du diagnostic et écrit un rapport JSON. Indépendant de l’interface Streamlit
et volontairement adapté à la configuration locale, sans appel IA.
"""


import json
import time
from pathlib import Path

import pandas as pd

from quality_checker import generate_quality_report


DATA_PATH = Path("data/portefeuille.csv")
OUTPUT_DIR = Path("reports")
OUTPUT_PATH = OUTPUT_DIR / "quality_report.json"

KEY_COLUMNS = ["insured_id", "year"]


def main():
    """Contrôle le fichier configuré, écrit le diagnostic JSON et affiche le bilan des sections."""
    start = time.perf_counter()

    print("\n=== VALIDATION QUALITY CHECKER V2 ===")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, sep=";", low_memory=False)

    print(f"\nLignes : {len(df):,}")
    print(f"Colonnes : {len(df.columns)}")

    print("\nExécution des contrôles de qualité...")

    report = generate_quality_report(
        df,
        key_columns=KEY_COLUMNS,
    )

    expected_sections = [
        "structure",
        "missing_values",
        "duplicates",
        "data_types",
        "outliers",
        "statistical_issues",
    ]

    print("\n--- Sections du rapport ---")
    missing_sections = []

    for section in expected_sections:
        exists = section in report
        print(f"{'OK' if exists else 'ABSENT'} : {section}")
        if not exists:
            missing_sections.append(section)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    elapsed = time.perf_counter() - start

    print(f"\nRapport sauvegardé : {OUTPUT_PATH}")
    print(f"Durée : {elapsed:.2f} secondes")

    if missing_sections:
        print("\nVALIDATION INCOMPLÈTE : sections absentes")
    else:
        print("\nVALIDATION TECHNIQUE : OK")
        print("Toutes les sections attendues sont présentes.")


if __name__ == "__main__":
    main()
