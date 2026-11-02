# Validation, limites et perspectives

[Retour au README](../README.md)

## Exécuter les vérifications

Depuis la racine du projet :

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest
```

`requirements.txt` fixe les versions directes présentes dans l’environnement local
Python 3.10 inspecté. `requirements-dev.txt` ajoute pytest et pypdf. Ce choix ne
remplace pas un verrouillage complet des dépendances transitives, une matrice de
compatibilité ou une installation propre testée sur plusieurs systèmes.

## Périmètre des tests

| Fichier de tests | Contrôles principaux |
|---|---|
| `test_quality_checker.py` | Structure, manquants, doublons, types, IQR, cohérence et cas limites |
| `test_data_cleaner.py` | Actions sélectionnées, absence de modification de l’original, journaux et alertes |
| `test_descriptive_analyzer.py` | Rôles, mapping, indicateurs, segments, associations, temporalité et contexte JSON |
| `test_agent_v2.py` | Outils sur agrégats et chaîne simulée d’appels multiples |
| `test_agent_exploration_v2.py` | Paire d’associations, sélection, tri avant pagination et seuils invalides |
| `test_agent_page_v2.py` | Réinitialisation de conversation lors d’un changement d’analyse |
| `test_report_generator.py` | Exclusion d’identifiants, validation, corrections bornées, citations et structure PDF |
| `test_ui_navigation.py` | Quatre espaces, persistance des paramètres, invalidation, textes de rapport conservés |

Le nombre de tests exécutés est fourni par pytest et peut évoluer. Les tests utilisent
de petits jeux contrôlés et des faux clients IA. Ils ne prouvent ni l’exactitude
sémantique d’un modèle réel ni la capacité à traiter un volume arbitraire. Les tests
PDF vérifient notamment le texte extrait et la pagination d’exemples courts ; ils
ne remplacent pas une revue visuelle de longs rapports réels.

## Validation indépendante du moteur qualité

`validate_quality.py` lit `data/portefeuille.csv` avec le séparateur point-virgule,
utilise actuellement la clé `insured_id` / `year`, vérifie les sections attendues
et écrit `reports/quality_report.json`. Il est donc spécifique à la configuration
locale du portefeuille exploratoire, contrairement au parcours interactif généraliste.
Sa durée affichée n’est pas un benchmark publié et aucune performance n’en est déduite.

## Limites actuelles

- Analyse descriptive et ratios agrégés : pas de GLM de fréquence/coût, de tarification
  technique complète, de provisionnement ou de modélisation prédictive intégrée.
- Associations descriptives sans contrôle automatique des facteurs de confusion,
  sans démonstration causale ni correction générale des comparaisons multiples.
- Pas d’intervalles de confiance actuariels systématiques ; les petits segments
  peuvent produire des taux instables malgré les seuils d’affichage.
- Temporalité annuelle ; aucune prévision ni correction automatique d’inflation.
- Consultation IA d’agrégats existants : pas de filtres arbitraires sur les contrats,
  de nouvelles tranches personnalisées ou de segmentations croisées créées par le chat.
- Distributions et contexte bornés ; pagination des résultats disponibles, sans
  reconstitution des modalités déjà regroupées.
- Rapports avec sélection bornée de preuves, contrôle lexical partiel et nombre de
  pages indicatif ; le graphique intégré concerne une distribution disponible.
- Session en mémoire, sans reprise persistante après arrêt ni dispositif de
  déploiement multi-utilisateur validé.

## Évolutions possibles, non implémentées

1. Évaluer les réponses réelles sur un corpus annoté de questions et conclusions
   actuariellement vérifiables, y compris questions hors périmètre.
2. Définir des outils de filtrage et de segmentation croisée avec traçabilité du
   périmètre, contrôle des dénominateurs et politique de faibles effectifs.
3. Enrichir la restitution : graphiques sélectionnables, tableaux comparatifs,
   vérification sémantique des commentaires et meilleure maîtrise de la pagination.
4. Ajouter des analyses techniques après définition des objectifs, de la granularité
   et des hypothèses métier : modèles de fréquence/coût ou autres besoins prioritaires.
5. Concevoir persistance, authentification et gouvernance pour un éventuel déploiement.

Ces pistes constituent des axes de développement et ne sont pas annoncées comme livrées.
