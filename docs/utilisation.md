# Utilisation et lecture actuarielle

[Retour au README](../README.md)

## Préparer le portefeuille

Importer un fichier CSV ou XLSX dans **Qualité & Nettoyage**, choisir au besoin son
séparateur puis charger la base. Le nom actif et l’avancement apparaissent dans la
barre latérale. Aucun jeu de données n’est téléchargé automatiquement.

Sélectionner une clé métier si elle est connue, puis lancer le contrôle qualité.
Les atypies IQR, doublons sur clé, variables constantes, modalités rares et fortes
cardinalités sont des signaux à examiner, pas des décisions automatiques de rejet.
L’écran distingue les corrections proposées des alertes informatives.

Les actions possibles comprennent la conversion numérique lorsque les valeurs sont
convertibles, le retrait des doublons exacts, la suppression d’espaces dans les noms
de colonnes sans collision et plusieurs traitements des valeurs manquantes. Une
imputation par médiane ou par « Inconnu » doit rester justifiée par le métier.
Les actions sont explicitement sélectionnées ; le nettoyage travaille sur une copie,
produit un journal, relance les contrôles et permet un export CSV.

## Définir l’analyse

Choisir la base importée ou nettoyée. Vérifier les rôles statistiques proposés :
`numeric`, `categorical`, `temporal`, `identifier`, `exclude`. Un code numérique
n’est pas nécessairement une mesure ; corriger son rôle si nécessaire.
Les identifiants ne doivent pas être utilisés comme variables explicatives.

Le mapping actuariel est manuel et facultatif champ par champ :

| Rôle | Signification à confirmer dans la base |
|---|---|
| `claim_count` | Nombre de sinistres |
| `exposure` | Exposition au risque, avec unité et période connues |
| `premium` | Prime correspondant au périmètre étudié |
| `claim_amount` | Charge de sinistres sur le même périmètre |
| `year` | Année ou date de référence |
| `policy_id` | Identifiant de contrat ou d’entité selon la granularité |

Une colonne ne peut pas remplir deux rôles actuariels. Les mesures sélectionnées
ne doivent pas contenir de manquants, de valeurs non numériques ou infinies. Les
expositions et nombres de sinistres négatifs sont refusés ; le nombre de sinistres
doit être entier. Les montants négatifs peuvent avoir une justification comptable
et ne sont pas interdits de manière générale par ce validateur.

## Indicateurs calculés

Les ratios sont des rapports de sommes sur la population analysée, pas des moyennes
non pondérées de ratios individuels.

| Indicateur | Formule | Champs nécessaires |
|---|---|---|
| Fréquence | Somme des sinistres / somme des expositions | `claim_count`, `exposure` |
| Coût moyen | Somme des charges / somme des sinistres | `claim_amount`, `claim_count` |
| Prime pure | Somme des charges / somme des expositions | `claim_amount`, `exposure` |
| S/P | Somme des charges / somme des primes | `claim_amount`, `premium` |
| Prime par exposition | Somme des primes / somme des expositions | `premium`, `exposure` |
| Prime moyenne par observation | Somme des primes / nombre de lignes | `premium` |

Un dénominateur non strictement positif conduit à un ratio indisponible. Le moteur
ne devine ni devise ni unité d’exposition. Le coût moyen suppose notamment que la
charge et le nombre de sinistres ont le même périmètre ; une prime moyenne par ligne
n’est pas nécessairement une prime moyenne par contrat.

## Explorer les résultats

Les vues présentent distributions, associations, segments, temporalité et exports.
Les variables numériques sont segmentées par valeurs ou par tranches quantiles
selon leur cardinalité. Les variables catégorielles dépassant la limite configurée
ne sont pas segmentées ; les exclusions sont signalées. Les valeurs manquantes
peuvent constituer une modalité explicite.

Les associations utilisent Pearson entre numériques, le V de Cramér corrigé entre
catégorielles et eta carré entre catégorielle et numérique. Les variables du mapping
actuariel sont exclues des matrices d’associations explicatives. Les constantes,
faibles effectifs et cardinalités élevées limitent les mesures disponibles.
L’exploration locale d’un couple peut afficher un échantillon de points pour son
nuage graphique ; les matrices globales ne sont pas calculées sur cet échantillon.

Les évolutions temporelles sont annuelles. Elles ne constituent ni une prévision,
ni une analyse de développement de sinistres, ni un retraitement de l’inflation.

## Assistant, rapports et exports

Ouvrir **Assistant IA** après le calcul pour poser des questions sur les agrégats.
Le filtrage de segments ne filtre pas à nouveau les contrats. Les parts renvoyées
restent celles du portefeuille complet. Une modalité incluse dans « Autres » ne
peut pas être récupérée en détail par simple pagination.

Dans **Rapports IA**, choisir les sections et la longueur cible. Les six sections
proposées sont : synthèse, présentation du portefeuille, descriptif, actuariat,
associations et temporalité, conclusions et recommandations. Les références sont
consultables dans les tableaux de résultats. Le texte peut être révisé avant de
préparer puis télécharger le PDF. Les formats de deux à trois ou cinq à dix pages
sont des objectifs indicatifs dépendant du contenu.

Les exports descriptifs comprennent un ZIP de tableaux CSV et une synthèse JSON.
Les fichiers téléchargés sont sous la responsabilité de l’utilisateur et peuvent
contenir des informations sensibles même lorsqu’ils sont agrégés.
