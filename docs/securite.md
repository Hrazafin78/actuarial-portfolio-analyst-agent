# Données, confidentialité et préparation à la publication

[Retour au README](../README.md)

## Frontière local / service IA

| Élément | Traitement actuel |
|---|---|
| Fichier CSV/XLSX importé | Lu en mémoire par pandas ; pas de sauvegarde automatique par l’import interactif |
| Données individuelles | Non exposées par les outils conversationnels ni par le catalogue de rédaction |
| Statistiques et noms de variables | Peuvent être envoyés à l’API du modèle |
| Libellés de modalités et de segments | Peuvent être envoyés avec leurs agrégats |
| Questions de l’utilisateur et brouillons à corriger | Envoyés à l’API pour la conversation ou la rédaction |
| Conversation | Chaîne Responses API suivie via `previous_response_id` ; historique visible en session |
| Exports téléchargés | Conservés par l’utilisateur à l’emplacement choisi |

**Agrégé ne signifie pas anonyme.** Une modalité rare, un nom de variable ou un segment
à faible effectif peut révéler une information. L’identification correcte des colonnes
`identifier` est nécessaire. Le projet ne met en œuvre ni anonymisation formelle,
ni seuil général de confidentialité, ni confidentialité différentielle. Ne pas
inclure de données individuelles dans les questions libres.

Les conditions de conservation du fournisseur d’API ne sont pas configurées ou
vérifiées par cette application. Effacer la conversation dans l’interface réinitialise
l’état local ; cela ne constitue pas une suppression des données côté fournisseur.
L’application n’est pas présentée comme certifiée conforme à une réglementation.

## Secrets et configuration

La clé se configure dans `.env` ou dans l’environnement du processus. `.env.example`
contient uniquement une clé vide et des paramètres commentés. Les variables de
modèles doivent correspondre aux droits et capacités du compte utilisé.
Ne jamais copier de clé dans un notebook, le README, un rapport ou une capture.

Une valeur ressemblant à une clé API a été détectée dans la version suivie de
`.env.example` lors de la préparation documentaire. Le fichier de travail a été
assaini. Si la clé est active, sa révocation et son remplacement restent nécessaires.
L’historique Git distant n’a pas été réécrit ; la V1 est conservée par son tag.

## Exclusions et fichiers existants

`.gitignore` exclut les fichiers `.env` et variantes, sauf `.env.example`, les
fichiers de clés, les environnements Python, caches, sorties temporaires et contenus
locaux de `data/` et `reports/`. Les README de ces deux dossiers restent publiables.
Les sorties et compteurs d’exécution du notebook exploratoire ont été retirés ;
son code doit encore être relu avant toute publication de nouvelles modifications.

Une règle d’exclusion ne retire pas un fichier déjà suivi. De même, retirer une
clé du dernier état ne l’efface pas des commits précédents. Les fichiers supprimés
de la branche V2 peuvent rester présents dans le tag V1 ou dans d’anciens commits.

## Vérifications avant publication

1. Révoquer toute clé exposée, puis configurer sa remplaçante uniquement localement.
2. Examiner `git status --short`, `git diff` et la liste `git ls-files`.
3. Vérifier les notebooks, captures, fichiers exportés, données déjà suivies et
   l’historique ; ne pas se limiter au contenu de `.gitignore`.
4. Si des secrets existent dans l’historique, préparer son assainissement séparément
   avant publication, en coordonnant toute réécriture avec les autres utilisateurs.
5. Relancer les tests et vérifier que la branche publiée contient bien la V2.

La publication V2 ne réécrit pas les commits déjà distants. Les sorties de notebook
des anciens commits locaux de développement ne sont pas incorporées dans l’historique
publié : l’intégration utilise une fusion squash de l’état final assaini.
L’application ne comporte pas
d’authentification applicative ou de gouvernance multi-utilisateur validée ; un
partage public nécessite une conception complémentaire.
