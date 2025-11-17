- Investigations pour comprendre les données

- Analyse exploratoire pour choisir les méthodes a tester

- test de nouvelle methodes de normalisation: Sans normalisation, simple, log normalisation, library Size

- Faire une ACP pour la reduction de dimensions

- Tester differents Scaler avant ACP

- Déséquilibre du dataset: augmentation des données de la classe sous représentée de facon random

Etape actuelle:

preprocessing: on a testé différentes methodes, simple, sans, log, library_size
Dimensionality reduction avec ACP, choix du best N components
methode ML: LinearSVC, RandomForest

avec linear svc + data augmentation 30% + No Standard Scaler + log normalization accuracy: 81%

On a fait un peu plus d'analyse exploratoire.
On a observé les gènes hautement representés, et les genes les plus variables.
On a supprimé les genes hautement representés et visualisés l'es resultats d'acp pour voir s'il y a une meilleure representation.

On a fait des recherches sur les méthodes de normalisation
On a trouvé log normalisation, et library size normalisation

on a essayé de trouver le nombre de composantes optimales pour l'acp..

On a essayé d'autres méthodes de reduction de dimensions

On a remarqué le déséquilibre des classes. Donc on vas essayer une augmentation des données.

CHoix des modèles.

On teste deja le random forest naif, avec quelques ajustements des paramètres

On a moins de 100k observations. En regardant la carte des methodes de scikit learn, on choisis LinearSVC.

On a remarqué qu'il y a deux classes qui se confondent beaucoup. que faire ?

##########################################################################

# Exploration des données

## Classes:

T_cells_CD8+ 237 0.342
T_cells_CD4+ 85 0.336
Cancer_cells 336 0.237
NK_cells 342 0.085

## Observation: Déséquilibre des classes( NK_cells minoritaire) - idée: augmenter les données.

## Repartition des gènes dans les cellules

### 13551 gènes

### Remarque: Matrices très creuses

### Visualisation de la repartition des gènes

Observation des stats descriptives (mean, std) des gènes pour chaque classe, avant et après normalisation

## Normalisation des données

### Exploration de l'etat de l'art en terme de normalisation de données single Cell RNA Seq

### Choix et implementation de quelques techniques de normalisation: simple, log, CPM, library size

## Reduction de dimensions

### ACP et choix des n-component: 0.9 (90% de variables expliquées) permettant de garder 717 features

## Construction des pipelines

### Sarah

### Fatima

### Guérin
