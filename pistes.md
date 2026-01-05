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

- Exploration de l'etat de l'art en terme de normalisation de données single Cell RNA Seq

- Choix et implementation de quelques techniques de normalisation: simple, log, CPM, library size

## Reduction de dimensions

- ACP et choix des n-component: 0.9 (90% de variables expliquées) permettant de garder 717 features

## Construction des pipelines

### Sarah

- Xgboost: acc - 0.68

### Fatima-Zahra

Entraînement des modèles de Lasso, SVM, XGboost et Stacking.
Résultats :
| Modèle | Train_Balanced Acc | Train — CV Acc(mean)| Test — Balanced Acc | Test — CV Acc(mean) |
|------------|----------------------|-------------------- |---------------------|---------------------|
| Lasso | 100% | 83.1% | 77.6% | 82.2% |
| SVM | 91.35% | 70.6% | 59.3% | 58% |
| XGBoost | 100% | 83.2% | 77.9% | 78.2% |
| Stacking | 100% | 86.9% | 79.9% | 81% |

### Guérin

- RandomForest Classifier: acc -
- LinearSVC: acc - 0.81

## Prochaines étapes

- augmentation des données (bootsrapping - injection de bruit - génération de données par GAN)
- suppression des gènes moins variables
- Regroupement des classes similaires et training de deux modèles

# Final Guérin

- Filtrage de gènes par variance
- Normalisation (library size + log1p)
- Réduction de dimension (PCA)
- Classification hiérarchique en deux étapes
- Stacking Classifier (RF + SVM + KNN)
