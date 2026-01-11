[![build](https://github.com/ramp-kits/scMARK_classification/actions/workflows/testing.yml/badge.svg)](https://github.com/ramp-kits/scMARK_classification/actions/workflows/testing.yml)

# TEAM AVENGERS

<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&weight=900&size=48&duration=3000&pause=1000&color=667EEA&background=FFFFFF00&center=true&vCenter=true&multiline=true&repeat=false&width=600&height=100&lines=TEAM+AVENGERS" alt="Team Avengers" />
</div>

---

## Single-Cell RNA-seq Classification Challenge

Ce projet propose une approche hiérarchique en deux étapes pour la classification de types cellulaires à partir de données scRNA-seq du benchmark scMARK. Le dataset contient 1,500 cellules réparties en 4 classes (Cancer_cells, NK_cells, T_cells_CD4+, T_cells_CD8+) avec une forte sous-représentation de NK_cells (8.5%). Après exploration et prétraitement des données (normalisation, filtrage par variance, réduction de dimension via PCA), plusieurs modèles ont été testés dont LightGBM (82.9% balanced accuracy), Stacking classique (77.8%) et régression logistique (77.6%). L'approche finale retenue utilise un classificateur hiérarchique à deux étages : le premier fusionne NK_cells et T_cells_CD8+ en une classe unique pour une classification en 3 classes (PCA 60 composantes, variance ≥ 1.2), puis le second raffine cette classe fusionnée via un modèle binaire spécialisé (PCA 80% variance, variance ≥ 0.8). Chaque étage utilise un ensemble Stacking combinant Random Forest, SVM et KNN avec une régression logistique comme meta-learner. Cette architecture modulaire atteint une balanced accuracy de 88.0% en tirant parti de la spécialisation de chaque modèle sur un sous-problème simplifié.

---

**Master 2 Data Science | Université Paris-Saclay (Évry) | Janvier 2026**
