from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier



import numpy as np

from scipy.sparse import issparse


def remove_genes(X, threshold=0.01):
    """
    Supprime les gènes dont la variance est inférieure à un seuil.

    Args:
        X (np.ndarray ou csr_matrix): matrice cellules x gènes
        threshold (float): seuil minimal de variance

    Returns:
        X_filtered: matrice filtrée (mêmes type que X)
        kept_idx: indices des gènes conservés
    """
    if issparse(X):
        # variance = mean(x^2) - mean(x)^2
        mean = np.array(X.mean(axis=0)).ravel()
        mean_sq = np.array(X.multiply(X).mean(axis=0)).ravel()
        var = mean_sq - mean**2
    else:
        var = X.var(axis=0)

    # indices des gènes à garder
    kept_idx = np.where(var >= threshold)[0]

    # filtrer X
    if issparse(X):
        X_filtered = X[:, kept_idx]
    else:
        X_filtered = X[:, kept_idx]

    return X_filtered, kept_idx


def _preprocess_X(X):
    """
    Preprocess an RNA-seq expression matrix.
    Parameters
    ----------
    X : np.ndarray or scipy.sparse matrix
        Input expression matrix (cells x genes)
    method : str or None
        Type of normalization:
        - None : return X as is
        - "log" : log1p transform
        - "library_size" : normalize each cell by total counts (library size)
          then log1p
    Returns
    -------
    X_processed : np.ndarray
        Preprocessed expression matrix
    """
    X, genes_to_remove = remove_genes(X, threshold=0.01)
    method = "log"
    # Convert sparse to dense if needed
    if hasattr(X, "toarray"):
        X = X.toarray()
    X_processed = X.copy()
    if method is None:
        return X_processed
    if method == "simple_normalize":
        return X / X.sum(axis=1)[:, np.newaxis]
    elif method == "log":
        X_processed = np.log1p(X_processed)
        return X_processed
    elif method == "library_size":
        # compute total counts per cell
        library_size = X_processed.sum(axis=1)[:, None]
        # avoid division by zero
        library_size[library_size == 0] = 1
        # normalize
        X_processed = X_processed / library_size * 1e4
        # log1p transform
        X_processed = np.log1p(X_processed)
        return X_processed, genes_to_remove
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")


class Classifier(object):
    def __init__(self):
        self.le = LabelEncoder()
        self.kept_idx = None  # indices des gènes conservés
        
        LGBM = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            objective='multiclass',
            random_state=42,
            n_jobs=-1,
            verbose = -1
        )
        rf = RandomForestClassifier(n_estimators=200, max_depth=50,
                                    max_features="sqrt",
                                    random_state=42)
        knn = KNeighborsClassifier(n_neighbors=15)

        stack = StackingClassifier(
            estimators=[("rf", rf), ("LGBM", LGBM), ("knn", knn)],
            final_estimator=LogisticRegression(max_iter=10000, solver="lbfgs"),
            n_jobs=-1,
            passthrough=False
        )

        self.pipe = make_pipeline(
            # StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=0.9, random_state=42),
            stack
        )

    def _preprocess_X(self, X, fit=False):
        # supprimer les gènes peu variables
        if fit:
            X_filtered, self.kept_idx = remove_genes(X, threshold=0.01)
        else:
            # appliquer le même masque que pour le train
            if self.kept_idx is None:
                raise ValueError("Kept indices are not set. ")
            if hasattr(X, "toarray"):
                X = X.toarray()
            X_filtered = X[:, self.kept_idx]

        # library size normalization + log1p
        if hasattr(X_filtered, "toarray"):
            X_filtered = X_filtered.toarray()
        libsize = X_filtered.sum(axis=1)[:, None]
        libsize[libsize == 0] = 1
        X_processed = np.log1p(X_filtered / libsize * 1e4)
        return X_processed

    def fit(self, X_sparse, y):
        X = self._preprocess_X(X_sparse, fit=True)
        y_enc = self.le.fit_transform(y)
        self.pipe.fit(X, y_enc)
        self.classes_ = self.le.classes_

    def predict_proba(self, X_sparse):
        X = self._preprocess_X(X_sparse, fit=False)
        return self.pipe.predict_proba(X)
