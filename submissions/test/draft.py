import numpy as np
from scipy.sparse import issparse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


# ======================
# Utils
# ======================
def remove_genes(X, threshold):
    if issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        var = mean_sq - mean ** 2
    else:
        var = X.var(axis=0)

    kept_idx = np.where(var >= threshold)[0]
    return X[:, kept_idx], kept_idx


def preprocess(X, kept_idx=None, fit=False, threshold=None):
    if issparse(X):
        X = X.toarray()

    if fit:
        X, kept_idx = remove_genes(X, threshold)
    else:
        X = X[:, kept_idx]

    libsize = X.sum(axis=1, keepdims=True)
    libsize[libsize == 0] = 1

    X = np.log1p(X / libsize * 1e4)
    return (X, kept_idx) if fit else X


# ======================
# Classifier
# ======================
class Classifier:
    def __init__(self):
        self.le_merge = LabelEncoder()
        self.le_bin = LabelEncoder()

        self.kept_idx_merge = None
        self.kept_idx_bin = None

        # ----- Stage 1 : 3 classes -----
        stack_merge = StackingClassifier(
            estimators=[
                ("svc", SVC(
                    kernel="rbf", C=10, gamma="scale",
                    probability=True, random_state=42
                )),
                ("lgbm", LGBMClassifier(
                    n_estimators=300, max_depth=8,
                    learning_rate=0.05, num_leaves=50,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1,
                    random_state=42, verbose=-1
                )),
            ],
            final_estimator=LogisticRegression(
                max_iter=10000, C=1.0, random_state=42
            ),
            cv=5,
            n_jobs=-1
        )

        self.pipe_merge = make_pipeline(stack_merge)

        # ----- Stage 2 : NK vs T CD8 -----
        stack_bin = StackingClassifier(
            estimators=[
                ("svc", SVC(
                    kernel="rbf", C=15, gamma="scale",
                    probability=True, class_weight="balanced",
                    random_state=42
                )),
                ("lgbm", LGBMClassifier(
                    n_estimators=400, max_depth=10,
                    learning_rate=0.03, num_leaves=60,
                    subsample=0.85, colsample_bytree=0.85,
                    reg_alpha=0.05, reg_lambda=0.05,
                    random_state=42, verbose=-1
                )),
            ],
            final_estimator=LogisticRegression(
                max_iter=10000, C=1.5, random_state=42
            ),
            cv=5,
            n_jobs=-1
        )

        self.pipe_bin = make_pipeline(stack_bin)

    # ======================
    # Fit
    # ======================
    def fit(self, X, y):
        y = np.asarray(y).astype(str)
        self.classes_ = np.unique(y)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        # ----- Stage 1 -----
        self.merged_label = "NK_or_T_cell"
        y_merge = y.copy()
        y_merge[np.isin(y_merge, ["NK_cells", "T_cells_CD8+"])] = self.merged_label

        X_merge, self.kept_idx_merge = preprocess(
            X, fit=True, threshold=0.8
        )
        y_merge_enc = self.le_merge.fit_transform(y_merge)

        self.pipe_merge.fit(X_merge, y_merge_enc)

        # ----- Stage 2 -----
        mask = np.isin(y, ["NK_cells", "T_cells_CD8+"])
        X_bin, self.kept_idx_bin = preprocess(
            X[mask], fit=True, threshold=0.6
        )
        y_bin_enc = self.le_bin.fit_transform(y[mask])

        self.pipe_bin.fit(X_bin, y_bin_enc)

    # ======================
    # Predict proba
    # ======================
    def predict_proba(self, X):
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, len(self.classes_)))

        X_merge = preprocess(X, kept_idx=self.kept_idx_merge, threshold=1.2)
        proba_merge = self.pipe_merge.predict_proba(X_merge)
        y_merge = self.le_merge.inverse_transform(
            self.pipe_merge.predict(X_merge)
        )

        # Direct predictions
        for i, label in enumerate(y_merge):
            if label != self.merged_label:
                j = self.le_merge.transform([label])[0]
                proba[i, self.class_to_idx[label]] = proba_merge[i, j]

        # Refine NK / T
        mask = y_merge == self.merged_label
        if mask.any():
            X_bin = preprocess(
                X[mask], kept_idx=self.kept_idx_bin,
                threshold=0.8
            )
            proba_bin = self.pipe_bin.predict_proba(X_bin)

            for i, row in zip(np.where(mask)[0], proba_bin):
                for cls, p in zip(self.le_bin.classes_, row):
                    proba[i, self.class_to_idx[cls]] = p

        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]



### 0.87



