import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from scipy.sparse import issparse


# =========================
# Feature filtering robuste
# =========================
def remove_genes_quantile(X, q=0.3):
    if issparse(X):
        X = X.toarray()
    var = X.var(axis=0)
    thresh = np.quantile(var, q)
    kept_idx = np.where(var >= thresh)[0]
    return X[:, kept_idx], kept_idx


# =========================
# Classifier
# =========================
class Classifier(object):
    def __init__(self):

        self.le_merge = LabelEncoder()
        self.le_bin = LabelEncoder()

        self.kept_idx_merge = None
        self.kept_idx_bin = None

        # -------- Stage 1 : 3-class merged model --------
        self.pipe_merge = Pipeline([
            ("lgbm", LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=6,
                min_data_in_leaf=30,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=2.0,
                class_weight="balanced",
                objective="multiclass",
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ))
        ])

        # -------- Stage 2 : Binary NK vs T_CD8 --------
        self.pipe_bin = Pipeline([
            ("lgbm", LGBMClassifier(
                n_estimators=800,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=6,
                min_data_in_leaf=20,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=2.0,
                class_weight="balanced",
                objective="binary",
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ))
        ])

        self.merged_label = "NK_or_T_cell"
        self.stage1_threshold = 0.35   # ⭐ seuil clé


    # =========================
    # Preprocessing
    # =========================
    def _preprocess_X(self, X, fit=False, kept_idx=None, q=0.3):

        if issparse(X):
            X = X.toarray()

        if fit:
            X, kept_idx = remove_genes_quantile(X, q=q)
        else:
            X = X[:, kept_idx]

        libsize = X.sum(axis=1, keepdims=True)
        libsize[libsize == 0] = 1
        X = np.log1p(X / libsize * 1e4)

        return (X, kept_idx) if fit else X


    # =========================
    # Fit
    # =========================
    def fit(self, X_sparse, y):

        y = np.array(y).astype(str)

        # ----- Stage 1 labels -----
        y_merge = y.copy()
        y_merge[np.isin(y_merge, ["NK_cells", "T_cells_CD8+"])] = self.merged_label

        X_merge, self.kept_idx_merge = self._preprocess_X(
            X_sparse, fit=True, q=0.3
        )

        y_merge_enc = self.le_merge.fit_transform(y_merge)

        self.pipe_merge.named_steps["lgbm"].fit(
            X_merge, y_merge_enc,
            eval_set=[(X_merge, y_merge_enc)],
            eval_metric="multi_logloss",
            early_stopping_rounds=50,
            verbose=False
        )

        # ----- Stage 2 data -----
        mask_bin = np.isin(y, ["NK_cells", "T_cells_CD8+"])
        X_bin = X_sparse[mask_bin]
        y_bin = y[mask_bin]

        X_bin, self.kept_idx_bin = self._preprocess_X(
            X_bin, fit=True, q=0.4
        )

        y_bin_enc = self.le_bin.fit_transform(y_bin)

        self.pipe_bin.named_steps["lgbm"].fit(
            X_bin, y_bin_enc,
            eval_set=[(X_bin, y_bin_enc)],
            eval_metric="binary_logloss",
            early_stopping_rounds=50,
            verbose=False
        )

        self.classes_ = np.unique(y)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}


    # =========================
    # Predict probabilities
    # =========================
    def predict_proba(self, X_sparse):

        n_samples = X_sparse.shape[0]
        n_classes = len(self.classes_)
        proba_final = np.zeros((n_samples, n_classes))

        # ----- Stage 1 -----
        X_merge = self._preprocess_X(
            X_sparse, fit=False, kept_idx=self.kept_idx_merge
        )

        proba_merge = self.pipe_merge.predict_proba(X_merge)
        idx_merged = self.le_merge.transform([self.merged_label])[0]

        nk_or_t_mask = proba_merge[:, idx_merged] > self.stage1_threshold

        # Non NK/T
        for i in np.where(~nk_or_t_mask)[0]:
            label = self.le_merge.inverse_transform(
                [np.argmax(proba_merge[i])]
            )[0]
            class_idx = self.class_to_idx[label]
            proba_final[i, class_idx] = 1.0

        # ----- Stage 2 -----
        if nk_or_t_mask.any():
            X_bin = X_sparse[nk_or_t_mask]
            X_bin = self._preprocess_X(
                X_bin, fit=False, kept_idx=self.kept_idx_bin
            )

            proba_bin = self.pipe_bin.predict_proba(X_bin)
            bin_classes = self.le_bin.classes_
            idxs = np.where(nk_or_t_mask)[0]

            for j, i in enumerate(idxs):
                for k, cls in enumerate(bin_classes):
                    proba_final[i, self.class_to_idx[cls]] = proba_bin[j, k]

        return proba_final
