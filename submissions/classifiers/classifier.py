"""
Hierarchical RNA-seq classifier for immune cell type classification.
"""
import warnings
import numpy as np
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import lightgbm as lgb

# Disable warnings
warnings.filterwarnings('ignore')
sc.settings.verbosity = 0


class RNAseqPreprocessor:
    """
    Preprocessor for RNA-seq data using scanpy.

    Performs normalization, log-transformation,
    and highly variable gene selection.
    """

    def __init__(self, n_top_genes=5000):
        """
        Initialize preprocessor.

        Args:
            n_top_genes (int): Number of highly variable genes to select
        """
        self.n_top_genes = n_top_genes
        self.var_genes_idx = None

    def fit(self, X, y=None):
        """
        Fit preprocessor on training data.

        Args:
            X: Feature matrix (sparse or dense)
            y: Labels (not used, for sklearn compatibility)

        Returns:
            self: Fitted preprocessor
        """
        X = X.toarray() if hasattr(X, "toarray") else X
        adata = sc.AnnData(X)

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        sc.pp.highly_variable_genes(adata, n_top_genes=self.n_top_genes)
        self.var_genes_idx = np.where(adata.var["highly_variable"])[0]
        return self

    def transform(self, X):
        """
        Transform data using fitted preprocessor.

        Args:
            X: Feature matrix (sparse or dense)

        Returns:
            X_transformed: Preprocessed feature matrix
        """
        X = X.toarray() if hasattr(X, "toarray") else X
        adata = sc.AnnData(X)

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        return adata[:, self.var_genes_idx].X


class Classifier:
    """
    Hierarchical classifier for immune cell types.

    Three-stage classification:
    1. Cancer cells vs Others
    2. NK cells vs T cells (within Others)
    3. CD4+ vs CD8+ T cells (within T cells)
    """

    def __init__(self, n_top_genes=5000):
        """
        Initialize hierarchical classifier.

        Args:
            n_top_genes (int): Number of highly variable genes
        """
        self.preprocessor = RNAseqPreprocessor(n_top_genes)

        # Stage 1: Cancer vs Others
        self.cancer_detector = Pipeline([
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                multi_class="auto"
            ))
        ])

        # Stage 2: NK vs T cells
        self.nk_t_detector = Pipeline([
            ("lgb", lgb.LGBMClassifier(
                n_estimators=2000,
                learning_rate=0.05,
                max_depth=5,
                class_weight='balanced',
                reg_alpha=0.7,
                reg_lambda=0.3,
                scale_pos_weight=678 / 85,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            ))
        ])

        # Stage 3: CD4+ vs CD8+ T cells
        self.cd4_cd8_detector = Pipeline([
            ("svc", SVC(
                kernel='linear',
                C=1.0,
                class_weight='balanced',
                random_state=42,
                probability=True,
                max_iter=2000
            ))
        ])

        # Safety flags for cross-validation
        self.has_nk_t = False
        self.has_cd4_cd8 = False

    def fit(self, X_sparse, y):
        """
        Fit hierarchical classifier.

        Args:
            X_sparse: Feature matrix (sparse or dense)
            y: Labels array

        Returns:
            self: Fitted classifier
        """
        X = self.preprocessor.fit(X_sparse, y).transform(X_sparse)
        y = np.array(y)

        # Stage 1: Cancer vs Others
        is_cancer = (y == "Cancer_cells")
        self.cancer_detector.fit(X, is_cancer)

        # Filter to non-cancer cells
        X_other = X[~is_cancer]
        y_other = y[~is_cancer]

        # Stage 2: NK vs T cells
        mask_nk = (y_other == "NK_cells")
        mask_t = np.isin(y_other, ["T_cells_CD4+", "T_cells_CD8+"])

        X_nk_t = X_other[mask_nk | mask_t]
        y_nk_t = mask_nk[mask_nk | mask_t]  # NK=True, T=False

        if len(np.unique(y_nk_t)) == 2:
            self.nk_t_detector.fit(X_nk_t, y_nk_t)
            self.has_nk_t = True
        else:
            self.has_nk_t = False

        # Stage 3: CD4+ vs CD8+ T cells
        X_t = X_other[mask_t]
        y_t = y_other[mask_t]

        y_cd4 = (y_t == "T_cells_CD4+")

        if len(np.unique(y_cd4)) == 2:
            self.cd4_cd8_detector.fit(X_t, y_cd4)
            self.has_cd4_cd8 = True
        else:
            self.has_cd4_cd8 = False

        return self

    def predict(self, X_sparse):
        """
        Predict cell types using hierarchical classification.

        Args:
            X_sparse: Feature matrix (sparse or dense)

        Returns:
            predictions: Array of predicted cell types
        """
        X = self.preprocessor.transform(X_sparse)
        preds = []

        is_cancer = self.cancer_detector.predict(X)

        for i, cancer in enumerate(is_cancer):
            if cancer:
                preds.append("Cancer_cells")
                continue

            # Stage 2: NK vs T
            if self.has_nk_t:
                is_nk = self.nk_t_detector.predict(X[i].reshape(1, -1))[0]
            else:
                is_nk = False

            if is_nk:
                preds.append("NK_cells")
                continue

            # Stage 3: CD4+ vs CD8+
            if self.has_cd4_cd8:
                is_cd4 = self.cd4_cd8_detector.predict(
                    X[i].reshape(1, -1)
                )[0]
                preds.append("T_cells_CD4+" if is_cd4 else "T_cells_CD8+")
            else:
                preds.append("T_cells_CD4+")

        return np.array(preds)

    def predict_proba(self, X_sparse):
        """
        Predict class probabilities using hierarchical classification.

        Args:
            X_sparse: Feature matrix (sparse or dense)

        Returns:
            proba: Probability matrix (n_samples, n_classes)
        """
        X = self.preprocessor.transform(X_sparse)
        n = X.shape[0]

        # Fixed class order
        classes = [
            "Cancer_cells",
            "NK_cells",
            "T_cells_CD4+",
            "T_cells_CD8+"
        ]

        proba = np.zeros((n, len(classes)))

        # Stage 1: Cancer vs Others
        p_cancer = self.cancer_detector.predict_proba(X)[:, 1]
        p_other = 1.0 - p_cancer

        # Stage 2: NK vs T cells
        if self.has_nk_t:
            p_nk = self.nk_t_detector.predict_proba(X)[:, 1]
            p_t = 1.0 - p_nk
        else:
            # Fallback for cross-validation
            p_nk = np.zeros(n)
            p_t = np.ones(n)

        # Stage 3: CD4+ vs CD8+
        if self.has_cd4_cd8:
            p_cd4 = self.cd4_cd8_detector.predict_proba(X)[:, 1]
            p_cd8 = 1.0 - p_cd4
        else:
            # Fallback for cross-validation
            p_cd4 = np.ones(n)
            p_cd8 = np.zeros(n)

        # Hierarchical probability combination
        proba[:, 0] = p_cancer
        proba[:, 1] = p_other * p_nk
        proba[:, 2] = p_other * p_t * p_cd4
        proba[:, 3] = p_other * p_t * p_cd8

        return proba
