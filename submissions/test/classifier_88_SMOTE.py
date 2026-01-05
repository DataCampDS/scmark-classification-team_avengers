from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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


def smote_resample(X, y, k_neighbors=5, random_state=42):
    """
    SMOTE implementation without imblearn.
    
    Args:
        X: feature matrix (n_samples, n_features)
        y: labels array
        k_neighbors: number of neighbors for SMOTE
        random_state: random seed
        
    Returns:
        X_resampled, y_resampled: balanced dataset
    """
    np.random.seed(random_state)
    
    # Convert to numpy arrays
    if issparse(X):
        X = X.toarray()
    X = np.array(X)
    y = np.array(y)
    
    # Get class counts
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    
    X_resampled = []
    y_resampled = []
    
    for cls, count in zip(classes, counts):
        # Get samples of this class
        cls_mask = y == cls
        X_cls = X[cls_mask]
        
        # Add original samples
        X_resampled.append(X_cls)
        y_resampled.extend([cls] * count)
        
        # Number of synthetic samples to generate
        n_synthetic = max_count - count
        
        if n_synthetic > 0:
            # Generate synthetic samples using SMOTE logic
            synthetic_samples = []
            
            for _ in range(n_synthetic):
                # Pick a random sample
                idx = np.random.randint(0, len(X_cls))
                sample = X_cls[idx]
                
                # Find k nearest neighbors in the same class
                # Use Euclidean distance
                distances = np.sqrt(((X_cls - sample) ** 2).sum(axis=1))
                # Exclude the sample itself (distance = 0)
                neighbor_indices = np.argsort(distances)[1:k_neighbors+1]
                
                # Pick a random neighbor
                neighbor_idx = np.random.choice(neighbor_indices)
                neighbor = X_cls[neighbor_idx]
                
                # Generate synthetic sample
                # interpolation between sample and neighbor
                alpha = np.random.random()
                synthetic = sample + alpha * (neighbor - sample)
                synthetic_samples.append(synthetic)
            
            if synthetic_samples:
                X_resampled.append(np.array(synthetic_samples))
                y_resampled.extend([cls] * n_synthetic)
    
    X_resampled = np.vstack(X_resampled)
    y_resampled = np.array(y_resampled)
    
    # Shuffle the resampled data
    shuffle_idx = np.random.permutation(len(y_resampled))
    X_resampled = X_resampled[shuffle_idx]
    y_resampled = y_resampled[shuffle_idx]
    
    return X_resampled, y_resampled


class Classifier(object):
    def __init__(self,
                 # SMOTE parameters
                 use_smote=True,
                 smote_k_neighbors=5,
                 
                 # Stage 1 (Merged model) parameters
                 merge_variance_threshold=1.2,
                 merge_pca_components=60,
                 merge_rf_n_estimators=200,
                 merge_rf_max_depth=50,
                 merge_rf_max_features="sqrt",
                 merge_svc_C=1.0,
                 merge_svc_gamma="scale",
                 merge_svc_kernel="rbf",
                 merge_knn_n_neighbors=15,
                 merge_lr_max_iter=10000,
                 merge_lr_solver="lbfgs",
                 
                 # Stage 2 (Binary model) parameters
                 binary_variance_threshold=0.8,
                 binary_pca_components=0.8,
                 binary_rf_n_estimators=200,
                 binary_rf_max_depth=50,
                 binary_rf_max_features="sqrt",
                 binary_svc_C=1.0,
                 binary_svc_gamma="scale",
                 binary_svc_kernel="rbf",
                 binary_knn_n_neighbors=15,
                 binary_lr_max_iter=10000,
                 binary_lr_solver="lbfgs",
                 
                 # General parameters
                 random_state=42,
                 n_jobs=-1):
        """
        Two-stage classifier with optional SMOTE balancing
        """
        # SMOTE parameters
        self.use_smote = use_smote
        self.smote_k_neighbors = smote_k_neighbors
        
        # Store all parameters
        self.merge_variance_threshold = merge_variance_threshold
        self.merge_pca_components = merge_pca_components
        self.merge_rf_n_estimators = merge_rf_n_estimators
        self.merge_rf_max_depth = merge_rf_max_depth
        self.merge_rf_max_features = merge_rf_max_features
        self.merge_svc_C = merge_svc_C
        self.merge_svc_gamma = merge_svc_gamma
        self.merge_svc_kernel = merge_svc_kernel
        self.merge_knn_n_neighbors = merge_knn_n_neighbors
        self.merge_lr_max_iter = merge_lr_max_iter
        self.merge_lr_solver = merge_lr_solver
        
        self.binary_variance_threshold = binary_variance_threshold
        self.binary_pca_components = binary_pca_components
        self.binary_rf_n_estimators = binary_rf_n_estimators
        self.binary_rf_max_depth = binary_rf_max_depth
        self.binary_rf_max_features = binary_rf_max_features
        self.binary_svc_C = binary_svc_C
        self.binary_svc_gamma = binary_svc_gamma
        self.binary_svc_kernel = binary_svc_kernel
        self.binary_knn_n_neighbors = binary_knn_n_neighbors
        self.binary_lr_max_iter = binary_lr_max_iter
        self.binary_lr_solver = binary_lr_solver
        
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Label encoders for both stages
        self.le_merge = LabelEncoder()
        self.le_bin = LabelEncoder()
        
        # Gene filtering indices
        self.kept_idx_merge = None
        self.kept_idx_bin = None
        
        # ===== STAGE 1: Merged Model =====
        rf_merge = RandomForestClassifier(
            n_estimators=merge_rf_n_estimators,
            max_depth=merge_rf_max_depth,
            max_features=merge_rf_max_features,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        svc_merge = SVC(
            kernel=merge_svc_kernel,
            probability=True,
            C=merge_svc_C,
            gamma=merge_svc_gamma,
            random_state=random_state
        )
        
        knn_merge = KNeighborsClassifier(
            n_neighbors=merge_knn_n_neighbors,
            n_jobs=n_jobs
        )
        
        stack_merge = StackingClassifier(
            estimators=[("rf", rf_merge), ("svc", svc_merge), ("knn", knn_merge)],
            final_estimator=LogisticRegression(
                max_iter=merge_lr_max_iter,
                solver=merge_lr_solver,
                random_state=random_state
            ),
            n_jobs=n_jobs,
            passthrough=False
        )
        
        self.pipe_merge = make_pipeline(
            PCA(n_components=merge_pca_components, random_state=random_state),
            stack_merge
        )
        
        # ===== STAGE 2: Binary Model =====
        rf_bin = RandomForestClassifier(
            n_estimators=binary_rf_n_estimators,
            max_depth=binary_rf_max_depth,
            max_features=binary_rf_max_features,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        svc_bin = SVC(
            kernel=binary_svc_kernel,
            probability=True,
            C=binary_svc_C,
            gamma=binary_svc_gamma,
            random_state=random_state
        )
        
        knn_bin = KNeighborsClassifier(
            n_neighbors=binary_knn_n_neighbors,
            n_jobs=n_jobs
        )
        
        stack_bin = StackingClassifier(
            estimators=[("rf", rf_bin), ("svc", svc_bin), ("knn", knn_bin)],
            final_estimator=LogisticRegression(
                max_iter=binary_lr_max_iter,
                solver=binary_lr_solver,
                random_state=random_state
            ),
            n_jobs=n_jobs,
            passthrough=False
        )
        
        self.pipe_bin = make_pipeline(
            PCA(n_components=binary_pca_components, random_state=random_state),
            stack_bin
        )

    def _preprocess_X(self, X, fit=False, kept_idx=None, threshold=0.01):
        """
        Preprocess features: filter genes + library size normalization + log1p
        """
        # Convert sparse to dense if needed
        if issparse(X):
            X = X.toarray()
        
        # Gene filtering
        if fit:
            X_filtered, kept_idx = remove_genes(X, threshold=threshold)
            return_idx = kept_idx
        else:
            if kept_idx is None:
                raise ValueError("kept_idx must be provided when fit=False")
            X_filtered = X[:, kept_idx]
            return_idx = None
        
        # Library size normalization + log1p
        libsize = X_filtered.sum(axis=1)[:, None]
        libsize[libsize == 0] = 1
        X_processed = np.log1p(X_filtered / libsize * 1e4)
        
        if fit:
            return X_processed, return_idx
        else:
            return X_processed

    def fit(self, X_sparse, y):
        """
        Train the two-stage classifier with optional SMOTE
        
        Stage 1: Train merged model (3 classes)
        Stage 2: Train binary model (NK_cells vs T_cells_CD8+)
        """
        # Convert y to numpy array
        y_arr = np.array(y).astype(str)
        
        # ===== STAGE 1: Merged Model (3 classes) =====
        # Merge NK_cells and T_cells_CD8+ into one class
        merge_from = {"NK_cells", "T_cells_CD8+"}
        self.merged_label = "NK_or_T_cell"
        
        y_merged = y_arr.copy()
        y_merged[np.isin(y_merged, list(merge_from))] = self.merged_label
        
        # Preprocess and fit merged model
        X_merge, self.kept_idx_merge = self._preprocess_X(
            X_sparse, fit=True, threshold=self.merge_variance_threshold
        )
        
        # Apply SMOTE if enabled
        if self.use_smote:
            X_merge, y_merged = smote_resample(
                X_merge, y_merged, 
                k_neighbors=self.smote_k_neighbors,
                random_state=self.random_state
            )
        
        y_merge_enc = self.le_merge.fit_transform(y_merged)
        self.pipe_merge.fit(X_merge, y_merge_enc)
        
        # ===== STAGE 2: Binary Model (NK_cells vs T_cells_CD8+) =====
        # Filter to only NK_cells and T_cells_CD8+
        binary_classes = {"NK_cells", "T_cells_CD8+"}
        bin_mask = np.isin(y_arr, list(binary_classes))
        
        X_bin = X_sparse[bin_mask] if issparse(X_sparse) else X_sparse[bin_mask]
        y_bin = y_arr[bin_mask]
        
        # Preprocess with higher variance threshold for binary model
        X_bin_processed, self.kept_idx_bin = self._preprocess_X(
            X_bin, fit=True, threshold=self.binary_variance_threshold
        )
        
        # Apply SMOTE if enabled
        if self.use_smote:
            X_bin_processed, y_bin = smote_resample(
                X_bin_processed, y_bin,
                k_neighbors=self.smote_k_neighbors,
                random_state=self.random_state
            )
        
        y_bin_enc = self.le_bin.fit_transform(y_bin)
        self.pipe_bin.fit(X_bin_processed, y_bin_enc)
        
        # Store original classes for compatibility
        self.classes_ = np.unique(y_arr)
        
        # Create mapping from class names to indices
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}

    def predict_proba(self, X_sparse):
        """
        Two-stage prediction:
        1. Predict with merged model (3 classes)
        2. If predicted as NK_or_T_cell, refine with binary model
        """
        n_samples = X_sparse.shape[0]
        n_classes = len(self.classes_)
        
        # Initialize probability matrix
        proba_final = np.zeros((n_samples, n_classes))
        
        # ===== STAGE 1: Predict with merged model =====
        X_merge = self._preprocess_X(X_sparse, fit=False, kept_idx=self.kept_idx_merge)
        proba_merge = self.pipe_merge.predict_proba(X_merge)
        y_pred_merge_enc = self.pipe_merge.predict(X_merge)
        y_pred_merge = self.le_merge.inverse_transform(y_pred_merge_enc)
        
        # ===== STAGE 2: Process predictions =====
        nk_or_t_mask = y_pred_merge == self.merged_label
        
        # For samples NOT predicted as merged class
        for i in np.where(~nk_or_t_mask)[0]:
            pred_label = y_pred_merge[i]
            class_idx = self.class_to_idx[pred_label]
            # Use probabilities from merged model
            merge_class_idx = self.le_merge.transform([pred_label])[0]
            proba_final[i, class_idx] = proba_merge[i, merge_class_idx]
        
        # For samples predicted as merged class, refine with binary model
        if nk_or_t_mask.sum() > 0:
            # Get samples predicted as merged class
            X_bin = X_sparse[nk_or_t_mask] if issparse(X_sparse) else X_sparse[nk_or_t_mask]
            
            # Preprocess with binary model's gene filtering
            X_bin_processed = self._preprocess_X(X_bin, fit=False, kept_idx=self.kept_idx_bin)
            
            # Predict with binary model
            proba_bin = self.pipe_bin.predict_proba(X_bin_processed)
            
            # Map binary probabilities to final matrix
            bin_classes = self.le_bin.classes_
            nk_or_t_indices = np.where(nk_or_t_mask)[0]
            
            for j, sample_idx in enumerate(nk_or_t_indices):
                for k, bin_class in enumerate(bin_classes):
                    class_idx = self.class_to_idx[bin_class]
                    proba_final[sample_idx, class_idx] = proba_bin[j, k]
        
        return proba_final