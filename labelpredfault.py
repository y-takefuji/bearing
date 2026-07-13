import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor, KernelDensity, NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN, MeanShift, OPTICS, estimate_bandwidth, KMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from scipy.spatial.distance import cdist

# =========================================================
# 1. Load dataset
# =========================================================
df = pd.read_csv("feature_time_48k_2048_load_1.csv")

print("=" * 60)
print("Dataset shape:", df.shape)
print("=" * 60)

target_col = "fault"

# ---------------------------------------------------------
# Drop identifier-like columns (avoid leakage from IDs/index)
# ---------------------------------------------------------
id_like_cols = [c for c in df.columns
                if c.lower() in ("id", "index", "unnamed: 0") or c.lower().startswith("unnamed")]
if id_like_cols:
    print(f"Dropping identifier-like columns to avoid leakage: {id_like_cols}")
    df = df.drop(columns=id_like_cols)

# ---------------------------------------------------------
# Drop fully duplicated rows (features + target)
# ---------------------------------------------------------
n_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
n_after = len(df)
if n_after < n_before:
    print(f"Dropped {n_before - n_after} fully duplicated rows (features+target) "
          f"to avoid train/test leakage.")

# ---------------------------------------------------------
# Drop rows that are duplicated on FEATURES ONLY
# ---------------------------------------------------------
feature_cols_for_dup_check = [c for c in df.columns if c != target_col]
n_before_feat = len(df)
df = df.drop_duplicates(subset=feature_cols_for_dup_check).reset_index(drop=True)
n_after_feat = len(df)
if n_after_feat < n_before_feat:
    print(f"Dropped {n_before_feat - n_after_feat} additional rows with duplicated "
          f"FEATURES ONLY (different/duplicate label) to avoid train/test leakage.")

X = df.drop(columns=[target_col])
y_raw = df[target_col].astype(str)

print("\nTarget distribution ('fault'):")
print(y_raw.value_counts())
print("=" * 60)

X_values_all = X.values

# =========================================================
# 1b. Train/Test split FIRST (prevents leakage)
# =========================================================
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_values_all, y_raw.values,
    test_size=0.2,
    random_state=42,
    stratify=y_raw.values
)

def _rows_to_hashable(arr, decimals=8):
    return set(map(tuple, np.round(arr.astype(float), decimals)))

train_set = _rows_to_hashable(X_train_raw)
test_set = _rows_to_hashable(X_test_raw)
overlap = train_set & test_set
if len(overlap) > 0:
    raise ValueError(
        f"Leakage detected: {len(overlap)} identical feature rows found in "
        f"both train and test sets after split."
    )
else:
    print("Leakage check passed: no identical feature rows between train and test.")

le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
classes = le.classes_
n_classes = len(classes)

try:
    y_test = le.transform(y_test_raw)
except ValueError as e:
    raise ValueError(
        "Test set contains classes not seen in training set "
        f"(should not happen with stratified split): {e}"
    )

# =========================================================
# 1c. No scaling — use raw features directly.
# =========================================================
X_train = X_train_raw
X_test = X_test_raw

print("\nClasses (fit on TRAIN only):", list(classes))
print("Number of classes:", n_classes)
print("=" * 60)

print(f"\nTrain size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")
print("=" * 60)


def safe_mode(arr, fallback=0):
    arr = np.asarray(arr)
    if arr.size == 0:
        return fallback
    try:
        result = mode(arr, keepdims=False)
        return int(result.mode)
    except TypeError:
        result = mode(arr)
        return int(np.ravel(result.mode)[0])


# =========================================================
# 2. One-vs-Rest unsupervised models (IF, LOF, KDE, GMM)
# =========================================================
def run_ovr_unsupervised(model_fn, X_train, y_train, X_test, n_classes,
                          score_type="decision", normalize_scores=True):
    n_test_samples = X_test.shape[0]
    score_matrix = np.zeros((n_test_samples, n_classes))

    for c in range(n_classes):
        X_class_train = X_train[y_train == c]
        n_c = X_class_train.shape[0]

        if n_c == 0:
            print(f"Warning: class {c} has no training samples; assigning -inf scores.")
            score_matrix[:, c] = -np.inf
            continue

        model = model_fn(n_samples=n_c, X_class_train=X_class_train)

        try:
            model.fit(X_class_train)

            if score_type == "decision":
                train_scores = model.decision_function(X_class_train)
                test_scores = model.decision_function(X_test)
            elif score_type in ("density", "gmm"):
                train_scores = model.score_samples(X_class_train)
                test_scores = model.score_samples(X_test)
            else:
                raise ValueError(f"Unknown score_type: {score_type}")

            if normalize_scores:
                mu = np.mean(train_scores)
                sigma = np.std(train_scores)
                if sigma < 1e-12:
                    sigma = 1e-12
                scores = (test_scores - mu) / sigma
            else:
                scores = test_scores

        except Exception as e:
            print(f"Warning: class {c} model fitting failed: {e}")
            scores = np.full(n_test_samples, -np.inf)

        score_matrix[:, c] = scores

    y_pred = np.argmax(score_matrix, axis=1)
    return y_pred


# =========================================================
# 3. URF: unsupervised-RF proximity + clustering
# =========================================================
def urf_proximity_classifier(X_train, y_train, X_test, n_classes,
                              n_estimators=500, random_state=42):
    rng = np.random.RandomState(random_state)
    n_train = X_train.shape[0]

    X_synthetic = np.column_stack([
        rng.choice(X_train[:, col], size=n_train, replace=True)
        for col in range(X_train.shape[1])
    ])

    X_urf = np.vstack([X_train, X_synthetic])
    y_urf = np.concatenate([np.ones(n_train), np.zeros(n_train)])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        bootstrap=True,
        n_jobs=-1,
        max_features="sqrt"
    )
    clf.fit(X_urf, y_urf)

    leaves_train = clf.apply(X_train)
    leaves_test = clf.apply(X_test)

    k = min(n_classes, n_train)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_labels_train = kmeans.fit_predict(leaves_train.astype(float))

    global_majority = safe_mode(y_train)

    cluster_to_class = {}
    for cl in np.unique(cluster_labels_train):
        mask = cluster_labels_train == cl
        if mask.sum() == 0:
            continue
        cluster_to_class[cl] = safe_mode(y_train[mask], fallback=global_majority)

    cluster_labels_test = kmeans.predict(leaves_test.astype(float))
    y_pred = np.array([
        cluster_to_class.get(cl, global_majority) for cl in cluster_labels_test
    ])

    return y_pred


# =========================================================
# 3b. Clustering-based classification helper
#     (DBSCAN / Mean Shift / OPTICS)
# =========================================================
def clustering_based_classifier(model_fn, X_train, y_train, X_test, n_classes,
                                 model_name=""):
    model = model_fn()
    cluster_labels_train = model.fit_predict(X_train)

    unique_clusters = np.unique(cluster_labels_train)

    global_majority = safe_mode(y_train)

    cluster_to_class = {}
    cluster_centroids = {}

    for cl in unique_clusters:
        if cl == -1:
            continue
        mask = cluster_labels_train == cl
        if mask.sum() == 0:
            continue
        cluster_to_class[cl] = safe_mode(y_train[mask], fallback=global_majority)
        cluster_centroids[cl] = X_train[mask].mean(axis=0)

    if len(cluster_centroids) == 0:
        print(f"Warning: {model_name} produced no valid clusters; "
              f"falling back to majority class for all predictions.")
        return np.full(X_test.shape[0], global_majority)

    centroid_ids = list(cluster_centroids.keys())
    centroid_matrix = np.vstack([cluster_centroids[cl] for cl in centroid_ids])
    centroid_classes = np.array([cluster_to_class[cl] for cl in centroid_ids])

    n_noise_train = np.sum(cluster_labels_train == -1)
    if n_noise_train > 0:
        print(f"  ({model_name}) {n_noise_train} noise points in training "
              f"set assigned to nearest centroid's class.")

    dists = cdist(X_test, centroid_matrix, metric="euclidean")
    nearest_idx = np.argmin(dists, axis=1)
    y_pred = centroid_classes[nearest_idx]

    n_clusters_found = len(centroid_ids)
    print(f"  ({model_name}) Found {n_clusters_found} usable cluster(s) "
          f"(excluding noise).")

    return y_pred


# =========================================================
# 3c. Heuristics for eps (DBSCAN), bandwidth (MeanShift), KDE bandwidth
# =========================================================
def estimate_dbscan_eps(X_train, k=5):
    k_eff = min(k, X_train.shape[0] - 1)
    if k_eff < 1:
        return 1e-3
    nn = NearestNeighbors(n_neighbors=k_eff + 1)
    nn.fit(X_train)
    dists, _ = nn.kneighbors(X_train)
    kth_dists = dists[:, -1]
    eps = np.median(kth_dists)
    return max(eps, 1e-3)


def estimate_meanshift_bandwidth(X_train, quantile=0.15):
    """
    FIX: default quantile lowered from 0.3 -> 0.15.
    quantile=0.3 was systematically overestimating bandwidth on
    this type of dataset, collapsing Mean Shift into a single
    giant cluster (i.e. degenerating to "predict majority class"),
    which explains the poor observed performance.
    """
    bw = estimate_bandwidth(X_train, quantile=quantile, n_samples=min(500, len(X_train)))
    if bw <= 0:
        bw = 1.0  # fallback for degenerate cases
    return bw


def estimate_meanshift_bandwidth_auto(X_train, y_train, n_classes,
                                       quantiles=(0.15, 0.1, 0.07, 0.05, 0.03),
                                       min_clusters=2):
    """
    FIX (main fix for Mean Shift poor performance):
    Iteratively shrink the bandwidth quantile until Mean Shift
    finds at least `min_clusters` clusters (ideally >= n_classes)
    on the TRAINING data. This avoids the common failure mode
    where a single too-large bandwidth merges every class into
    one cluster, making Mean Shift equivalent to a majority-class
    predictor.

    All estimation happens on X_train only -> no leakage.
    """
    best_bw = None
    best_n_clusters = -1

    for q in quantiles:
        bw = estimate_bandwidth(X_train, quantile=q, n_samples=min(500, len(X_train)))
        if bw <= 0:
            continue

        trial_model = MeanShift(bandwidth=bw, bin_seeding=True)
        trial_labels = trial_model.fit_predict(X_train)
        n_clusters_found = len(np.unique(trial_labels[trial_labels != -1]))

        print(f"    [MeanShift bandwidth search] quantile={q:.3g} -> "
              f"bandwidth={bw:.3g} -> {n_clusters_found} cluster(s)")

        if n_clusters_found > best_n_clusters:
            best_n_clusters = n_clusters_found
            best_bw = bw

        # Stop as soon as we have a reasonably informative clustering
        if n_clusters_found >= max(min_clusters, n_classes):
            return bw, n_clusters_found

    # If nothing reached the target, return the best we found
    if best_bw is None:
        best_bw = 1.0
    return best_bw, best_n_clusters


def estimate_kde_bandwidth(X_class_train, k=5):
    n_samples, n_features = X_class_train.shape
    if n_samples < 2:
        return 1.0

    k_eff = min(k, n_samples - 1)
    if k_eff >= 1:
        nn = NearestNeighbors(n_neighbors=k_eff + 1)
        nn.fit(X_class_train)
        dists, _ = nn.kneighbors(X_class_train)
        bw = np.median(dists[:, -1])
        if bw > 1e-12:
            return bw

    sigma = np.std(X_class_train)
    bw = sigma * (n_samples ** (-1.0 / (n_features + 4)))
    return max(bw, 1e-3)


# =========================================================
# 4. Run each model
# =========================================================
results = {}

print("\nRunning Isolation Forest (IF)...")


def _make_if(n_samples, X_class_train=None):
    max_samples = min(256, n_samples)
    return IsolationForest(random_state=42, max_samples=max_samples)


y_pred_if = run_ovr_unsupervised(
    _make_if, X_train, y_train, X_test, n_classes, score_type="decision"
)
results["Isolation Forest (IF)"] = y_pred_if

print("Running Local Outlier Factor (LOF)...")


def _make_lof(n_samples, X_class_train=None):
    n_neighbors = min(20, max(1, n_samples - 1))
    return LocalOutlierFactor(novelty=True, n_neighbors=n_neighbors)


y_pred_lof = run_ovr_unsupervised(
    _make_lof, X_train, y_train, X_test, n_classes, score_type="decision"
)
results["Local Outlier Factor (LOF)"] = y_pred_lof

print("Running Kernel Density Estimation (KDE) [bandwidth tuned per-class on TRAIN only]...")


def _make_kde(n_samples, X_class_train):
    bw = estimate_kde_bandwidth(X_class_train, k=5)
    return KernelDensity(bandwidth=bw)


y_pred_kde = run_ovr_unsupervised(
    _make_kde, X_train, y_train, X_test, n_classes, score_type="density"
)
results["Kernel Density Estimation (KDE)"] = y_pred_kde

print("Running Gaussian Mixture Model (GMM)...")


def _make_gmm(n_samples, X_class_train=None):
    n_components = 1
    if n_samples < 2:
        return GaussianMixture(n_components=n_components, random_state=42, reg_covar=1e-3)
    return GaussianMixture(n_components=n_components, random_state=42)


y_pred_gmm = run_ovr_unsupervised(
    _make_gmm, X_train, y_train, X_test, n_classes, score_type="gmm"
)
results["Gaussian Mixture Model (GMM)"] = y_pred_gmm

print("Running Unsupervised Random Forest (URF) [proximity + clustering, fit on TRAIN only]...")
y_pred_urf = urf_proximity_classifier(X_train, y_train, X_test, n_classes)
results["Unsupervised Random Forest (URF)"] = y_pred_urf

print("Running DBSCAN [eps tuned via k-distance heuristic on TRAIN only]...")
dbscan_eps = estimate_dbscan_eps(X_train, k=5)
print(f"  Estimated DBSCAN eps = {dbscan_eps:.3g}")
y_pred_dbscan = clustering_based_classifier(
    lambda: DBSCAN(eps=dbscan_eps, min_samples=5),
    X_train, y_train, X_test, n_classes, model_name="DBSCAN"
)
results["DBSCAN"] = y_pred_dbscan

print("Running Mean Shift [bandwidth auto-searched to avoid single-cluster collapse]...")
ms_bandwidth, ms_n_clusters_found = estimate_meanshift_bandwidth_auto(
    X_train, y_train, n_classes
)
print(f"  Selected Mean Shift bandwidth = {ms_bandwidth:.3g} "
      f"(found {ms_n_clusters_found} cluster(s) on TRAIN)")
if ms_n_clusters_found < 2:
    print("  Warning: Mean Shift still collapsed to <2 clusters even after "
          "bandwidth search; predictions will fall back to majority class.")

y_pred_meanshift = clustering_based_classifier(
    lambda: MeanShift(bandwidth=ms_bandwidth, bin_seeding=True),
    X_train, y_train, X_test, n_classes, model_name="Mean Shift"
)
results["Mean Shift"] = y_pred_meanshift

print("Running OPTICS...")
y_pred_optics = clustering_based_classifier(
    lambda: OPTICS(),
    X_train, y_train, X_test, n_classes, model_name="OPTICS"
)
results["OPTICS"] = y_pred_optics

print("\nAll models completed.")
print("=" * 60)


# =========================================================
# 5. Evaluation
# =========================================================
summary_rows = []

for method_name, y_pred in results.items():
    acc = accuracy_score(y_test, y_pred)
    f1_per_class = f1_score(y_test, y_pred, average=None, labels=range(n_classes))

    row = {"method": method_name, "accuracy": round(acc, 3)}
    for i, cls_name in enumerate(classes):
        row[f"F1_{cls_name}"] = round(f1_per_class[i], 3)

    summary_rows.append(row)

    print(f"\n[{method_name}]")
    print(f"  Accuracy: {acc:.3g}")
    for i, cls_name in enumerate(classes):
        print(f"  F1 ({cls_name}): {f1_per_class[i]:.3g}")

# =========================================================
# 6. Build and save summary table
# =========================================================
summary_df = pd.DataFrame(summary_rows)

print("\n" + "=" * 60)
print("Summary Table:")
print(summary_df.to_string(index=False))
print("=" * 60)

summary_df.to_csv("result.csv", index=False)
print("\nSaved summary table to 'result.csv'")