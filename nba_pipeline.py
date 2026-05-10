"""
CS439 Final Project: NBA Player Archetype Segmentation and Career Longevity Prediction
Full Pipeline Script
"""

# ─────────────────────────────────────────────
# 0. Imports & Setup
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance

plt.rcParams.update({"figure.dpi": 130, "font.size": 11})
SEED = 42

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
df_raw = pd.read_csv("all_seasons.csv")
print(f"Raw shape: {df_raw.shape}")
print(df_raw.head(3))

# ─────────────────────────────────────────────
# 2. Data Preprocessing
# ─────────────────────────────────────────────

# 2a. Drop index column
df = df_raw.drop(columns=["Unnamed: 0"]).copy()

# 2b. Filter out players with very few games (noise reduction)
df = df[df["gp"] >= 10].reset_index(drop=True)

# 2c. Engineer TARGET: career_longevity (1 if player appeared in 5+ distinct seasons)
seasons_per_player = df.groupby("player_name")["season"].nunique()
long_career_players = set(seasons_per_player[seasons_per_player >= 5].index)
df["career_longevity"] = df["player_name"].apply(lambda x: 1 if x in long_career_players else 0)
print(f"\nTarget distribution:\n{df['career_longevity'].value_counts()}")

# 2d. Aggregate to one row per player (use their career averages for rookie/first season)
#     This prevents data leakage — we predict longevity from FIRST-SEASON stats only
df_sorted = df.sort_values(["player_name", "season"])
df_first = df_sorted.groupby("player_name").first().reset_index()

print(f"\nAfter aggregation (one row per player): {df_first.shape}")

# 2e. Encode draft features
df_first["draft_round"] = df_first["draft_round"].replace("Undrafted", "0")
df_first["draft_number"] = df_first["draft_number"].replace("Undrafted", "0")
df_first["draft_round"] = pd.to_numeric(df_first["draft_round"], errors="coerce").fillna(0).astype(int)
df_first["draft_number"] = pd.to_numeric(df_first["draft_number"], errors="coerce").fillna(0).astype(int)
df_first["is_undrafted"] = (df_first["draft_round"] == 0).astype(int)

# 2f. Binary encode country (USA vs international)
df_first["is_international"] = (df_first["country"] != "USA").astype(int)

# 2g. Define feature set for modeling
FEATURE_COLS = [
    "age", "player_height", "player_weight",
    "gp", "pts", "reb", "ast",
    "net_rating", "oreb_pct", "dreb_pct",
    "usg_pct", "ts_pct", "ast_pct",
    "draft_round", "draft_number", "is_undrafted", "is_international"
]
TARGET_COL = "career_longevity"

X = df_first[FEATURE_COLS].copy()
y = df_first[TARGET_COL].copy()

# 2h. Handle any remaining NaNs
X = X.fillna(X.median())

# 2i. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)

print(f"\nFinal modeling shape: X={X.shape}, y={y.shape}")
print(f"Class balance: {y.value_counts().to_dict()}")

# ─────────────────────────────────────────────
# 3. Customer Segmentation via K-Means
# ─────────────────────────────────────────────

# 3a. Use only on-court performance features (exclude target-leaking & draft features)
CLUSTER_FEATURES = ["pts", "reb", "ast", "usg_pct", "ts_pct",
                    "oreb_pct", "dreb_pct", "ast_pct", "net_rating"]
X_cluster = df_first[CLUSTER_FEATURES].fillna(0)
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# 3b. Elbow Method + Silhouette Score
sse, sil_scores = [], []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X_cluster_scaled)
    sse.append(km.inertia_)
    sil_scores.append(silhouette_score(X_cluster_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K_range, sse, "bo-")
axes[0].set_title("Elbow Method (SSE vs. k)")
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("SSE (Inertia)")
axes[0].axvline(x=4, color="red", linestyle="--", label="k=4 selected")
axes[0].legend()

axes[1].plot(K_range, sil_scores, "gs-")
axes[1].set_title("Silhouette Score vs. k")
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].axvline(x=4, color="red", linestyle="--", label="k=4 selected")
axes[1].legend()

plt.tight_layout()
plt.savefig("fig1_cluster_selection.png", bbox_inches="tight")
plt.close()
print("Saved: fig1_cluster_selection.png")

# 3c. Fit final K-Means with k=4
K_BEST = 4
km_final = KMeans(n_clusters=K_BEST, random_state=SEED, n_init=10)
df_first["cluster"] = km_final.fit_predict(X_cluster_scaled)

# 3d. PCA for 2D visualization
pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_cluster_scaled)
df_first["pca1"] = X_pca[:, 0]
df_first["pca2"] = X_pca[:, 1]

CLUSTER_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868", 3: "#C44E52"}
CLUSTER_NAMES = {
    0: "Interior Bigs",
    1: "Role Players / Bench",
    2: "Playmakers",
    3: "Scorers / Wings"
}

fig, ax = plt.subplots(figsize=(9, 6))
for c in range(K_BEST):
    mask = df_first["cluster"] == c
    ax.scatter(df_first.loc[mask, "pca1"], df_first.loc[mask, "pca2"],
               c=CLUSTER_COLORS[c], label=f"Cluster {c}: {CLUSTER_NAMES[c]}",
               alpha=0.6, s=30)
ax.set_title("NBA Player Segmentation — PCA 2D Visualization")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("fig2_pca_clusters.png", bbox_inches="tight")
plt.close()
print("Saved: fig2_pca_clusters.png")

# 3e. Cluster profile summary
cluster_profile = df_first.groupby("cluster")[CLUSTER_FEATURES + ["career_longevity"]].mean().round(3)
cluster_profile.index = [f"Cluster {i}: {CLUSTER_NAMES[i]}" for i in range(K_BEST)]
print("\nCluster Profile:\n", cluster_profile.T)

# 3f. Churn rate (longevity) per cluster
fig, ax = plt.subplots(figsize=(8, 4))
churn_by_cluster = df_first.groupby("cluster")["career_longevity"].mean()
bars = ax.bar(
    [f"C{i}:\n{CLUSTER_NAMES[i]}" for i in range(K_BEST)],
    churn_by_cluster.values,
    color=[CLUSTER_COLORS[i] for i in range(K_BEST)]
)
ax.set_title("Career Longevity Rate by Player Archetype")
ax.set_ylabel("Proportion with 5+ Season Career")
ax.set_ylim(0, 1)
for bar, val in zip(bars, churn_by_cluster.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("fig3_longevity_by_cluster.png", bbox_inches="tight")
plt.close()
print("Saved: fig3_longevity_by_cluster.png")

# ─────────────────────────────────────────────
# 4. Supervised Churn Prediction Models
# ─────────────────────────────────────────────

# Add cluster as a feature (one-hot encode it)
X_model = X_scaled_df.copy()
for c in range(K_BEST):
    X_model[f"cluster_{c}"] = (df_first["cluster"].values == c).astype(int)

# 4a. Train-test split (stratified 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_model, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# 4b. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8, random_state=SEED, class_weight="balanced"),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=SEED),
}

# 4c. Train, evaluate, collect metrics
results = {}
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob), 4),
    }
    trained_models[name] = (model, y_pred, y_prob)

results_df = pd.DataFrame(results).T
print("\nModel Performance:\n", results_df)

# 4d. ROC Curves
fig, ax = plt.subplots(figsize=(7, 5))
for name, (model, _, y_prob) in trained_models.items():
    RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax)
ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_title("ROC Curves — Career Longevity Prediction")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("fig4_roc_curves.png", bbox_inches="tight")
plt.close()
print("Saved: fig4_roc_curves.png")

# 4e. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, (model, y_pred, _)) in zip(axes, trained_models.items()):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Short (<5yr)", "Long (5+yr)"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.suptitle("Confusion Matrices", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("fig5_confusion_matrices.png", bbox_inches="tight")
plt.close()
print("Saved: fig5_confusion_matrices.png")

# ─────────────────────────────────────────────
# 5. Feature Importance Analysis
# ─────────────────────────────────────────────

# 5a. Random Forest feature importance
rf_model = trained_models["Random Forest"][0]
importances = rf_model.feature_importances_
feat_names = X_model.columns.tolist()
feat_imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
feat_imp_df = feat_imp_df.sort_values("importance", ascending=False).head(12)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(feat_imp_df["feature"][::-1], feat_imp_df["importance"][::-1], color="#4C72B0")
ax.set_title("Top Feature Importances — Random Forest")
ax.set_xlabel("Mean Decrease in Impurity")
plt.tight_layout()
plt.savefig("fig6_feature_importance.png", bbox_inches="tight")
plt.close()
print("Saved: fig6_feature_importance.png")

# 5b. Gradient Boosting feature importance
gb_model = trained_models["Gradient Boosting"][0]
gb_importances = gb_model.feature_importances_
gb_imp_df = pd.DataFrame({"feature": feat_names, "importance": gb_importances})
gb_imp_df = gb_imp_df.sort_values("importance", ascending=False).head(12)

# 5c. Combined top-10 table
print("\nTop-10 RF Feature Importances:\n", feat_imp_df.head(10).to_string(index=False))
print("\nTop-10 GB Feature Importances:\n", gb_imp_df.head(10).to_string(index=False))

# ─────────────────────────────────────────────
# 6. Summary Table
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL MODEL COMPARISON TABLE")
print("="*60)
print(results_df.to_string())
print("\nCluster Longevity Rates:")
for i in range(K_BEST):
    rate = df_first[df_first["cluster"] == i]["career_longevity"].mean()
    print(f"  Cluster {i} ({CLUSTER_NAMES[i]}): {rate:.2%} long careers")

print("\nPipeline complete. All figures saved.")
