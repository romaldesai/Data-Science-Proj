# CS439 Final Project
# NBA Player Archetype Segmentation and Career Longevity Prediction
# Romal Desai (rsd101)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

plt.rcParams.update({"figure.dpi": 130, "font.size": 11})
SEED = 42

# rsd101 - load data
df_raw = pd.read_csv("all_seasons.csv")
df = df_raw.drop(columns=["Unnamed: 0"]).copy()
df = df[df["gp"] >= 10].reset_index(drop=True)

# rsd101 - build target: 1 if player appeared in 5+ seasons
seasons_per_player = df.groupby("player_name")["season"].nunique()
long_career_players = set(seasons_per_player[seasons_per_player >= 5].index)
df["career_longevity"] = df["player_name"].apply(lambda x: 1 if x in long_career_players else 0)

# rsd101 - keep first season only to prevent data leakage
df_sorted = df.sort_values(["player_name", "season"])
df_first = df_sorted.groupby("player_name").first().reset_index()

# rsd101 - encode draft and country features
df_first["draft_round"] = df_first["draft_round"].replace("Undrafted", "0")
df_first["draft_number"] = df_first["draft_number"].replace("Undrafted", "0")
df_first["draft_round"] = pd.to_numeric(df_first["draft_round"], errors="coerce").fillna(0).astype(int)
df_first["draft_number"] = pd.to_numeric(df_first["draft_number"], errors="coerce").fillna(0).astype(int)
df_first["is_undrafted"] = (df_first["draft_round"] == 0).astype(int)
df_first["is_international"] = (df_first["country"] != "USA").astype(int)

FEATURE_COLS = [
    "age", "player_height", "player_weight",
    "gp", "pts", "reb", "ast",
    "net_rating", "oreb_pct", "dreb_pct",
    "usg_pct", "ts_pct", "ast_pct",
    "draft_round", "draft_number", "is_undrafted", "is_international"
]
TARGET_COL = "career_longevity"

X = df_first[FEATURE_COLS].fillna(df_first[FEATURE_COLS].median())
y = df_first[TARGET_COL]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)

# rsd101 - cluster on performance features only
CLUSTER_FEATURES = ["pts", "reb", "ast", "usg_pct", "ts_pct",
                    "oreb_pct", "dreb_pct", "ast_pct", "net_rating"]
X_cluster = df_first[CLUSTER_FEATURES].fillna(0)
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

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

K_BEST = 4
km_final = KMeans(n_clusters=K_BEST, random_state=SEED, n_init=10)
df_first["cluster"] = km_final.fit_predict(X_cluster_scaled)

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
ax.set_title("NBA Player Segmentation -- PCA 2D Visualization")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("fig2_pca_clusters.png", bbox_inches="tight")
plt.close()

cluster_profile = df_first.groupby("cluster")[CLUSTER_FEATURES + ["career_longevity"]].mean().round(3)
cluster_profile.index = [f"Cluster {i}: {CLUSTER_NAMES[i]}" for i in range(K_BEST)]
print("\nCluster Profile:\n", cluster_profile.T)

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

# rsd101 - one-hot encode cluster membership and train classifiers
X_model = X_scaled_df.copy()
for c in range(K_BEST):
    X_model[f"cluster_{c}"] = (df_first["cluster"].values == c).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_model, y, test_size=0.2, random_state=SEED, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8, random_state=SEED, class_weight="balanced"),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=SEED),
}

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

fig, ax = plt.subplots(figsize=(7, 5))
for name, (model, _, y_prob) in trained_models.items():
    RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax)
ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_title("ROC Curves -- Career Longevity Prediction")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("fig4_roc_curves.png", bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, (model, y_pred, _)) in zip(axes, trained_models.items()):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Short (<5yr)", "Long (5+yr)"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.suptitle("Confusion Matrices", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("fig5_confusion_matrices.png", bbox_inches="tight")
plt.close()

# rsd101 - feature importance from Random Forest and Gradient Boosting
rf_model = trained_models["Random Forest"][0]
feat_names = X_model.columns.tolist()
feat_imp_df = pd.DataFrame({"feature": feat_names, "importance": rf_model.feature_importances_})
feat_imp_df = feat_imp_df.sort_values("importance", ascending=False).head(12)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(feat_imp_df["feature"][::-1], feat_imp_df["importance"][::-1], color="#4C72B0")
ax.set_title("Top Feature Importances -- Random Forest")
ax.set_xlabel("Mean Decrease in Impurity")
plt.tight_layout()
plt.savefig("fig6_feature_importance.png", bbox_inches="tight")
plt.close()

gb_model = trained_models["Gradient Boosting"][0]
gb_imp_df = pd.DataFrame({"feature": feat_names, "importance": gb_model.feature_importances_})
gb_imp_df = gb_imp_df.sort_values("importance", ascending=False).head(12)

print("\nTop-10 RF Feature Importances:\n", feat_imp_df.head(10).to_string(index=False))
print("\nTop-10 GB Feature Importances:\n", gb_imp_df.head(10).to_string(index=False))

print("\n" + "="*60)
print("FINAL MODEL COMPARISON")
print("="*60)
print(results_df.to_string())
print("\nCluster Longevity Rates:")
for i in range(K_BEST):
    rate = df_first[df_first["cluster"] == i]["career_longevity"].mean()
    print(f"  Cluster {i} ({CLUSTER_NAMES[i]}): {rate:.2%} long careers")
