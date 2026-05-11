NBA Player Archetype Segmentation and Career Longevity Prediction
CS439 Final Project
Romal Desai (rsd101)


OVERVIEW

This project presents an integrated data science pipeline combining unsupervised
player segmentation with supervised career longevity prediction. Using first-season
NBA statistics, we identify four distinct player archetypes via K-Means clustering
and train multiple classifiers to predict whether a player will sustain a 5+ season
career. The framework provides actionable insights for front office roster decisions.

Key results:
  Career longevity rates vary from 31.6% to 75.3% across player archetypes
  Best classifier achieves ROC-AUC of 0.784 using only rookie-year statistics
  Games played, net rating, and draft number are the strongest longevity predictors


PROJECT STRUCTURE

    nba_pipeline.py     Full pipeline: preprocessing, clustering, classification, evaluation
    figures/            Generated plots (auto-created on run)
    .gitignore
    README.txt


DATASET

NBA Players Dataset by Justinas Cirtautas, available on Kaggle:
https://www.kaggle.com/datasets/justinas/nba-players-data

Download all_seasons.csv and place it in the project root before running.
The dataset contains 12,844 player-season records from 1996-97 through 2022-23
across 2,551 unique players, with 22 features covering demographics, draft
information, and per-season performance statistics.


PIPELINE SUMMARY

Preprocessing
  Records with fewer than 10 games played are filtered out. The binary target
  career_longevity is engineered from the full dataset (1 if a player appeared
  in 5 or more seasons) and then applied to each player's first season only,
  preventing data leakage. Draft and country features are binary encoded.
  All numerical features are standardized using StandardScaler. The dataset
  is split 80/20 with stratified sampling.

Player Segmentation (K-Means)
  Clustering is performed on 9 on-court performance features: PTS, REB, AST,
  USG%, TS%, OREB%, DREB%, AST%, and Net Rating. The optimal k=4 is selected
  using the Elbow Method and Silhouette Score.

  Cluster 0  Interior Bigs          75.3% long careers
  Cluster 1  Role Players / Bench   31.6% long careers
  Cluster 2  Playmakers             66.6% long careers
  Cluster 3  Scorers / Wings        44.6% long careers

Classification Models
  Three models are trained on scaled features plus one-hot encoded cluster membership.

  Model                 Accuracy    F1      ROC-AUC
  Logistic Regression   0.722       0.696   0.776
  Random Forest         0.709       0.649   0.784
  Gradient Boosting     0.704       0.637   0.783

Feature Importance
  Top predictors (Random Forest): Games Played, Points, Rebounds,
  Net Rating, Draft Number


HOW TO RUN

1. Clone the repo
      git clone https://github.com/romaldesai/Data-Science-Proj.git
      cd Data-Science-Proj

2. Install dependencies
      pip install pandas numpy matplotlib seaborn scikit-learn

3. Download all_seasons.csv from the Kaggle link above and place it in
   the project root.

4. Run the pipeline
      python nba_pipeline.py

   All 6 figures are saved to the project directory automatically.


REQUIREMENTS

  pandas
  numpy
  matplotlib
  seaborn
  scikit-learn


OUTPUT FIGURES

  fig1_cluster_selection.png    Elbow Method and Silhouette Score for k selection
  fig2_pca_clusters.png         PCA 2D visualization of the 4 player archetypes
  fig3_longevity_by_cluster.png Career longevity rate per archetype
  fig4_roc_curves.png           ROC curves for all 3 classifiers
  fig5_confusion_matrices.png   Confusion matrices for all 3 classifiers
  fig6_feature_importance.png   Top feature importances from Random Forest
