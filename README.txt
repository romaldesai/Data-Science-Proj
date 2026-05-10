NBA Player Archetype Segmentation & Career Longevity Prediction
================================================================
CS439 Final Project — Data Science Pipeline for NBA Roster Decision Support


OVERVIEW
--------
This project builds an end-to-end data science pipeline to help NBA front offices
predict whether a player will sustain a long career (5+ seasons) based solely on
their first season statistics. We combine unsupervised player archetype segmentation
with supervised career longevity classification, producing both predictive accuracy
and actionable business insights.

Key findings:
  - Four distinct player archetypes emerge from first-season stats, with career
    longevity rates ranging from 31.6% to 75.3%
  - Best classifier achieves ROC-AUC of 0.784 predicting career longevity from
    rookie-year data alone
  - Games played, net rating, and draft number are the strongest early predictors
    of career durability


PROJECT STRUCTURE
-----------------
nba-career-longevity/
    nba_pipeline.py     Full pipeline: preprocessing, clustering, classification, evaluation
    figures/            All generated plots (auto-created when script runs)
    .gitignore
    README.txt


DATASET
-------
NBA Players Dataset by Justinas Cirtautas — available on Kaggle:
https://www.kaggle.com/datasets/justinas/nba-players-data

Download all_seasons.csv and place it in the project root before running.

  - 12,844 player-season records, 1996-97 through 2022-23
  - 2,551 unique players
  - 22 features: demographics, draft info, per-season performance stats


PIPELINE SUMMARY
----------------

1. Preprocessing
   - Filter records with fewer than 10 games played
   - Engineer binary target: career_longevity (1 if player appeared in 5+ seasons)
   - Aggregate to one row per player using first season only (prevents data leakage)
   - Binary encode is_undrafted and is_international
   - Standardize all numerical features with StandardScaler
   - Stratified 80/20 train-test split

2. Player Segmentation (K-Means)
   - Cluster on 9 on-court performance features:
     PTS, REB, AST, USG%, TS%, OREB%, DREB%, AST%, Net Rating
   - Optimal k=4 selected via Elbow Method and Silhouette Score
   - PCA 2D visualization of cluster structure

   Cluster 0  Interior Bigs          75.3% long careers
   Cluster 1  Role Players / Bench   31.6% long careers
   Cluster 2  Playmakers             66.6% long careers
   Cluster 3  Scorers / Wings        44.6% long careers

3. Classification Models
   Three models trained on scaled features + cluster membership:

   Model                 Accuracy    F1      ROC-AUC
   Logistic Regression   0.722       0.696   0.776
   Random Forest         0.709       0.649   0.784
   Gradient Boosting     0.704       0.637   0.783

4. Feature Importance
   Top predictors (Random Forest): Games Played, Points, Rebounds,
   Net Rating, Draft Number


HOW TO RUN
----------

1. Clone the repo
      git clone https://github.com/romaldesai/Data-Science-Proj.git
      cd Data-Science-Proj

2. Install dependencies
      pip install pandas numpy matplotlib seaborn scikit-learn

3. Download the dataset
   Download all_seasons.csv from Kaggle (link above) and place it in
   the project root.

4. Run the pipeline
      python nba_pipeline.py

   All 6 figures will be saved to the project directory automatically.


REQUIREMENTS
------------
   pandas
   numpy
   matplotlib
   seaborn
   scikit-learn


OUTPUT FIGURES
--------------
   fig1_cluster_selection.png    Elbow Method and Silhouette Score for k selection
   fig2_pca_clusters.png         PCA 2D visualization of the 4 player archetypes
   fig3_longevity_by_cluster.png Career longevity rate per archetype
   fig4_roc_curves.png           ROC curves for all 3 classifiers
   fig5_confusion_matrices.png   Confusion matrices for all 3 classifiers
   fig6_feature_importance.png   Top feature importances from Random Forest


AUTHORS
-------
CS439 Final Project
