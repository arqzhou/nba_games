### Goal: Train a model based on NBA_games dataset to predict the winner of each game.

Features:

| W_PCT_home | Total win percentage of the home team |
| --- | --- |
| W_PCT_away | Total win percentage of the road team |
| HOME_RECORD_home | Win percentage of the home team at home |
| ROAD_RECORD_away | Win percentage of the away team on the road |
| LAST_X_home | Win percentage of the home team’s last X games (default to 10) |
| LAST_X_away | Win percentage of the away team’s last X games (default to 10) |
| Predicting: HOME_TEAM_WINS | 1 if home team wins, 0 if home team loses |

# Models

- Logistic Regression (w/wo LAST_X_ features)
- Decision Tree (w/wo LAST_X_ features, depths = 4, 6, and 8)
- Bagged Decision Trees?
- Random Forest?
- XGBoost

## Logistic Regression

- Logistic Regression with 4 features
    
    
    | Confusion Matrix | True Loss | True Win |
    | --- | --- | --- |
    | Predicted Loss | 1414 | 704 |
    | Predicted Win | 585 | 2391 |
    
    | Classification Report | precision | recall | f1-score | support |
    | --- | --- | --- | --- | --- |
    | 0 | 0.71 | 0.67 | 0.69 | 2118 |
    | 1 | 0.77 | 0.80 | 0.79 | 2976 |
    |  |  |  |  |  |
    | accuracy |  |  | 0.75 | 5094 |
    
    | Feature | Importance |
    | --- | --- |
    | W_PCT_home | 0.566 |
    | W_PCT_away | -0.269 |
    | HOME_RECORD_home | 4.70 |
    | ROAD_RECORD_away | -4.681 |
    | LAST_X_home | n/a |
    | LAST_X_away | n/a |
- Logistic Regression with 6 features
    
    
    | Confusion Matrix | True Loss | True Win |
    | --- | --- | --- |
    | Predicted Loss | 1414 | 704 |
    | Predicted Win | 585 | 2391 |
    
    | Classification Report | precision | recall | f1-score | support |
    | --- | --- | --- | --- | --- |
    | 0 | 0.71 | 0.67 | 0.69 | 2118 |
    | 1 | 0.77 | 0.80 | 0.79 | 2976 |
    |  |  |  |  |  |
    | accuracy |  |  | 0.75 | 5094 |
    
    | Feature | Importance |
    | --- | --- |
    | W_PCT_home | 1.857 |
    | W_PCT_away | -1.371 |
    | HOME_RECORD_home | 4.698 |
    | ROAD_RECORD_away | -4.661 |
    | LAST_X_home | -1.378 |
    | LAST_X_away | 1.165 |
    
    Note:
    
    W_PCT_home + LAST_X_home = (1.857) + (-1.378) = (0.479)
    
    W_PCT_away + LAST_X_away = (-1.371) + (1.165) = (-0.206)
    
    These sum up to approximately the same as W_PCT_home and W_PCT_away in the 4-feature model, indicating that they are very strongly correlated (share a similar impact on the model). It is essentially just the same model, so adding the LAST_X features did not improve the model nearly as much as I thought it would.
    
    The record of the team in recent games (last 2, 3, 5, or 10 games) doesn’t really affect the model’s accuracy. This means the strength of the team over the course of the season is effectively a good enough predictor. 
    

## Decision Tree

- Decision Tree with 4 features, max_depth = 4
    
    
    | Confusion Matrix | True Loss | True Win |
    | --- | --- | --- |
    | Predicted Loss | 1226 | 892 |
    | Predicted Win | 471 | 2505 |
    
    | Classification Report | precision | recall | f1-score | support |
    | --- | --- | --- | --- | --- |
    | 0 | 0.72 | 0.58 | 0.64 | 2118 |
    | 1 | 0.74 | 0.88 | 0.79 | 2976 |
    |  |  |  |  |  |
    | accuracy |  |  | 0.73 | 5094 |
    
    | Feature | Importance |
    | --- | --- |
    | W_PCT_home | 0.0 |
    | W_PCT_away | 0.0 |
    | HOME_RECORD_home | 0.575 |
    | ROAD_RECORD_away | 0.425 |
    | LAST_X_home | n/a |
    | LAST_X_away | n/a |
    
    Note:
    
    The tree only ever split on the home record of the home team and the away record of the away team, indicating that they were consistently the best features to split on to gain information/reduce entropy.
    
- Decision Tree with 6 features, max_depth = 4
    
    Exact same values as with 4 features. The tree also did not use the LAST_X_ variables to make any splits.
    
- Decision Tree with 4 features, max_depth = 6
    
    
    | Confusion Matrix | True Loss | True Win |
    | --- | --- | --- |
    | Predicted Loss | 1466 | 652 |
    | Predicted Win | 686 | 2290 |
    
    | Classification Report | precision | recall | f1-score | support |
    | --- | --- | --- | --- | --- |
    | 0 | 0.68 | 0.69 | 0.69 | 2118 |
    | 1 | 0.78 | 0.77 | 0.77 | 2976 |
    |  |  |  |  |  |
    | accuracy |  |  | 0.74 | 5094 |
    
    | Feature | Importance |
    | --- | --- |
    | W_PCT_home | 0.014 |
    | W_PCT_away | 0.007 |
    | HOME_RECORD_home | 0.546 |
    | ROAD_RECORD_away | 0.434 |
    | LAST_X_home | n/a |
    | LAST_X_away | n/a |
    
    Note:
    
    The tree would very rarely split on other features, which would have to take place at depths 5 and 6, but the tree performed negligibly better in terms of accuracy, indicating that increasing any depth beyond 6 likely isn’t worth it.
    
- Decision Tree with 6 features, max_depth = 6
    
    
    | Confusion Matrix | True Loss | True Win |
    | --- | --- | --- |
    | Predicted Loss | 1467 | 651 |
    | Predicted Win | 689 | 2287 |
    
    | Classification Report | precision | recall | f1-score | support |
    | --- | --- | --- | --- | --- |
    | 0 | 0.68 | 0.69 | 0.69 | 2118 |
    | 1 | 0.78 | 0.77 | 0.77 | 2976 |
    |  |  |  |  |  |
    | accuracy |  |  | 0.74 | 5094 |
    
    | Feature | Importance |
    | --- | --- |
    | W_PCT_home | 0.013 |
    | W_PCT_away | 0.004 |
    | HOME_RECORD_home | 0.545 |
    | ROAD_RECORD_away | 0.432 |
    | LAST_X_home | 0.001 |
    | LAST_X_away | 0.005 |
    
    Note:
    
    The tree would also rarely split on LAST_X_ variables, but again, the increase in performance was negligible, likely for the same reason as logistic regression: it just doesn’t add that much new information. The effect of home court advantage has much more impact than a team’s recent string of results.
    

Accuracy of the model with depth = 3: 0.70. (w/wo LAST_X_ variables)

Accuracy of the model with depth = 5: 0.74. (w/wo LAST_X_ variables)

So, a max_depth of 4 is likely the deepest depth the model needs before obtaining negligible returns.

So logistic regression and decision trees have both resulted in accuracy rankings of about 0.73-0.75, with logistic regression performing slightly better. For the sake of completeness, what about RandomForest or XGBoost, which are supposedly better algorithms for trees? Personally, I think there will be minimal difference.

## Random Forest

- Hypothesis: n_estimators = 100, max_depth = 4.
    
    
    | Confusion Matrix | True Loss | True Win |
    | --- | --- | --- |
    | Predicted Loss | 1380 | 738 |
    | Predicted Win | 582 | 2394 |
    
    | Classification Report | precision | recall | f1-score | support |
    | --- | --- | --- | --- | --- |
    | 0 | 0.70 | 0.65 | 0.68 | 2118 |
    | 1 | 0.76 | 0.80 | 0.78 | 2976 |
    |  |  |  |  |  |
    | accuracy |  |  | 0.74 | 5094 |
    

The random forest did no better than the singular decision tree, which makes sense because the decision tree was so heavily reliant on two features (Home Record and Road Record). As the tree considers its features, these will always win out meaning most trees in the forest will closely match the singular decision tree, adding no new information.

## XGBoost

- Hypothesis: n_estimators = 100, max_depth = 6.
    
    
    | Confusion Matrix | True Loss | True Win |
    | --- | --- | --- |
    | Predicted Loss | 1393 | 725 |
    | Predicted Win | 664 | 2312 |
    
    | Classification Report | precision | recall | f1-score | support |
    | --- | --- | --- | --- | --- |
    | 0 | 0.68 | 0.66 | 0.67 | 2118 |
    | 1 | 0.76 | 0.78 | 0.77 | 2976 |
    |  |  |  |  |  |
    | accuracy |  |  | 0.73 | 5094 |
    

XGBoost (at depth 6) actually performed slightly worse than a regular decision tree, which lines up with how XGBoost works. After the first or second tree, the model learns to fit the residual error of the previous tree in the sequence, which typically helps it to learn from previous mistakes. However, the first few trees in this example are already performing at the ceiling (a singular decision tree was already at 0.74 accuracy), so all trees after the initial few are just fitting to random noise, which is why XGBoost actually performs worse than a single tree.

# Comparison

| Ranking | Accuracy | Model |
| --- | --- | --- |
| 1= | 0.75 | Logistic Regression |
| 1= | 0.75 | Neural Network (6, 3, 1) |
| 2= | 0.74 | Decision Tree (depth = 6) |
| 2= | 0.74 | Random Forest (n_estimators = 100, max_depth = 4) |
| 2= | 0.74 | XGBoost (n_estimators = 100, max_depth = 4) |
| 4= | 0.73 | XGBoost (n_estimators = 100, max_depth = 6) |
| 4= | 0.73 | Decision Tree (depth = 4) |

Based on the data available, the best performing model is logistic regression. This dataset does not play well to the Decision Trees’ strengths.

## Neural Network w/ TensorFlow (just to practice it)

- Hypothesis: epochs = 10, batch_size = 32
    
    
    | Confusion Matrix | True Loss | True Win |
    | --- | --- | --- |
    | Predicted Loss | 1414 | 704 |
    | Predicted Win | 588 | 2388 |
    
    | Classification Report | precision | recall | f1-score | support |
    | --- | --- | --- | --- | --- |
    | 0 | 0.71 | 0.67 | 0.69 | 2118 |
    | 1 | 0.77 | 0.80 | 0.79 | 2976 |
    |  |  |  |  |  |
    | accuracy |  |  | 0.75 | 5094 |
    
    The neural network performed about as well as logistic regression, with very diminishing returns in improvement after 10 epochs.
