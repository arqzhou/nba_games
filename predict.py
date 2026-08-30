import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from last_x import RDF
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Download latest version
path = kagglehub.dataset_download("nathanlauga/nba-games")

# Pull data from two databases
games_raw = pd.read_csv(f"{path}/games.csv")
games = games_raw[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_TEAM_WINS']]

rankings_raw = pd.read_csv(f"{path}/ranking.csv")
rankings = rankings_raw[['TEAM_ID', 'STANDINGSDATE', 'W_PCT', 'HOME_RECORD', 'ROAD_RECORD', 'W', 'L']]

# Variables to adjust and experiment with
window_size = 10
max_tree_depth = 4
n_est = 100

home_merged = pd.merge(
    games,
    rankings.add_suffix('_home'),
    left_on=['HOME_TEAM_ID', 'GAME_DATE_EST'],
    right_on=['TEAM_ID_home', 'STANDINGSDATE_home'],
    how='left',
).drop(columns=['TEAM_ID_home', 'STANDINGSDATE_home'])

total_merged = pd.merge(
    home_merged,
    rankings.add_suffix('_away'),
    left_on=['VISITOR_TEAM_ID', 'GAME_DATE_EST'],
    right_on=['TEAM_ID_away', 'STANDINGSDATE_away'],
    how='left',
).drop(columns=['TEAM_ID_away', 'STANDINGSDATE_away'])


# Calculate 'LAST_X' variables and merge with existing df.
new_test = RDF(games)
new_test.calc_last_xs(window_size)

win_pct = new_test.get_df()
total_merged = pd.merge(
    total_merged,
    win_pct.query('HOME_ROW == True').drop(columns='GAME_DATE_EST'),
    left_on=['GAME_ID', 'HOME_TEAM_ID'],
    right_on=['GAME_ID', 'TEAM_ID'],
    how='left'
).drop(columns=['TEAM_ID', 'TEAM_WIN', 'HOME_ROW'])

total_merged = pd.merge(
    total_merged,
    win_pct.query('HOME_ROW == False').drop(columns='GAME_DATE_EST'),
    left_on=['GAME_ID', 'VISITOR_TEAM_ID'],
    right_on=['GAME_ID', 'TEAM_ID'],
    how='left'
).drop(columns=['TEAM_ID', 'TEAM_WIN', 'HOME_ROW'])

total_merged = total_merged.rename(columns={'LAST_X_x':'LAST_X_home', 'LAST_X_y':'LAST_X_away'})

# Converts Home and Road Records into percentages.
def convert_to_percentage(df, colname, isHomeTeam):
    suffix = '_home' if isHomeTeam else '_away'
    col, col_w, col_l = (colname + suffix), (colname + suffix + '_W'), (colname + suffix + '_L')
    df[[col_w, col_l]] = df[col].str.split('-', expand=True)
    df = df.astype({col_w: int, col_l: int})
    df[col] = round(df[col_w] / (df[col_w] + df[col_l]), 3)
    df = df.drop(columns=[col_w, col_l])
    df = df.dropna(subset=[col]) # drops divide by 0 errors (team hasn't played yet that season)
    return(df)

def convert_to_percentages(data):
    df = convert_to_percentage(data, 'HOME_RECORD', True)
    df = convert_to_percentage(df, 'HOME_RECORD', False)
    df = convert_to_percentage(df, 'ROAD_RECORD', True)
    df = convert_to_percentage(df, 'ROAD_RECORD', False)
    return(df)
    
total_merged = convert_to_percentages(total_merged)
# print(total_merged.drop(columns='GAME_DATE_EST').head())


def basic_logistic_regression_w_last_x(total_merged):
    # Independent (predictors) and Dependent (won?) Columns
    X = total_merged[['W_PCT_home', 'W_PCT_away', 'HOME_RECORD_home', 'ROAD_RECORD_away', 'LAST_X_home', 'LAST_X_away']]
    y = total_merged['HOME_TEAM_WINS']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = 38,
        shuffle = False,
        stratify = None
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

    for feature, coef in zip(X.columns, model.coef_[0]):
        print(f"{feature}: {coef:.3f}")

def decision_tree_w_last_x(total_merged, max_tree_depth):
    # Independent (predictors) and Dependent (won?) Columns
    X = total_merged[['W_PCT_home', 'W_PCT_away', 'HOME_RECORD_home', 'ROAD_RECORD_away', 'LAST_X_home', 'LAST_X_away']]
    y = total_merged['HOME_TEAM_WINS']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = 38,
        shuffle = False,
        stratify = None
    )

    # Not as effective since there aren't complex interactions.
    tree = DecisionTreeClassifier(max_depth=max_tree_depth)
    tree.fit(X_train, y_train)
    tree_preds = tree.predict(X_test)

    print(confusion_matrix(y_test, tree_preds))
    print(classification_report(y_test, tree_preds))

    for name, importance in zip(X.columns, tree.feature_importances_):
        print(f"{name}: {importance:.3f}")


#Without LAST_X_ variable
def basic_logistic_regression(total_merged):
    # Independent (predictors) and Dependent (won?) Columns
    X = total_merged[['W_PCT_home', 'W_PCT_away', 'HOME_RECORD_home', 'ROAD_RECORD_away']]
    y = total_merged['HOME_TEAM_WINS']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = 38,
        shuffle = False,
        stratify = None
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

    for feature, coef in zip(X.columns, model.coef_[0]):
        print(f"{feature}: {coef:.3f}")

def decision_tree(total_merged, max_tree_depth):
    # Independent (predictors) and Dependent (won?) Columns
    X = total_merged[['W_PCT_home', 'W_PCT_away', 'HOME_RECORD_home', 'ROAD_RECORD_away']]
    y = total_merged['HOME_TEAM_WINS']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = 38,
        shuffle = False,
        stratify = None
    )

    # Not as effective since there aren't complex interactions.
    tree = DecisionTreeClassifier(max_depth=max_tree_depth)
    tree.fit(X_train, y_train)
    tree_preds = tree.predict(X_test)

    print(confusion_matrix(y_test, tree_preds))
    print(classification_report(y_test, tree_preds))

    for name, importance in zip(X.columns, tree.feature_importances_):
            print(f"{name}: {importance:.3f}")

def random_forest(total_merged, n_estimators, max_tree_depth):
    X = total_merged[['W_PCT_home', 'W_PCT_away', 'HOME_RECORD_home', 'ROAD_RECORD_away']]
    y = total_merged['HOME_TEAM_WINS']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = 38,
        shuffle = False,
        stratify = None
    )

    forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_tree_depth)
    forest.fit(X_train, y_train)
    forest_preds = forest.predict(X_test)

    print(confusion_matrix(y_test, forest_preds))
    print(classification_report(y_test, forest_preds))

    # for name, importance in zip(X.columns, forest.feature_importances_):
    #     print(f"{name}: {importance:.3f}")

def xgb_model(total_merged, n_estimators, max_tree_depth):
    # Independent (predictors) and Dependent (won?) Columns
    X = total_merged[['W_PCT_home', 'W_PCT_away', 'HOME_RECORD_home', 'ROAD_RECORD_away']]
    y = total_merged['HOME_TEAM_WINS']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = 38,
        shuffle = False,
        stratify = None
    )

    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

def nn_model(total_merged):
    # Independent (predictors) and Dependent (won?) Columns
    X = total_merged[['W_PCT_home', 'W_PCT_away', 'HOME_RECORD_home', 'ROAD_RECORD_away']]
    y = total_merged['HOME_TEAM_WINS']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = 38,
        shuffle = False,
        stratify = None
    )

    nn_games_model = tf.keras.Sequential([
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(3, activation='relu'),
        tf.keras.layers.Dense(1),
    ])

    nn_games_model.fit(X_train, y_train)
    preds = nn_games_model.predict(X_test)
    preds = np.where(preds>0.5, 1, 0)
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))


# Uncomment to test models.

# basic_logistic_regression(total_merged)
# decision_tree(total_merged, max_tree_depth)
# basic_logistic_regression_w_last_x(total_merged)
# decision_tree_w_last_x(total_merged, max_tree_depth)
# random_forest(total_merged, n_est, max_tree_depth)
# xgb_model(total_merged, n_est, max_tree_depth)
nn_model(total_merged)
