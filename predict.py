import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier


# Download latest version
path = kagglehub.dataset_download("nathanlauga/nba-games")

games_raw = pd.read_csv(f"{path}/games.csv")
games = games_raw[['GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_TEAM_WINS']]

rankings_raw = pd.read_csv(f"{path}/ranking.csv")
rankings = rankings_raw[['TEAM_ID', 'STANDINGSDATE', 'W_PCT', 'HOME_RECORD', 'ROAD_RECORD', 'W', 'L']]

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

print(total_merged.head())

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
    
total_merged = convert_to_percentages(total_merged)[::-1] # flipped to chronological order

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

def decision_tree(total_merged):
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
    tree = DecisionTreeClassifier(max_depth=4)
    tree.fit(X_train, y_train)
    tree_preds = tree.predict(X_test)

    print(confusion_matrix(y_test, tree_preds))
    print(classification_report(y_test, tree_preds))


# basic_logistic_regression(total_merged)
decision_tree(total_merged)
