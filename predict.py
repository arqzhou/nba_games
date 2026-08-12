import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from rolled import RDF
import mlflow

# Download latest version
path = kagglehub.dataset_download("nathanlauga/nba-games")

mlflow.set_experiment('NBA_predictor')
mlflow.sklearn.autolog()

# Pull data from two databases
games_raw = pd.read_csv(f"{path}/games.csv")
games = games_raw[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_TEAM_WINS']]

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


# Calculate 'LAST_TEN' variables and merge with existing df.
new_test = RDF(games)
new_test.calc_last_tens()

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

total_merged = total_merged.rename(columns={'LAST_TEN_x':'LAST_TEN_home', 'LAST_TEN_y':'LAST_TEN_away'})

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
print(total_merged.drop(columns='GAME_DATE_EST').head())


def basic_logistic_regression_w_last_ten(total_merged):
    # Independent (predictors) and Dependent (won?) Columns
    X = total_merged[['W_PCT_home', 'W_PCT_away', 'HOME_RECORD_home', 'ROAD_RECORD_away', 'LAST_TEN_home', 'LAST_TEN_away']]
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

    print(X.head())

    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

    # with mlflow.start_run(run_name="logreg_with_last_ten"):
    #     mlflow.log_param("features", X.columns.tolist())
    #     mlflow.log_param("model", "LogisticRegression")

    #     model.fit(X_train, y_train)
    #     preds = model.predict(X_test)

    #     mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    #     mlflow.log_metric("precision", precision_score(y_test, preds))
    #     mlflow.log_metric("recall", recall_score(y_test, preds))

    #     mlflow.sklearn.log_model(model, "model")

    for feature, coef in zip(X.columns, model.coef_[0]):
        print(f"{feature}: {coef:.3f}")

def decision_tree_w_last_ten(total_merged):
    # Independent (predictors) and Dependent (won?) Columns
    X = total_merged[['W_PCT_home', 'W_PCT_away', 'HOME_RECORD_home', 'ROAD_RECORD_away', 'LAST_TEN_home', 'LAST_TEN_away']]
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

    for name, importance in zip(X.columns, tree.feature_importances_):
        print(f"{name}: {importance:.3f}")


#Without last ten variable
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

    for name, importance in zip(X.columns, tree.feature_importances_):
            print(f"{name}: {importance:.3f}")

basic_logistic_regression(total_merged)
decision_tree(total_merged)
print("With last ten:")
basic_logistic_regression_w_last_ten(total_merged)
decision_tree_w_last_ten(total_merged)
