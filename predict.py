import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("nathanlauga/nba-games")

# print("Path to dataset files:", path)

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
    return(df)

def convert_to_percentages(data):
    df = convert_to_percentage(data, 'HOME_RECORD', True)
    df = convert_to_percentage(df, 'HOME_RECORD', False)
    df = convert_to_percentage(df, 'ROAD_RECORD', True)
    df = convert_to_percentage(df, 'ROAD_RECORD', False)
    return(df)
    
total_merged = convert_to_percentages(total_merged)
print(total_merged.head())
print(total_merged.shape)
    