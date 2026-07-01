import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("nathanlauga/nba-games")

print("Path to dataset files:", path)

games_raw = pd.read_csv(f"{path}/games.csv")
games = games_raw[['GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']]
games.info()

rankings_raw = pd.read_csv(f"{path}/ranking.csv")
rankings = rankings_raw[['TEAM_ID', 'STANDINGSDATE', 'W_PCT', 'HOME_RECORD', 'ROAD_RECORD', 'W', 'L']]

print(rankings_raw.columns)
print(rankings.iloc[:3])

print(games.iloc[:3])

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