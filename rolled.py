import pandas as pd

class RDF:
    def __init__(self, games):
        self._home = games[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'HOME_TEAM_WINS']].copy()
        self._home = self._home.rename(columns={'HOME_TEAM_ID':'TEAM_ID', 'HOME_TEAM_WINS': 'TEAM_WIN'})

        self._away = games[['GAME_ID', 'GAME_DATE_EST', 'VISITOR_TEAM_ID', 'HOME_TEAM_WINS']].copy()
        self._away['TEAM_WIN'] = 1 - self._away['HOME_TEAM_WINS']
        self._away = self._away.rename(columns={'VISITOR_TEAM_ID':'TEAM_ID'}).drop(columns='HOME_TEAM_WINS')

        self._team_games = pd.concat([self._home, self._away]).sort_values('GAME_DATE_EST')
        self._team_games = self._team_games.sort_values(by=['GAME_DATE_EST', 'GAME_ID'], ascending=[True, True]).reset_index(drop=True)
        print("Unique Index?", self._team_games.index.is_unique)
        print(self._team_games.head())

        print(type(self._team_games))
        print(self._team_games.shape)

    def calc_last_tens(self):
        rolled = (
            self._team_games.groupby('TEAM_ID')['TEAM_WIN']
            .rolling(window = 10, min_periods = 1)
            .mean()
        )
        shifted = rolled.groupby(level=0).shift(1)
        print("Shifted", shifted.head())

        self._team_games['LAST_TEN'] = shifted.reset_index(level=0, drop=True)


        self._team_games['LAST_TEN'] = self._team_games['LAST_TEN'].fillna(0.5)
        print(self._team_games[0:20])
        print(self._team_games[self._team_games['TEAM_ID'] == 1610612762].head(20))

        # maybe we could have an agent go thru and label which of these were preseason etc.
