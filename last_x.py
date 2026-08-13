import pandas as pd

class RDF:
    def __init__(self, games):
        self._home = games[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'HOME_TEAM_WINS']].copy()
        self._home = self._home.rename(columns={'HOME_TEAM_ID':'TEAM_ID', 'HOME_TEAM_WINS': 'TEAM_WIN'})
        self._home['HOME_ROW'] = True

        self._away = games[['GAME_ID', 'GAME_DATE_EST', 'VISITOR_TEAM_ID', 'HOME_TEAM_WINS']].copy()
        self._away['TEAM_WIN'] = 1 - self._away['HOME_TEAM_WINS']
        self._away = self._away.rename(columns={'VISITOR_TEAM_ID':'TEAM_ID'}).drop(columns='HOME_TEAM_WINS')
        self._away['HOME_ROW'] = False

        self._team_games = pd.concat([self._home, self._away]).sort_values('GAME_DATE_EST')
        self._team_games = self._team_games.sort_values(by=['GAME_DATE_EST', 'GAME_ID'], ascending=[True, True]).reset_index(drop=True)

    # Calculates all the last X win pcts to create a 'LAST_X' variable.
    def calc_last_xs(self, max_window_size):
        rolled = (
            self._team_games.groupby('TEAM_ID')['TEAM_WIN']
            .rolling(window = max_window_size, min_periods = 1) # adjust X here
            .mean()
        )
        shifted = rolled.groupby(level=0).shift(1)
        self._team_games['LAST_X'] = shifted.reset_index(level=0, drop=True)
        self._team_games['LAST_X'] = self._team_games['LAST_X'].fillna(0.5)

        # maybe we could have an agent go thru and label which of these were preseason etc.

    def get_df(self):
        return self._team_games
