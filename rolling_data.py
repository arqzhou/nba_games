import pandas as pd
import datetime as dt
import kagglehub

TEAM_IDS = ['1610612740', '1610612762', '1610612739', '1610612755', '1610612737', '1610612738', '1610612751', '1610612752', '1610612745', '1610612750', '1610612760', '1610612758', '1610612746', '1610612765', '1610612748', '1610612756', '1610612743', '1610612754', '1610612761', '1610612747', '1610612759', '1610612749', '1610612766', '1610612741', '1610612742', '1610612763', '1610612753', '1610612764', '1610612757', '1610612744']

class RollingDataFrame:
    def __init__(self, games):
        self._df = games
        self._df['LAST_TEN_home'] = 0.0
        self._df['LAST_TEN_away'] = 0.0

    def roll_teams(self, window=10):
        for team_id in TEAM_IDS:
            self.team_rolling_record(team_id, 10)

        print(self._df[0:40])


    def team_rolling_record(self, team_id, window=10):
        filtered_df = self._df[(self._df['HOME_TEAM_ID'].astype(str).str.strip() == team_id) | (self._df['VISITOR_TEAM_ID'].astype(str).str.strip() == team_id)]
        filtered_df['TEAM_WIN'] = ((filtered_df['HOME_TEAM_ID'].astype(str).str.strip() == team_id) & filtered_df['HOME_TEAM_WINS'] == 1) | ((filtered_df['VISITOR_TEAM_ID'].astype(str).str.strip() == team_id) & filtered_df['HOME_TEAM_WINS'] == 0) 
        filtered_df['GAME_DATE_EST'] = pd.to_datetime(filtered_df['GAME_DATE_EST'])

        for date in filtered_df['GAME_DATE_EST']:
            print("this is the date:", date)

            last_ten, game_id = self.window_record(filtered_df, date, window)

            if last_ten == "No previous games":
                pass
            else:
                home_id = str(filtered_df.loc[filtered_df['GAME_ID'] == game_id, 'HOME_TEAM_ID'].item())
                if home_id == team_id:
                    self._df.loc[self._df['GAME_ID'] == game_id, 'LAST_TEN_home'] = last_ten
                else:
                    self._df.loc[self._df['GAME_ID'] == game_id, 'LAST_TEN_away'] = last_ten
        
    def window_record(self, filtered_df, date_before, window):
        date_cutoff = date_before
        # date_cutoff = dt.datetime.strptime(date_before, "%Y-%m-%d")
        try:
            game_id = filtered_df.loc[filtered_df['GAME_DATE_EST'] == date_cutoff, 'GAME_ID'].item()
        except:
            # print(filtered_df.head())
            print("Error", date_cutoff)

        window_df = filtered_df[filtered_df['GAME_DATE_EST'] < date_cutoff][0:10]
        if len(window_df) == 0:
            return("No previous games", game_id)

        last_ten = round(((window_df['TEAM_WIN']).sum() / min(window, len(window_df))), 3)

        return(last_ten, game_id)

    def print_team_last_ten(self, team_id, date_before):
        filtered_df = self._df[(self._df['HOME_TEAM_ID'].astype(str).str.strip() == team_id) | (self._df['VISITOR_TEAM_ID'].astype(str).str.strip() == team_id)]
        date_cutoff = date_before
        # print(date_cutoff)
        testing = filtered_df[filtered_df['GAME_ID'] == 22000067]
        # print(testing)
        # print(testing.GAME_DATE_EST)


        print(filtered_df.loc[filtered_df['GAME_DATE_EST'] == date_cutoff, 'GAME_ID'].tolist())

        game_id = filtered_df.loc[filtered_df['GAME_DATE_EST'] == date_cutoff, 'GAME_ID'].item()
        window_df = filtered_df[filtered_df['GAME_DATE_EST'] <= date_cutoff][0:10]

        print(window_df)

    if __name__ == '__main__':
        # Download latest version
        path = kagglehub.dataset_download("nathanlauga/nba-games")
        # distinct
        games_raw = pd.read_csv(f"{path}/games.csv")
        games = games_raw[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_TEAM_WINS']]

        