import numpy as np
import pandas as pd
import stats


def split_home_away(df_match_complete):
    conditions = [(df_match_complete['team_home'] == df_match_complete['team']),
                  (df_match_complete['team_away'] == df_match_complete['team'])]
    values = ['H', 'A']
    df_match_complete['home_away'] = np.select(conditions, values)
    df_match_home = df_match_complete[df_match_complete['team_home']
                                      == df_match_complete['team']]
    df_match_away = df_match_complete[df_match_complete['team_away']
                                      == df_match_complete['team']]
    return {
        'home': df_match_home,
        'away': df_match_away
    }


def create_home_away_fields(df_match_complete, home_away_split_char, match_field_name) -> pd.DataFrame:
    df_ret = df_match_complete.copy(deep=True)

    df_ret['team_home'] = df_ret.apply(
        lambda x: x[match_field_name].split(home_away_split_char)[0], axis=1)
    df_ret['team_away'] = df_ret.apply(
        lambda x: x[match_field_name].split(home_away_split_char)[1], axis=1)
    df_ret['team_home'] = df_ret['team_home'].str.title()
    df_ret['team_away'] = df_ret['team_away'].str.title()
    df_ret['team'] = df_ret['team'].str.title()
    return df_ret


def team_alignment(df_match_complete, home_away_split_char, match_field_name, dict_for_alignment=None):
    df_match_complete = create_home_away_fields(
        df_match_complete=df_match_complete, home_away_split_char=home_away_split_char,
        match_field_name=match_field_name)

    if not dict_for_alignment:
        dict_for_alignment = {
            'Internazionale': 'Inter',
            'Inter Milan': 'Inter',
            'Hellas Verona': 'Verona',
            'Sportiva Salernitana': 'Salernitana',
            'Spezia Calcio': 'Spezia'
        }

    for k, v in dict_for_alignment.items():
        df_match_complete[df_match_complete.team == f'{k}']['team'] = df_match_complete.team.replace(
            {f'{k}': f'{v}'}, inplace=True)
        df_match_complete[df_match_complete.team == f'{k}']['team_home'] = df_match_complete.team_home.replace({
            f'{k}': f'{v}'}, inplace=True)
        df_match_complete[df_match_complete.team == f'{k}']['team_away'] = df_match_complete.team_away.replace({
            f'{k}': f'{v}'}, inplace=True)

    '''
    df_match_complete[df_match_complete.team == 'Hellas Verona']['team_home'] = df_match_complete.team_home.replace({
        'Hellas Verona': 'Verona'}, inplace=True)
    df_match_complete[df_match_complete.team == 'Hellas Verona']['team_away'] = df_match_complete.team_away.replace({
        'Hellas Verona': 'Verona'}, inplace=True)

    df_match_complete[df_match_complete.team == 'Sportiva Salernitana'][
        'team_home'] = df_match_complete.team_home.replace(
        {'Sportiva Salernitana': 'Salernitana'}, inplace=True)
    df_match_complete[df_match_complete.team == 'Sportiva Salernitana'][
        'team_away'] = df_match_complete.team_away.replace(
        {'Sportiva Salernitana': 'Salernitana'}, inplace=True)

    df_match_complete[df_match_complete.team == 'Spezia Calcio']['team_home'] = df_match_complete.team_home.replace({
        'Spezia Calcio': 'Spezia'}, inplace=True)
    df_match_complete[df_match_complete.team == 'Spezia Calcio']['team_away'] = df_match_complete.team_away.replace({
        'Spezia Calcio': 'Spezia'}, inplace=True)
    
    print(df_match_complete[df_match_complete.index ==
                            2229098][['team_home', 'team_away']])
    '''
    return df_match_complete


def get_numeric_col_names_from_df(df, exceptions=None):
    _num_col_names = df.select_dtypes(include=np.number).columns
    if not exceptions:
        return list(_num_col_names)
    else:
        _num_col_names = [
            elem for elem in _num_col_names if elem not in exceptions]

    return _num_col_names


def create_history_of_matches(df_match_complete, df_match_home, df_match_away):
    _match_days = sorted(df_match_complete['matchday'].unique())
    _dict_days = {}

    _cols_grouping = ['game_id', 'team_home',
                      'team_away', 'matchday']  # , 'home_away']

    _last_days = 3
    for _day in _match_days:

        df_complete_day = df_match_complete[df_match_complete['matchday'] == _day]
        df_home_day = df_match_home[df_match_home['matchday'] == _day]
        df_away_day = df_match_away[df_match_away['matchday'] == _day]

        df_complete_day.fillna(0)
        df_home_day.fillna(0)
        df_away_day.fillna(0)

        df_complete_day_grouped = df_complete_day.groupby(
            _cols_grouping, as_index=False).sum().set_index('game_id')
        df_home_day_grouped = df_home_day.groupby(
            _cols_grouping, as_index=False).sum().set_index('game_id')
        df_away_day_grouped = df_away_day.groupby(
            _cols_grouping, as_index=False).sum().set_index('game_id')

        df_complete_day_previous = df_match_complete[df_match_complete['matchday'] < _day]
        df_home_day_previous = df_match_home[df_match_home['matchday'] < _day]
        df_away_day_previous = df_match_away[df_match_away['matchday'] < _day]
        df_complete_day_previous.fillna(0)
        df_home_day_previous.fillna(0)
        df_away_day_previous.fillna(0)

        _numeric_col_names_complete = get_numeric_col_names_from_df(
            df_complete_day_previous, exceptions=['matchday', 'game_id'])
        _numeric_col_names_home = get_numeric_col_names_from_df(
            df_home_day_previous, exceptions=['matchday', 'game_id'])
        _numeric_col_names_away = get_numeric_col_names_from_df(
            df_away_day_previous, exceptions=['matchday', 'game_id'])

        df_complete_grouped_previous_match_day_complete = df_complete_day_previous.groupby(
            _cols_grouping, as_index=False)[stats.numeric_fields].sum().set_index('game_id')
        df_home_grouped_previous_matchday = df_home_day_previous.groupby(
            _cols_grouping, as_index=False)[stats.numeric_fields].sum().set_index('game_id')
        df_away_grouped_previous_matchday = df_away_day_previous.groupby(
            _cols_grouping, as_index=False)[stats.numeric_fields].sum().set_index('game_id')

        _completed_agg_key = 'completed_aggregated_previous_'
        _home_agg_key = 'home_aggregated_previous_'
        _away_agg_key = 'away_aggregated_previous_'
        _agg_key = {}

        _prev_match_days = sorted(
            df_home_grouped_previous_matchday['matchday'].unique(), reverse=True)

        '''
        *** Da modificare come vengono valutate le partite da prendere per le ultime giornate.
        Deve essere inserito un ciclo su tutte le squadre in modo tale da poter inserire la condizione sulla
        singola squadra
        '''
        for i in [3, 5]:
            if i < len(_prev_match_days):
                _prev_match_days_temp = _prev_match_days[0:i + 1]
            else:
                _prev_match_days_temp = _prev_match_days

            _agg_key[_home_agg_key + f'{i}'] = df_home_grouped_previous_matchday.sort_values(
                by=['team_home', 'matchday'], ascending=False).groupby(['team_home']).head(
                i).groupby('team_home')[stats.numeric_fields].mean().reset_index()

            _agg_key[_away_agg_key + f'{i}'] = df_away_grouped_previous_matchday.sort_values(
                by=['matchday'], ascending=False).groupby(['team_away']).head(
                i).groupby('team_away')[stats.numeric_fields].mean().reset_index()

        _dict_day = {
            'complete': df_complete_day,
            'complete_aggregated': df_complete_day_grouped,
            'complete_aggregated_previous': df_complete_grouped_previous_match_day_complete,
            'home': df_home_day,
            'home_aggregated': df_home_day_grouped,
            'home_aggregated_previous': df_home_grouped_previous_matchday,
            'away': df_away_day,
            'away_aggregated': df_away_day_grouped,
            'away_aggregated_previous': df_away_grouped_previous_matchday
        }

        _dataframes_current_previous = []
        _metrics_to_use = ['xG', 'xT', 'xA']
        for k, v in _agg_key.items():
            _dict_day[k] = v

        # print(_dict_day['complete_aggregated'])
        _current_with_previous_data = pd.DataFrame()
        _current_with_previous_data['game_id'] = list(
            _dict_day['complete_aggregated'].index)
        _current_with_previous_data['team_home'] = list(
            _dict_day['complete_aggregated']['team_home'])
        _current_with_previous_data['team_away'] = list(
            _dict_day['complete_aggregated']['team_away'])
        _current_with_previous_data['matchday'] = list(
            _dict_day['complete_aggregated']['matchday'])

        if _day > 0:
            print(f"Day: {_day}")
            # print(_dict_day['home_aggregated_previous_5'][['team_home', 'xG']])
            # print(_dict_day['home_aggregated_previous_3'])
            # if _dict_day['home_aggregated_previous_3']:
            _current_with_previous_data = pd.merge(
                left=_current_with_previous_data[
                    ['game_id', 'matchday', 'team_home', 'team_away']],
                # 'indicator_away'
                # [['xG']],
                right=_dict_day['home_aggregated_previous_3'][stats.numeric_fields],
                left_on=_current_with_previous_data['team_home'],
                right_on=_dict_day['home_aggregated_previous_3']['team_home'],
                how='inner').drop('key_0', axis=1)

            # if _dict_day['home_aggregated_previous_5']:
            _current_with_previous_data = pd.merge(
                left=_current_with_previous_data[
                    ['game_id', 'matchday', 'team_home', 'team_away'] + stats.numeric_fields],
                # 'indicator_away'
                right=_dict_day['home_aggregated_previous_5'][stats.numeric_fields],
                left_on=_current_with_previous_data['team_home'],
                right_on=_dict_day['home_aggregated_previous_5']['team_home'],
                how='inner').drop('key_0', axis=1)

            _dict_for_renaming = {}
            _cols_for_merging = []

            # for _m in _metrics_to_use:
            for _c in list(_current_with_previous_data.columns):

                # print(_c[len(_c)-2:len(_c)])
                if _c.endswith('_x'):  # _c[len(_c)-2:len(_c)] == '_x':
                    _k = _c.replace('_x', '')
                    _dict_for_renaming[f'{_c}'] = f'{_k}_home_previous_3'
                elif _c.endswith('_y'):  # _c[len(_c)-2:len(_c)] == '_y':
                    _k = _c.replace('_y', '')
                    _dict_for_renaming[f'{_c}'] = f'{_k}_home_previous_5'

            _current_with_previous_data.rename(
                columns=_dict_for_renaming, inplace=True)

            # print(_current_with_previous_data.head(10))

            _current_with_previous_data = pd.merge(
                left=_current_with_previous_data,  # [
                # ['game_id', 'matchday', 'team_home', 'team_away'] + _cols_for_merging],
                # 'indicator_away'
                right=_dict_day['away_aggregated_previous_3'][stats.numeric_fields],
                left_on=_current_with_previous_data['team_away'],
                right_on=_dict_day['away_aggregated_previous_3']['team_away'],
                how='inner').drop('key_0', axis=1)
            if _day == 20:
                print(_current_with_previous_data[[
                    'game_id', 'matchday', 'team_home', 'team_away']])
            # if _dict_day['away_aggregated_previous_5']:
            _current_with_previous_data = pd.merge(
                _current_with_previous_data,  # [
                # ['game_id', 'matchday', 'team_home', 'team_away'] + _cols_for_merging + _metrics_to_use],
                # 'xG_home_previous_3', 'xG_home_previous_5', 'xG']],
                # 'indicator_away'
                _dict_day['away_aggregated_previous_5'][stats.numeric_fields],
                left_on=_current_with_previous_data['team_away'],
                right_on=_dict_day['away_aggregated_previous_5']['team_away'],
                how='inner').drop('key_0', axis=1)

            for _c in list(_current_with_previous_data.columns):
                if _c.endswith('_x'):  # _c[len(_c)-2:len(_c)] == '_x':
                    _k = _c.replace('_x', '')
                    _dict_for_renaming[f'{_c}'] = f'{_k}_away_previous_3'
                elif _c.endswith('_y'):  # _c[len(_c)-2:len(_c)] == '_y':
                    _k = _c.replace('_y', '')
                    _dict_for_renaming[f'{_c}'] = f'{_k}_away_previous_5'

            _current_with_previous_data.rename(
                columns=_dict_for_renaming, inplace=True)

        _dict_day['dataframe_for_simulation'] = _current_with_previous_data

        _dict_days[_day] = _dict_day

    return _dict_days


def trunc_string(full_name):
    if len(full_name.split(' ')) == 1:
        return full_name
    _ret = ' '.join([f'{l[0]}' if i == 0 else f'{l}' for i,
                                                         l in enumerate(full_name.split(' '))])

    return _ret


def evaluate_home_away_rank_values_from_parameters(df: pd.DataFrame, parameters: list, team_id_home: int, team_id_away: int):
    _rank_metrics = []
    _values_home = []
    _values_away = []
    for _m in parameters:
        for _c in df.columns:
            if _m == _c:
                _v = f'{_c}_rank'
                _rank_metrics.append(_v)
                df[_v] = 100 * df[_c].rank(pct=True)
    _values_home = list(df.loc[team_id_home][_rank_metrics].values)
    _values_away = list(df.loc[team_id_away][_rank_metrics].values)
    return _rank_metrics, _values_home, _values_away
