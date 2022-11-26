from streamlit.proto.PlotlyChart_pb2 import PlotlyChart as PlotlyChartProto
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import altair.vegalite.v4 as alt
import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.spines as spines
import streamlit as st
import pandas as pd
import warnings
import altair as alt
from PIL import Image
from chart_studio import plotly as ply
from scipy import interpolate
import numpy as np
import io
import base64
import ast

from scipy.ndimage import gaussian_filter

import df_operations as dfo
import time
from mplsoccer.pitch import Pitch, VerticalPitch
from mplsoccer import PyPizza, Radar
from chart_utils import pizza
from custom_functions import HomeAwayOffensiveShotsGoals, HomeAwayDefensiveEvents, HomeAwayPassingEvents
from highlight_text import fig_text

start_time = time.time()


warnings.filterwarnings('ignore')
st.set_page_config(layout="wide")


@st.cache(allow_output_mutation=True, show_spinner=False)
def from_image_to_b64(path):
    image = Image.open(path)
    output = io.BytesIO()
    image.save(output, format='PNG')
    encoded_string = "data:image/jpeg;base64," + \
                     base64.b64encode(output.getvalue()).decode()
    return encoded_string


@st.cache(allow_output_mutation=True, show_spinner=False)
def read_csv(path, delimiter=None):
    delimiter = ',' if delimiter is None else delimiter
    return pd.read_csv(path, delimiter=delimiter)


@st.cache(allow_output_mutation=True, show_spinner=False)
def create_history(df: pd.DataFrame, df_home: pd.DataFrame, df_away: pd.DataFrame):
    return dfo.create_history_of_matches(df, df_home, df_away)


@st.cache(allow_output_mutation=True, show_spinner=False)
def team_alignment(df: pd.DataFrame, home_away_split_char: str, match_field_name: str):
    return dfo.team_alignment(
        df, home_away_split_char=home_away_split_char, match_field_name=match_field_name)


@st.cache(allow_output_mutation=True, show_spinner=False)
def split_home_away(df: pd.DataFrame):
    return dfo.split_home_away(df_match_complete=df)


@st.cache(allow_output_mutation=True, show_spinner=False)
def group_by_sum(df: pd.DataFrame, cols_grouping: list, use_for_index: bool):
    return df.groupby(cols_grouping, as_index=use_for_index).sum()


def truncate_string(full_name: str):
    if len(full_name.split(' ')) == 1:
        return full_name
    _ret = ' '.join([f'{l[0]}' if i == 0 else f'{l}' for i,
                    l in enumerate(full_name.split(' '))])
    return _ret


def filter_all_matches_by_team(df: pd.DataFrame, selected_round: int, team: str):
    return df[
        (df['matchday'] < selected_round) &
        ((df.team_home == team) | (
            df.team_away == team))].sort_values('matchday')


def create_dict_for_renaming(df: pd.DataFrame, endings: dict):
    _ren_columns = {}
    for _c in df.columns:
        for _old, _new in endings.items():
            if not _c.endswith(_old):
                continue
            elif _c.endswith(_old):
                _ren_columns[_c] = f"{_c[:-2]}{_new}"
    return _ren_columns


st.title('xPred')
st.header('The tool for analyzing and predicting football matches')

spinner_container = st.container()
with spinner_container:
    with st.spinner("Wait a moment..."):
        # ********* Importing Datasets *********
        path_ipo = 'datasets/SICS_SerieA_2021-22_OffensiveIndexTimetable.csv'
        path_events = 'datasets/SICS-SerieA_2021-22_eventi.csv'
        path_simulated_matches = 'datasets/simulated_matches.csv'
        path_players = 'datasets/dati_completi_serieA_2021-22_partite.csv'
        df_ipo = read_csv(path=path_ipo, delimiter=';')
        df_ipo = team_alignment(
            df_ipo, home_away_split_char='-', match_field_name='match_name')

        df_events = read_csv(path=path_events, delimiter=';')
        # DF ipo by match (2 rows for each match - team1/team2)
        df_ipo_by_match = df_ipo.groupby(
            ['matchDay', 'team_home', 'team_away', 'team'], as_index=False).sum().sort_values('matchDay')
        df_ipo_by_match.rename(columns={'matchDay': 'matchday'}, inplace=True)

        df_simulated_matches = read_csv(
            path_simulated_matches).set_index('game_id')

        df_players = read_csv(path_players)

        # ** !! Fix per Samuele Ricci: da togliere nelle righe dell'empoli
        df_players.drop(df_players[(df_players.opta_id == 450023) & (
            df_players.game_id == 2229098)].index, inplace=True)

        df_players = team_alignment(
            df_players, home_away_split_char=' v ', match_field_name='Match')

        game_id_match_all = group_by_sum(df_players,
                                         ['game_id', 'team_home',
                                          'team_away', 'matchday'],
                                         use_for_index=False).drop_duplicates('game_id').sort_values(
            ['matchday', 'team_home'])

        # DF con id di tutti i team
        # team_id, team
        team_id_name_logo = df_players.set_index(
            'team_id')['team'].reset_index().sort_values('team_id').drop_duplicates()
        team_id_name_logo['team_logo'] = team_id_name_logo.apply(
            lambda x: from_image_to_b64(path=f"logos/{x['team']}.png"), axis=1)
        # st.write(team_id_name_logo)
        df_players_ha = split_home_away(df_players)

        df_grouped_home = df_players_ha['home'].groupby(
            ['game_id', 'team_home', 'team_away', 'matchday'], as_index=False).sum()
        df_grouped_away = df_players_ha['away'].groupby(
            ['game_id', 'team_home', 'team_away', 'matchday'], as_index=False).sum()

        df_previous_tot = pd.merge(
            left=df_grouped_home[['game_id', 'matchday',
                                  'team_home', 'team_away'] + stats.attacking_metrics],
            right=df_grouped_away[['game_id'] + stats.attacking_metrics],
            left_on=df_grouped_home['game_id'],
            right_on=df_grouped_away['game_id']
        )
        df_previous_tot.set_index('key_0', inplace=True)
        df_previous_tot.drop('game_id_x', axis=1, inplace=True)
        df_previous_tot.drop('game_id_y', axis=1, inplace=True)

        # df_ipo_by_match_home = df_ipo_by_match[df_ipo_by_match['team_home']
        #                                       == df_ipo_by_match['team']]
        # df_ipo_by_match_away = df_ipo_by_match[df_ipo_by_match['team_away']
        #                                       == df_ipo_by_match['team']]
        # df_ipo_by_match_home['lk'] = df_ipo_by_match_home.apply(
        #    lambda x: f"{x['team_home']}_{x['team_away']}_{x['matchday']}", axis=1)
        # df_ipo_by_match_away['lk'] = df_ipo_by_match_away.apply(
        #    lambda x: f"{x['team_home']}_{x['team_away']}_{x['matchday']}", axis=1)

        # df_ipo_by_match_final = pd.merge(
        #    left=df_ipo_by_match_home[['lk', 'matchday',
        #                               'team_home', 'team_away', 'weight']],
        #    right=df_ipo_by_match_away[['lk', 'weight']],
        #    left_on='lk',
        #    right_on='lk',
        #    how='inner'
        # )

        # df_ipo_by_match_final.rename(columns={
        #    'weight_x': 'ipo_home',
        #    'weight_y': 'ipo_away'}, inplace=True)
        # df_ipo_by_match_final.set_index('lk', inplace=True)

        text_expander = "Choose Round and match to analyze"
        match_selection_expander = st.expander(text_expander)

expander_container = st.container()
with expander_container:
    with match_selection_expander:
        round_list = df_ipo_by_match['matchday'].unique().tolist()
        round_list.sort(reverse=False)
        selected_round = st.slider('Matchday', 1, round_list[-1], 15)

        match_list_by_selected_round = game_id_match_all[game_id_match_all.matchday == selected_round]

        match_list_by_selected_round['match'] = match_list_by_selected_round.apply(
            lambda x: f"{x['team_home']} - {x['team_away']}", axis=1)

        game_ids = [None] + match_list_by_selected_round['game_id'].tolist()
        matches = [f"Select Match of round {selected_round}"] + \
            match_list_by_selected_round['match'].tolist()

        dic = dict(zip(game_ids, matches))

        selected_match = st.selectbox("Match", game_ids, format_func=lambda x: dic[x], index=0, key='expander_text',
                                      args=(match_selection_expander,))

if not selected_match:
    st.stop()
team_home = match_list_by_selected_round.loc[match_list_by_selected_round.game_id ==
                                             selected_match]['team_home'].values[0]
team_away = match_list_by_selected_round.loc[match_list_by_selected_round.game_id ==
                                             selected_match]['team_away'].values[0]

home_team, away_team = st.columns(2)
st.markdown(
    """

    <style>
        div[data-testid=column] {
            text-align:center;
        }
        div[data-testid=stImage] {         
            display: block;
            margin: auto;
        }
    </style>
   
    """, unsafe_allow_html=True
)
with home_team:
    img_home = Image.open(f'logos/{team_home}.png')
    st.image(img_home, width=60)
with away_team:
    img_home = Image.open(f'logos/{team_away}.png')
    st.image(img_home, width=60)

# IPO of matches played by team_home (every matches is depicted on 2 rows)
df_ipo_by_match_home = filter_all_matches_by_team(
    df=df_ipo_by_match,
    selected_round=selected_round,
    team=team_home)

# IPO of matches played by team_away (every matches is depicted on 2 rows)
df_ipo_by_match_away = filter_all_matches_by_team(
    df=df_ipo_by_match,
    selected_round=selected_round,
    team=team_away)

# Team Home statistics:
# DF df_prev_h_played_ha contains all previous games played by the home team in Home and Away stadium

df_prev_h_played_ha = filter_all_matches_by_team(
    df=df_previous_tot, selected_round=selected_round, team=team_home)

df_prev_h_played_ha['pts'] = df_prev_h_played_ha.apply(
    lambda x: 1 if x['Gol_x'] == x['Gol_y'] else 3 if (x['Gol_x'] > x['Gol_y'] and team_home == x['team_home']) or (
        x['Gol_y'] > x['Gol_x'] and team_home == x['team_away']) else 0, axis=1)
df_prev_h_played_ha['wdl'] = df_prev_h_played_ha.apply(
    lambda x: 1 if x['Gol_x'] == x['Gol_y'] else 3, axis=1)
df_prev_h_played_ha['home_away'] = df_prev_h_played_ha.apply(
    lambda x: 'H' if x['team_home'] == team_home else 'A', axis=1)

ren_columns = create_dict_for_renaming(df_prev_h_played_ha, endings={
                                       '_x': '_home', '_y': '_away'})


df_prev_h_played_ha.rename(columns=ren_columns, inplace=True)

df_prev_a_played_ha = filter_all_matches_by_team(
    df=df_previous_tot, selected_round=selected_round, team=team_away)

df_prev_a_played_ha['pts'] = df_prev_a_played_ha.apply(
    lambda x: 1 if x['Gol_x'] == x['Gol_y'] else 3 if (x['Gol_x'] > x['Gol_y'] and team_away == x['team_home']) or (
        x['Gol_y'] > x['Gol_x'] and team_away == x['team_away']) else 0, axis=1)
df_prev_a_played_ha['wdl'] = df_prev_a_played_ha.apply(
    lambda x: 1 if x['Gol_x'] == x['Gol_y'] else 3, axis=1)
df_prev_a_played_ha['home_away'] = df_prev_a_played_ha.apply(
    lambda x: 'H' if x['team_home'] == team_away else 'A', axis=1)

df_prev_a_played_ha.rename(columns=ren_columns, inplace=True)


color_scale_home_away = alt.Scale(domain=['H', 'A'],
                                  range=['#C2C2C2', '#AC0800'])

df_ipo_by_match_home['home_away'] = df_ipo_by_match_home.apply(
    lambda x: 'H' if x['team'] == x['team_home'] else 'A', axis=1
)

df_ipo_by_match_home['lk'] = df_ipo_by_match_home.apply(
    lambda x: f"{x['team_home']}_{x['team_away']}_{x['matchday']}",
    axis=1
)

df_ipo_by_match_home_single_row = pd.merge(
    left=df_ipo_by_match_home[df_ipo_by_match_home['home_away'] == 'H'][
        ['matchday', 'team_home', 'team_away', 'weight', 'lk']],
    right=df_ipo_by_match_home[df_ipo_by_match_home['home_away'] == 'A'][[
        'weight', 'lk']],
    left_on='lk',
    right_on='lk',
    how='inner'
)
df_ipo_by_match_home_single_row.rename(
    columns={'weight_x': 'ipo_home', 'weight_y': 'ipo_away'},
    inplace=True)

df_ipo_by_match_home_single_row['home_away'] = df_ipo_by_match_home_single_row.apply(
    lambda x: 'H' if x['team_home'] == team_home else 'A',
    axis=1
)

df_ipo_by_match_home_single_row['ipo_to_show'] = df_ipo_by_match_home_single_row.apply(
    lambda x: x['ipo_home'] if x['home_away'] == 'H' else x['ipo_away'],
    axis=1
)

# st.write(df_ipo_by_match_home_single_row)

df_ipo_by_match_home_only_team_home = df_ipo_by_match_home[df_ipo_by_match_home['team'] == team_home].set_index(
    'lk')
df_ipo_by_match_home_only_other_team = df_ipo_by_match_home[df_ipo_by_match_home['team'] != team_home].set_index(
    'lk')
df_ipo_by_match_home_only_team_home['logo_position'] = df_ipo_by_match_home_only_team_home.apply(
    lambda x: 0, axis=1)

df_ipo_by_match_home_only_team_home['against_team_logo'] = df_ipo_by_match_home_only_team_home.apply(
    lambda x: team_id_name_logo.set_index(
        'team').loc[f"{x['team_home']}"].team_logo
    if x['team'] == x['team_away']
    else team_id_name_logo.set_index('team').loc[f"{x['team_away']}"].team_logo, axis=1)


df_ipo_by_match_away['home_away'] = df_ipo_by_match_away.apply(
    lambda x: 'H' if x['team'] == x['team_home'] else 'A', axis=1
)
df_ipo_by_match_away['lk'] = df_ipo_by_match_away.apply(
    lambda x: f"{x['team_home']}_{x['team_away']}_{x['matchday']}",
    axis=1
)

df_ipo_by_match_away_single_row = pd.merge(
    left=df_ipo_by_match_away[df_ipo_by_match_away['home_away'] == 'H'][
        ['matchday', 'team_home', 'team_away', 'weight', 'lk']],
    right=df_ipo_by_match_away[df_ipo_by_match_away['home_away'] == 'A'][[
        'weight', 'lk']],
    left_on='lk',
    right_on='lk',
    how='inner'
)
df_ipo_by_match_away_single_row.rename(
    columns={'weight_x': 'ipo_home', 'weight_y': 'ipo_away'},
    inplace=True)

df_ipo_by_match_away_single_row['home_away'] = df_ipo_by_match_away_single_row.apply(
    lambda x: 'H' if x['team_home'] == team_away else 'A',
    axis=1
)

df_ipo_by_match_away_single_row['ipo_to_show'] = df_ipo_by_match_away_single_row.apply(
    lambda x: x['ipo_home'] if x['home_away'] == 'H' else x['ipo_away'],
    axis=1
)

df_ipo_by_match_away_only_team_away = df_ipo_by_match_away[df_ipo_by_match_away['team'] == team_away].set_index(
    'lk')
df_ipo_by_match_away_only_other_team = df_ipo_by_match_away[df_ipo_by_match_away['team'] != team_away].set_index(
    'lk')
df_ipo_by_match_away_only_team_away['logo_position'] = df_ipo_by_match_away_only_team_away.apply(
    lambda x: 0, axis=1)
df_ipo_by_match_away_only_team_away['against_team_logo'] = df_ipo_by_match_away_only_team_away.apply(
    lambda x: team_id_name_logo.set_index(
        'team').loc[f"{x['team_home']}"].team_logo
    if x['team'] == x['team_away']
    else team_id_name_logo.set_index('team').loc[f"{x['team_away']}"].team_logo, axis=1)

# This dataframe is necessary in order to have ipo_home and ipo_away on the same row
df_ipo_by_match_away_ha_data = pd.merge(
    left=df_ipo_by_match_away_only_team_away[[
        'matchday', 'team_home', 'team_away', 'weight', 'home_away']],
    right=df_ipo_by_match_away_only_other_team[['weight']],
    left_on='lk',
    right_on='lk',
    how='inner'
)
df_ipo_by_match_away_ha_data.rename(
    columns={'weight_x': 'ipo_home',
             'weight_y': 'ipo_away'},
    inplace=True
)

lin_space_x = np.linspace(1, len(
    df_ipo_by_match_home[df_ipo_by_match_home.team == team_home]['matchday']), 10000)

pch_home_team_home = interpolate.pchip(
    df_ipo_by_match_home[df_ipo_by_match_home.team ==
                         team_home]['matchday'],
    df_ipo_by_match_home[df_ipo_by_match_home.team == team_home]['weight'])

pch_home_other_team = interpolate.pchip(
    df_ipo_by_match_home[df_ipo_by_match_home.team !=
                         team_home]['matchday'],
    df_ipo_by_match_home[df_ipo_by_match_home.team != team_home]['weight'])

pch_away_home = interpolate.pchip(
    df_ipo_by_match_away[df_ipo_by_match_away.team ==
                         team_away]['matchday'],
    df_ipo_by_match_away[df_ipo_by_match_away.team == team_away]['weight'])

pch_away_away = interpolate.pchip(
    df_ipo_by_match_away[df_ipo_by_match_away.team !=
                         team_away]['matchday'],
    df_ipo_by_match_away[df_ipo_by_match_away.team != team_away]['weight'])

df_spline_home = pd.DataFrame({
    'x2': lin_space_x,
    'y_considered_team': pch_home_team_home(lin_space_x),
    'y_other_team': pch_home_other_team(lin_space_x),
})


df_spline_away = pd.DataFrame({
    'x2': lin_space_x,
    'y_considered_team': pch_away_home(lin_space_x),
    'y_other_team': pch_away_away(lin_space_x),
})
x_scale = alt.Scale(domain=[1, df_ipo_by_match_home['matchday'].max()])
max_ipo_scale = alt.Scale(
    domain=[0, np.max([df_spline_home['y_considered_team'].max(), df_spline_away['y_considered_team'].max()])])

# Prepare chart home
ipochart_line_home = alt.Chart(df_spline_home).mark_line(strokeWidth=4).encode(
    x=alt.X('x2',
            axis=None,
            scale=x_scale),
    y=alt.Y('y_considered_team:Q',
            axis=alt.Axis(tickMinStep=1),
            scale=max_ipo_scale)
).properties(
    title=f'Previous matches {team_home}',

).properties(
    height=300)

ipochart_area_home = alt.Chart(df_spline_home).mark_area(opacity=.8).encode(
    x=alt.X('x2',
            axis=None,
            scale=x_scale),
    y=alt.Y('y_considered_team:Q',
            axis=alt.Axis(tickMinStep=1)),
    # y2='y_other_team:Q'
)

ipochart_area_away = alt.Chart(df_spline_home).mark_area(opacity=.6, color='#838383').encode(
    alt.X('x2',
          axis=None,
          scale=x_scale),
    alt.Y('y_other_team:Q',
          axis=alt.Axis(tickMinStep=1)))

team_logos = alt.Chart(df_ipo_by_match_home_only_team_home).mark_image(width=20, height=20, opacity=1).encode(
    x=alt.X(
        'matchday',
        scale=x_scale),
    y=alt.Y(
        'logo_position',
        axis=alt.Axis(domainOpacity=0)),
    url="against_team_logo"
)
ipochart_line_match = alt.Chart(df_ipo_by_match_home[df_ipo_by_match_home['team'] == team_home]).mark_bar(width=2,
                                                                                                          color='steelblue').encode(
    x=alt.X('matchday',
            axis=None,
            scale=x_scale),
    y=alt.Y('weight:Q',
            axis=alt.Axis(tickMinStep=1))
)

# df_ipo_by_match_home_ha_data['ipo_team'] = df_ipo_by_match_home_ha_data.apply(
#    lambda x: x['ipo_home'] if x['team'] == x['team_home'] else x['ipo_away'], axis=1
# )
# df_ipo_by_match_home_ha_data['ipo_other_team'] = df_ipo_by_match_home_ha_data.apply(
#    lambda x: x['ipo_home'] if x['team'] == x['team_away'] else x['ipo_away'], axis=1
# )
dots_ipo = alt.Chart(df_ipo_by_match_home_single_row).mark_circle(size=60,
                                                                  opacity=1).encode(
    x=alt.X(
        'matchday',
        scale=x_scale
    ),
    y=alt.Y('ipo_to_show:Q', title=None),
    color=alt.Color('home_away:N', title=None, sort='-x',
                    scale=color_scale_home_away),
    tooltip=['team_home', 'team_away', 'ipo_home:Q', 'ipo_away:Q']
)

# ipochart_area_away + ipochart_line_home + ipochart_area_home + dots_ipo + team_logos  # (team_logos + dots_ipo)
ipochart_home = ipochart_area_away + ipochart_line_home + \
    ipochart_area_home + ipochart_line_match + dots_ipo + team_logos

ipo_bars_home = alt.Chart(df_prev_h_played_ha).mark_bar(size=8,
                                                        cornerRadiusTopLeft=3,
                                                        cornerRadiusTopRight=3,
                                                        cornerRadiusBottomRight=0,
                                                        cornerRadiusBottomLeft=0).encode(
    x=alt.X('matchday',
            title='Matchday',
            axis=alt.Axis(tickMinStep=1),
            scale=x_scale),
    y=alt.Y(
        'wdl',
        stack=None,
        axis=None,
        title='IPO',
        scale=alt.Scale(domain=[0, 3])),
    color=alt.condition(
        alt.datum.pts == 0,
        alt.value('#838383'),
        alt.value('steelblue')
    ),

    tooltip=['team_home', 'team_away', 'Gol_home', 'Gol_away']
    # column='team:O'
    # 'team',
).properties(
    height=100
)

color_away_team = 'green'

ipochart_line_away = alt.Chart(df_spline_away).mark_line(strokeWidth=4, color=color_away_team).encode(
    x=alt.X('x2',
            axis=None,
            scale=x_scale),
    y=alt.Y('y_considered_team:Q',
            axis=alt.Axis(tickMinStep=1),
            scale=max_ipo_scale),
).properties(
    title=f'Previous matches {team_away}',

).properties(
    height=300)

ipochart_area_away = alt.Chart(df_spline_away).mark_area(opacity=.7, color=color_away_team).encode(
    x=alt.X('x2',
            axis=None,
            scale=x_scale),
    y=alt.Y('y_considered_team:Q',
            axis=alt.Axis(tickMinStep=1)),
    # y2='y_other_team:Q'
)

ipochart_area_home = alt.Chart(df_spline_away).mark_area(opacity=.6, color='#838383').encode(
    alt.X('x2',
          axis=None,
          scale=x_scale),
    alt.Y('y_other_team:Q',
          axis=alt.Axis(tickMinStep=1)))

team_logos = alt.Chart(df_ipo_by_match_away_only_team_away).mark_image(width=20, height=20, opacity=1).encode(
    x=alt.X(
        'matchday',
        scale=x_scale),
    y=alt.Y(
        'logo_position',
        axis=alt.Axis(domainOpacity=0)),
    url="against_team_logo"
)
ipochart_line_match = alt.Chart(df_ipo_by_match_away[df_ipo_by_match_away['team'] == team_away]).mark_bar(
    width=2,
    color='lightgreen').encode(
    x=alt.X('matchday',
            axis=None,
            scale=x_scale),
    y=alt.Y('weight:Q',
            axis=alt.Axis(tickMinStep=1))
)

dots_ipo = alt.Chart(df_ipo_by_match_away_single_row).mark_circle(size=60, opacity=1).encode(
    x=alt.X(
        'matchday',
        scale=x_scale
    ),
    y=alt.Y('ipo_to_show:Q', title=None),
    color=alt.Color('home_away:N', title=None, sort='-x',
                    scale=color_scale_home_away),
    tooltip=['team_home', 'team_away', 'ipo_home:Q', 'ipo_away:Q']
)
# st.write(df_prev_a_played_ha)
ipo_bars_away = alt.Chart(df_prev_a_played_ha).mark_bar(size=8,
                                                        cornerRadiusTopLeft=3,
                                                        cornerRadiusTopRight=3,
                                                        cornerRadiusBottomRight=0,
                                                        cornerRadiusBottomLeft=0).encode(
    x=alt.X('matchday',
            title='Matchday',
            axis=alt.Axis(tickMinStep=1),
            scale=x_scale),
    y=alt.Y(
        'wdl',
        stack=None,
        axis=None,
        title='IPO',
        scale=alt.Scale(domain=[0, 3])),
    color=alt.condition(
        alt.datum.pts == 0,
        alt.value('#838383'),
        alt.value('lightgreen')
    ),

    tooltip=['team_home', 'team_away', 'Gol_home', 'Gol_away']
).properties(
    height=100,
)

home_col, away_col = st.columns(2)

with home_col:
    _all = alt.vconcat(ipochart_home, ipo_bars_home).configure_view(
        stroke=None).configure_axis(grid=False).properties(
        padding=0
    )
    # st.write(df_ipo_by_match_home_ha_data)

    # st.write(df_ipo_by_match_home)

    # st.write(df_ipo_by_match_home_only_other_team)
    home_col.altair_chart(_all.resolve_scale(
        color='independent', x='shared'), use_container_width=True)
    # home_col.altair_chart(ipochart)
    # _all.resolve_scale(color='independent', x='shared'), use_container_width=True)

with away_col:

    _all = alt.vconcat(
        ipochart_area_home + ipochart_area_away + ipochart_line_away +
        ipochart_line_match + team_logos + dots_ipo,
        ipo_bars_away
    ).configure_view(
        stroke=None).configure_axis(grid=False).properties(
        padding=0
    )

    away_col.altair_chart(_all.resolve_scale(
        color='independent', x='shared'), use_container_width=True)

team_id_home = team_id_name_logo[team_id_name_logo.team ==
                                 team_home]['team_id'].values[0]
team_id_away = team_id_name_logo[team_id_name_logo.team ==
                                 team_away]['team_id'].values[0]

df_players_home_team = df_players[(df_players['matchday'] < selected_round) &
                                  (df_players['team_id'] == team_id_home)].groupby(
    ['opta_id', 'full_name', 'team', 'soccRole'], as_index=False).sum().set_index('opta_id')

df_players_away_team = df_players[(df_players['matchday'] < selected_round) &
                                  (df_players['team_id'] == team_id_away)].groupby(
    ['opta_id', 'full_name', 'team', 'soccRole'], as_index=False).sum().set_index('opta_id')

# For having a uniform X-axis
min_mins_played = int(np.min(
    [df_players_home_team['mins_played'].max(),
     df_players_away_team['mins_played'].max()]))

max_x_axis = int(min_mins_played - int(min_mins_played * 0.1))
mins_step = 30
range = [i for i in range(0, max_x_axis - max_x_axis % mins_step, mins_step)]

default_min_value = int(900 * selected_round / 38)
default_min_value = default_min_value - default_min_value % mins_step
selected_mins_played_min, selected_mins_played_max = st.select_slider('Mins Played',
                                                                      options=range,
                                                                      value=(default_min_value, range[-1]), )

if selected_mins_played_min == selected_mins_played_max:
    if range[0] == selected_mins_played_min:
        _selected_mins_played_max = range[1]
    elif range[-1] == selected_mins_played_max:
        _selected_mins_played_min = range[-2]
    else:
        _selected_mins_played_max = selected_mins_played_min + \
            selected_mins_played_min % mins_step

df_players_home_team_minutes_filtered = df_players_home_team[
    (df_players_home_team['mins_played'] >= selected_mins_played_min) &
    (df_players_home_team['mins_played'] < selected_mins_played_max)]

df_players_away_team_minutes_filtered = df_players_away_team[
    (df_players_away_team['mins_played'] >= selected_mins_played_min) &
    (df_players_away_team['mins_played'] < selected_mins_played_max)]

attacking_metrics_p90 = []
for _m in stats.attacking_metrics:
    df_players_home_team_minutes_filtered[f'{_m}_p90'] = df_players_home_team[_m] / \
        (df_players_home_team['mins_played'] / 90)
    df_players_away_team_minutes_filtered[f'{_m}_p90'] = df_players_away_team[_m] / \
        (df_players_away_team['mins_played'] / 90)
    attacking_metrics_p90.append(f'{_m}_p90')

defensive_metrics_p90 = []
for _m in stats.defensive_metrics:
    df_players_home_team_minutes_filtered[f'{_m}_p90'] = df_players_home_team[_m] / \
        (df_players_home_team['mins_played'] / 90)
    df_players_away_team_minutes_filtered[f'{_m}_p90'] = df_players_away_team[_m] / \
        (df_players_away_team['mins_played'] / 90)
    defensive_metrics_p90.append(f'{_m}_p90')

passing_metrics_p90 = []
for _m in stats.passing_metrics:
    df_players_home_team_minutes_filtered[f'{_m}_p90'] = df_players_home_team[_m] / \
        (df_players_home_team['mins_played'] / 90)
    df_players_away_team_minutes_filtered[f'{_m}_p90'] = df_players_away_team[_m] / \
        (df_players_away_team['mins_played'] / 90)
    passing_metrics_p90.append(f'{_m}_p90')

physical_metrics_p90 = []
for _m in stats.physical_metrics:
    df_players_home_team_minutes_filtered[f'{_m}_p90'] = df_players_home_team[_m] / \
        (df_players_home_team['mins_played'] / 90)
    df_players_away_team_minutes_filtered[f'{_m}_p90'] = df_players_away_team[_m] / \
        (df_players_away_team['mins_played'] / 90)
    physical_metrics_p90.append(f'{_m}_p90')

for _spread in stats.spreads:
    _metrics = _spread.split('-')
    print(_spread)
    print(_metrics)

    if len(_metrics) == 2:
        _m1 = _metrics[0]
        _m2 = _metrics[1]
        df_players_home_team_minutes_filtered[f'{_spread}'] = df_players_home_team_minutes_filtered.apply(
            lambda x: x[f'{_m1}_p90'] - x[f'{_m2}_p90'], axis=1)
        df_players_away_team_minutes_filtered[f'{_spread}'] = df_players_away_team_minutes_filtered.apply(
            lambda x: x[f'{_m1}_p90'] - x[f'{_m2}_p90'], axis=1)

        df_players_home_team_minutes_filtered[f'{_spread}_p90'] = df_players_home_team_minutes_filtered.apply(
            lambda x: (x[f'{_m1}_p90'] - x[f'{_m2}_p90']) / (x['mins_played'] / 90), axis=1)
        df_players_away_team_minutes_filtered[f'{_spread}_p90'] = df_players_away_team_minutes_filtered.apply(
            lambda x: (x[f'{_m1}_p90'] - x[f'{_m2}_p90']) / (x['mins_played'] / 90), axis=1)
        # attacking_metrics_p90.append(f'{_spread}_p90')

max_metrics_for_scale_domain = {}

chart_home_offensive = []
chart_away_offensive = []

chart_home_defensive = []
chart_away_defensive = []

chart_home_passing = []
chart_away_passing = []

chart_home_physical = []
chart_away_physical = []

metrics_to_be_considered = attacking_metrics_p90 + \
    defensive_metrics_p90 + passing_metrics_p90 + physical_metrics_p90

for _m in metrics_to_be_considered:
    _df_home = df_players_home_team_minutes_filtered.sort_values(
        _m, ascending=False).head(5)
    _df_away = df_players_away_team_minutes_filtered.sort_values(
        _m, ascending=False).head(5)

    _max_away = _df_away[_m].max()
    _max_home = _df_home[_m].max()

    if _max_away > _max_home:
        max_metrics_for_scale_domain[_m] = _max_away
    else:
        max_metrics_for_scale_domain[_m] = _max_home

    _df_home['full_name_t'] = _df_home.apply(
        lambda x: dfo.trunc_string(x['full_name']), axis=1)

    _df_away['full_name_t'] = _df_away.apply(
        lambda x: dfo.trunc_string(x['full_name']), axis=1)

    if _m in attacking_metrics_p90:
        print(f"OFFENSIVE: {_m}")
        chart_home_offensive.append(alt.Chart(_df_home).mark_bar(opacity=.8,
                                                                 cornerRadiusTopRight=10,
                                                                 cornerRadiusBottomRight=10).encode(
            y=alt.Y('full_name_t:O', sort='-x', title=None),
            x=alt.X(f"{_m}:Q", title=_m, scale=alt.Scale(
                domain=[0, max_metrics_for_scale_domain[_m] * 1.1]))
        ))
        chart_away_offensive.append(alt.Chart(_df_away).mark_bar(opacity=.8, color='lightgreen',
                                                                 cornerRadiusTopRight=10,
                                                                 cornerRadiusBottomRight=10).encode(
            y=alt.Y('full_name_t:O', sort='-x', title=None),
            x=alt.X(f"{_m}:Q", title=_m, scale=alt.Scale(
                domain=[0, max_metrics_for_scale_domain[_m] * 1.1]))
        ))

    if _m in defensive_metrics_p90:
        chart_home_defensive.append(alt.Chart(_df_home).mark_bar(opacity=.8,
                                                                 cornerRadiusTopRight=10,
                                                                 cornerRadiusBottomRight=10).encode(
            y=alt.Y('full_name_t:O', sort='-x', title=None),
            x=alt.X(f"{_m}:Q", title=_m, scale=alt.Scale(
                domain=[0, max_metrics_for_scale_domain[_m] * 1.1]))
        ))
        chart_away_defensive.append(alt.Chart(_df_away).mark_bar(opacity=.8, color='lightgreen',
                                                                 cornerRadiusTopRight=10,
                                                                 cornerRadiusBottomRight=10).encode(
            y=alt.Y('full_name_t:O', sort='-x', title=None),
            x=alt.X(f"{_m}:Q", title=_m, scale=alt.Scale(
                domain=[0, max_metrics_for_scale_domain[_m] * 1.1]))
        ))

    if _m in passing_metrics_p90:
        chart_home_passing.append(alt.Chart(_df_home).mark_bar(opacity=.8,
                                                               cornerRadiusTopRight=10,
                                                               cornerRadiusBottomRight=10).encode(
            y=alt.Y('full_name_t:O', sort='-x', title=None),
            x=alt.X(f"{_m}:Q", title=_m, scale=alt.Scale(
                domain=[0, max_metrics_for_scale_domain[_m] * 1.1]))
        ))
        chart_away_passing.append(alt.Chart(_df_away).mark_bar(opacity=.8, color='lightgreen',
                                                               cornerRadiusTopRight=10,
                                                               cornerRadiusBottomRight=10).encode(
            y=alt.Y('full_name_t:O', sort='-x', title=None),
            x=alt.X(f"{_m}:Q", title=_m, scale=alt.Scale(
                domain=[0, max_metrics_for_scale_domain[_m] * 1.1]))
        ))

    if _m in physical_metrics_p90:
        chart_home_physical.append(alt.Chart(_df_home).mark_bar(opacity=.8,
                                                                cornerRadiusTopRight=10,
                                                                cornerRadiusBottomRight=10).encode(
            y=alt.Y('full_name_t:O', sort='-x', title=None),
            x=alt.X(f"{_m}:Q", title=_m, scale=alt.Scale(
                domain=[0, max_metrics_for_scale_domain[_m] * 1.1]))
        ))
        chart_away_physical.append(alt.Chart(_df_away).mark_bar(opacity=.8, color='lightgreen',
                                                                cornerRadiusTopRight=10,
                                                                cornerRadiusBottomRight=10).encode(
            y=alt.Y('full_name_t:O', sort='-x', title=None),
            x=alt.X(f"{_m}:Q", title=_m, scale=alt.Scale(
                domain=[0, max_metrics_for_scale_domain[_m] * 1.1]))
        ))

df_team_previous = df_players[(df_players['matchday'] < selected_round)].groupby(['team', 'team_id'],
                                                                                 as_index=False).sum().set_index(
    'team_id')
# df_team_previous = df_players[(df_players['matchday'] < selected_round)].groupby(['team', 'team_id'], as_index=False).sum().set_index('team_id')

rank_metrics = []

_, values_home, values_away = dfo.evaluate_home_away_rank_values_from_parameters(
    df=df_team_previous,
    parameters=stats.attacking_metrics + stats.shots,
    team_id_home=team_id_home,
    team_id_away=team_id_away)
# stats.defensive_metrics

fig_pizza_offensive, ax = pizza(
    params=stats.attacking_metrics + stats.shots,
    values_home=values_home,
    values_away=values_away,
    color_home='steelblue',
    color_away='lightgreen').draw()

fig_pizza_offensive.set_size_inches(10, 10)

_, values_home, values_away = dfo.evaluate_home_away_rank_values_from_parameters(
    df=df_team_previous,
    parameters=stats.defensive_metrics,
    team_id_home=team_id_home,
    team_id_away=team_id_away)

fig_pizza_defensive, ax = pizza(
    params=stats.defensive_metrics,
    values_home=values_home,
    values_away=values_away,
    color_home='steelblue',
    color_away='lightgreen').draw()

fig_pizza_defensive.set_size_inches(10, 10)

_, values_home, values_away = dfo.evaluate_home_away_rank_values_from_parameters(
    df=df_team_previous,
    parameters=stats.passing_metrics,
    team_id_home=team_id_home,
    team_id_away=team_id_away)

fig_pizza_passing, ax = pizza(
    params=stats.passing_metrics,
    values_home=values_home,
    values_away=values_away,
    color_home='steelblue',
    color_away='lightgreen').draw()

fig_pizza_passing.set_size_inches(10, 10)

_, values_home, values_away = dfo.evaluate_home_away_rank_values_from_parameters(
    df=df_team_previous,
    parameters=stats.physical_metrics,
    team_id_home=team_id_home,
    team_id_away=team_id_away)

fig_pizza_physical, ax = pizza(
    params=stats.physical_metrics,
    values_home=values_home,
    values_away=values_away,
    color_home='steelblue',
    color_away='lightgreen').draw()

fig_pizza_passing.set_size_inches(10, 10)

df_previous_selected_round = df_events[(df_events['Giornata'] < selected_round) &
                                       (df_events['Giornata'] >= selected_round - 5)]

df_previous_selected_round['key'] = df_previous_selected_round.apply(
    lambda x: f"{x['Partita'].split('-')[0].title()}_{x['Partita'].split('-')[1].title()}_{x['Giornata']}", axis=1
)

df_previous_selected_round.set_index('key', inplace=True)

df_previous_selected_round_home = df_previous_selected_round[
    (df_previous_selected_round['Squadra'].str.title() == team_home)]
df_previous_selected_round_away = df_previous_selected_round[
    (df_previous_selected_round['Squadra'].str.title() == team_away)]

df_shots_home = df_previous_selected_round_home[
    df_previous_selected_round_home['Descrizione'] == 'Tiro Fatto']
df_goals_home = df_previous_selected_round_home[
    df_previous_selected_round_home['Descrizione'] == 'Rete Fatta']

df_shots_away = df_previous_selected_round_away[
    df_previous_selected_round_away['Descrizione'] == 'Tiro Fatto']
df_goals_away = df_previous_selected_round_away[
    df_previous_selected_round_away['Descrizione'] == 'Rete Fatta']

fig_pitch_offensive, _ = HomeAwayOffensiveShotsGoals(df_shots_home=df_shots_home,
                                                     df_goals_home=df_goals_home,
                                                     df_shots_away=df_shots_away,
                                                     df_goals_away=df_goals_away,
                                                     team_home=team_home,
                                                     team_away=team_away).instantiate_figure()

df_goals_against_home = df_previous_selected_round_home[
    df_previous_selected_round_home['Descrizione'] == 'Rete Subita']
df_shots_against_home = df_previous_selected_round_home[
    df_previous_selected_round_home['Descrizione'] == 'Tiro Subito']

df_goals_against_away = df_previous_selected_round_away[
    df_previous_selected_round_away['Descrizione'] == 'Rete Subita']
df_shots_against_away = df_previous_selected_round_away[
    df_previous_selected_round_away['Descrizione'] == 'Tiro Subito']

df_keypass_against_home = df_previous_selected_round_home[
    df_previous_selected_round_home['Descrizione'] == 'Passaggio Chiave Subito']
df_assist_against_home = df_previous_selected_round_home[
    df_previous_selected_round_home['Descrizione'] == 'Assist Subito']

df_keypass_against_away = df_previous_selected_round_home[
    df_previous_selected_round_home['Descrizione'] == 'Passaggio Chiave Subito']
df_assist_against_away = df_previous_selected_round_home[
    df_previous_selected_round_home['Descrizione'] == 'Assist Subito']

fig_pitch_defensive_shots_goals, _ = HomeAwayDefensiveEvents(df_shots_against_home=df_shots_against_home,
                                                             df_goals_against_home=df_goals_against_home,
                                                             df_shots_against_away=df_shots_against_away,
                                                             df_goals_against_away=df_goals_against_away,
                                                             df_home=df_previous_selected_round_home,
                                                             df_away=df_previous_selected_round_away,
                                                             team_home=team_home,
                                                             team_away=team_away).instantiate_figure_shots_goal()

fig_pitch_defensive_heatmap, _ = HomeAwayDefensiveEvents(df_shots_against_home=df_shots_against_home,
                                                         df_goals_against_home=df_goals_against_home,
                                                         df_shots_against_away=df_shots_against_away,
                                                         df_goals_against_away=df_goals_against_away,
                                                         df_home=df_previous_selected_round_home,
                                                         df_away=df_previous_selected_round_away,
                                                         team_home=team_home,
                                                         team_away=team_away).instantiate_figure_heatmap()


df_keypass_home = df_previous_selected_round_home[
    df_previous_selected_round_home['Descrizione'] == 'Passaggio Chiave']
df_assist_home = df_previous_selected_round_home[
    df_previous_selected_round_home['Descrizione'] == 'Assist']
df_triangle_home = df_previous_selected_round_home[
    df_previous_selected_round_home['Descrizione'] == 'Triangolazione']
df_pass_home = df_previous_selected_round_home[
    df_previous_selected_round_home['Descrizione'] == 'Passaggio']

df_keypass_away = df_previous_selected_round_away[
    df_previous_selected_round_away['Descrizione'] == 'Passaggio Chiave']
df_assist_away = df_previous_selected_round_away[
    df_previous_selected_round_away['Descrizione'] == 'Assist']
df_triangle_away = df_previous_selected_round_away[
    df_previous_selected_round_away['Descrizione'] == 'Triangolazione']
df_pass_away = df_previous_selected_round_away[
    df_previous_selected_round_away['Descrizione'] == 'Passaggio']


hape = HomeAwayPassingEvents(
    df_keypass_home=df_keypass_home,
    df_assist_home=df_assist_home,
    df_triangle_home=df_triangle_home,
    df_pass_home=df_pass_home,
    df_keypass_away=df_keypass_away,
    df_assist_away=df_assist_away,
    df_triangle_away=df_triangle_away,
    df_pass_away=df_pass_away,
    team_home=team_home,
    team_away=team_away
)
fig_pitch_assist, _ = hape.instantiate_figure_passing()
fig_pitch_passing_heatmap, _ = hape.instantiate_figure_heatmap()

tab_attacking_metrics, tab_defensive_metrics, tab_passing_metrics, tab_physical_metrics = st.tabs(
    ["Offensive Metrics", "Defensive Metrics", "Passing Metrics", "Physical Metrics"])

with tab_attacking_metrics:
    _home_col, _away_col = st.columns(2)
    with _home_col:
        _chart_home = alt.vconcat()
        for _chart in chart_home_offensive:
            _chart_home &= _chart

        st.altair_chart(_chart_home, use_container_width=True)

    with _away_col:
        _chart_away = alt.vconcat()
        for _chart in chart_away_offensive:
            _chart_away &= _chart

        st.altair_chart(_chart_away, use_container_width=True)

    _left, _right = st.columns([1, 1])

    with _left:
        st.pyplot(fig_pizza_offensive)
    with _right:
        st.pyplot(fig_pitch_offensive)

with tab_defensive_metrics:
    _home_col_def, _away_col_def = st.columns(2)
    with _home_col_def:
        _chart_home = alt.vconcat()
        for _chart in chart_home_defensive:
            _chart_home &= _chart

        st.altair_chart(_chart_home, use_container_width=True)

    with _away_col_def:
        _chart_away = alt.vconcat()
        for _chart in chart_away_defensive:
            _chart_away &= _chart

        st.altair_chart(_chart_away, use_container_width=True)
    _left, _right = st.columns([1, 1])
    with _left:
        st.pyplot(fig_pizza_defensive)
    with _right:
        st.pyplot(fig_pitch_defensive_shots_goals)
        st.pyplot(fig_pitch_defensive_heatmap)

with tab_passing_metrics:
    _home_col, _away_col = st.columns(2)
    with _home_col:
        _chart_home = alt.vconcat()
        for _chart in chart_home_passing:
            _chart_home &= _chart

        st.altair_chart(_chart_home, use_container_width=True)

    with _away_col:
        _chart_away = alt.vconcat()
        for _chart in chart_away_passing:
            _chart_away &= _chart
        st.altair_chart(_chart_away, use_container_width=True)
    _left, _right = st.columns([1, 1])
    with _left:
        st.pyplot(fig_pizza_passing)
    with _right:
        st.pyplot(fig_pitch_assist)
        st.pyplot(fig_pitch_passing_heatmap)

with tab_physical_metrics:
    _home_col, _away_col = st.columns(2)
    with _home_col:
        _chart_home = alt.vconcat()
        for _chart in chart_home_physical:
            _chart_home &= _chart

        st.altair_chart(_chart_home, use_container_width=True)

    with _away_col:
        _chart_away = alt.vconcat()
        for _chart in chart_away_physical:
            _chart_away &= _chart
        st.altair_chart(_chart_away, use_container_width=True)
    _left, _right = st.columns([1, 1])
    with _left:
        st.pyplot(fig_pizza_physical)

# st.plotly_chart(ply.plot_mpl(f), use_container_width=True)


# df_players_home_team_defensive = df_players_home_team[]
container = st.container()
st.write("Predictions")
df_sel_match = df_simulated_matches.loc[selected_match]
st.write(df_simulated_matches.loc[selected_match])
h_score_probability = ast.literal_eval(df_sel_match['p_home_score'])
a_score_probability = ast.literal_eval(df_sel_match['p_away_score'])

x, y = np.meshgrid(range(0, 5), range(0, 5))
z = h_score_probability * a_score_probability
source = pd.DataFrame({'x': x.ravel(),
                       'y': y.ravel(),
                       'z': z.ravel()})
print("--- %s seconds ---" % (time.time() - start_time))
