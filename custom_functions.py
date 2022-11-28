from mplsoccer import VerticalPitch
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import stats


class HomeAwayOffensiveShotsGoals:
    def __init__(self, df_shots_home, df_goals_home, df_shots_away, df_goals_away, team_home, team_away):
        self.df_shots_home = df_shots_home
        self.df_goals_home = df_goals_home
        self.df_shots_away = df_shots_away
        self.df_goals_away = df_goals_away
        self.team_home = team_home
        self.team_away = team_away

    def instantiate_figure(self):
        figure, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

        # primo grafico: shot map home team
        pitch = VerticalPitch(pitch_type='custom', pitch_length=105,
                              pitch_width=68, line_zorder=12, half=True)

        pitch.draw(ax=ax[0][0])

        pitch.scatter(x=self.df_shots_home['P0 norm x'],
                      y=self.df_shots_home['P0 norm y'],
                      s=100,
                      color='steelblue',
                      edgecolor='steelblue',
                      zorder=12,
                      ax=ax[0][0])
        pitch.scatter(x=self.df_goals_home['P0 norm x'],
                      y=self.df_goals_home['P0 norm y'],
                      s=500,
                      color='blue',
                      edgecolor='blue',
                      zorder=12,
                      marker='*',
                      ax=ax[0][0])
        ax[0][0].set_title(f'Shotmap {self.team_home}')
        # secondo grafico: shot map away team
        pitch.draw(ax=ax[0][1])
        pitch.scatter(x=self.df_shots_away['P0 norm x'],
                      y=self.df_shots_away['P0 norm y'],
                      s=100,
                      color='lightgreen',
                      edgecolor='lightgreen',
                      zorder=12,
                      ax=ax[0][1])
        pitch.scatter(x=self.df_goals_away['P0 norm x'],
                      y=self.df_goals_away['P0 norm y'],
                      s=500,
                      color='green',
                      edgecolor='green',
                      zorder=12,
                      marker='*',
                      ax=ax[0][1])
        ax[0][1].set_title(f'Shotmap {self.team_away}')

        # terzo grafico: Home Shots Heatmap
        pitch.draw(ax=ax[1][0])
        bin_statistic = pitch.bin_statistic(self.df_shots_home['P0 norm x'], self.df_shots_home['P0 norm y'],
                                            statistic='count', bins=(50, 50))
        bin_statistic['statistic'] = gaussian_filter(
            bin_statistic['statistic'], 5)
        pcm = pitch.heatmap(bin_statistic, ax=ax[1][0], cmap='Blues')
        pitch.scatter(x=self.df_shots_home['P0 norm x'],
                      y=self.df_shots_home['P0 norm y'],
                      s=10,
                      color='black',
                      zorder=10,
                      alpha=0.1,
                      ax=ax[1][0])
        ax[1][0].set_title(f'{self.team_home} Shots Heatmap')

        # quarto grafico: heatmap con filtro gaussiano
        pitch.draw(ax=ax[1][1])
        bin_statistic = pitch.bin_statistic(self.df_shots_away['P0 norm x'], self.df_shots_away['P0 norm y'],
                                            statistic='count', bins=(50, 50))
        bin_statistic['statistic'] = gaussian_filter(
            bin_statistic['statistic'], 5)
        pcm = pitch.heatmap(bin_statistic, ax=ax[1][1], cmap='Greens')
        pitch.scatter(x=self.df_shots_away['P0 norm x'],
                      y=self.df_shots_away['P0 norm y'],
                      s=10,
                      color='black',
                      zorder=10,
                      alpha=0.1,
                      ax=ax[1][1])
        ax[1][1].set_title(f'{self.team_away} Shots Heatmap')

        plt.subplots_adjust(wspace=0., hspace=-0.05)
        return figure, ax


class HomeAwayDefensiveEvents:
    def __init__(self,
                 df_shots_against_home,
                 df_goals_against_home,
                 df_shots_against_away,
                 df_goals_against_away,
                 team_home,
                 team_away,
                 df_home,
                 df_away):
        self.df_shots_against_home = df_shots_against_home
        self.df_goals_against_home = df_goals_against_home
        self.df_shots_against_away = df_shots_against_away
        self.df_goals_against_away = df_goals_against_away

        self.df_home = df_home
        self.df_away = df_away
        self.team_home = team_home
        self.team_away = team_away

    def instantiate_figure_shots_goal(self):
        figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
        pitch = VerticalPitch(pitch_type='custom', pitch_length=105,
                              pitch_width=68, line_zorder=12, half=True)

        # primo grafico: defensive events home team
        pitch.draw(ax=ax[0])
        pitch.scatter(x=105 - self.df_goals_against_home['P0 norm x'],
                      y=self.df_goals_against_home['P0 norm y'],
                      s=500,
                      color='blue',
                      edgecolor='blue',
                      marker='*',
                      zorder=15,
                      ax=ax[0])
        pitch.scatter(x=105 - self.df_shots_against_home['P0 norm x'],
                      y=self.df_shots_against_home['P0 norm y'],
                      s=100,
                      color='steelblue',
                      edgecolor='steelblue',
                      zorder=12,
                      ax=ax[0])
        ax[0].set_title(f'Shots and Goals against {self.team_home}')

        # secondo grafico: shot map away team
        pitch.draw(ax=ax[1])
        pitch.scatter(x=105 - self.df_goals_against_away['P0 norm x'],
                      y=self.df_goals_against_away['P0 norm y'],
                      s=500,
                      color='green',
                      edgecolor='green',
                      marker='*',
                      zorder=15,
                      ax=ax[1])
        pitch.scatter(x=105 - self.df_shots_against_away['P0 norm x'],
                      y=self.df_shots_against_away['P0 norm y'],
                      s=100,
                      color='lightgreen',
                      edgecolor='lightgreen',
                      zorder=12,
                      ax=ax[1])
        ax[1].set_title(f'Shots and Goals against {self.team_away}')

        return figure, ax

    def instantiate_figure_heatmap(self):
        figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
        pitch = VerticalPitch(pitch_type='custom',
                              pitch_length=105, pitch_width=68, line_zorder=12)

        for idx, event in enumerate(stats.defensive_events):
            df_current_event_home = self.df_home[self.df_home['Descrizione']
                                                 == f'{event}']
            df_current_event_away = self.df_away[self.df_away['Descrizione']
                                                 == f'{event}']
            # terzo grafico: Home Shots Heatmap
            pitch.draw(ax=ax[0])
            bin_statistic = pitch.bin_statistic(df_current_event_home['P0 norm x'],
                                                df_current_event_home['P0 norm y'],
                                                statistic='count', bins=(50, 50))
            bin_statistic['statistic'] = gaussian_filter(
                bin_statistic['statistic'], 5)
            pcm = pitch.heatmap(bin_statistic, ax=ax[0], cmap='Blues')
            pitch.scatter(x=df_current_event_home['P0 norm x'],
                          y=df_current_event_home['P0 norm y'],
                          s=10,
                          color='black',
                          zorder=10,
                          alpha=0.1,
                          ax=ax[0])

        _considered_defensive_events = ', '.join(stats.defensive_events)
        ax[0].set_title(
            f'{self.team_home} Heatmap of {_considered_defensive_events} events')

        # quarto grafico: heatmap con filtro gaussiano
        pitch.draw(ax=ax[1])
        bin_statistic = pitch.bin_statistic(df_current_event_away['P0 norm x'],
                                            df_current_event_away['P0 norm y'],
                                            statistic='count', bins=(50, 50))
        bin_statistic['statistic'] = gaussian_filter(
            bin_statistic['statistic'], 5)
        pcm = pitch.heatmap(bin_statistic, ax=ax[1], cmap='Greens')
        pitch.scatter(x=df_current_event_away['P0 norm x'],
                      y=df_current_event_away['P0 norm y'],
                      s=10,
                      color='black',
                      zorder=10,
                      alpha=0.1,
                      ax=ax[1])
        _str = ', '.join(stats.defensive_events)
        ax[1].set_title(
            f'{self.team_away} Heatmap of {_considered_defensive_events} events')

        plt.subplots_adjust(wspace=0., hspace=-0.05)
        return figure, ax


class HomeAwayPassingEvents:
    def __init__(self,
                 df_keypass_home,
                 df_assist_home,
                 df_triangle_home,
                 df_pass_home,
                 df_keypass_away,
                 df_assist_away,
                 df_triangle_away,
                 df_pass_away,
                 team_home,
                 team_away
                 ) -> None:
        self.df_keypass_home = df_keypass_home
        self.df_assist_home = df_assist_home
        self.df_triangle_home = df_triangle_home
        self.df_pass_home = df_pass_home
        self.df_keypass_away = df_keypass_away
        self.df_assist_away = df_assist_away
        self.df_triangle_away = df_triangle_away
        self.df_pass_away = df_pass_away
        self.team_home = team_home
        self.team_away = team_away

    def instantiate_figure_passing(self):
        figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
        pitch = VerticalPitch(pitch_type='custom', pitch_length=105,
                              pitch_width=68, line_zorder=12, half=True)

        # primo grafico: defensive events home team
        pitch.draw(ax=ax[0])
        pitch.lines(self.df_assist_home['P0 norm x'],
                    self.df_assist_home['P0 norm y'],
                    self.df_assist_home['P1 norm x'],
                    self.df_assist_home['P1 norm y'],
                    lw=3,
                    comet=True,
                    #color = 'b',
                    cmap='Blues',
                    alpha=1,
                    ax=ax[0])
        pitch.scatter(x=self.df_assist_home['P1 norm x'],
                      y=self.df_assist_home['P1 norm y'],
                      s=100,
                      color='steelblue',
                      edgecolor='blue',
                      marker='D',
                      zorder=15,
                      ax=ax[0])

        ax[0].set_title(f'Assists {self.team_home}')

        pitch.draw(ax=ax[1])
        pitch.lines(self.df_assist_away['P0 norm x'],
                    self.df_assist_away['P0 norm y'],
                    self.df_assist_away['P1 norm x'],
                    self.df_assist_away['P1 norm y'],
                    lw=3,
                    comet=True,
                    #color = 'b',
                    cmap='Greens',
                    alpha=1,
                    ax=ax[1])
        pitch.scatter(x=self.df_assist_away['P1 norm x'],
                      y=self.df_assist_away['P1 norm y'],
                      s=100,
                      color='lightgreen',
                      edgecolor='green',
                      marker='D',
                      zorder=15,
                      ax=ax[1])

        ax[1].set_title(f'Assists {self.team_away}')

        plt.subplots_adjust(wspace=0., hspace=-0.05)
        return figure, ax

    def instantiate_figure_heatmap(self):
        figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
        pitch = VerticalPitch(pitch_type='custom', pitch_length=105,
                              pitch_width=68, line_zorder=12)
        pitch.draw(ax=ax[0])
        bin_statistic = pitch.bin_statistic(self.df_pass_home['P0 norm x'], self.df_pass_home['P0 norm y'],
                                            statistic='count', bins=(50, 50))
        bin_statistic['statistic'] = gaussian_filter(
            bin_statistic['statistic'], 5)
        pcm = pitch.heatmap(bin_statistic, ax=ax[0], cmap='Blues')
        pitch.scatter(x=self.df_pass_home['P0 norm x'],
                      y=self.df_pass_home['P0 norm y'],
                      s=10,
                      color='black',
                      zorder=10,
                      alpha=0.1,
                      ax=ax[0])
        ax[0].set_title(f'{self.team_home} Passes Heatmap')

        # quarto grafico: heatmap con filtro gaussiano
        pitch.draw(ax=ax[1])
        bin_statistic = pitch.bin_statistic(self.df_pass_away['P0 norm x'], self.df_pass_away['P0 norm y'],
                                            statistic='count', bins=(50, 50))
        bin_statistic['statistic'] = gaussian_filter(
            bin_statistic['statistic'], 5)
        pcm = pitch.heatmap(bin_statistic, ax=ax[1], cmap='Greens')
        pitch.scatter(x=self.df_pass_away['P0 norm x'],
                      y=self.df_pass_away['P0 norm y'],
                      s=10,
                      color='black',
                      zorder=10,
                      alpha=0.1,
                      ax=ax[1])
        ax[1].set_title(f'{self.team_away} Passes Heatmap')

        return figure, ax
