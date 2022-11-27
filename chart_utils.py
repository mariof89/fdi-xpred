from mplsoccer import PyPizza, add_image
from highlight_text import fig_text
import matplotlib.pyplot as plt
import utilities


class pizza:
    def __init__(self, params, values_home, values_away, color_home, color_away, team_home, team_away):
        self.params = params
        self.values_home = [round(i, 1) for i in values_home]
        self.values_away = [round(i, 1) for i in values_away]
        self.color_home = color_home
        self.color_away = color_away
        self.team_home = team_home
        self.team_away = team_away

        self.serie_a_logo = utilities.get_image('logos/serie_a.png')

    def draw(self):
        baker = PyPizza(
            params=self.params,  # list of parameters
            background_color="#66000000",  # background color
            straight_line_color="#222222",  # color for straight lines
            straight_line_lw=1,  # linewidth for straight lines
            last_circle_lw=0,  # linewidth of last circle
            last_circle_color="#222222",  # color of last circle
            other_circle_ls="-.",  # linestyle for other circles
            other_circle_lw=0,  # linewidth for other circles
            inner_circle_size=20
        )

        # plot pizza
        fig, ax = baker.make_pizza(
            self.values_home,  # list of values
            compare_values=self.values_away,  # comparison values
            value_colors=['#FFFFFF' for i in self.params],
            # figsize=(8, 8),  # adjust figsize according to your need
            color_blank_space="same",
            blank_alpha=0.4,
            kwargs_slices=dict(
                facecolor=f"{self.color_home}", edgecolor="#222222",
                zorder=2, linewidth=1
            ),  # values to be used when plotting slices
            kwargs_compare=dict(
                facecolor=f"{self.color_away}", edgecolor="#222222",
                zorder=2, linewidth=1,
            ),
            kwargs_params=dict(
                color="#FFFFFF", fontsize=12,
                va="center"
            ),  # values to be used when adding parameter
            kwargs_values=dict(
                color="#000000", fontsize=12,
                zorder=3,
                bbox=dict(
                    edgecolor="#000000", facecolor=f"{self.color_home}",
                    boxstyle="round,pad=0.2", lw=1
                )
            ),  # values to be used when adding parameter-values labels
            kwargs_compare_values=dict(
                color="#000000", fontsize=12, zorder=3,
                bbox=dict(
                    edgecolor="#000000", facecolor=f"{self.color_away}", boxstyle="round,pad=0.2", lw=1)
            ),  # values to be used when adding parameter-values labels
        )

        fig_text(
            0.515, 0.99, f"<{self.team_home}> vs <{self.team_away}>", size=17, fig=fig,
            highlight_textprops=[
                {"color": f'{self.color_home}'}, {"color": f'{self.color_away}'}],
            ha="center", color="#FFFFFF"
        )
        # add subtitle
        fig.text(
            0.515, 0.955,
            "Percentile Rank vs Serie A | Season 2021-22",
            size=13,
            ha="center", color="#F2F2F2"
        )
        # add image
        ax_image = add_image(
            self.serie_a_logo, fig,
            left=0.4478,
            bottom=0.4315,
            width=0.13,
            height=0.127
        )

        return fig, ax
