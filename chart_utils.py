from mplsoccer import PyPizza


class pizza:
    def __init__(self, params, values_home, values_away, color_home, color_away):
        self.params = params
        self.values_home = [round(i, 1) for i in values_home]
        self.values_away = [round(i, 1) for i in values_away]
        self.color_home = color_home
        self.color_away = color_away

    def draw(self):
        baker = PyPizza(
            params=self.params,  # list of parameters
            background_color="#EBEBE9",  # background color
            straight_line_color="#222222",  # color for straight lines
            straight_line_lw=1,  # linewidth for straight lines
            last_circle_lw=1,  # linewidth of last circle
            last_circle_color="#222222",  # color of last circle
            other_circle_ls="-.",  # linestyle for other circles
            other_circle_lw=1  # linewidth for other circles
        )

        # plot pizza
        fig, ax = baker.make_pizza(
            self.values_home,  # list of values
            compare_values=self.values_away,  # comparison values
            # figsize=(8, 8),  # adjust figsize according to your need
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
                color="#000000", fontsize=12,
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
                bbox=dict(edgecolor="#000000", facecolor=f"{self.color_away}", boxstyle="round,pad=0.2", lw=1)
            ),  # values to be used when adding parameter-values labels
        )

        return fig, ax
