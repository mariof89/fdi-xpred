async def plot(container, figure):
    container.pyplot(figure)


async def aplot(cont_fig: dict):
    for cnt, fig in cont_fig.items():
        for f in fig:
            await plot(cnt, f)
