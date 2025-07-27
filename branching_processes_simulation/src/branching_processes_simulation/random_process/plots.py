import numpy as np
import plotly.graph_objects as go


def create_fig_with_axis(timeline, profile):
    # Create a figure
    fig = go.Figure()
    # Add red vertical line at x=0
    fig.add_shape(
        type="line",
        x0=0,
        y0=np.min(profile),  # starting point of the line
        x1=0,
        y1=np.max(profile),  # ending point of the line
        line=dict(color="red", width=2),  # line color and width
    )

    # Add red horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=np.min(timeline),  # starting point of the line
        y0=0,  # y-value of the line
        x1=np.max(timeline),  # ending point of the line
        y1=0,  # y-value of the line
        line=dict(color="red", width=2),  # line color and width
    )
    return fig


def plot_profile(fig, timeline, profile):
    fig.add_trace(
        go.Scatter(
            x=timeline,
            y=profile,  # y-values are the data array
            # mode='lines+markers',       # display both lines and markers
            # name='Data Points',          # legend name
            # fill='tozeroy',                                # fill area to the x-axis (y=0)
            # fillcolor='black',                             # color of the filled area
            line=dict(color="black"),  # color of the line
        )
    )

    # fig.add_trace(go.Scatter(
    #     x=timeline,
    #     y=profile,
    #     # y=,                     # y-values are the data array
    #     mode='lines+markers',       # display both lines and markers
    #     name='Data Points',          # legend name
    #     fill='tonexty',                                # fill area to the x-axis (y=0)
    #     fillcolor='lightgoldenrodyellow',                             # color of the filled area
    #     line=dict(color='black')                       # color of the line
    # ))

    return fig
