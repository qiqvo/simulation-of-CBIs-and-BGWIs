import numpy as np
import plotly.graph_objects as go

def plot_stacked_below(time, profile):
    cum_profile = np.cumsum(profile[::-1,:], axis=0)[::-1]
    # print(cum_profile)
    return plot_stacked(time, profile, cum_profile, True)

def plot_stacked_above(time, profile):
    cum_profile = np.cumsum(profile, axis=0)
    return plot_stacked(time, profile, cum_profile, False)

def plot_stacked(time, profile, cum_profile, below):
    timeline = list(range(time))

    fig = create_fig_with_axis(timeline, cum_profile)
    for i in range(len(profile)):
        plot_profile(fig, list(range(i, time)), cum_profile[i,i:], profile[i, i:], below)

    return fig, cum_profile

def create_fig_with_axis(timeline, profile):
    # Create a figure
    fig = go.Figure()
    # Add red vertical line at x=0
    fig.add_shape(
        type='line',
        x0=0, y0=np.min(profile),       # starting point of the line
        x1=0, y1=np.max(profile),       # ending point of the line
        line=dict(color='red', width=2)  # line color and width
    )

    # Add red horizontal line at y=0
    fig.add_shape(
        type='line',
        x0=np.min(timeline),  # starting point of the line
        y0=0,                       # y-value of the line
        x1=np.max(timeline),  # ending point of the line
        y1=0,                       # y-value of the line
        line=dict(color='red', width=2)  # line color and width
    )
    return fig

def plot_profile(fig, timeline, profile, mask, below):
    # print(timeline, profile, mask)
    if mask[0] == 0:
        return None
    
    last_non_zero = len(mask)-1
    while last_non_zero != 0 and mask[last_non_zero - 1] == 0:
        last_non_zero -= 1

    last_non_zero += 1
    timeline = timeline[:last_non_zero]
    profile = profile[:last_non_zero]
    mask = mask[:last_non_zero]

    fig.add_trace(go.Scatter(
        x=timeline,
        y=profile - mask,                     # y-values are the data array
        # mode='lines+markers',       # display both lines and markers
        # name='Data Points',          # legend name        
        # fill='tozeroy',                                # fill area to the x-axis (y=0)
        # fillcolor='black',                             # color of the filled area
        line=dict(color='black')                       # color of the line
    ))
    
    if below and timeline[0] != 0:
        timeline = np.insert(timeline, 0, timeline[0]-1)
        profile = np.insert(profile, 0, 0)

    fig.add_trace(go.Scatter(
        x=timeline,
        y=profile,
        # y=,                     # y-values are the data array
        mode='lines+markers',       # display both lines and markers
        name='Data Points',          # legend name        
        fill='tonexty',                                # fill area to the x-axis (y=0)
        fillcolor='lightgoldenrodyellow',                             # color of the filled area
        line=dict(color='black')                       # color of the line
    ))
