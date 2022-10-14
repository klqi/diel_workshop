#! Users/Kathy/anaconda3/envs/seaflow/bin/python3
## script that contains helper functions for diel cycle experiments

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

##### functions for plotting ##########
# helper function to plot seasonal decomp results
def plotseasonal(res, axes, pop, lat):
    # plot observed subplot
    res.observed.plot(ax=axes[0])
    axes[0].set_ylabel('Observed')
    # set secondary y axis as latitude in subplot
    ax2 = axes[0].twinx()
    ax2.set_ylabel('Latitude', c='g')
    lat.plot(ax=ax2, c='g')
    # plot trend subplot
    res.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Trend')
    # set secondary y axis as latitude in subplot
    ax3 = axes[1].twinx()
    ax3.set_ylabel('Latitude', c='g')
    lat.plot(ax=ax3, c='g')
    # plot seasonal subplot
    res.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel('Seasonal')
    # set secondary y axis as latitude in subplot
    ax4 = axes[2].twinx()
    ax4.set_ylabel('Latitude', c='g')
    lat.plot(ax=ax4, c='g')
    # plot residual subplot
    res.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel('Residual')
    # set secondary y axis as latitude in subplot
    ax5 = axes[3].twinx()
    ax5.set_ylabel('Latitude', c='g')
    lat.plot(ax=ax5, c='g')
    axes[0].set_title(f"{pop} TS Decomposition")
    
#helper function to show time decomposed results in a plot
def decomp_results(df, pop):
    # make figure
    fig, ax1 = plt.subplots(figsize=(20,8))
    # make second axis
    ax2 = ax1.twinx()
    
    ax1.plot(df.loc[df['pop']==pop,'time'], 
         df.loc[df['pop']==pop,'seasonal'], '-r.', alpha=0.75, label='seasonal')
    # plot the product of resid and seasonality
    ax1.plot(df.loc[df['pop']==pop,'time'], 
             df.loc[df['pop']==pop,'resid'], '-g.', alpha=0.75, label='residual')
    ax1.set_ylim([0.85, 1.2])
    #ax1.legend()
    ## plot pro
#     ax2.plot(df.loc[df['pop']==pop,'time'], 
#              df.loc[df['pop']==pop,'diam_med'], '.-',alpha=0.75, label='hourly diameter')
    # plot trend
    ax2.plot(df.loc[df['pop']==pop,'time'], 
             df.loc[df['pop']==pop,'trend'], '.-',alpha=0.75, label='trend')

    # plot par edges as lines
    ax1.fill_between(df['time'], 0, 1, where=df['night'] != 'day',
                    color='gray', alpha=0.3, transform=ax1.get_xaxis_transform())

    # set axis labels
    ax1.set_xlabel('time')
    ax1.set_ylabel('Residual and Seasonal')
    ax2.set_ylabel('Trend')
    ax1.set_title(f"{pop} TS Decomposition")
    #ax2.legend()
    # get all handles/labels for each axis
    handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
    #fig.legend(handles, labels)
    fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.rcParams.update({'font.size': 15})
    return(fig)



##### functions for peak detection ##########
from scipy.signal import find_peaks
from scipy.signal import argrelextrema

# inputs: df=dataframe, column=column to analyze, max/min_orders= # of data points to look around when finding extrema
# outputs: pd.Series of inflection categories, left and right edges of plateaus (for PAR data)
def peaks_and_plateaus(df=None, column=None, max_order=12, min_order=12):
    # find the floor to round to ints, better for plateau detection
    floor_par = np.floor(df[column])
    # find indices at which plateaus start (left_edges) and end (right_edges)
    peaks, peak_plateaus = find_peaks(- floor_par, plateau_size = 6)
    # get list of all indices that represent plateaus
    plateau_idx = []
    # loop through left edges
    for i in range(len(peak_plateaus['left_edges'])):
        # get starting and ending indices of plateaus
        left = peak_plateaus['left_edges'][i]
        right = peak_plateaus['right_edges'][i]
        # get range of all indices in between
        plateau = list(range(left, right+1))
        # append to list
        plateau_idx.append(plateau)
    # flatten list
    flat_plateau_idx = [x for xs in plateau_idx for x in xs]
    # get the times of the plateaus start and stops
    left_edges = df.loc[peak_plateaus['left_edges'], 'time']
    right_edges = df.loc[peak_plateaus['right_edges'], 'time']

    # find indices at which maximas are detected
    ilocs_max = argrelextrema(df[column].values, np.greater_equal, order=max_order)[0]
    # indices at which minimas are detected
    ilocs_min = argrelextrema(df[column].values, np.less_equal, order=max_order)[0]

    # set the inflection column default
    df['inflection'] = 'none'
    # set minimas in inflection
    df.loc[df.iloc[ilocs_min].index, 'inflection'] = 'min'
    # set maximas in inflection
    df.loc[df.iloc[ilocs_max].index, 'inflection'] = 'max'
    # set plateaus in inflection
    df.loc[df.iloc[flat_plateau_idx].index, 'inflection'] = 'plateau'
    # return inflection categories, along with the times of start/stop plateaus
    return(df['inflection'], left_edges, right_edges)


##### functions for solar calculations ##########


from datetime import datetime, timedelta, time
# use astral (pyephem not in python 3.8 yet)
from astral import Observer
from astral.sun import sunrise
from astral.sun import sunset
from astral.sun import night
from datetime import datetime

# function to round to nearest hour
def hour_rounder(t):
    # returns datetime rounded to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
               +timedelta(hours=t.minute//30))

def is_time_between(begin_time, end_time, check_time):
    # If check time is not given, default to current UTC time
    check_time = check_time
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else: # crosses midnight
        return check_time >= begin_time or check_time <= end_time

# helper function to calculate day/night cycle using astral
# requires lat (float), lon (float), and time (datetime) columns as input
# returns dataframe with added columns, night (day, night, sunrise, sunset) and time_day (offset for calculation)
def find_night(df):
    df = df[df['lat'].notnull()].reset_index().drop(columns=['index'])
    # offset by 1 day
    df['time_day'] = df['time'].dt.round('1d') - pd.DateOffset(1)
    df['night'] = 'night'

    # loooooop thru whee! (using sunrise/sunset is better than dawn/dusk- why?)
    for index, row in df.iterrows():
        # is it night time?
        obs = Observer(row['lat'], row['lon'], 0)
        sr = sunrise(obs, date = row['time_day'])
        ss = sunset(obs, date = row['time_day'])
        # round to nearest hour
        night_time = [hour_rounder(pd.to_datetime(x)) for x in (ss, sr)]
        # say if time is at sunset
        if row['time'] == night_time[0]:
            df.loc[index, 'night'] = 'sunset'
        # sunrise check
        elif row['time'] == night_time[1]:
            df.loc[index, 'night'] = 'sunrise'
        # night check
        elif is_time_between(night_time[0], night_time[1], row['time']):
            df.loc[index, 'night'] = 'night'
        else:
            df.loc[index, 'night'] = 'day'
    return(df)