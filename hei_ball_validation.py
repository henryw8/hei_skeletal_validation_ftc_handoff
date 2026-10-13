import pandas as pd
import numpy as np
from datetime import datetime

class HEIBallValidationClass:
    def __init__(self,hei_ball,rate,speed_cap=150,verbose=False):
        
        
        self.verbose = verbose
        self.rate = rate
        self.speed_cap = speed_cap #in km/h
        self.percent_missing_collected_at = None
        self.percent_missing_frames = None
        self.percent_missing_coords = None
        self.percent_too_fast = None
        
        if "frame_number" in hei_ball.columns: #rename frame_number column to what is used in other scripts for consistency
            hei_ball.rename({"frame_number":"frame"},axis=1,inplace=True)
            
        if self.verbose:    
            print("Creating Unix Timestamps and Formatting Seconds...",end=' ')
        hei_ball['unix_time']=pd.to_datetime(hei_ball['collected_at']).apply(datetime.timestamp)*1000 #add unix (ms) col
        hei_ball['speed_kph'] = hei_ball['ball_speed_mps']/1000*3600 #convert meters-per-second to kilometers-per-hour
        self.hei_ball = hei_ball.sort_values("unix_time")
        
        unix = hei_ball.unix_time.to_list()
        total_duration_sec = (unix[-1] - unix[0])/1000 #get total duration covered by frames in minutes
        self.total_duration_min = total_duration_sec/60

        
    
    def percent_missing_collected_times(self,latency=None):
        """By period, returns dict with percent missing collected_at values in the consective range 
        between the first and last values in the data. Option to filter by latency."""
        
        if latency: #filter by latency if desired
            df = self.hei_ball.query("latency == @latency").drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
        else:
            df = self.hei_ball.drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
            
        all_periods = df['period'].unique().tolist()
        missing_times = {}
        time_us = int(1e6/self.rate) #1.1 to account for slight changes in frequency
        for period in all_periods:
            df_this = df.query("period == @period").copy()
            df_this['rate'] = df_this['collected_at'].diff().dt.microseconds
            skip = df_this['rate'].value_counts().reset_index().query("index > 1.1*@time_us")
            frames_lost = sum((skip['index'].values/time_us-1)*skip['rate'].values)
            missing_times[period] = 100*frames_lost/df_this.shape[0]
        self.percent_missing_collected_at = missing_times
        return missing_times
    
    def percent_missing_frame_ids(self,latency=None):
        """By period, returns dict with percent of missing frame IDs in the consective range 
        between the first and last frame IDs in the data. Option to filter by latency."""
        missing_frames = {}
        if latency: #filter by latency if desired
            df = self.hei_ball.query("latency == @latency").drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
        else:
            df = self.ball.drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
       
        def find_missing(lst): #returns missing numbers in a consecutive range
            return sorted(set(range(lst[0], lst[-1])) - set(lst))
        all_periods = df['period'].unique().tolist()
        for period in all_periods:
            df_this = df.query("period == @period").copy()
            all_present_frames = sorted(list(df_this.frame.unique())) 
            missing_frames[period] = len(find_missing(all_present_frames))*100/df_this.shape[0]
        self.percent_missing_frames = missing_frames
        return missing_frames
    
    def percent_missing_coordinates(self,latency=None):
        """By period, returns percentage of frames that are missing at least one of the ball x,y,z coordinates"""
        missing_coords = {}
        if latency:
            df = self.hei_ball.query("lateny==@latency").copy()
        else:
            df = self.hei_ball.copy()
        
        all_periods = df['period'].unique().tolist()
        for period in all_periods:
            df_this = df.query("period == @period").copy()
            missing_df = df.where(df.ball_x.isna()).dropna(how='all')
            missing_coords[period] = missing_df.shape[0]*100/df_this.shape[0]
        self.percent_missing_coords = missing_coords
        return missing_coords
    

    def percent_speed_outliers(self,latency=None):
        """By period, returns percentage of frames that have the ball speed above the given speed cap"""
        outlier_percent = {}
        if latency:
            df = self.hei_ball.query("lateny==@latency").copy()
        else:
            df = self.hei_ball.copy()
        
        all_periods = df['period'].unique().tolist()
        for period in all_periods:
            df_this = df.query("period == @period").copy()
            outlier_df = df_this[df_this.speed_kph>self.speed_cap]
            outlier_percent[period] = outlier_df.shape[0]*100/df_this.shape[0]
        self.percent_too_fast = outlier_percent
        return outlier_percent