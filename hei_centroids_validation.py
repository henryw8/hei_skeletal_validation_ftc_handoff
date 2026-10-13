import pandas as pd
import numpy as np
from datetime import datetime
import math
class HEICentroidsValidationClass:
   
    def __init__(self,hei_centroids,rate,speed_cap=10.2,latency=None,verbose=True):
        self.fifa_hex = "#005391"
        self.speed_cap = speed_cap
        self.latency = latency
        self.verbose = verbose
        self.rate = rate
        self.num_players_tracked_df = None
        self.num_players_tracked_plot = None
        self.num_referees_tracked_df = None
        self.num_referees_tracked_plot = None
        self.missing_referee_frames = None
        #Helpers
        def snakecase_to_camelcase(cols):
            """ Changes column names to necessary format. """
            out = []
            for c in cols:
                if c[:2] in ['l_','r_']:
                    if c[0]=='l':
                        out.append('l'+c[2].upper()+c[3:])
                    else:
                        out.append('r'+c[2].upper()+c[3:])
                elif c=='frame_number':
                    out.append('frame')
                else:
                    out.append(c)
            return out
             
        if self.verbose:    
            print("Creating Unix Timestamps and Formatting Seconds...",end=' ')
        hei_centroids['unix_time']=pd.to_datetime(hei_centroids['collected_at']).apply(datetime.timestamp)*1000 #add unix (ms) col
        hei_centroids['sec_truncated'] = hei_centroids['sec'].apply(math.trunc)
        hei_centroids['time_truncated'] = 60*hei_centroids['min'] + hei_centroids['sec_truncated']
        if self.verbose:
            print("Done!")
        if self.verbose:
            print("Checking column names...",end=' ')
        hei_centroids.columns = snakecase_to_camelcase(hei_centroids.columns.values) #convert columns to camelcase if necessary
        self.hei_centroids = hei_centroids #the combined pose dataset
        self.hei_player_centroids = hei_centroids[hei_centroids.role_name.isin(['goalie','player'])].copy()
        self.hei_referee_centroids = hei_centroids[hei_centroids.role_name.isin(['goalie','player'])==False].copy()
        self.players = hei_centroids.player_id.unique() #array of unique player IDs in the dataset
        self.total_num_frames = hei_centroids.shape[0] #number of total frames in the dataset
        if self.verbose:
            print("Done!")
            
        if self.verbose:
            print("Chunking games into 5 min intervals...",end=" ")
        hei_centroids['min_group'] = hei_centroids['min']//5
        self.hei_centroids = hei_centroids #the combined pose dataset
        self.hei_player_centroids = hei_centroids[hei_centroids.role_name=='player'].copy()
        if self.verbose:
            print("Done!")
            
        if self.verbose:
            print("Segmenting players...",end=' ')
        self.players = list(hei_centroids.player_id.unique())
        frames_by_player = {i:None for i in self.players}
        for i in self.players:
            d = hei_centroids[hei_centroids.player_id==i].copy()
            d.sort_values("unix_time",inplace=True) #sort player frames so they're in chronological order
            frames_by_player[i] = d 
  
        self.frames_by_player = frames_by_player #dict mapping player IDs to their frames
        if self.verbose:
            print("Done!")
            print('-'*50)
            print("Initialization Complete!")
        
    def percent_missing_collected_times(self,latency=None):
        """By period, returns dict with percent missing collected_at values in the consective range 
        between the first and last values in the data. Option to filter by latency."""
        
        if latency: #filter by latency if desired
            df = self.hei_centroids.query("latency == @latency").drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
        else:
            df = self.hei_centroids.drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
            
        all_periods = df['period'].unique().tolist()
        missing_times = {}
        time_us = int(1e6/self.rate) #1.1 to account for slight changes in frequency
        for period in all_periods:
            df_this = df.query("period == @period").copy()
            df_this['rate'] = df_this['collected_at'].diff().dt.microseconds
            skip = df_this['rate'].value_counts().reset_index().query("index > 1.1*@time_us")
            frames_lost = sum((skip['index'].values/time_us-1)*skip['rate'].values)
            missing_times[period] = 100*frames_lost/df_this.shape[0]

        return missing_times
    
    def percent_missing_frame_ids(self,latency=None):
        """By period, returns dict with percent of missing frame IDs in the consective range 
        between the first and last frame IDs in the data. Option to filter by latency."""
        missing_frames = {}
        if latency: #filter by latency if desired
            df = self.hei_centroids.query("latency == @latency").drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
        else:
            df = self.hei_centroids.drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
       
        def find_missing(lst): #returns missing numbers in a consecutive range
            return sorted(set(range(lst[0], lst[-1])) - set(lst))
        all_periods = df['period'].unique().tolist()
        for period in all_periods:
            df_this = df.query("period == @period").copy()
            all_present_frames = sorted(list(df_this.frame.unique())) 
            missing_frames[period] = len(find_missing(all_present_frames))*100/df_this.shape[0]
        return missing_frames
    
    def percent_nonmonotonic_increase(self,latency=None):
        """By period, returns dict with percent of frames that exhibit a non-monotic increase in frame ID when sorted by collected at.
        Option to filter by latency."""
        out = {}
        if latency: #filter by latency if desired
            df = self.hei_centroids.query("latency == @latency").drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
        else:
            df = self.hei_centroids.drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
        all_periods = df['period'].unique().tolist()
        for period in all_periods:
            df_this = df.query("period == @period").copy()
            df_this['frame_diff'] = df_this['frame'] - df_this['frame'].shift(1)
            out[period] = df_this.query("frame_diff!=1").dropna().shape[0]/df_this.shape[0]*100
        return out
    
    def period_durations(self,latency=None):
        """By period, returns dict with duration in ms of each period in the data"""
        out = {}
        if latency: #filter by latency if desired
            df = self.hei_centroids.query("latency == @latency").drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
        else:
            df = self.hei_centroids.drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
        all_periods = df['period'].unique().tolist()
        for period in all_periods:
            df_this = df.query("period == @period").copy()
            out[period] = df_this.iloc[[-1]].unix_time.item() - df_this.iloc[[0]].unix_time.item() #since sorted, index max - index min
        return out
    
    def player_duplication_frames(self):
        """Returns frames where one frame ID has multiple frames of the same player"""
        out = []
        frames_by_player = self.frames_by_player
        for p in frames_by_player:
            df = frames_by_player[p]
            df = df[df.role_name=='player'] #exclude referees, otherwise we get referee frames
            if df.shape[0]==0:
                continue
            v = df.frame.value_counts()
            out.append(df[df.frame.isin(v.index[v.gt(2)])])
        return pd.concat(out)
    
    def percent_player_duplications(self):
        """Returns percent of frames that are player duplications"""
        return self.player_duplication_frames().shape[0]*100/self.hei_player_centroids.shape[0]
        
    
    def speed_outlier_frames(self):
        """Returns frames where player speed is greater than self.speed_cap (10.2 m/s)"""
        df = self.hei_centroids
        out = df[df.speed_mps>10.2].copy()
        self.speed_outlier_frames = out
        return out
    
    def percent_speed_outliers(self):
        """Returns percent of frames where player speed is above self.speed_cap"""
        outliers = self.speed_outlier_frames()
        return outliers.shape[0]*100/self.hei_player_centroids.shape[0]
    
    def find_missing_referee_frames(self):
        """Returns list of frame IDs with at least one missing referee"""
        all_frames = hei_centroids.frame.unique()
        out = []
        for i in all_frames:
            f = self.hei_referee_centroids.query(f"frame=={i}")
            if sorted(f.role_id.to_list())!=[3,4,5]:
                out.append(i)
        self.missing_referee_frames = out
        return out
    
    def percent_missing_referees(self):
        """Returns percent of total frame IDs with at least one missing referee"""
        if self.missing_referee_frames is None:
            missing_referee_frames = self.find_missing_referee_frames()
        return len(missing_referee_frames)*100/len(hei_centroids.frame.unique())
    
    def num_people_tracked(self,person):
        """
        Given a half/game, splits the game into 5 minute intervals and looks at how many players on each team are tracked at each second. For each time interval,
        returns the mean number of players per team tracked every second.
        """
        if person=='player':    
            centroids = self.hei_player_centroids
        elif person=='referee':
            centroids = self.hei_referee_centroids
#         print(f"{centroids.shape[0]} Frames to Analyze")
        centroids_by_min_group = {i:None for i in centroids.min_group.unique()}
        for mg in centroids_by_min_group:
            frames = centroids[centroids.min_group==mg] # filter only frames from that 5 minute interval
            if person=="player":
                num_players_by_team = {i:[] for i in frames.team_name.unique()} # map team to a list which contains the number of players tracked for that team at every distint time_rounded
                for t in frames.time_truncated.unique():
                    for team in num_players_by_team:
                        num_players_by_team[team].append(frames[(frames.team_name==team)&(frames.time_truncated==t)][['player_id']].drop_duplicates().shape[0]) #find number of distinct players being tracked for that team at time=t
                for team in num_players_by_team:
                    num_players_by_team[team] = np.mean(np.array(num_players_by_team[team]))
                centroids_by_min_group[mg] = num_players_by_team
            elif person=="referee":
                num_refs = []
                for t in frames.time_truncated.unique():
                    num_refs.append(frames[['player_id']].drop_duplicates().shape[0])
                centroids_by_min_group[mg] = np.mean(np.array(num_refs))
            
        out = {f"{5*i}-{5*i+5}":centroids_by_min_group[i] for i in centroids_by_min_group}  
        if person=='player':
            self.num_players_tracked_df = out
        elif person=='referee':
            self.num_referees_tracked_df = out
            
        return out

    def plot_num_people_tracked(self,person):
        """
        Plots in grouped bar form the results from self.num_people_tracked()
        """
        if person=="player":
            d = self.num_players_tracked_df
            df = pd.DataFrame(d).transpose().reset_index().rename({"index":"min_group"},axis=1).round(2)
            teams = list(df.columns.values)[1:]
            colors = {teams[0]:'steelblue',teams[1]:'firebrick'}

            fig = go.Figure([go.Bar(name=teams[0],x=df.min_group, y=df[teams[0]],marker_color=colors[teams[0]],text=df[teams[0]],textposition='auto'),
                         go.Bar(name=teams[1],x=df.min_group, y=df[teams[1]],marker_color=colors[teams[1]],text=df[teams[1]],textposition='auto')
                        ])

            fig.update_layout(height=600,width=1100,title="Mean Number of Players Tracked per Team over Time",plot_bgcolor='white')
            fig.update_xaxes(title="Minute Group")
            fig.update_yaxes(range=[0,11],showticklabels=False,title="")
            self.num_players_tracked_plot = fig
        
        elif person=="referee":
            d = self.num_referees_tracked_df
            df = pd.DataFrame(d,index=[0]).transpose().reset_index().rename({"index":"min_group",0:"num_refs"},axis=1).round(2)
            fig = go.Figure([go.Bar(x=df.min_group, y=df.num_refs,marker_color=self.fifa_hex,text=df.num_refs,textposition='auto')])
            fig.update_layout(height=400,width=600,title="Mean Number of Referees Tracked over Time",plot_bgcolor='white')
            fig.update_xaxes(title="Minute Group")
            fig.update_yaxes(range=[0,5],showticklabels=False,title="")
            self.num_referees_tracked_plot = fig
        
        return fig