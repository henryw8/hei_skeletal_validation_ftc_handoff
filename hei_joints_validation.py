class HEIJointsValidationClass:
   
    def __init__(self,hei_joints,rate, include_limb_extensions=True, include_abnormally_high=True, include_abnormally_low=True, latency=None,verbose=True):

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
        def euclidean_3d(p1,p2):
            """Calculate euclidean distance between two 3D points"""
            return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)**.5
        
        def label_outliers(df,var,type_outlier,iqr_factor=1.5):
            """
            Given a df and a column var, add a new column with a dummy indicating whether or not
            the row is an outlier for the given var
            """   
            upper_heights = ['lEar_z', 'lShoulder_z', 'neck_z', 'nose_z', 'rEye_z', 'rShoulder_z']
            arms_heights = ['lElbow_z', 'lWrist_z', 'rElbow_z', 'rWrist_z']
            legs_heights = ['lAnkle_z', 'lHip_z', 'lKnee_z', 'rAnkle_z', 'rHip_z', 'rKnee_z', ]
            joint_heights = ['lEar_z', 'lShoulder_z', 'neck_z','nose_z', 'rEye_z', 'rShoulder_z','lElbow_z', 
                             'lWrist_z', 'rElbow_z', 'rWrist_z','lAnkle_z', 'lHip_z', 'lKnee_z', 'rAnkle_z', 'rHip_z', 'rKnee_z', ]
            if type_outlier == 'limb_extension':
                col_name = var + "_outlier"
                q1, q3= df[var].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 -(iqr_factor * iqr) 
                upper_bound = q3 +(iqr_factor * iqr)
                df[col_name] = np.where((df[var]<lower_bound) | (df[var]>upper_bound),1,0)
            elif type_outlier == "abnormally_low":
                col_name = 'abnormally_low'
                df[col_name] = np.where((df[joint_heights] < 0).any(axis=1),1,0)
            elif type_outlier == 'abnormally_high':
                col_name = 'abnormally_high'
                df[col_name] = np.where(((df[upper_heights] > 2.5)|(df[arms_heights] > 2.75)|(df[legs_heights] > 2)).any(axis=1),1,0)
                
        ##################################################################################################################################################################                
        #Initialize relevant coordinates and other information
        self.joint_coords = ['lAnkle_x', 'lAnkle_y', 'lAnkle_z', 'lEar_x', 'lEar_y', 'lEar_z', 'lElbow_x', 'lElbow_y', 
                'lElbow_z', 'lHip_x', 'lHip_y', 'lHip_z', 'lKnee_x', 'lKnee_y', 'lKnee_z', 'lShoulder_x', 
                'lShoulder_y', 'lShoulder_z', 'lWrist_x', 'lWrist_y', 'lWrist_z', 'neck_x', 'neck_y', 'neck_z', 
                'nose_x', 'nose_y', 'nose_z', 'rAnkle_x', 'rAnkle_y', 'rAnkle_z', 'rElbow_x', 'rElbow_y', 'rElbow_z',
                'rEye_x', 'rEye_y', 'rEye_z', 'rHip_x', 'rHip_y', 'rHip_z', 'rKnee_x', 'rKnee_y', 'rKnee_z', 'rShoulder_x', 
                'rShoulder_y', 'rShoulder_z', 'rWrist_x', 'rWrist_y', 'rWrist_z', 'lEye_x', 'lEye_y', 'lEye_z','rEar_x', 
                'rEar_y', 'rEar_z']
        self.goal_scoring_joint_coords = ['lAnkle_x', 'lAnkle_y', 'lAnkle_z', 'lEar_x', 'lEar_y', 'lEar_z', 'lHip_x', 'lHip_y', 'lHip_z', 'lKnee_x', 'lKnee_y', 'lKnee_z', 'lShoulder_x', 'lShoulder_y', 'lShoulder_z', 'neck_x', 
                                          'neck_y', 'neck_z', 'nose_x', 'nose_y', 'nose_z', 'rAnkle_x', 'rAnkle_y', 'rAnkle_z', 'rElbow_x', 'rElbow_y', 
                                          'rElbow_z', 'rEye_x', 'rEye_y', 'rEye_z', 'rHip_x', 'rHip_y', 'rHip_z', 'rKnee_x', 'rKnee_y', 'rKnee_z', 'rShoulder_x', 'rShoulder_y', 'rShoulder_z']
        self.joint_pairs = [('Elbow','Wrist'), ('Shoulder','Elbow'), ('Hip','Knee'),('Knee','Ankle')]
        self.distance_metrics = ['l_Shoulder_Elbow_dist','r_Shoulder_Elbow_dist','l_Elbow_Wrist_dist','r_Elbow_Wrist_dist','l_Hip_Knee_dist','r_Hip_Knee_dist','l_Knee_Ankle_dist','r_Knee_Ankle_dist']
        self.limb_ext_outlier_cols = ['l_Shoulder_Elbow_dist_outlier','r_Shoulder_Elbow_dist_outlier','l_Elbow_Wrist_dist_outlier',
                'r_Elbow_Wrist_dist_outlier','l_Hip_Knee_dist_outlier','r_Hip_Knee_dist_outlier','l_Knee_Ankle_dist_outlier',
                'r_Knee_Ankle_dist_outlier']  
        self.joints_heights = ['lEar_z', 'lShoulder_z', 'neck_z','nose_z', 'rEye_z', 'rShoulder_z','lElbow_z', 
                             'lWrist_z', 'rElbow_z', 'rWrist_z','lAnkle_z', 'lHip_z', 'lKnee_z', 'rAnkle_z', 'rHip_z', 'rKnee_z', ]
        self.fifa_hex = "#005391"
        self.verbose = verbose
        self.rate = rate
        self.time_ms = int(1000/self.rate)
        
        ##################################################################################################################################################################                      
        #Initialize data into desired format and instantiate relevant characteristics/attributes
        if self.verbose:
            print("Creating Unix Timestamps...",end=' ')
        hei_joints['unix_time']=pd.to_datetime(hei_joints['collected_at']).apply(datetime.timestamp)*1000 #add unix (ms) col
        if self.verbose:
            print("Done!")
        
        if self.verbose:
            print("Checking column names...",end=' ')
        hei_joints.columns = snakecase_to_camelcase(hei_joints.columns.values) #convert columns to camelcase if necessary
        if latency:
            hei_joints = hei_joints[hei_joints.latency==latency].copy()
        self.hei_joints = hei_joints #the combined pose dataset
        self.players = hei_joints.player_id.unique() #array of unique player IDs in the dataset
        self.total_num_frames = hei_joints.shape[0] #number of total frames in the dataset
        if self.verbose:
            print("Done!")
        
        if self.verbose:
            print("Segmenting players...",end=' ')
        frames_by_player = {i:None for i in self.players}
        for i in self.players:
            d = hei_joints[hei_joints.player_id==i].copy()
            d['rate'] = d['collected_at'].diff().dt.microseconds/1000
            d.sort_values("collected_at",inplace=True) #sort player frames so they're in chronological order
            frames_by_player[i] = d 
  
        self.frames_by_player = frames_by_player #dict mapping player IDs to their frames
        if self.verbose:
            print("Done!")
        
        if include_limb_extensions:
            if self.verbose:
                print("Calculating limb extension lengths...",end=' ')
            #modify self.frames_by_player to include limb extension lengths
            for i in self.frames_by_player:
                all_pose = self.frames_by_player[i]
                for l1,l2 in self.joint_pairs:
                    for side in ('l','r'):
                        all_pose[f"{side}_{l1}_{l2}_dist"] = euclidean_3d((all_pose[side+l1+"_x"],all_pose[side+l1+"_y"],all_pose[side+l1+"_z"]), (all_pose[side+l2+"_x"],all_pose[side+l2+"_y"],all_pose[side+l2+"_z"]))
            if self.verbose:
                print("Done!")

            if self.verbose:
                print("Labelling limb extension outliers...",end=' ')
            #for each player label outliers for each metric
            for p in self.frames_by_player:
                d = self.frames_by_player[p]
                for m in self.distance_metrics:
                    label_outliers(d,m,type_outlier='limb_extension')
            if self.verbose:
                print("Done!")
        
        if include_abnormally_high:
            if self.verbose:
                print("Labelling abnormally high frames...",end=' ')
            for p in self.frames_by_player:
                if p==None:
                    continue
                df = self.frames_by_player[p]
                label_outliers(df,var=None,type_outlier='abnormally_high')
            if self.verbose:
                print("Done!")
        
        if include_abnormally_low:
            if self.verbose:
                print("Labelling abnormally low frames...",end=' ')
            for p in self.frames_by_player:
                if p==None:
                    continue
                df = self.frames_by_player[p]
                label_outliers(df,var=None,type_outlier='abnormally_low')
            if self.verbose:
                print("Done!")

  
        ##################################################################################################################################################################
        #Initialize other attributes that are set later on when calling class funcs
        self.limb_ext_outliers_by_player = None #dict mapping player IDs to dicts mapping each distance metric to outliers for that metric
        self.limb_ext_outlier_percent_df = None #df with mean % outlier frames by metric
        self.limb_ext_outlier_percent_plot = None #bar chart with mean % outlier frames by metric
        self.player_limb_ext_boxplots = None #dict mapping player IDs to player-specific limb-extension distribution length boxplots
        self.limb_ext_outlier_duration_df = None #df with mean outlier sequence duration (ms)
        self.limb_ext_outlier_duration_plot = None #bar chart with mean outlier sequence duration (ms)
        self.player_limb_ext_outlier_duration_boxplots = None #dict mapping player IDs to player-specific limb-extension duration boxplots
        self.limb_ext_outlier_dur_df_by_player = None
        
        self.abnormally_high_frames_dict = None #dict mapping body region (upper,arms,legs) to anomaly frames
        self.abnormally_high_percent_df = None
        self.abnormally_high_percent_plot = None
        self.abnormally_high_outlier_dur_df_by_player = None
        self.player_abnormally_high_duration_boxplots = None
        
        self.abnormally_low_frames_dict = None
        self.abnormally_low_percent_df = None
        self.abnormally_low_percent_plot = None
        self.abnormally_low_outlier_dur_df_by_player = None
        self.player_abnormally_low_duration_boxplots = None
        ##################################################################################################################################################################
        if self.verbose:
            print('-'*50)
            print("Initialization Complete!")
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def get_frames_by_player(self):
        """Getter for dict mapping player IDs to that player's frames"""
        return self.frames_by_player
    
    def get_player_ids(self):
        """Getter for list of distinct player IDs"""
        return list(self.players)
    
    def plot_limb_ext_percent_outlier(self):
        """Getter for limb extension outlier percent by metric plot"""
        return self.limb_ext_outlier_percent_plot
    
    def plot_player_limb_ext_boxplots(self):
        """Getter for player limb extension boxplots dict"""
        return self.player_limb_ext_boxplots
    
    def plot_limb_ext_outlier_durations(self):
        """Getter for limb extension durations by metric plot"""
        return self.limb_ext_outlier_duration_plot
    
    def plot_abnormally_high_percent_outlier(self):
        """Getter for high anomalies percent by metric plot"""
        return self.abnormally_high_percent_plot
    
    def plot_abnormally_low_percent_outlier(self):
        """Getter for low anomalies percent by metric plot"""
        return self.abnormally_low_percent_plot
    
    def percent_missing_joints(self):
        """Returns percentage of frames with missing values, averaged across all joint coordinates"""
        percentages = np.array(list(hei_joints[self.joint_coords].isna().sum()))/self.total_num_frames
        return percentages.mean()
    
    def percent_missing_collected_times(self,latency=None):
        """By period, returns dict with percent missing collected_at values in the consective range 
        between the first and last values in the data. Option to filter by latency."""
        
        if latency: #filter by latency if desired
            df = self.hei_joints.query("latency == @latency").drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
        else:
            df = self.hei_joints.drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
            
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
            df = self.hei_joints.query("latency == @latency").drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
        else:
            df = self.hei_joints.drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
       
        def find_missing(lst): #returns missing numbers in a consecutive range
            return sorted(set(range(lst[0], lst[-1])) - set(lst))
        all_periods = df['period'].unique().tolist()
        for period in all_periods:
            df_this = df.query("period == @period").copy()
            all_present_frames = sorted(list(df_this.frame.unique())) 
            missing_frames[period] = len(find_missing(all_present_frames))*100/df_this.shape[0]
        return missing_frames
    
    def percent_nonmonotonic_increase(self,latency=None):
        """By period, returns percent of frames that exhibit a non-monotic increase in frame ID when sorted by collected at """
        out = {}
        if latency: #filter by latency if desired
            df = self.hei_joints.query("latency == @latency").drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
        else:
            df = self.hei_joints.drop_duplicates(subset='collected_at').sort_values(by='collected_at',ascending=True).copy()
        all_periods = df['period'].unique().tolist()
        for period in all_periods:
            df_this = df.query("period == @period").copy()
            df_this['frame_diff'] = df_this['frame'] - df_this['frame'].shift(1)
            out[period] = df_this.query("frame_diff!=1").dropna().shape[0]/df_this.shape[0]*100
        return out
    
    def percent_missing_live(self):
        """Returns percentage of frame IDs missing the live latency."""
        unique_frames = hei_joints.frame.unique()
        df = hei_joints[['frame','latency']].drop_duplicates()
        missing_live = []
        for f in tqdm(unique_frames): #iterate thru all unique frame IDs
            d = df[df.frame==f]
            if d.shape[0]!=2: #if true, missing one of live or delayed
                latencies = d.latency.unique()
                if 'live' not in latencies: #check it's missing live
                    missing_live.append(f)
        return len(missing_live)*100/len(unique_frames) #return final percentage
    
    def percent_missing_delayed(self):
        """Returns percentage of frame IDs missing the delayed latency."""
        unique_frames = hei_joints.frame.unique()
        df = hei_joints[['frame','latency']].drop_duplicates()
        missing_delayed = []
        for f in tqdm(unique_frames): #iterate thru all unique frame IDs
            d = df[df.frame==f]
            if d.shape[0]!=2: #if true, missing one of live or delayed
                latencies = d.latency.unique()
                if 'delayed' not in latencies: #check it's missing delayed
                    missing_live.append(f)
        return len(missing_delayed)*100/len(unique_frames) #return final percentage
    
    
        
    
    def outlier_duration_faster(self,player_frames,metric):
        """
        Given a binary sequence (1:outlier, 0:not outlier) returns mapping of (start_frame,end_frame) to
        list of outlier frame sequence lengths
        """
        sequence = player_frames[['frame',metric]]
        sequence['difference'] = sequence[metric] - sequence[metric].shift(1)
        seq_transitions = sequence[sequence['difference']!=0]
        out = {}
        start,end = None,None
        for i,row in seq_transitions.itertuples():
            frame = row.frame.item()
            diff = row.difference.item()
            if math.isnan(diff):
                continue
            if diff == 1:
                start = frame
            elif diff==-1:
                end = frame
                out[(start,end-1)] = end - start
                start,end = None,None
        return out
    
    def outlier_duration_slower(self,player_frames,metric):
        """Given sorted player frames and an outlier column, returns mapping of (start_frame,end_frame) to
        list of outlier frame sequence lengths """
        out = {}
        cols_map = dict(zip(player_frames.columns.values,range(len(player_frames.columns.values))))
        frame_key = cols_map['frame']
        metric_key = cols_map[metric]
        start,end,prev = player_frames.frame.to_list()[0],None,0
        for row in player_frames.itertuples(index=False):
            frame = row[frame_key]
            dummy = row[metric_key]
            if prev==1 and dummy==0: #come to the end of a sequence
                end = frame
    
                out[(start,end)] = end-start+1
                prev = 0
            elif prev==0 and dummy==1: #come to begininng of a sequence
                start = frame
                prev = 1
            else:
                prev = dummy
        return out
    
    def plot_outlier_duration_boxplots(self,type_outlier):
        """Returns dict mapping player IDs to a boxplot"""
        if type_outlier=='limb_extension':
            player_limb_ext_outlier_duration_boxplots = {}
            for p in self.players:
                fig = go.Figure()
                # Use x instead of y argument for horizontal plot
                x_labs = []
                outlier_dur_df_by_player = self.limb_ext_outlier_dur_df_by_player
                for m in outlier_dur_df_by_player[p]:
                    x_labs.append(m[2:])
                    df = outlier_dur_df_by_player[p][m].copy()
                    df['duration'] = df['duration']*self.time_ms/1000 #convert num frames to sec
                    fig.add_trace(go.Box(x=df.duration))
                fig.update_layout(title = f"Outlier Duration Distribution for all Limb Extensions for Player {p}", showlegend=False,width=1000,height=500,plot_bgcolor='#fff')
                fig.update_yaxes(ticktext=x_labs,tickmode='array',tickvals=np.arange(8))
                fig.update_xaxes(title='Duration [s]')
                fig.update_traces(marker=dict(color=self.fifa_hex))
                player_limb_ext_outlier_duration_boxplots[p] = fig
            return player_limb_ext_outlier_duration_boxplots
        elif type_outlier=='abnormally_high':
            player_high_boxplots = {}
            for p in self.players:
                fig = go.Figure()
                df = self.abnormally_high_outlier_dur_df_by_player[p].copy()
                df['duration'] = df['duration']*self.time_ms/1000 #convert num frames to sec
                fig.add_trace(go.Box(x=df.duration))
                fig.update_layout(title = f"Abnormally High Outlier Duration Distribution for Player {p}", showlegend=False,width=1000,height=500,plot_bgcolor='#fff')
                fig.update_yaxes(title="")
                fig.update_xaxes(title='Duration [s]')
                fig.update_traces(marker=dict(color=self.fifa_hex))
                player_high_boxplots[p] = fig
            return player_high_boxplots
        elif type_outlier=='abnormally_low':
            player_low_boxplots = {}
            for p in self.players:
                fig = go.Figure()
                df = self.abnormally_low_outlier_dur_df_by_player[p].copy()
                df['duration'] = df['duration']*self.time_ms/1000 #convert num frames to sec
                fig.add_trace(go.Box(x=df.duration))
                fig.update_layout(title = f"Abnormally Low Outlier Duration Distribution for Player {p}", showlegend=False,width=1000,height=500,plot_bgcolor='#fff')
                fig.update_yaxes(title="")
                fig.update_xaxes(title='Duration [s]')
                fig.update_traces(marker=dict(color=self.fifa_hex))
                player_low_boxplots[p] = fig
            return player_low_boxplots
    
    
    def get_frame_sequence(self,player,frame,frames_before,frames_after):
        """
        Given a player, frame ID, and number of adjacent frames to plot on each side, returns the desired frame sequence in chronological order
        """
        player_frames = self.frames_by_player[player].copy() #get frames for desired player, which are already sorted
        return player_frames[(player_frames.frame>=frame-frames_before) & (player_frames.frame<=frame+frames_after)] #find and return frames in desired range
    
    
    def plot_outlier_sequence(self,player,var,frame,frames_before,frames_after,outlier_type,show_median=True):
        """
        Given a player, column name, frame ID, and number of adjacent frames to plot on each side, finds the frame sequence
        and plots the column over time as a scatterplot. Option to show the median value.
        """
        seq = self.get_frame_sequence(player,frame,frames_before,frames_after) #get desired frame sequence
        seq['unix_time'] = (seq['unix_time'] - seq.iloc[[0]].unix_time.item())/1000 
        color_discrete_map = {0: self.fifa_hex, 1: "darkorange"}
        symbol_map = {0: 0, 1: 4}
        if outlier_type == "limb_extension":
            color = f"{var}_outlier"
            color_discrete_map = {0: self.fifa_hex, 1: "darkorange"}
            fig = px.scatter(seq,x='unix_time',y=var,color=color,
                         color_discrete_map=color_discrete_map,
                         hover_data=['frame'])
        else:
            fig = px.scatter(seq,x='unix_time',y=var,hover_data=['frame'])
            

        fig.update_layout(height=450,width=800,plot_bgcolor='#fff',title=f"{var} over Time for {player}",font=dict(family='Arial',size=14))
        fig.update_xaxes(title='Time [s]',showticklabels=True)
        if outlier_type == "limb_extension":
            fig.update_yaxes(title=f'{var} Length [m]')
        elif outlier_type == "abnormally_high" or outlier_type == "abnormally_low":
            fig.update_yaxes(title='Joint Height [m]') 
            if outlier_type=="abnormally_low":
                fig.add_hline(y=0,line_dash='dash',line_color="crimson")
        fig.update_traces(marker_coloraxis=None)
        if show_median:
            med_val = self.frames_by_player[player][var].median()
            fig.add_hline(y=med_val,line_dash='dash',line_color='crimson')
        return fig
    
    
    def limb_extension_outliers(self,include_ext_boxplots=True,include_durations=True):
        """
        Initializes:
        - self.limb_ext_outliers_by_player 
        - self.limb_ext_outlier_percent_df
        - self.limb_ext_outlier_percent_plot
        - self.player_limb_ext_boxplots
        """
        
        if self.verbose:
            print("Initializing limb_ext_outliers_by_player...")
        #map player IDs to dicts mapping metrics to outlier frames for that metric
        outliers_by_player = {p:{} for p in self.players}
        for p in self.players: 
            val = {}
            player_frames = self.frames_by_player[p]
            for m in self.distance_metrics:
                outliers_by_player[p][m] = player_frames[player_frames[m+"_outlier"]==1].copy()             
        self.limb_ext_outliers_by_player = outliers_by_player
        
        ##################################################################################################################################################################
       
        if self.verbose:
            print("Initializing limb_ext_outlier_percent_df...")
        #Aggregate across all players to find avg % outlier frames for each distance metric
        percent_outliers_by_metric = {m:[] for m in self.distance_metrics}
        percent_outliers_by_player = {} #track percent outliers for each player distinctly
        for i in self.limb_ext_outliers_by_player:
            tot_num_frames = self.frames_by_player[i].shape[0] #total number of frames player i has
            val = {}
            for m in self.distance_metrics:
                try:
                    percent_outliers_by_metric[m].append(self.limb_ext_outliers_by_player[i][m].shape[0]/tot_num_frames*100)
                    val[m] = self.limb_ext_outliers_by_player[i][m].shape[0]/tot_num_frames*100
                except: #player has no outlier frames for that distance_metric
                    percent_outliers_by_metric[m].append(0)
                    val[m] = 0
            percent_outliers_by_player[i] = val
        percent_outliers_by_metric = {m: [np.mean(np.array(percent_outliers_by_metric[m]))] for m in percent_outliers_by_metric}
        outlier_percent_df = pd.DataFrame.from_dict(percent_outliers_by_metric).transpose().round(2)
        outlier_percent_df.columns = ['percent_outliers']
        self.limb_ext_outlier_percent_df = outlier_percent_df
        
        ##################################################################################################################################################################
        #Generate plot
        if self.verbose:
            print("Initializing limb_ext_outlier_percent_plot...")
#         outlier_percent_df.drop(index=['l_Shoulder_Elbow_dist','l_Elbow_Wrist_dist','l_Hip_Knee_dist','l_Knee_Ankle_dist'], inplace=True) #drop a side since HEI controls L and R to be equal
        fig = go.Figure(data=[go.Bar(
            y=outlier_percent_df.index, x=outlier_percent_df['percent_outliers'],
            text=outlier_percent_df['percent_outliers'],
            textposition='auto',marker_color=self.fifa_hex,orientation='h'
        )])
        fig.update_layout(height=500,width=800,plot_bgcolor='#fff',font=dict(family='Arial',size=15),
                          title="Mean percentage of limb extension outliers across players (IQR Factor = 1.5)")
        fig.update_xaxes(range=[0, 10],title="",tickvals=[])
#         fig.update_yaxes(title="Limb",ticktext=['knee-ankle ','hip-knee ',
#                                    'elbow-wrist ','shoulder-elbow ' ][::-1], tickmode='array', tickvals=np.arange(0,8)
#         )
        fig.update_traces(width=0.7)
        self.limb_ext_outlier_percent_plot = fig
        
        ##################################################################################################################################################################
        if include_ext_boxplots:
            #Generate player box plots
            if self.verbose:
                print("Generating player box plots (self.player_limb_ext_boxplots)...")
            player_limb_plots = {} #maps player to his box and whisker diagram
            for i in self.frames_by_player:
                if i is None:
                    continue
                df = self.frames_by_player[i].copy()
                num_outliers = []
                for m in self.distance_metrics[::2]:
                    df[m] = df[m]*100
                    try:
                        num_outliers.append(self.limb_ext_outliers_by_player[i][m].shape[0]/df.shape[0]*100)
                    except:
                        num_outliers.append(0)
                fig = px.box(df,x=self.distance_metrics[::2],hover_data=['frame'],points='outliers')
                fig.update_layout(height=450,width=800,plot_bgcolor='#fff',title=f"Limb-extension distribution for #{i} ({df.shape[0]} frames)",font=dict(family='Arial',size=15),)
                fig.update_xaxes(title='Extension [cm]', tickmode='array')
                fig.update_yaxes(title='Limb',ticktext=[f'knee-ankle ({np.round(num_outliers[-1],1)}%)',f'hip-knee ({np.round(num_outliers[-2],1)}%)',
                                       f'elbow-wrist ({np.round(num_outliers[-3],1)}%)',f'shoulder-elbow ({np.round(num_outliers[-4],1)}%)'][::-1], tickmode='array', tickvals=np.arange(0,8))
                fig.update_traces(marker=dict(color=self.fifa_hex), marker_opacity=0.5)
                player_limb_plots[i] = fig
            self.player_limb_ext_boxplots = player_limb_plots
        
        ##################################################################################################################################################################
        #Temporal Analysis
        if include_durations:
            if self.verbose:
                print("Analyzing outlier sequence durations...")
            outlier_duration_by_player = {p:{} for p in self.players}
            for p in self.frames_by_player:
                if p==None:
                    continue
                player_frames = self.frames_by_player[p]
                cont_frames = player_frames[player_frames.rate.isin([self.time_ms-1,self.time_ms,self.time_ms+1])].copy() #filter for continuous frames
                for m in self.distance_metrics:
                    val = {}
                    for k, segment in cont_frames.groupby(cont_frames['frame'] - np.arange(cont_frames.shape[0])): #call the helper on each segment of continuous frames and continuously update val dictionary with sequences/durations
                        val.update(self.outlier_duration_slower(segment,m+"_outlier"))

                    outlier_duration_by_player[p][m] = val
            outlier_dur_df_by_player = {p:{} for p in self.players}
            for p in self.frames_by_player:
                if p==None:
                    continue
                for m in self.distance_metrics:
                    outlier_dur_df_by_player[p][m] = pd.DataFrame(outlier_duration_by_player[p][m],index=[0]).transpose().reset_index().rename({'level_0':'start_frame','level_1':'end_frame',0:'duration'},axis=1)
            self.limb_ext_outlier_dur_df_by_player = outlier_dur_df_by_player
            mean_outlier_durations_by_metric = {m:None for m in self.distance_metrics}
            vals_outlier_durations_by_metric = {m:None for m in self.distance_metrics}
            outlier_durations_by_metric_player = {m:{} for m in self.distance_metrics}
            for m in self.distance_metrics:
                metric_duration_vals = np.array([])
                player_means = []
                for p in self.frames_by_player: #for each player
                    if p==None:
                        continue
                    seq = outlier_dur_df_by_player[p][m]['duration'] #get the list of outlier sequence durations (list of integers)
                    outlier_durations_by_metric_player[m][p] = seq
                    if outlier_dur_df_by_player[p][m].shape[0]!=0: #handle cases when player has no outlier frames for a certain metric (we just omit)
                        player_means.append(np.median(seq)) #median       
                    metric_duration_vals = np.concatenate((metric_duration_vals, seq))
                vals_outlier_durations_by_metric[m] = metric_duration_vals
                mean_outlier_durations_by_metric[m] = np.mean(player_means)
            outlier_duration_df = pd.DataFrame(mean_outlier_durations_by_metric,index=[0]).transpose().reset_index().rename({0:'duration','index':'limb'},axis=1)
    #         outlier_duration_df['limb'] = outlier_duration_df['limb'].str[2:]
            outlier_duration_df['duration'] = round(outlier_duration_df['duration']*self.time_ms,0)
            outlier_duration_df.drop_duplicates(inplace=True) #combine L and R side because they are the same for HEI
            self.limb_ext_outlier_duration_df = outlier_duration_df
            ##################################################################################################################################################################
            if self.verbose:
                print("Generating outlier sequence duration plot (self.limb_ext_outlier_duration_plot)...")
            #Plot % outlier frames
            fig = go.Figure(data=[go.Bar(
                        y=outlier_duration_df['limb'], x=outlier_duration_df['duration'],
                        text=outlier_duration_df['duration'],
                        textposition='auto',marker_color=self.fifa_hex,orientation='h'
                    )])
            fig.update_layout(height=500,width=800,plot_bgcolor='#fff',font=dict(family='Arial',size=15),
                              title="Median duration of limb-extension outlier sequences averaged across players",
                             barmode='stack', yaxis={'categoryorder':'total ascending'})
            fig.update_xaxes(range=[0, 1500],title="Duration (ms)",tickvals=[])
            fig.update_yaxes(title="Limb",ticktext=['knee-ankle ','hip-knee ',
                                       'elbow-wrist ','shoulder-elbow ' ][::-1], tickmode='array', tickvals=np.arange(0,8)
            )
            fig.update_traces(width=0.7)
            self.limb_ext_outlier_duration_plot = fig
            ##################################################################################################################################################################
            if self.verbose:
                print("Generating player specific outlier duration boxplots... (self.player_limb_ext_outlier_duration_boxplots)")
            player_limb_ext_outlier_duration_boxplots = self.plot_outlier_duration_boxplots(type_outlier="limb_extension")
            self.player_limb_ext_outlier_duration_boxplots = player_limb_ext_outlier_duration_boxplots
            ##################################################################################################################################################################
            if self.verbose:
                print("Done!")
        
        
    def abnormally_high_outliers(self,include_durations=True):
        """Initializes self.abnormally_high_frames_dict, self.abnormally_high_percent_df and self.abnormally_high_frames_plot"""
        
        if self.verbose:
            print("Initializing self.abnormally_high_frames_dict and self.abnormally_high_percent_df...")
        perc_outliers, abnormally_high_df_by_region = {}, {}
        for limbs in ("upper","arms","legs"):
            if limbs == 'upper':
                ### upper
                joint_heights = ['lEar_z', 'lShoulder_z', 'neck_z', 
                                 'nose_z', 'rEye_z', 'rShoulder_z',]
                height_threshold = 2.5
            elif limbs == 'arms':
                joint_heights = ['lElbow_z', 'lWrist_z', 'rElbow_z', 'rWrist_z']
                height_threshold = 2.75 
            elif limbs == 'legs':
                joint_heights = ['lAnkle_z', 'lHip_z', 'lKnee_z', 'rAnkle_z', 'rHip_z', 'rKnee_z', ]
                height_threshold = 2
            df = self.hei_joints[joint_heights]
            weird = df.where(df>height_threshold).dropna(how='all')
            abnormal_df = self.hei_joints.loc[weird.index.values,][['frame','player_id','unix_time']+self.joint_coords]
#             print(f"{abnormal_df.shape[0]} abnormally high frames out of {self.total_num_frames} total frames for {limbs}")
            perc_outliers[limbs] = abnormal_df.shape[0]*100/hei_joints.shape[0]
            abnormally_high_df_by_region[limbs] = abnormal_df
        self.abnormally_high_frames_dict = abnormally_high_df_by_region
        self.abnormally_high_percent_df = pd.DataFrame(perc_outliers,index=[0]).transpose().reset_index().rename({'index':'region',0:'percent_outliers'},axis=1)
        
        if self.verbose:
            print("Initializing self.abnormally_high_frames_plot...")
        fig = go.Figure(data=[go.Bar(
            y=self.abnormally_high_percent_df['region'], x=self.abnormally_high_percent_df['percent_outliers'],
            text=self.abnormally_high_percent_df['percent_outliers'].round(7),
            textposition='auto',marker_color=self.fifa_hex,orientation='h'
        )])
        fig.update_layout(height=500,width=800,plot_bgcolor='#fff',font=dict(family='Arial',size=15),
                          title="Percent Outlier Frames by Body Region")
        fig.update_xaxes(range=[0, .2],title="Percent Outlier Frames")
        fig.update_yaxes(title="Body Region",ticktext=['Upper Torso and Head  ','Arms  ','Legs  '][::-1], tickmode='array', tickvals=np.arange(0,3)
        )
        fig.update_traces(width=0.7)  
        self.abnormally_high_percent_plot = fig
        ################################
        #Temporal Analysis
        if include_durations:
            if self.verbose:
                print("Analyzing abnormally high outlier durations...")
            abnormally_high_outlier_dur_df_by_player = {}
  
            for p in self.frames_by_player:
                if p==None:
                    continue
                player_frames = self.frames_by_player[p]
                cont_frames = player_frames[player_frames.rate.isin([self.time_ms-1,self.time_ms,self.time_ms+1])].copy() #filter for continuous frames
                val = {}
                for k, segment in cont_frames.groupby(cont_frames['frame'] - np.arange(cont_frames.shape[0])): #call the helper on each segment of continuous frames and continuously update val dictionary with sequences/durations
                    val.update(self.outlier_duration_slower(segment,"abnormally_high"))
                abnormally_high_outlier_dur_df_by_player[p] = pd.DataFrame(val,index=[0]).transpose().reset_index().rename({'level_0':'start_frame','level_1':'end_frame',0:'duration'},axis=1)
            self.abnormally_high_outlier_dur_df_by_player = abnormally_high_outlier_dur_df_by_player
             
            if self.verbose:
                print("Generating player specific outlier duration boxplots... (self.player_abnormally_high_duration_boxplots)")
            player_abnormally_high_duration_boxplots = self.plot_outlier_duration_boxplots(type_outlier="abnormally_high")
            self.player_abnormally_high_duration_boxplots = player_abnormally_high_duration_boxplots
       
        if self.verbose:
            print("Done!")
        
    def abnormally_low_outliers(self,include_durations=True):
        """Initializes self.abnormally_low_frames_dict, self.abnormally_low_frames_df and self.abnormally_low_frames_plot"""
        
        if self.verbose:
            print("Initializing self.abnormally_low_frames_dict and self.abnormally_low_frames_df...")
        joint_heights = ['lAnkle_z', 'lEar_z', 'lElbow_z', 'lHip_z', 'lKnee_z', 'lShoulder_z', 'lWrist_z', 'neck_z', 
                         'nose_z', 'rAnkle_z', 'rElbow_z', 'rEye_z', 'rHip_z', 'rKnee_z', 'rShoulder_z', 'rWrist_z']
        below_ground_frames = self.hei_joints.loc[(self.hei_joints[joint_heights] < 0).any(axis=1)].copy()
        abnormally_low_frames_dict = {}
        below_ground_percent_dict = {}
        for j in joint_heights:
            d = below_ground_frames[below_ground_frames[j]<0]
            abnormally_low_frames_dict[j] = d
            below_ground_percent_dict[j] = d.shape[0]*100/self.total_num_frames
        self.abnormally_low_frames_dict = abnormally_low_frames_dict
        self.abnormally_low_percent_df = pd.DataFrame(below_ground_percent_dict,index=[0]).transpose().reset_index().rename({'index':'joint',0:'percent_outliers'},axis=1)
        if self.verbose:
            print("Initializing self.abnormally_low_frames_plot...")
        fig = go.Figure(data=[go.Bar(
            y=self.abnormally_low_percent_df['joint'][::1], x=self.abnormally_low_percent_df['percent_outliers'],
            text=self.abnormally_low_percent_df['percent_outliers'].round(7),
            textposition='auto',marker_color=self.fifa_hex,orientation='h'
        )])
                      
        fig.update_layout(height=500,width=800,plot_bgcolor='#fff',font=dict(family='Arial',size=15),
                          title="Mean Percentage of Frames Below the Ground")
        fig.update_xaxes(range=[0, 1],title="Percent Outlier Frames [%]",showticklabels=False)
        fig.update_traces(width=0.7)
        self.abnormally_low_percent_plot = fig
        ################################
        #Temporal Analysis
        if include_durations:
            if self.verbose:
                print("Analyzing abnormally low outlier durations...")
            abnormally_low_outlier_dur_df_by_player = {}
  
            for p in self.frames_by_player:
                if p==None:
                    continue
                player_frames = self.frames_by_player[p]
                cont_frames = player_frames[player_frames.rate.isin([self.time_ms-1,self.time_ms,self.time_ms+1])].copy() #filter for continuous frames, allowing for .001 s error
                val = {}
                for k, segment in cont_frames.groupby(cont_frames['frame'] - np.arange(cont_frames.shape[0])): #call the helper on each segment of continuous frames and continuously update val dictionary with sequences/durations
                    val.update(self.outlier_duration_slower(segment,"abnormally_low"))
                abnormally_low_outlier_dur_df_by_player[p] = pd.DataFrame(val,index=[0]).transpose().reset_index().rename({'level_0':'start_frame','level_1':'end_frame',0:'duration'},axis=1)
            self.abnormally_low_outlier_dur_df_by_player = abnormally_low_outlier_dur_df_by_player
            
            if self.verbose:
                print("Generating player specific outlier duration boxplots... (self.player_abnormally_low_duration_boxplots)")
            player_abnormally_low_duration_boxplots = self.plot_outlier_duration_boxplots(type_outlier="abnormally_low")
            self.player_abnormally_low_duration_boxplots = player_abnormally_low_duration_boxplots
            
        if self.verbose:
            print("Done!")