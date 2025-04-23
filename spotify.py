import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from scipy.stats import chi2_contingency
import plotly.express as px
import plotly.graph_objects as go

#read csv into dataframe
df = pd.read_csv('spotify_history.csv')

#convert ts(timestamp) to datetime for easy analysis
df['ts'] = pd.to_datetime(df['ts'])

#drop rows with null values
df = df.dropna()

#convert ms_played column to seconds_played
df['seconds_played'] = df['ms_played'] / 1000
df['seconds_played'] = df['seconds_played'].round(2)

#save clean dataset for analysis with another tool
#df.to_csv('spotify_history_cleaned.csv', index=False)

#derive new feature for further analysis; now the datset has 16 columns from the original 11
df['hour'] = df['ts'].dt.hour
df['weekday'] = df['ts'].dt.day_name()
df['month'] = df['ts'].dt.month_name()
df['date'] = df['ts'].dt.date

#Descriptive Statistics
#Histogram of seconds_played
plt.figure(figsize=(10, 6))
sns.histplot(df['seconds_played'], bins=50, kde=True)
plt.title('Distribution of Seconds Played')
plt.xlabel('Seconds Played')
plt.ylabel('Frequecy')
#plt.savefig('Seconds_played.png')
plt.close()

#Categorical: Platform, reason_start, reason_end, shiffle, skipped
for col in ['platform', 'reason_start', 'reason_end', 'shuffle', 'skipped']:
    #show the value counts for each of the columns noted above
    #print(f"\n{col} value counts: \n", df[col].value_counts())
    #bar_plot
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col)
    plt.title(f"Frequency of {col}")
    plt.xticks(rotation=45)
    #plt.savefig(f"Bar_plot_of_{col}.png")

#Unique Tracks and artists
#print('Unique tracks:', df['track_name'].nunique())
#print('Unique artists:', df['artist_name'].nunique())
#for item in ['track_name', 'artist_name']:
    #print(f"\n{item} value counts: \n", df[item].value_counts())

#Listening Frequency by date, hour and weekday
daily_counts = df.groupby('date').size()
plt.figure(figsize=(12, 6))
daily_counts.plot(kind='line')
plt.title('Listening Frequency by Date')
plt.xlabel('Date')
plt.ylabel('Number of Streams')
#plt.savefig('Listening_Frequency_by_Date.png')
plt.close()

hourly_counts = df.groupby('hour').size()
plt.figure(figsize=(10, 6))
sns.barplot(x=hourly_counts.index, y=hourly_counts.values)
plt.title("Listening Frequecy by Hour")
plt.xlabel('Hour of Day')
plt.ylabel('Number of Streams')
#plt.savefig('Listening_Frequency_by_Hour.png')
plt.close()

weekday_counts = df.groupby('weekday').size().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.figure(figsize=(10, 6))
sns.barplot(x=weekday_counts.index, y=weekday_counts.values)
plt.title('Listening Frequecy by Weekday')
plt.xlabel('Weekday')
plt.ylabel('Number of Streams')
plt.xticks(rotation=45)
#plt.savefig('Listening_Frequency_by_Weekday.png')
plt.close()

pivot_table = df.pivot_table(index='weekday', columns='hour', values='artist_name', aggfunc='nunique', fill_value=0).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, cmap='Blues', annot=True, fmt='d')
plt.title('Streams by Hour and Weekday')
#plt.savefig('Streams_by_Hour_and_Weekday.png')
plt.close()

#To investigate music preference based on the track and artists most listened to
#Top 10 Tracks
top_tracks = df['track_name'].value_counts().head(10)
plt.figure(figsize=(8, 4))
sns.barplot(x=top_tracks.values, y=top_tracks.index)
plt.title('Top 10 Most Played Tracks')
plt.xlabel('Number of Streams')
plt.ylabel('Track Name')
#plt.savefig('Top_10_Tracks.png')
plt.close()

#Top 10 artists
top_artists = df['artist_name'].value_counts().head(10)
plt.figure(figsize=(8, 4))
sns.barplot(x=top_artists.values, y=top_artists.index)
plt.title('Top 10 Most Played Artists')
plt.xlabel('Number of Streams')
plt.ylabel('Artist Name')
#plt.savefig('Top_10_Artists.png')
plt.close()

#wordcloud for artists
artist_text = ' '.join(df['artist_name'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='skyblue').generate(artist_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Artists')
#plt.savefig('Word_Cloud_for_Artists.png')
plt.close()

#Repeat listening behaviour
repeat_tracks = df.groupby(['track_name', 'artist_name']).size().sort_values(ascending=False).head(10)
#print('Top 10 repeated tracks:\n', repeat_tracks)

#Behavioural Analysis
#Skip Rate
skip_rate = df['skipped'].value_counts(normalize=True)
#print('Skip Rate:\n', skip_rate)

#boxplot of seconds_played by skipped
plt.figure(figsize=(10, 6))
sns.boxplot(x='skipped', y='seconds_played', data=df)
plt.title('Seconds Played by Skipped Status')
plt.xlabel('Skipped')
plt.ylabel('Seconds Played')
#plt.savefig('Seconds_Played_by_Skipped.png')
plt.close()

#reasons for starting/ending
for col in ['reason_start', 'reason_end']:
    plt.figure(figsize=(12, 8))
    sns.countplot(data=df, x=col, hue='skipped')
    plt.title(f"{col} by Skipped Status")
    plt.xticks(rotation=45)
    #plt.savefig(f"{col}_by_Skipped.png")
    plt.close()

#Shuffle vs Non Shuffle
plt.figure(figsize=(10, 6))
sns.boxplot(x='shuffle', y='seconds_played', hue='skipped', data=df)
plt.title('Seconds Played by Shuffle and Skipped Status')
plt.xlabel('Shuffle')
plt.ylabel('Seconds Played')
#plt.savefig('Seconds_Played_by_Shuffle_and_Skipped.png')
plt.close()

#Platform Analysis
platform_counts = df['platform'].value_counts()
plt.figure(figsize=(12, 6))
platform_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('Stream Distribution by Platform')
plt.ylabel(' ')
#plt.savefig('Stream_Distribution_by_platform.png')
plt.close()

#Skip rate by Platform
skip_rate_by_platform = df.groupby('platform')['skipped'].value_counts(normalize=True).unstack()
plt.figure(figsize=(10, 6))
skip_rate_by_platform.plot(kind='bar', stacked=True)
plt.title('Skip Rate by Platform')
plt.xlabel('Platform')
plt.ylabel('Proportion')
#plt.savefig('skip_rate_by_platform.png')
plt.close()

#Correlation and Realtionship Analysis
contingency_table = pd.crosstab(df['skipped'], df['platform'])
chi2, p, dof, ex = chi2_contingency(contingency_table)
#print(f"Chi-Square test (skipped vs. platform): p-value = {p}")

#SKip rate by the hour
skip_by_hour = df.groupby('hour')['skipped'].mean()
plt.figure(figsize=(10, 6))
sns.lineplot(x=skip_by_hour.index, y=skip_by_hour.values)
plt.title('Skip Rate by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Skip Rate')
plt.close()

#Session Analysis
df = df.sort_values('ts')
df['time_diff'] = df['ts'].diff().dt.total_seconds() / 60 #sorts the difference in minutes
df['new_session'] = (df['time_diff'] > 30) | df['time_diff'].isna()
df['session_id'] = df['new_session'].cumsum()

#session summary
session_summary = df.groupby('session_id').agg({
'ts': ['min', 'max'],
'track_name': 'count',
'seconds_played': 'sum',
'skipped': 'mean'
}).reset_index()
session_summary.columns = ['session_id', 'start_time', 'end_time', 'track_count', 'total_seconds', 'skip_rate']
session_summary['duration_min'] = (session_summary['end_time'] - session_summary['start_time']).dt.total_seconds() / 60

#Histogram of session durations
plt.figure(figsize=(10, 6))
sns.histplot(session_summary['duration_min'], bins=50)
plt.title('Distribution of Session Durations (Minutes)')
plt.xlabel('Session Duration (Minutes)')
plt.ylabel('Frequency')
#plt.savefig('Distribution_of_Session_Durations.png')
plt.close()

#Interactive Dashboards with Plotly
fig = px.scatter(df, x='hour', y='seconds_played', color='skipped', size='seconds_played', hover_data=['track_name', 'artist_name'], title='Listening Behaviour by Hour')
fig.write_html('Listening_Behaviour_by_Hour.html')
del fig

#Sankey diagram: Platform -> reason_end -> Shuffle
sankey_data = df.groupby(['platform', 'reason_end', 'shuffle']).size().reset_index(name='count')
labels = list(pd.concat([sankey_data['platform'], sankey_data['reason_end'], sankey_data['shuffle']]).unique())
source = pd.Categorical(sankey_data['platform'], categories=labels).codes
target = pd.Categorical(sankey_data['reason_end'], categories=labels).codes
target2 = pd.Categorical(sankey_data['shuffle'], categories=labels).codes

fig = go.Figure(data=[go.Sankey(
node=dict(label=labels),
link=dict(
source=np.concatenate([source, target]),
target=np.concatenate([target, target2]),
value=np.concatenate([sankey_data['count'], sankey_data['count']])
)
)])
fig.update_layout(title_text='Flow: Platform -> Reason End -> Shuffle', font_size=10)
fig.write_html('Sankey Diagram.html')
