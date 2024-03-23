import pandas as pd
import streamlit as st
import numpy as np
import os
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='WhatsApp', page_icon=':speech_balloon:', layout='wide')

st.title(':speech_balloon: WhatsApp Group Chat Analysis Dashboard')
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

fl = st.file_uploader(":file_folder: upload a file", type=(["csv", "txt", "xlsx", "xls"]))

if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename)
else:
    os.chdir(r"C:\Users\PC\.streamlit")
    df = pd.read_csv("WhatsApp Chat.txt", sep = "delimiter",skip_blank_lines = True, header = None, engine='python')
    print(df)

# Adding column headers
df.columns = ['raw_data']

# Split the column by the second colon
df[['Timestamp', 'Sender']] = df['raw_data'].str.split(' - ', n=1, expand=True)
df[['Sender', 'Text']] = df['Sender'].str.split(': ', n=1, expand=True)


df.drop(columns = ['raw_data'], inplace=True)
df['Time'] = df['Timestamp'].str.split(',').str[1]
df['Date'] = df['Timestamp'].str.split(',').str[0]
df.drop(columns=['Timestamp'], inplace=True)

df.dropna(inplace=True)

df = df[df['Text'] != '<Media omitted>']

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract the day names and month names as separate columns
df['Day Name'] = df['Date'].dt.day_name()
df['Month Name'] = df['Date'].dt.month_name()

# Print the updated DataFrame
print(df[['Date', 'Day Name', 'Month Name']])

df.dropna(inplace=True)

#create sidebar filter for contacts
st.sidebar.header("Filter Pane")
contact = st.sidebar.multiselect("Select Contact", df['Sender'].unique())
if not contact:
    df2 = df.copy()
else:
    df2 = df[df['Sender'].isin(contact)]

#create filter for date
day = st.sidebar.multiselect("Select Day", df2['Day Name'].unique())
if not day:
    df3 = df2.copy()
else:
    df3 =df2[df2['Day Name'].isin(day)]

#create filter for month
month = st.sidebar.multiselect("select Month", df3["Month Name"].unique())


## filtering the dataset
if not contact and not day and not month:
    filtered_df = df
elif not month and not day:
    filtered_df = df[df["Sender"].isin(contact)]
elif not contact and not month:
    filtered_df = df[df["Day Name"].isin(day)]
elif day and month:
    filtered_df = df3[df['Day Name'].isin(day) & df3["Month Name"].isin(month)]
elif contact and month:
    filtered_df = df3[df['Day Name'].isin(contact) & df3["Month Name"].isin(month)]
elif contact and day:
    filtered_df = df3[df['Day Name'].isin(contact) & df3["Month Name"].isin(day)]
elif month:
    filtered_df = df3[df3["Month Name"].isin(month)]
else:
    filtered_df = df3[df3["Sender"].isin(contact) & df3["Day Name"].isin(day) & df3['Month Name'].isin(month)]


# Count the occurrences of each sender
sender_counts = filtered_df['Sender'].value_counts()

# Print the top 10 active senders
top_senders = sender_counts.head(10)
print(top_senders)

top_senders_df = pd.DataFrame(top_senders)
top_senders_df.reset_index(inplace=True)

cl1, chart2 = st.columns((2))
with cl1:
    st.subheader("Top Participants")
    fig = px.bar(top_senders_df, x = "Sender", y = "count", color = 'count', color_continuous_scale='greens', text = ['{:,.0f}'.format(x) for x in top_senders_df["count"]],template = "seaborn")
    st.plotly_chart(fig,use_container_width=True, height = 200)


ma1 = filtered_df['Month Name'].value_counts()
ma = pd.DataFrame(ma1)
ma.reset_index(inplace=True)


# Clean the 'Time' column by removing leading/trailing whitespace
filtered_df['Time'] = filtered_df['Time'].str.strip()

# Convert the 'Time' column to datetime format with specified format
filtered_df['Time'] = pd.to_datetime(filtered_df['Time'], format='%I:%M %p')

# Extract the hours and create a new column
filtered_df['Hour'] = filtered_df['Time'].dt.hour

clock24 = filtered_df['Hour'].value_counts()
clock = pd.DataFrame(clock24)
clock.reset_index(inplace=True)


fig2 = px.bar(clock, x = "Hour", y="count", color = 'count', color_continuous_scale='greens', text = ['{:,.0f}'.format(x) for x in clock["count"]],template = "seaborn")
fig2.update_layout(xaxis=dict(tickmode='array', tickvals=clock['Hour'], ticktext=clock['Hour']))
st.plotly_chart(fig2,use_container_width=True)

count_df1 = filtered_df['Day Name'].value_counts()
count_df = pd.DataFrame(count_df1)
count_df.reset_index(inplace=True)

color_map = {'Monday': 'green', 'Tuesday': 'lightgreen', 'Wednesday': 'limegreen', 'Thursday': 'darkgreen', 'Friday': 'seagreen', 'Saturday': 'mediumseagreen', 'Sunday': 'forestgreen'}

chart1, cl2 = st.columns((2))
with chart1:
    st.subheader("DAily Group Activity")
    fig3 = px.treemap(count_df, path = ["Day Name"], values = "count",hover_data = ["count"],
                    color = "Day Name",  color_discrete_map=color_map)
    fig3.update_layout(width = 800, height = 650)
    st.plotly_chart(fig3, use_container_width=True)

with cl2:
    st.subheader("Monthly Group Activity")
    fig3 = px.pie(ma, values = "count", names = "Month Name", hole = 0.5)
    fig3.update_traces(text = ma["Month Name"], textposition = "outside")
    fig3.update_traces(marker=dict(colors=['green']))
    st.plotly_chart(fig3,use_container_width=True)



import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Assuming you have a DataFrame called 'data' with a 'text' column

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis on each text
sentiment_scores = filtered_df['Text'].apply(lambda x: sid.polarity_scores(x))

# Extract the sentiment scores
filtered_df['sentiment_score'] = sentiment_scores.apply(lambda x: x['compound'])

# Categorize sentiment based on the compound score
filtered_df['sentiment'] = filtered_df['sentiment_score'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

# Display the updated DataFrame
print(filtered_df[['Text', 'sentiment_score', 'sentiment']])

sent1 = filtered_df['sentiment'].value_counts()
sent = pd.DataFrame(sent1)
sent.reset_index(inplace=True)

with chart2:
    st.subheader("Sentiment Analysis")
    fig = px.bar(sent, x = "sentiment", y="count", color = 'count', color_continuous_scale='greens',template = "seaborn")
    st.plotly_chart(fig,use_container_width=True)


with st.expander("View Data"):
    st.write(filtered_df.iloc[:500,1:20:2].style.background_gradient(cmap="Oranges"))




import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Combine all the text into a single string
all_text = ' '.join(df['Text'])

# Tokenize the text by splitting it into words
words = all_text.split()

# Count the occurrence of each word
word_counts = Counter(words)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='cool').generate_from_frequencies(word_counts)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud')
plt.tight_layout()

# Display the word cloud using Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()