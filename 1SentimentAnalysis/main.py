import streamlit as st
from textblob import TextBlob
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import cleantext
import matplotlib.pyplot as plot

df = pd.read_csv('Reviews.csv')
review_text = df['Text']
analyser = SentimentIntensityAnalyzer()

sentiment_scores = []
blob_subj = []
for review in review_text:
    sentiment_scores.append(analyser.polarity_scores(review)['compound'])
    blob = TextBlob(review)
    blob_subj.append(blob.subjectivity)

sentiment_classes = []
for sentiment_score in sentiment_scores:
    if sentiment_score > 0.8 :
        sentiment_classes.append('Highly Positive')
    elif sentiment_score > 0.4 : 
        sentiment_classes.append('Positive')
    elif -0.4 <= sentiment_score <= 0.4 : 
        sentiment_classes.append('Neutral')
    elif sentiment_score < -0.4 : 
        sentiment_classes.append('Negative')
    else : 
        sentiment_classes.append('Highly Negative')

st.title('Sentiment Analysis on Customer Feedback')

user_input = st.text_area("Enter the FeedBack")
blob = TextBlob(user_input)

user_sentiment_score = analyser.polarity_scores(user_input)['compound']
if user_sentiment_score > 0.8 :
    user_sentiment_class = 'Highly Positive'
elif user_sentiment_score > 0.4 : 
    user_sentiment_class = 'Positive'
elif -0.4 <= user_sentiment_score <= 0.4 : 
    user_sentiment_class = 'Neutral'
elif user_sentiment_score < -0.4 : 
    user_sentiment_class = 'Negative'
else : 
    user_sentiment_class = 'Highly Negative'

st.write("** Vader Sentiment Class : **",user_sentiment_class,"** Vader Sentiment Scores: **", user_sentiment_score)
st.write("** TextBlob Polarity **",blob.sentiment.polarity, "** TextBlob Subjectivity **", blob.sentiment.subjectivity)


pre = st.text_input('Clean Text: ')
if pre:
    st.write(cleantext.clean(pre,clean_all= False, extra_spaces= True, stopwords= True , lowercase=True , numbers= True, punct= True))
else:
    st.write("No Text is been provided from the user for cleaning.")



st.subheader("The Graphical Representation of the Data")
plot.figure(figsize=(10,5))

sentiment_score_by_class = {k:[]for k in set(sentiment_classes)}
for sentiment_score, sentiment_class in zip(sentiment_scores,sentiment_classes):
    sentiment_score_by_class[sentiment_class].append(sentiment_score)

for sentiment_class,score in sentiment_score_by_class.items():
    plot.hist(score,label=sentiment_class, alpha= 0.5)

plot.xlabel("Sentiment Score")
plot.ylabel("Count")
plot.title("Score Distribution")
plot.legend()
st.pyplot(plot)

df["Sentiment Class"] = sentiment_classes
df["Sentiment Score"] = sentiment_scores
df["Subjectivity"] = blob_subj

new_df = df[["Score","Text","Sentiment Score","Sentiment Class","Subjectivity"]]
st.subheader("Input Dataframe")
st.dataframe(new_df.head(50),use_container_width=True)