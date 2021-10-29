
import streamlit as st
import pandas as pd
import pickle
import datetime
import yfinance as yf
import numpy as np

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

header = st.container()
dataset = st.container()
features = st.container()
model_training=st.container()

with header:
    st.title("Welcome Sentiment Analysis for Stock Trend!")
    st.text("With the changes in news title, what will be the trend of stock?")
    

st.balloons()

with features: 
    st.header("Features")
    st.markdown("* **News Article Data Collected:** In order to train data to predict the trend of stock price with results from sentiment analysis, news article are scraped from FinViz.")
    st.markdown("* **Stock Price Display:** This app allows tou to key in the ticker and show the trend of the stock in line graph.")
    st.markdown("* **Prediction Model:** Logistic Regression is used to to build a model to predict whether the stock will increase or drop based on the scores obtained from Sentiment Analysis.")
    


with dataset:
    st.header("Data to train Prediction Model")
    FB_df=pd.read_csv("C:/Users/MBizM LSS/Downloads/Data Science/Project/FB_dF.csv")
    st.subheader("The prediction model are trained with stock market news titles for Facebook and Bank of America.")
    st. write(FB_df)

        

with model_training:
    st.header("Determine the trend of stock.")
    st.text ("By putting in the news title, it will calculate the polarity and predict the trend of stock.")
    
    text= 'Place your text here.'
    
    sel_col, disp_col = st.columns(2)
    art_input = sel_col.text_input("Put in the article title.",text)
    
    submit = st.button('Predict')
    
    df = pd.DataFrame(columns=['art_input'])
    df.loc[0] = art_input
            
    vader= SentimentIntensityAnalyzer()

    f = lambda art_input: vader.polarity_scores(art_input)['compound']
    p = lambda art_input: vader.polarity_scores(art_input)['pos']
    n = lambda art_input: vader.polarity_scores(art_input)['neg']
    m = lambda art_input: vader.polarity_scores(art_input)['neu']
    

    df['compound']=df['art_input'].apply(f)
    df['positive']=df['art_input'].apply(p)
    df['negative']=df['art_input'].apply(n)
    df['neutral']=df['art_input'].apply(m)
    
    compound=df.compound
    positive=df.positive
    negative=df.negative
    neutral=df.neutral
    
    pickle_in = open('logisticRegr2.pkl','rb')
    classifier=pickle.load(pickle_in)

    prediction = classifier.predict(df[['compound','positive','negative','neutral']])
    if prediction == 0:
        st.write("The stock might drop!")
        
    else:
        st.write("Oh yes, the stock might increase!")
    