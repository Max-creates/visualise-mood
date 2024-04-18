import glob
from pathlib import Path
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px


filepaths = glob.glob("diary/*.txt")

dates = [Path(filepath).stem for filepath in filepaths]
    
pos = []
neg = []
analyzer = SentimentIntensityAnalyzer()

for filepath in filepaths:
    with open(filepath, "r") as file:
        text = file.read()
    scores = analyzer.polarity_scores(text)
    pos.append(scores["pos"])
    neg.append(scores["neg"])

st.header("Diary Tone")
st.subheader("Positivity")

figure_pos = px.line(x=dates, y=pos, labels={"x": "Date", "y": "Positivity"})
st.plotly_chart(figure_pos)

st.subheader("Negativity")

figure_neg = px.line(x=dates, y=neg, labels={"x": "Date", "y": "Negativity"})
st.plotly_chart(figure_neg)
