import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json


st.markdown("""
<style>
.dataf {
    font-size:50px !important;
    text-align: center;
    color: black
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="dataf">DATAFRAME</p>', unsafe_allow_html=True)



def add_bg_from_url():

    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url("https://media.istockphoto.com/id/1278709873/photo/brown-recycled-paper-crumpled-texture-background-cream-old-vintage-page-or-grunge-vignette.jpg?b=1&s=170667a&w=0&k=20&c=NqKmm_gkRwJAqpTbiiqv3TwfWjq9ymOwUDwfG2ck9no=");
        background-attachment: fixed;
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

add_bg_from_url() 
df=pd.read_csv('news_articles.csv')
df=df[1:15]
st.dataframe(data=df, width=None, height=None, use_container_width=False)
st.write("**[Dataset](https://www.kaggle.com/datasets/ruchi798/source-based-news-classification)**")

