import json
from datetime import date
from urllib.request import urlopen
import time
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import hydralit_components as hc
from streamlit_lottie import st_lottie
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Chocolate Case Study",
    page_icon= "https://cdn-icons-png.flaticon.com/512/2824/2824980.png",
    layout='wide'
)


# Load css style file from local disk
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
# Load css style from url
def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">',unsafe_allow_html = True)

# Display lottie animations
def load_lottieurl(url):

    # get the url
    r = requests.get(url)
    # if error 200 raised return Nothing
    if r.status_code !=200:
        return None
    return r.json()

# Extract Lottie Animations

lottie_eda = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_ic37y4kv.json")
lottie_ml = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_q5qeoo3q.json")

# Load css library
remote_css("https://unpkg.com/tachyons@4.12.0/css/tachyons.min.css")
# Load css style
local_css('style.css')

menu_data = [
    {'label': "Overview", 'icon': 'bi bi-bar-chart-line'},
    {'label':"EDA", 'icon': "bi bi-graph-up-arrow"},
    {'label': 'Tableau', 'icon': 'bi bi-clipboard-data'},
    {'label':"Application", 'icon':'fa fa-brain'}]

over_theme = {'txc_inactive': 'white','menu_background':'rgb(0,0,128)', 'option_active':'white'}

menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    hide_streamlit_markers=True,
    sticky_nav=True, #at the top or not
    sticky_mode='sticky', #jumpy or not-jumpy, but sticky or pinned
)

# Read the CSV file into a DataFrame
df = pd.read_excel('data.xlsx')



# Retreive detailed report of the Exploratory Data Analysis
def profile(df):
    pr = ProfileReport(df, explorative=True)
    tbl = st_profile_report(pr)
    return  tbl



# EDA page

if menu_id == "EDA":

    # Drop unnecessary columns
    df1 = df
    

    # 2 Column Layouts of Same Size
    col4,col5 = st.columns([1,1])

    # First Column - Shows Description of EDA
    with col4:
        st.markdown("""
        <h3 class="f2 f1-m f-headline-l measure-narrow lh-title mv0">
         Know Your Data
         </h3>
         <p class="f5 f4-ns lh-copy measure mb4" style="text-align: justify;font-family: Sans Serif">
          Before implementing your machine learning model, it is important at the initial stage to explore your data.
          It is a good practice to understand the data first and try gather as many insights from it. EDA is all about
          making sense of data in hand, before diving deep into it.
         </p>
            """,unsafe_allow_html = True)
        global eda_button

        # Customize Button
        button = st.markdown("""
        <style>
        div.stButton > button{
        background-color: #0178e4;
        color:#ffffff;
        box-shadow: #094c66 4px 4px 0px;
        border-radius:8px 8px 8px 8px;
        transition : transform 200ms,
        box-shadow 200ms;
        }

         div.stButton > button:focus{
        background-color: #0178e4;
        color:#ffffff;
        box-shadow: #094c66 4px 4px 0px;
        border-radius:8px 8px 8px 8px;
        transition : transform 200ms,
        box-shadow 200ms;
        }


        div.stButton > button:active {

                transform : translateY(4px) translateX(4px);
                box-shadow : #0178e4 0px 0px 0px;

            }
        </style>""", unsafe_allow_html=True)
        # Display Button
        eda_button= st.button("Explore Your Data")


    # Second Column - Display EDA Animation
    with col5:
        st_lottie(lottie_eda, key = "eda",height = 300, width = 800)

    # User Clicks on Button, then profile report of the uplaoded or existing dataframe will be displayed
    if eda_button:
        profile(df1)





#edit footer
page_style= """
    <style>
    footer{
        visibility: visible;
        }
    footer:after{
        content: 'Developed by Ali Maatouk for Publicis Case Study';
        display:block;
        position:relative;
        color:#1e54e4;
    }
    </style>"""

st.markdown(page_style, unsafe_allow_html=True)

