import streamlit as st
import joblib

'''
# New York Taxi Fare Prediction
'''

st.markdown('''
The prediction of taxi fare in New York City. The dataset is based on [Kaggle New York Taxi Fare Challenge](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/overview)
''')

'''
## User Input

1. Date Time
'''
col1, col2 = st.columns(2)
d = col1.date_input("Input Date")
t = col2.time_input("Input Time")

col3, col4 = st.columns(2)
pickup_lon = col3.number_input("Input Pickup Longitude")
pickup_lat = col4.number_input("Input Pickup Latitude")

col5, col6 = st.columns(2)
pickup_lon = col5.number_input("Input Drop off Longitude")
pickup_lat = col6.number_input("Input Drop off Latitude")

model = joblib.load('TaxiFareModel/model.joblib')
url = 'https://taxifare.lewagon.ai/predict'

if url == 'https://taxifare.lewagon.ai/predict':

    st.markdown('Maybe you want to use your own API for the prediction, not the one provided by Le Wagon...')

'''

2. Let's build a dictionary containing the parameters for our API...

3. Let's call our API using the `requests` package...

4. Let's retrieve the prediction from the **JSON** returned by the API...

## Finally, we can display the prediction to the user
'''
