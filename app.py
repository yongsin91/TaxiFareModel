import streamlit as st
import joblib
import requests
import datetime
import pytz
import pandas as pd
import os

st.set_page_config(page_title='New York Taxi Fare Prediction',
                   page_icon =None, initial_sidebar_state = 'auto')

'''
# New York Taxi Fare Prediction
'''

def coordinates(input):
    """Checks whether the address entered is valid and extracts the
    coordinates through an API
    """

    api = 'https://nominatim.openstreetmap.org/search'
    response = requests.get(api, params={'q':input, 'format':'json'}).json()

    nyc_add = []
    for address in response:
        address_split = address['display_name'].split(', ')
        if "United States" in address_split and "New York" in address_split:
            nyc_add.append(address)

    for ny_add in nyc_add:
        if ny_add['display_name'].lower() == input:
            lon = float(ny_add['lon'])
            lat = float(ny_add['lat'])
            return lon, lat

    if len(nyc_add) == 1:
        st.markdown("*We found one possible address. Please confirm or refine\
                your search. When confirmed, please copy the address into the query box:*")
        st.markdown(f"""`{nyc_add[0]['display_name']}`""")
    elif len(nyc_add) > 1:
        st.write("*We found multiple possible addresses. Please refine your search if not listed or\
                please copy the intended address below into the query box:*")
        for index, address in enumerate(nyc_add[:3]):
            st.markdown(f"""`{index + 1})  {address['display_name']}`""")
    elif len(nyc_add) == 0 and input!="":
        st.write("*No addresses were found. Please refine your search.*")

    return -1000, -1000

st.markdown('''
This server is for the demonstration of the prediction of taxi fare in New York City. This model is currently trained on \
the dataset of [Kaggle New York Taxi Fare Challenge](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/overview). \
Currently this taxi fare prediction model is only possible for areas within New York City of United States.
''')

'''
## User Input
'''
col1, col2, col3 = st.columns(3)
d = col1.date_input("Input Date")
t = col2.time_input("Input Time")
a = col3.number_input("Input Number of Passengers", min_value=1, max_value=8)

date_and_time = datetime.datetime.combine(d, t)
# Obtaining coordinates for pickup and dropoff address
p_add = st.text_input('Input Pickup address').strip().lower()
p_lon, p_lat = coordinates(p_add)
d_add = st.text_input('Input dropoff address').strip().lower()
d_lon, d_lat = coordinates(d_add)

predict = st.button("Predict Taxi Fare")

if predict:

    if p_lon+p_lat+d_lon+d_lat<-900:
        st.markdown("### Error! Please enter a valid address.")
    elif p_add == d_add:
        st.markdown("### $0 Fare")
    elif p_add=="" and d_add=="":
        st.markdown("Invalid Input. Please enter both input and before clicking predict again.")
    else:
        params=dict(pickup_datetime=date_and_time,
                pickup_longitude=p_lon,
                pickup_latitude=p_lat,
                dropoff_longitude=d_lon,
                dropoff_latitude=d_lat,
                passenger_count=int(a))

        model = joblib.load(os.path.dirname(__file__) + "TaxiFareModel/model.joblib")

        # Modifying the format of params['pickup_datetime'] as per requirements in the model
        # localize the user provided datetime with the NYC timezone
        eastern = pytz.timezone("US/Eastern")
        localized_pickup_datetime = eastern.localize(params['pickup_datetime'], is_dst=None)
        # convert the user datetime to UTC
        utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)
        # format the datetime as expected by the pipeline
        params['pickup_datetime'] = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

        # Fixing a value for the key, as the pipeline was created with the original dataset
        # which contains a key column. This column was dropped later on in the pipeline
        # and not used in training the model.
        key='2020-07-03 23:00:00.000000001'
        X_pred = pd.DataFrame(params, index=[0])
        X_pred.insert(loc=0, column='key', value=key)
        fare = model.predict(X_pred)[0].round(2)
        if fare==41.89:
            st.markdown("### Range is too large. ")
            st.markdown("### Please input a location within New York City only.")
        else:
            st.markdown(f'### Predicted fare: `${fare}`')
