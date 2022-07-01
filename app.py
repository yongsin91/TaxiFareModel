import streamlit as st
import joblib
import requests


'''
# New York Taxi Fare Prediction
'''
def coordinates(input_add):
    """Checks whether the address entered is valid and extracts the
    coordinates through an API
    """
    api_url = 'https://nominatim.openstreetmap.org/search'
    response = requests.get(api_url, params={'q':input_add,
                                                'format':'json'}).json()

    new_york = []
    for ny_address in response:
        ny_address_split = ny_address['display_name'].split(', ')
        if ny_address_split[-1] == "United States" and\
                (ny_address_split[-2] == "New York" or\
                 ny_address_split[-3] == "New York"):
            new_york.append(ny_address)

    for add in new_york:
        if add['display_name'].lower() == input_add:
            lon = float(add['lon'])
            lat = float(add['lat'])
            return lon, lat

    if len(new_york) == 1:
        st.markdown("*We found one possible address but it doesn't match your query exactly.\
                Please refine your search or copy this address into the query box:*")
        st.markdown(f"""`{address['display_name']}`""")
    elif len(new_york) > 1:
        st.write("*We found multiple possible addresses. Please refine your search or\
                copy one of the addresses below into the query box:*")
        for index, address in enumerate(response[:3]):
            st.markdown(f"""`{index + 1})  {address['display_name']}`""")
    elif len(new_york) == 0 and input_add != "":
        st.write("*No addresses were found. Please refine your search.*")

    return -9999, -9999

st.markdown('''
The prediction of taxi fare in New York City. The dataset is based on [Kaggle New York Taxi Fare Challenge](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/overview)
''')

'''
## User Input

1. Date Time
'''
col1, col2, col3 = st.columns(3)
d = col1.date_input("Input Date")
t = col2.time_input("Input Time")
a = col3.number_input("Input Number of Passengers", min_value=1, max_value=8)
# Obtaining coordinates for pickup and dropoff address
p_add = st.text_input('Input Pickup address').strip().lower()
p_lon, p_lat = coordinates(p_add)
d_add = st.text_input('Input dropoff address').strip().lower()
d_lon, d_lat = coordinates(d_add)

predict = st.button("Predict Taxi Fare")

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
