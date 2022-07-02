import streamlit as st
import joblib
import requests
import datetime
import pytz

'''
# New York Taxi Fare Prediction
'''
def coordinates(input):
    """Checks whether the address entered is valid and extracts the
    coordinates through an API
    """

    api = 'https://nominatim.openstreetmap.org/search'

    if input is not "":
        if "new york" not in input.lower():
            input = input + " New York"
        if "united states" not in input.lower():
            input = input + " United States"

    response = requests.get(api, params={'q':input, 'format':'json'}).json()

    for add in response:
        if add['display_name'].lower() == input:
            lon = float(add['lon'])
            lat = float(add['lat'])
            return lon, lat

    if len(response) == 1:
        st.markdown("*We found one possible address. Please confirm or refine\
                your search. When confirmed, please copy the address into the query box:*")
        st.markdown(f"""`{response[0]['display_name']}`""")
    elif len(response) > 1:
        st.write("*We found multiple possible addresses. Please refine your search if not listed or\
                please copy the intended address below into the query box:*")
        for index, address in enumerate(response[:3]):
            st.markdown(f"""`{index + 1})  {address['display_name']}`""")
    elif len(response) == 0 and input != "":
        st.write("*No addresses were found. Please refine your search.*")

    return -9999, -9999

st.markdown('''
The prediction of taxi fare in New York City. The dataset is based on \
[Kaggle New York Taxi Fare Challenge](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/overview)
''')

'''
## User Input

1. Date Time
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

model = joblib.load('TaxiFareModel/model.joblib')


if p_add!="" and d_add!="" and predict:
    if p_add == d_add :
        st.markdown("0 Fare since same address")
    params=dict(pickup_datetime=date_and_time,
            pickup_longitude=p_lon,
            pickup_latitude=p_lat,
            dropoff_longitude=d_lon,
            dropoff_latitude=d_lat,
            passenger_count=int(a))

    ############################################################################
# Loading model directly
############################################################################

## Modifying the format of params['pickup_datetime'] as per requirements in the model
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
    key='2013-07-06 17:18:00.000000119'
    X_pred = pd.DataFrame(params, index=[0])
    X_pred.insert(loc=0, column='key', value=key)
    model = joblib.load(os.path.dirname(__file__) + "/../joblib/model.joblib")
    fare = float(model.predict(X_pred).round(2))
    st.markdown(f'## Predicted fare: `{fare}`')

else:
    st.markdown("Invalid Input. Please enter both input and before clicking predict again.")
