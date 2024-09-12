import pandas as pd 
import fastf1 as f1
import math

'''Step 2: Collect all the necessary lap & weather data for each session using the FastF1 API.

This data includes:
 - Driver
 - Compound
 - Tyre Life
 - Fresh Tyre?
 - TrackStatus
 - Air Temperature
 - Humidity
 - Air Pressure
 - Track Temperature
 - Rainfall?

And it also provides the target variable: Laptime.
'''

def get_all_data(row):
    print("*" * 80)
    print(row['Year'], row['Location'])
    session = f1.get_session(row['Year'], row['RoundNumber'], row['Session'])
    session.load()
    laps = session.laps

    # Selects all laps within 107% of the fastest time (i.e., typically within 4-5s of fastest time)
    # This essentially eliminates all cool-down/outlaps that are done by drivers during qualifying sessions
    if row['Session'] in ('Qualifying', 'Sprint Qualifying', 'Sprint Shootout'):
        laps = laps.pick_quicklaps()
    
    # Remove all box laps and all deleted laps, and only pick laps with accurate timing data
    laps = laps.pick_wo_box().pick_not_deleted().pick_accurate()

    # Resets the index & drops the old index column to make concatenation with laps easier
    weather_data = laps.get_weather_data().reset_index(drop=True)   

    # Filtering the laps & weather data to extract all of the relevant columns only
    filtered_laps = laps.loc[:, ['Driver', 'Compound', 'TyreLife', 'FreshTyre', 'TrackStatus', 'LapTime']].reset_index(drop=True)
    filtered_weather = weather_data.loc[:, ~(weather_data.columns == 'Time')]

    # This is our calculated target variable - how many laps does the driver have left on this tire

    # Appending the race information (location & year) next to each row of lap & weather data, to make a complete input data point.
    race_info = pd.DataFrame([row] * len(filtered_laps), index=filtered_laps.index)
    combined_data = pd.concat([race_info, filtered_laps, filtered_weather], axis=1)
    return combined_data


# Retrieving all of the required data for each sessions. Each row represents a session, so each processed row returns its own 
# dataframe containing all the data for that session. To generate one cumulative dataset, we concatenate all the resulting dfs vertically.
# We do this in batches, so that the cache can be cleared and we don't put too much load on the API 
sessions = pd.read_csv("session_data.csv")
start = sessions[(sessions['Year'] == 2024) & (sessions['Location'] == 'Spielberg') & (sessions['Session'] == 'Race')].index[0] + 1
BATCH_SIZE = 10
num_batches = math.ceil((sessions.shape[0] - start) / BATCH_SIZE)

for i in range(0, num_batches):
    f1.Cache.clear_cache()

    chosen_sessions = sessions.iloc[(start + BATCH_SIZE * i) : start + BATCH_SIZE * (i+1)]
    print(chosen_sessions)
    retrieved_data = chosen_sessions.apply(get_all_data, axis=1)

    total_data = pd.concat(retrieved_data.tolist(), ignore_index=True)
    total_data.to_csv("lap_weather_data.csv", index=False, mode='a', header=False)
