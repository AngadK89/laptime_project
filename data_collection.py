import pandas as pd 
import fastf1 as f1

# f1.Cache.clear_cache()

'''Step 1: Create a table of all races, sprints, and qualifying sessions available in the API.'''

# Produces a single dataframe of all the events from 2018-2024. 
# It includes all fps and includes some races yet to be completed (to be removed)
years = [i for i in range(2018, 2025)]
events = list(map(lambda x: f1.get_event_schedule(x, include_testing=False), years))
event_data = pd.concat(events)


# Filter out only the necessary columns, and create separate rows for each sprint/race/quali.
def filter_sessions(row):
    data = [] 
    base = [row['RoundNumber'], row['Location']]

    match row['EventFormat']:
        case 'testing':
            return []
        
        # In case we have a sprint weekend, create a new row for the sprint race specifically
        case 'sprint':
            # Append qualifying in case we have a sprint weekend
            data.append(base + [row['Session2'], row['Session2Date']])

        case 'sprint_qualifying' | 'sprint_shootout':
            # In case of sprint_quali, sessions 2 & 3 are sprint qualifying & sprint
            # In case of sprint_shootout, sessions 2 & 3 are quali & sprint shootout
            data.append(base + [row['Session2'], row['Session2Date']])
            data.append(base + [row['Session3'], row['Session3Date']])
    
    # Last 2 sessions are either qualifying, sprint, or race, regardless of weekend type.
    data.append(base + [row['Session4'], row['Session4Date']])
    data.append(base + [row['Session5'], row['Session5Date']])
    
    return data

'''
1. Apply the filter function to each row.

2. If the filter produces a nested list of 2 or more elements (E.g., in the case of sprint weekends), explode() flattens 
   it so that each event list is a separate row. 

3. We then drop any null lists (indicating testing sessions). 

4. Finally, flatten the df back into a list for easy splitting into the required columns. 
'''
filtered_data = event_data.apply(filter_sessions, axis=1).explode().dropna().tolist()
races = pd.DataFrame(filtered_data, columns=['RoundNumber', 'Location', 'Session', 'Year'])

# Filter out any races that have yet to occur (they are scheduled for the 2024 season but have not yet occurred)
races['Year'] = pd.to_datetime(races['Year'], utc=True)
races = races[races['Year'] <= pd.Timestamp.now(tz='UTC')]

# Convert the datetimes of the sessions into just the year. This is as we just need the year & round number to extract lap data from the API
races['Year'] = races['Year'].dt.year


'''Step 2: Define & apply a function to this table that loads all the laps in the "Race" session of each event.'''

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

# Retrieving all of the required data for each race event. Each row represents a race, so each processed row returns its own 
# dataframe containing all the data for that race. To generate one cumulative dataset, we concatenate all the resulting dfs vertically.
retrieved_data = races.iloc[[2]].apply(get_all_data, axis=1)
total_data = pd.concat(retrieved_data.tolist(), ignore_index=True)
total_data.to_csv("f1_data.csv", index=False)


# Call the OpenAI API 
# Give it a list of all the events (track + year)
# For each event, it'll return a row containing all necessary track info

'''
Need to figure out how to collect track info:
- Track length
- Number of corners
- No. Of high, mid, low speed corners
- Track rotation
- Number of straights 
- Length of straights
- Number of DRS zones


Collected: 
 - Compound 
 - Tyre age
 - fresh typre
 - track status 
 - Track temperature
 - Rainfall
 - Air temperature



Based on this, we will try to predict Laptime
'''