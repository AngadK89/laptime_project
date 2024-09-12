import fastf1 as f1
import pandas as pd 

'''Step 1: Create a table of all sessions, sprints, and qualifying sessions available in the API.'''

f1.Cache.clear_cache()

# Produces a single dataframe of all the events from 2020-2024. 
# It includes all fps and includes some sessions yet to be completed (to be removed)
years = [i for i in range(2020, 2025)]
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
Apply the filter function to each row. If the filter produces a nested list of 2 or more elements (E.g., in the case of sprint weekends), 
explode() flattens it so that each event list is a separate row. We then drop any null lists (indicating testing sessions), and 
flatten the df back into a list for easy splitting into the required columns. 
'''
filtered_data = event_data.apply(filter_sessions, axis=1).explode().dropna().tolist()
sessions = pd.DataFrame(filtered_data, columns=['RoundNumber', 'Location', 'Session', 'Year'])

# Filter out any sessions that have yet to occur (they are scheduled for the 2024 season but have not yet occurred)
sessions['Year'] = pd.to_datetime(sessions['Year'], utc=True)
sessions = sessions[sessions['Year'] <= pd.Timestamp.now(tz='UTC')]

# Convert the datetimes of the sessions into just the year. This is as we just need the year & round number to extract lap data from the API
sessions['Year'] = sessions['Year'].dt.year

sessions.to_csv("session_data.csv", index=False)