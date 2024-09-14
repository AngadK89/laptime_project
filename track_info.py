import openai 
import os 
import json
import pandas as pd
import fastf1 as f1
import numpy as np
import math

'''
This will collect all of the following required track data:
- Track length
- Number of corners
- Corner positions & offset angle (angle of corner relative to track rotation)
- Track rotation
- Number of DRS zones


I will use data about corner positions & angles to give the ML model geolocation data of the track. Thanks to this, it will 
be able to form a rough "visualisation" of the track, which will help in determining laptime. 
E.g., if 3 corners form a sharp angle to each other, it is likely that there is a hairpin, so it will be slower speed.
'''

# Deep clear the cache once before using it.
f1.Cache.clear_cache(deep=True)

API_KEY = os.getenv("OPENAI_KEY")

client = openai.OpenAI(api_key=API_KEY)

SYS_MESSAGE = """Imagine you are an information provider about F1 circuits. I will provide you an F1 circuit and a year, 
             and you will just give me a simple JSON response with the following format:
             {
                track_length: <length of the F1 track in the given year>,
                num_drs: <number of DRS zones in the F1 track in that given year>
             }
             Do not include the word 'km' in your track length response. Just return the number. Make sure you do not change the column names
             in your JSON response. They should ALWAYS remain fixed. 
             Make sure to use the year provided as date context in your information searches, and do not provide any explanations or notes
             alongside your answers.
             """


def get_track_data(row):    
    # Since we just want track data, we can arbitrarily use any session. So, we use the race ("R") here.
    track, year = row['Location'], row['Year']

    print("*" * 100)
    print(track, year)

    f1.Cache.clear_cache()

    session = f1.get_session(year, track, "R")
    session.load(weather=False, messages=False)

    circuit_info = session.get_circuit_info()

    # Extract all of the data for each corner, including corner position, number of corners, and track rotation.
    corners = circuit_info.corners
    num_corners = len(corners)
    track_rotation = circuit_info.rotation  # Angle at which the track is oriented relative to true North

    # Creates an array of shape (3n,), with each pair of indices holding an (x,y) coordinate of a corner
    corners = corners.drop(["Number", "Letter", "Distance", "Angle"], axis=1)
    corner_data = corners.to_numpy().reshape(-1)
    # corner_data = [tuple(corner) for corner in corners.itertuples(index=False, name=None)]

    # We want to create a list of length 27 as there are max. 27 corners in the list of modern F1 circuits, so we pad our corner
    # data with the required number of missing corners 
    padded_corners = np.pad(corner_data, (0, 2 * (27 - num_corners)), mode='constant', constant_values=0)
    # padded_corners = corner_data + [(0,0,0)] * (27 - num_corners)

    # Use OpenAI API to get the track length & drs zones for each F1 track as a JSON, and extract this data from there.
    complete = False 

    while not complete:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYS_MESSAGE},
                    {"role": "user", "content": f"{row['Location']}, {row['Year']}"}
                ],
                response_format={"type": "json_object"}
            )

            response_data = json.loads(response.choices[0].message.content)
            track_length = response_data['track_length']
            num_drs = response_data['num_drs']
            complete = True

        except:
            continue
        
    # Create an array of all of the remaining track data to be added - track, year, length, number of drs zones, track rotation, number of corners
    rem_track_data = np.array([track, year, track_length, num_drs, num_corners, track_rotation])

    # Concatenates them all together to produce 1 comprehensive row of "track data"
    track_data = np.concatenate((rem_track_data, padded_corners))
    # track_data = rem_track_data + padded_corners
    return pd.Series(track_data)


# Create the list of all column names that we require
column_names = ["Location", "Year", "TrackLength", "NumDrs", "NumCorners", "TrackRotation"]

for i in range(1, 28):
    column_names += [f"Turn{i}" + s for s in ["X", "Y"]]

# Retrieve all unique tracks in the datset. 
sessions = pd.read_csv("session_data.csv")
unique_tracks = sessions[['Location', 'Year']].drop_duplicates(ignore_index=True)

# This creates a dataframe with all the track data that we require. 
start = unique_tracks[(unique_tracks['Year'] == 2023) & (unique_tracks['Location'] == 'Marina Bay')].index[0] + 1
BATCH_SIZE = 10
num_batches = math.ceil((unique_tracks.shape[0] - start) / BATCH_SIZE)

for i in range(0, num_batches):
    chosen_tracks = unique_tracks.iloc[start + BATCH_SIZE * i : start + BATCH_SIZE * (i+1)]
    print(chosen_tracks)
    track_data = pd.DataFrame(chosen_tracks.apply(get_track_data, axis=1).values.tolist(), columns=column_names)

    track_data.to_csv("track_data.csv", index=False, mode='a', header=False)

