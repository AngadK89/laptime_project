import openai 
import os 
import json
import pandas as pd
import math

'''
Need to figure out how to collect track info:
- Track length
- Number of corners
- No. Of high, mid, low speed corners
- Track rotation
- Number of DRS zones

This will be done manually as using API calls leads to high variance in results for the same query 
+ 
has higher probability of inaccurate results as compared to manually using the GPT web applications
'''



API_KEY = os.getenv("OPENAI_KEY")

SYS_MESSAGE = """
I am going to give you a list of F1 circuits and years.
Given this list of circuit names and years, I want you to return a JSON response with multiple rows in the following format:
{
circuit_name: [<name of circuit1>, <circuit2> ...],
year: [<year1>, <year2>...],
track_length (km): [<length of track 1>, <track 2>...],
number_of_turns: [<number of turns in the track 1>, <track 2>, ...],
number_of_low_speed_turns: [<number of low speed turns in the track 1>, <track 2>, ...],
number_of_mid_speed_turns: [<number of mid speed turns in the track 1>, <track 2>, ...],
number_of_high_speed_turns: [<number of high speed turns in the track 1>, <track 2>, ...],
number_of_drs_zones: [<number of drs zones in the track 1>, <track 2>, ...]
}
Here, each row shoud contain the data for 1 circuit & year. Do not include the 'km' units in track_length.
If I provide a list of tracks and years, return a list alongside each column representing the response for all objects in the input.
E.g., [2018, 2018, 2023...] for year.
Use the internet and make sure to include the year as date context in your internet searches. E.g., for searches about 
tracks in 2018, do not use data about any years after 2018. Bring me the most relevant information. Outdated information will not be accepted.
Feel free to use this website, but note that it has not been updated since 2021: https://www.planetf1.com/tracks,
so, make sure to check if there have been any track or DRS modifications since.
Do not give explanations or notes in your result.
"""

client = openai.OpenAI(api_key=API_KEY)

def get_track_data(tracks):
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYS_MESSAGE},
        {"role": "user", "content": tracks},
    ],
    response_format={"type": "json_object"}
    )

    response_data = json.loads(response.choices[0].message.content)
    return pd.DataFrame(response_data)


sessions = pd.read_csv("session_data.csv")

# Split this over 3 API calls and append to the same csv file
unique_tracks = sessions[['Location', 'Year']].drop_duplicates(ignore_index=True)
batch_size = math.ceil(unique_tracks.shape[0] / 3)
track_data = []

print(sessions.shape[0], batch_size)

# for i in range(0, 1):
#     chosen_tracks = unique_tracks.iloc[batch_size * i : batch_size * (i+1)].values.tolist()
#     print("*" * 100)
#     print(chosen_tracks)
#     track_data.append(get_track_data(str(chosen_tracks)))

print(get_track_data(str(unique_tracks[(unique_tracks['Location'] == 'Jeddah') & (unique_tracks['Year'] == 2021)].values.tolist())))
# track_data.to_csv("track_data.csv", index=False)

