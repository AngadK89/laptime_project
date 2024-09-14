import pandas as pd 

lap_weather_data = pd.read_csv("lap_weather_data.csv")
track_data = pd.read_csv("track_data.csv")

# Natural join both dfs together on location & year (i.e., add track data to every lap at a given circuit)
total_data = pd.merge(lap_weather_data, track_data, how='inner')

# Verify that there are no missing/unmerged columns
verify_table = pd.merge(lap_weather_data, track_data, how='outer', indicator=True)
print(verify_table['_merge'].isin(['left_only', 'right_only']).sum())   # Returns 0, i.e., all rows are merged, there is no missing data

total_data.to_csv("total_data.csv")