import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

# Import dataset and modify accordingly
print('Load and preprocess data')
tap = pd.read_csv('acc2005_2016.csv', low_memory=False).merge(pd.read_csv('cas2005_2016.csv'), on='Accident_Index').merge(pd.read_csv('veh2005_2016.csv'), on='Accident_Index').drop([
    # Removed because they are index or references
    'Accident_Index', 'Vehicle_Reference_x', 'Vehicle_Reference_y', 'Casualty_Reference',
    # Removed because they are noisy data
    'Age_of_Casualty', 'LSOA_of_Accident_Location', 'Location_Easting_OSGR', 'Location_Northing_OSGR', 'Age_of_Driver', 
    # Removed because they are contextually unrelated
    'Did_Police_Officer_Attend_Scene_of_Accident', # Not interested in what happens after the accident
    # Removed because they have at least 50% missing value or are completely missing in one or more year
    'Casualty_IMD_Decile', 'Vehicle_IMD_Decile', 'Pedestrian_Road_Maintenance_Worker', 'Driver_IMD_Decile', 
    # Removed because they are cheating variables
    'Casualty_Severity'
], axis=1).dropna(subset=['Longitude', 'Latitude', 'Time']).rename(columns={
    'Local_Authority_.District.': 'Local_Authority_District',
    'Local_Authority_.Highway.': 'Local_Authority_Highway',
    'Speed_limit': 'Speed_Limit',
    'Pedestrian_Crossing.Human_Control': 'Pedestrian_Crossing_Human_Control',
    'Pedestrian_Crossing.Physical_Facilities': 'Pedestrian_Crossing_Physical_Facilities',
    'Was_Vehicle_Left_Hand_Drive.': 'Was_Vehicle_Left_Hand_Drive', 
    'Vehicle_Location.Restricted_Lane': 'Vehicle_Location_Restricted_Lane',
    'Engine_Capacity_.CC.': 'Engine_Capacity_CC'
})

# Monotonise Date and Time to independent features then drop compound feature
print('Atominise Date and Time into independent variables')
tap.loc[:, ('Date')] = pd.to_datetime(tap['Date'])
tap.loc[:, ('Time')] = pd.to_datetime(tap['Time'], format='%H:%M')
tap = tap.assign(Year=tap.Date.dt.year, Month=tap.Date.dt.month, Day=tap.Date.dt.day, Hour=tap.Time.dt.hour, Minute=tap.Time.dt.minute, Is_Weekend=[1 if day == 1 or day == 7 else 0 for day in tap.Day_of_Week])
tap = tap.drop(['Date', 'Time'], axis=1)

# Replace -1 missing value placeholder as np.nan (NaN missing values)
print('Replacing -1 with NA missing value')
tap = tap.replace(-1, np.nan)
tap[['Propulsion_Code']] = tap[['Propulsion_Code']].replace('M', np.nan)
tap[['Longitude']] = tap[['Longitude']].replace(np.nan, -1)

# Encode string feature to integer
print('Encoding')
le_jc = LabelEncoder()
tap.loc[:, ('Local_Authority_Highway')] = le_jc.fit_transform(tap['Local_Authority_Highway'])

print('Write to CSV to be imputed')
tap.to_csv('tap_tobeimputed.csv', sep=',', encoding='utf-8', index=False)
