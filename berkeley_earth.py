import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import time
# from sklearn

# Change to where house price files are
os.chdir("/Users/dustinwicker/.kaggle/datasets/berkeleyearth/climate-change-earth-surface-temperature-data")

# Suppress scientific notation
np.set_printoptions(suppress=True)

# Print full contents of ndarray, i.e. predict
np.set_printoptions(threshold=np.nan)

# Create bytes to GB converter
byte_to_gb = 1024**3

# List file in current directory
os.listdir()

########################
### Create functions ###
########################

# Define function to check for dataset
def check_for_dataset(dataset):
    if dataset in os.listdir():
        print(dataset, "exists in the current directory. Okay to proceed and pull it in.")
    else:
        print("The file does not exist in the current directory.")
        return

# Define function to check for completely missing rows
def completely_missing_rows(df):
    # Determine rows that are completely missing
    if df[df.isnull().all(axis=1)].empty:
        print("No rows in dataset are completely missing. We are good to proceed.")
    else:
        print("There are", len(df[df.isnull().all(axis=1)]), "completely empty rows.")
        print("These rows need to be removed.")
        return

# Function to determine columns with missing values
def determine_columns_with_missing_values(df):
    # Determine columns with missing values
    missing = [(str(item) + str(" has ") + str("{:,}".format(df[item].isnull().values.sum())) +
                str(" missing values and is ") + str("{:.1%}".format(df[item].isnull().values.sum() / len(df))) +
                str(" missing.")) for item in list(df) if df[item].isnull().values.any()]

    # View results vertically [just printing the list result (i.e., house_prices_train_missing) prints it vertically]
    if len(missing) > 0:
        for item in missing:
            print(item)
    else:
        print("There are no columns in with missing values.")

# Function that creates indicator flag column when imputing values where feature is null (creates column of 0's where not null and 1 where null)
def create_indicator_flag(df,column):
    df[str(column)+"_flag"] = 0
    df.loc[df[str(column)].isnull(), str(column)+"_flag"] = 1

#############
### Ideas ###
#############
# 1. Create time series (predict future temperatures)

########################
### Load in datasets ###
########################

### GlobalLandTemperaturesByCountry.csv ###
# Check for dataset with created function from above
check_for_dataset("GlobalLandTemperaturesByCountry.csv")

# Load in dataset #
# Use list comprehension to find GlobalTemperatures.csv to load in
global_temps_country_dataset = [value for value in os.listdir() if value == "GlobalLandTemperaturesByCountry.csv"]
# Load in GlobalLandTemperaturesByCountry.csv
global_temps_country = pd.read_csv(global_temps_country_dataset[0], sep=",", header="infer")
print("The dataset has been loaded in.")
# Determine number of columns and rows of house_prices_train
print("global_temps_country has", global_temps_country.shape[0], "rows and", global_temps_country.shape[1], "columns.")

### GlobalLandTemperaturesByState.csv ###
# Check for dataset with created function from above
check_for_dataset("GlobalLandTemperaturesByState.csv")

# Load in dataset #
# Use list comprehension to find GlobalTemperatures.csv to load in
global_temps_state_dataset = [value for value in os.listdir() if value == "GlobalLandTemperaturesByState.csv"]
# Load in GlobalLandTemperaturesByState.csv
global_temps_state = pd.read_csv(global_temps_state_dataset[0], sep=",", header="infer")
print("The dataset has been loaded in.")
# Determine number of columns and rows of house_prices_train
print("global_temps_state has", global_temps_state.shape[0], "rows and", global_temps_state.shape[1], "columns.")

# Widen output to display 200 columns
pd.options.display.max_columns = 200
# Increase number of rows printed out in console to 200
pd.options.display.max_rows = 200

#####################
### Data Cleaning ###
#####################

# Lowercase column headers for country
global_temps_country.columns = [value.lower() for value in global_temps_country.columns]

# Lowercase column headers for state
global_temps_state.columns = [value.lower() for value in global_temps_state.columns]

########################
### Data Exploration ###
########################

# See unique countries
global_temps_country.country.value_counts()

missing_by_country = []
# See percentage missing for each country for each column
for value in global_temps_country.country.unique():
    print("Country:", value)
    print(global_temps_country.loc[global_temps_country.country==value].isnull().sum()/
                                len(global_temps_country.loc[global_temps_country.country==value]),'\n')

# United States is 15% missing from 1768-09-01 to 2013-09-01
usa_temps = global_temps_country.loc[global_temps_country.country=="United States"].reset_index(drop=True)

# Convert Celsius to Fahrenheit
usa_temps.loc[:,"averagetemperature_f"] = (9/5)*usa_temps.loc[:, "averagetemperature"]+32

usa_temps[usa_temps.dt>"1849-12-01"].isnull().sum()/len(usa_temps[usa_temps.dt>"1849-12-01"])
# No missingness from 1850 and beyond
usa_temps_1850 = usa_temps.loc[usa_temps.dt>"1849-12-01"]

# Set dt (date) as index to create DatetimeIndex
usa_temps_1850 = usa_temps_1850.set_index(pd.DatetimeIndex(usa_temps_1850["dt"]))

# usa_temps_1850.plot(x="dt", y='averagetemperature',kind='line')

usa_temps_1850_year = usa_temps_1850.groupby(usa_temps_1850.index.year)['averagetemperature_f'].mean()

# Plot average usa temperatues by year
sns.lineplot(data=usa_temps_1850_year)

# North Carolina
nc_temps = global_temps_state.loc[(global_temps_state.state=="North Carolina") & (global_temps_state.country=="United States")]

# Convert Celsius to Fahrenheit
nc_temps.loc[:,"averagetemperature_f"] = (9/5)*nc_temps.loc[:, "averagetemperature"]+32

nc_temps[nc_temps.dt>"1849-12-01"].isnull().sum()/len(nc_temps[nc_temps.dt>"1849-12-01"])
# No missingness from 1850 and beyond
nc_temps_1850 = nc_temps.loc[nc_temps.dt>"1849-12-01"]

# Set dt (date) as index to create DatetimeIndex
nc_temps_1850 = nc_temps_1850.set_index(pd.DatetimeIndex(nc_temps_1850["dt"]))

nc_temps_1850_year = nc_temps_1850.groupby(nc_temps_1850.index.year)['averagetemperature_f'].mean()

# Plot average usa temperatues by year
sns.lineplot(data=nc_temps_1850_year)