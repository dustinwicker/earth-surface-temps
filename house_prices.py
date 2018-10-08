import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.model_selection import cross_val_score
import time
from sklearn.metrics import mean_squared_error
from math import sqrt
from CHAID import Tree

# Change to where house price files are
os.chdir("/Users/dustinwicker/.kaggle/competitions/house-prices-advanced-regression-techniques")

# Suppress scientific notation
np.set_printoptions(suppress=True)

# Print full contents of ndarray, i.e. predict
np.set_printoptions(threshold=np.nan)

# List file in current directory
os.listdir()

# Create bytes to GB converter
byte_to_gb = 1024**3

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

# Define function to convert to object
def convert_to_object(cols):
    for value in cols:
        house_prices_train[value] = house_prices_train[value].apply(str)

# Define function to make sure all variables in the dataset have been put into a variable category
def all_variables_given_type(continous_vars, nominal_vars, ordinal_vars, indicator_vars, id_var, df):
    if set(continous_vars+nominal_vars+ordinal_vars+indicator_vars+id_var)^set(list(df)) == set():
        print('All variables in the dataset have been accounted for. Okay to proceed.')
    else:
        print("All variables in the dataset have NOT been accounted for. Need to check")
        print(set(continous_vars + nominal_vars + ordinal_vars + indicator_vars + id_var) ^ set(list(df)))
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

########################
### Load in datasets ###
########################

### train ###
# Check for dataset with created function from above
check_for_dataset("train.csv")

# Load in dataset #
# Use list comprehension to find train.csv to load in
house_prices_train_dataset = [value for value in os.listdir() if value == "train.csv"]
# Load in train.csv
house_prices_train = pd.read_csv(house_prices_train_dataset[0], sep=",", header="infer")
print("The dataset has been loaded in.")

# Widen output to display as many columns as there are in dataset
pd.options.display.max_columns = len(list(house_prices_train))
# Increase number of rows printed out in console
pd.options.display.max_rows = len(list(house_prices_train))

# Determine number of columns and rows of house_prices_train
print("house_prices_train has", house_prices_train.shape[0], "rows and", house_prices_train.shape[1], "columns.")

### test ###
# Check for dataset with created function from above
check_for_dataset("test.csv")

# Load in dataset #
# Use list comprehension to find test.csv to load in
house_prices_test_dataset = [value for value in os.listdir() if value == "test.csv"]
# Load in test.csv
house_prices_test = pd.read_csv(house_prices_test_dataset[0], sep=",", header="infer")
print("The dataset has been loaded in.")

### Already ran when loading in train
# Widen output to display as many columns as there are in dataset
# pd.options.display.max_columns = len(list(house_prices_train))
# Increase number of rows printed out in console
# pd.options.display.max_rows = len(list(house_prices_train))

# Determine number of columns and rows of house_prices_test
print("house_prices_test has", house_prices_test.shape[0], "rows and", house_prices_test.shape[1], "columns.")

################################
### Data Preparation/Cleanup ###
################################

### train ###

# Check for completely missing rows with created function from above
completely_missing_rows(house_prices_train)

# Determine columns with missing values
determine_columns_with_missing_values(df=house_prices_train)

# house_prices_train_missing = [(str(item) + str(" has ") + str("{:,}".format(house_prices_train[item].isnull().values.sum()))+
#                                str(" missing values and is ") +
#                                str("{:.1%}".format(house_prices_train[item].isnull().values.sum()/len(house_prices_train))) +
#                               str(" missing.")) for item in list(house_prices_train) if house_prices_train[item].isnull().values.any()]
#
# # View results vertically [just printing the list result (i.e., house_prices_train_missing) prints it vertically]
# if len(house_prices_train_missing) > 0:
#     for item in house_prices_train_missing:
#         print(item)
# else:
#     print("There are no columns with missing values.")

### Fill in NaN columns appropriately and update other columns accordingly ###
# Indicator flags
create_indicator_flag(df=house_prices_train, column="Alley")
# An NA value in Alley means the house has no alley access
house_prices_train["Alley"] = house_prices_train["Alley"].fillna("None")

# Indicator flags
create_indicator_flag(df=house_prices_train, column="MasVnrType")
# This is an assumption - if MasVnrType is NaN, assume None
house_prices_train["MasVnrType"] = house_prices_train["MasVnrType"].fillna("None")

# Indicator flags
house_prices_train["MasVnrArea_flag"] = 0
house_prices_train.loc[(house_prices_train.MasVnrType == "None") & (house_prices_train.MasVnrArea != 0.0), "MasVnrArea_flag"] = 1
# This is an assumption, if MasVnrType is None, then MasVnrArea is 0
house_prices_train.loc[house_prices_train.MasVnrType == "None", "MasVnrArea"] = 0.0

# An NA value in BsmtQual and BsmtCond means No Basement - they are linked
# An NA value in BsmtExposure, BsmtFinType1, and BsmtFinType2 means No Basement
# Create list of the five columns
# Indicator flags
basement_cols_to_fix = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
for item in basement_cols_to_fix:
    create_indicator_flag(df=house_prices_train,column=item)
# For loop to fix columns
for item in basement_cols_to_fix:
    house_prices_train[item] = house_prices_train[item].fillna("None")

# Check that all Basement Type 1 that are classified as None have 0 square feet
if set(house_prices_train.loc[house_prices_train.BsmtFinType1.isin(["None", "Unf"])]["BsmtFinSF1"]) == {0}:
    print("All Basement Type 1 ratings of None and Unf are equal to zero. Good to proceed.")
else:
    print("All Basement Type 1 ratings of None and Unf are NOT equal to zero. Need to inspect.")
    print(house_prices_train.loc[(house_prices_train.BsmtFinType1.isin(["None", "Unf"])) & (house_prices_train.BsmtFinSF1 != 0)])

# Check that all Basement Type 1 that have 0 square feet are classified as None or Unf
if set(house_prices_train.loc[house_prices_train.BsmtFinSF1 == 0]["BsmtFinType1"]) == {"None", "Unf"}:
    print("All Basement Type 1 that are 0 square feet are either None or Unf. Good to proceed.")
else:
    print("All Basement Type 1 that are 0 square feet are NOT limited to None or Unf. Need to check.")

# Seeing the minimum is a good check for all other than None or Unf (all others should have minimums > 0)
# Every BsmtFinType1 other than None and Unf have minimum square feet > 0 - this is good
house_prices_train.groupby("BsmtFinType1")["BsmtFinSF1"].min() #mean #min
# Seeing the max is a good check for None or Unf (they should have max of 0)
# None and Unf in BsmtFinType1 have maxes = 0 (this is good)
house_prices_train.groupby("BsmtFinType1")["BsmtFinSF1"].max()

# Check that all Basement Type 2 that are classified as None and Unf have 0 square feet
if set(house_prices_train.loc[house_prices_train.BsmtFinType2.isin(["None", "Unf"])]["BsmtFinSF2"]) == {0}:
    print("All Basement Type 2 ratings of None and Unf are equal to zero. Good to proceed.")
else:
    print("All Basement Type 2 ratings of None and Unf are NOT equal to zero. Need to inspect.")
    print(house_prices_train.loc[(house_prices_train.BsmtFinType2.isin(["None", "Unf"])) & (house_prices_train.BsmtFinSF2 != 0)])

# Fix value where BsmtFinType2=="None" & house_prices_train.BsmtFinSF2 != 0
# Create binary flag column indicating value has been imputed - create all zeroes and then one I am going to impute to 1
house_prices_train["BsmtFinType2_flag"] = 0
# Put one in binary flag column indicating the value has been imputed
house_prices_train.loc[(house_prices_train.BsmtFinType2=="None") & (house_prices_train.BsmtFinSF2 != 0), "BsmtFinType2_flag"] = 1
# Replace value with mode - Unf is mode (has BsmtFinSF2 == 0) so choose next value which is Rec
house_prices_train.BsmtFinType2.value_counts()
# Impute BsmtFinType2 to "Rec"
house_prices_train.loc[(house_prices_train.BsmtFinType2=="None") & (house_prices_train.BsmtFinSF2 != 0), "BsmtFinType2"] = "Rec"

# Check that all Basement Type 2 that have 0 square feet are classified as None or Unf
if set(house_prices_train.loc[house_prices_train.BsmtFinSF2 == 0]["BsmtFinType2"]) == {"None", "Unf"}:
    print("All Basement Type 2 that are 0 square feet are either None or Unf. Good to proceed.")
else:
    print("All Basement Type 2 that are 0 square feet are NOT limited to None or Unf. Need to check.")

# Every BsmtFinType2 other than None and Unf have minimum square feet > 0 - this is good
house_prices_train.groupby("BsmtFinType2")["BsmtFinSF2"].min()
# BsmtFinType2 None and Unf have maximum square feet = 0 - this is good
house_prices_train.groupby("BsmtFinType2")["BsmtFinSF2"].max()

# See value counts of LotFrontage
house_prices_train["LotFrontage"].value_counts(dropna=False)

# This is an assumption - if LotFrontage is NaN, assume 0
house_prices_train["LotFrontage"] = house_prices_train["LotFrontage"].fillna(0)

# Electrical - see row where value is missing
house_prices_train.loc[house_prices_train.Electrical.isnull()]
# This is an assumption - if Electrical is NaN, replace with mode (SBrkr)
house_prices_train.Electrical.value_counts()
# Create binary flag column indicatign value has been imputed
house_prices_train["Electrical_flag"] = 0
# Put value of 1 in Electrical_flag for row where it is equal to NaN
house_prices_train.loc[house_prices_train.Electrical.isnull(), "Electrical_flag"] = 1
# Replace missing Elecrical value with Sbrkr
house_prices_train["Electrical"] = house_prices_train["Electrical"].fillna("SBrkr")

# If FireplaceQu is NaN, then No Fireplace
house_prices_train["FireplaceQu"] = house_prices_train["FireplaceQu"].fillna("None")

# An NA value in BsmtQual and BsmtCond means No Basement - they are linked
# An NA value in BsmtExposure, BsmtFinType1, and BsmtFinType2 means No Basement
# Create list of the five columns
basement_cols_to_fix = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
# For loop to fix columns
for item in basement_cols_to_fix:
    house_prices_train[item] = house_prices_train[item].fillna("None")

# Check to see if all missing values related to Garage are the same rows - there are 81 missing values of GarageType - if
# all 81 are returned from the subsetted DataFrame (i.e. the two values on either side of the double equal each other) then
# all 5 Garage columns' missingness represent the same rows
if len(house_prices_train.loc[house_prices_train.GarageType.isnull()]) ==\
    house_prices_train.loc[house_prices_train.GarageType.isnull(),
                           ['GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']].isnull().sum().unique()[0]:
    print("All Garage columns' missingness represent the same rows. Convert all NaN's to None.")
    garage_cols_to_fix = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
    for item in garage_cols_to_fix:
        house_prices_train[item]= house_prices_train[item].fillna("None")
    print("All Garage rows that were NaN have been set to None. Good to proceed.")
    house_prices_train['GarageYrBlt'] = house_prices_train['GarageYrBlt'].fillna(0)
    print("All house without a garage have had GarageYrBlt set to 0.")

else:
    print("All Garage columns' missingness DO NOT represent the same rows. Time to inspect.")

# If PoolQC is NaN, then No Pool
house_prices_train["PoolQC"] = house_prices_train["PoolQC"].fillna("None")

# If Fence is NaN, then No Fence
house_prices_train["Fence"] = house_prices_train["Fence"].fillna("None")

# If MiscFeature is NaN, then No Miscellaneous Feature
house_prices_train["MiscFeature"] = house_prices_train["MiscFeature"].fillna("None")

# No columns remain with missing values
determine_columns_with_missing_values(df=house_prices_train)

# Id - Check to make sure all rows represent a unique house
if len(house_prices_train.Id.unique()) == len(house_prices_train):
    print("All rows represnt a unique house.")
else:
    print("All rows DO NOT represent a unique house. Possible duplicate?")

### test ###

# Check for completely missing rows with created function from above
completely_missing_rows(house_prices_test)

### Fill in NaN columns appropriately and update other columns accordingly ###
# An NA value in Alley means the house has no alley access
house_prices_test["Alley"] = house_prices_test["Alley"].fillna("None")

# This is an assumption - if MasVnrType is NaN, assume None
house_prices_test["MasVnrType"] = house_prices_test["MasVnrType"].fillna("None")

# This is an assumption, if MasVnrType is None, then MasVnrArea is 0
house_prices_test.loc[house_prices_test.MasVnrType == "None", "MasVnrArea"] = 0.0

# If FireplaceQu is NaN, then No Fireplace
house_prices_test["FireplaceQu"] = house_prices_test["FireplaceQu"].fillna("None")

# If PoolQC is NaN, then No Pool
house_prices_test["PoolQC"] = house_prices_test["PoolQC"].fillna("None")

### Shouldn't FireplaceQU and PoolQC where None, be 0 for FireplaceArea and PoolArea??? ###
### Or should PoolArea > 0 with PoolQC as None just be given the mode of PoolQC where PoolArea is > 0??? ###
### Have been fixed in one way above ###

# If Fence is NaN, then No Fence
house_prices_test["Fence"] = house_prices_test["Fence"].fillna("None")

# If MiscFeature is NaN, then No Miscellaneous Feature
house_prices_test["MiscFeature"] = house_prices_test["MiscFeature"].fillna("None")

# This is an assumption - if LotFrontage is NaN, assume 0
house_prices_test["LotFrontage"] = house_prices_test["LotFrontage"].fillna(0)

# MSZoning - fill missing values with mode
house_prices_test["MSZoning"].value_counts(dropna=False,ascending=False)
# Create binary flag column indicatign value has been imputed
house_prices_test["MSZoning_flag"] = 0
# Put value of 1 in MSZoning_flag for row where it is equal to NaN
house_prices_test.loc[house_prices_test.MSZoning.isnull(), "MSZoning_flag"] = 1

# Obtain mode
house_prices_test["MSZoning"].mode()
# Fill in NaN values with mode
house_prices_test["MSZoning"] = house_prices_test["MSZoning"].fillna(house_prices_test["MSZoning"].mode().values[0])

### Utilities
house_prices_test["Utilities"].value_counts(dropna=False)
# Create binary flag column indicatign value has been imputed
house_prices_test["Utilities_flag"] = 0
# Put value of 1 in Utilities_flag for row where it is equal to NaN
house_prices_test.loc[house_prices_test.Utilities.isnull(), "Utilities_flag"] = 1

# Fill in NaN values with mode
house_prices_test["Utilities"] = house_prices_test["Utilities"].fillna(house_prices_test["Utilities"].mode().values[0])

### Exterior1st
# Create binary flag column indicatign value has been imputed
house_prices_test["Exterior1st_flag"] = 0
# Put value of 1 in Exterior1st_flag for row where it is equal to NaN
house_prices_test.loc[house_prices_test.Exterior1st.isnull(), "Exterior1st_flag"] = 1

# Fill in NaN value with mode
house_prices_test["Exterior1st"] = house_prices_test["Exterior1st"].fillna(house_prices_test["Exterior1st"].mode().values[0])

### Exterior2nd
# Create binary flag column indicatign value has been imputed
house_prices_test["Exterior2nd_flag"] = 0
# Put value of 1 in Exterior2nd_flag for row where it is equal to NaN
house_prices_test.loc[house_prices_test.Exterior2nd.isnull(), "Exterior2nd_flag"] = 1

# Fill in NaN value with mode
house_prices_test["Exterior2nd"] = house_prices_test["Exterior2nd"].fillna(house_prices_test["Exterior2nd"].mode().values[0])

### Rows where BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, and BsmtFinType2 are NaN
# Create binary flag column indicatign value has been imputed
house_prices_test["Bsmt_flag"] = 0
# Put value of 1 in Bsmt_flag for row where BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, and BsmtFinType2 are equal to NaN
house_prices_test.loc[(house_prices_test.BsmtQual.isnull()) & (house_prices_test.BsmtCond.isnull()) &\
                      (house_prices_test.BsmtExposure.isnull()) & (house_prices_test.BsmtFinType1.isnull()) &\
                      (house_prices_test.BsmtFinType2.isnull()), "Bsmt_flag"] = 1

# If NaN for all of these columns then No Basement
house_prices_test.loc[(house_prices_test.BsmtQual.isnull()) & (house_prices_test.BsmtCond.isnull()) &\
                      (house_prices_test.BsmtExposure.isnull()) & (house_prices_test.BsmtFinType1.isnull()) &\
                      (house_prices_test.BsmtFinType2.isnull()), ('BsmtQual','BsmtCond','BsmtExposure',
                      'BsmtFinType1', 'BsmtFinType2')] = ("None","None","None","None","None")

### BsmtQual - change to mode of BsmtCond (basement present but unfinished)
# Indicator Flag
house_prices_test["BsmtQual_flag"] = 0
# Put value of 1 in BsmtQual_flag for row where it is equal to NaN
house_prices_test.loc[(house_prices_test.BsmtExposure=='No') & (house_prices_test.BsmtFinSF1==0) &
                      (house_prices_test.BsmtFinType1=='Unf') & (house_prices_test.BsmtFinSF2==0) &
                      (house_prices_test.BsmtFinType2=='Unf') & (house_prices_test.BsmtQual.isnull()),'BsmtQual_flag'] = 1

# Fill with mode where BsmtQual is not null and BsmtCond='Fa'
house_prices_test.loc[(house_prices_test.BsmtCond=='Fa') & (house_prices_test.BsmtExposure=='No') &
                      (house_prices_test.BsmtFinSF1==0) & (house_prices_test.BsmtFinType1=='Unf') &
                      (house_prices_test.BsmtFinSF2==0) & (house_prices_test.BsmtFinType2=='Unf') &
                      (house_prices_test.BsmtQual.isnull()), ('BsmtQual')] = 'TA'

# Fill with mode where BsmtQual is not null and BsmtCond='TA'
house_prices_test.loc[(house_prices_test.BsmtCond=='TA') & (house_prices_test.BsmtExposure=='No') &
                      (house_prices_test.BsmtFinSF1==0) & (house_prices_test.BsmtFinType1=='Unf') &
                      (house_prices_test.BsmtFinSF2==0) & (house_prices_test.BsmtFinType2=='Unf') &
                      (house_prices_test.BsmtQual.isnull()), ('BsmtQual')] = 'TA'

### BsmtCond
# Inidicator Flag
house_prices_test['BsmtCond_flag'] = 0
# Fill BsmtCond_flag with 1 where BsmtCond has been imputed
house_prices_test.loc[house_prices_test.BsmtCond.isnull(), "BsmtCond_flag"] = 1

# Fill BsmtCond with mode of subsetted dataframe where Bsmt is not null
house_prices_test.loc[(house_prices_test.BsmtQual=='Gd') & (house_prices_test.BsmtCond.isnull()) &
                      (house_prices_test.BsmtExposure == 'Mn') & (house_prices_test.BsmtFinSF1>0) &
                      (house_prices_test.BsmtFinType1 == 'GLQ') & (house_prices_test.BsmtFinSF2>0) &
                      (house_prices_test.BsmtFinType2 == 'Rec'),"BsmtCond"] = 'TA'

# Fill BsmtCond with mode of subsetted dataframe where Bsmt is not null
house_prices_test.loc[(house_prices_test.BsmtQual=='TA') & (house_prices_test.BsmtCond.isnull()) &
                      (house_prices_test.BsmtExposure == 'No') & (house_prices_test.BsmtFinSF1>0) &
                      (house_prices_test.BsmtFinType1 == 'BLQ') & (house_prices_test.BsmtFinSF2==0) &
                      (house_prices_test.BsmtFinType2 == 'Unf'),"BsmtCond"] = 'TA'

house_prices_test.loc[(house_prices_test.BsmtQual=='TA') & (house_prices_test.BsmtCond.isnull()) &
                      (house_prices_test.BsmtExposure == 'Av') & (house_prices_test.BsmtFinSF1>0) &
                      (house_prices_test.BsmtFinType1 == 'ALQ') & (house_prices_test.BsmtFinSF2==0) &
                      (house_prices_test.BsmtFinType2 == 'Unf'),"BsmtCond"] = 'TA'

# BsmtFinSF1 and BsmtFinSF2
# Indicator flags
house_prices_test["BsmtFinSF1_flag"] = 0
house_prices_test["BsmtFinSF2_flag"] = 0

# Change row to 1 where BsmtFinSF1 and BsmtFinSF2 is null
house_prices_test.loc[(house_prices_test.BsmtFinSF1.isnull()) & (house_prices_test.BsmtFinSF2.isnull()), ["BsmtFinSF1_flag", "BsmtFinSF2_flag"]] = [1,1]

house_prices_test.loc[(house_prices_test.BsmtFinSF1.isnull()) & (house_prices_test.BsmtFinSF2.isnull()), ["BsmtFinSF1", "BsmtFinSF2"]] = [0,0]

# BsmtExposure
# Create indicator flag
house_prices_test["BsmtExposure_flag"] = 0
# Change values of BsmtExposure_flag to 1 where BmstExposure is null
house_prices_test.loc[house_prices_test.BsmtExposure.isnull(), ["BsmtExposure_flag"]] = 1

# Change null values of BsmtExposure to "No"
house_prices_test.loc[house_prices_test.BsmtExposure.isnull(), ["BsmtExposure"]] = "No"

# BsmtUnfSF
# Create indicator flag
house_prices_test["BsmtUnfSF_flag"] = 0
# Change values of BsmtExposure_flag to 1 where BmstExposure is null
house_prices_test.loc[house_prices_test.BsmtUnfSF.isnull(), ["BsmtUnfSF_flag"]] = 1

# house_prices_test.loc[house_prices_test.BsmtUnfSF.isnull(), ["BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2"]]
# Change null values of BsmtUnfSF to 0.0
house_prices_test.loc[house_prices_test.BsmtUnfSF.isnull(), ["BsmtUnfSF"]] = 0.0

# TotalBsmtSF
# Create indicator flag
house_prices_test["TotalBsmtSF_flag"] = 0
# Change values of TotalBsmtSF_flag to 1 where TotalBsmtSF is null
house_prices_test.loc[house_prices_test.TotalBsmtSF.isnull(), ["TotalBsmtSF_flag"]] = 1

#house_prices_test.loc[house_prices_test.TotalBsmtSF.isnull(), ["BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2"]]
# Change null values of BsmtUnfSF to 0.0
house_prices_test.loc[house_prices_test.TotalBsmtSF.isnull(), ["TotalBsmtSF"]] = 0.0

# BsmtFullBath and BsmtHalfBath
# Indicator flags
house_prices_test["BsmtFullBath_flag"] = 0
house_prices_test["BsmtHalfBath_flag"] = 0
house_prices_test.loc[(house_prices_test.BsmtFullBath.isnull()) & (house_prices_test.BsmtHalfBath.isnull()),
                      ["BsmtFullBath_flag", "BsmtHalfBath_flag"]] = [1,1]

house_prices_test.loc[(house_prices_test.BsmtFullBath.isnull()) & (house_prices_test.BsmtHalfBath.isnull()),
                      ["BsmtFullBath", "BsmtHalfBath"]] = [0,0]

# KitchenQual
# Indicator flags
house_prices_test["KitchenQual_flag"] = 0
house_prices_test.loc[house_prices_test.KitchenQual.isnull(), "KitchenQual_flag"] = 1

house_prices_test.loc[house_prices_test.KitchenQual.isnull(),
                      'KitchenQual'] = house_prices_test.loc[house_prices_test.KitchenQual.notnull(), 'KitchenQual'].mode().values[0]

# Functional
# Indicator Flags
create_indicator_flag(df=house_prices_test, column="Functional")

house_prices_test.loc[house_prices_test.Functional.isnull(),'Functional'] = "Typ"

# GarageType, GarageYrBlt, GarageFinish, GarageCars, GarageArea, GarageQual, GarageCond
# Indicator flags
# Create list of garage columns to loop through create_indicator_flag function
garage_cols = ["GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond"]
for value in garage_cols:
    create_indicator_flag(df=house_prices_test, column=value)

house_prices_test.loc[(house_prices_test.GarageType.isnull()) & (house_prices_test.GarageYrBlt.isnull()) &
                      (house_prices_test.GarageFinish.isnull()) & (house_prices_test.GarageQual.isnull()) &
                      (house_prices_test.GarageCond.isnull()),
                      ["GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond"]] = ["None", 0, "None", "None", "None"]

house_prices_test.loc[(house_prices_test.GarageYrBlt.isnull()) & (house_prices_test.GarageFinish.isnull()) &
                      (house_prices_test.GarageCars.isnull()) & (house_prices_test.GarageArea.isnull()) &
                      (house_prices_test.GarageQual.isnull()) & (house_prices_test.GarageCond.isnull()),
                      ["GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond"]] = [0, "None", 0, 0, "None", "None"]

house_prices_test.loc[(house_prices_test.GarageYrBlt.isnull()) & (house_prices_test.GarageFinish.isnull()) &
                      (house_prices_test.GarageQual.isnull()) & (house_prices_test.GarageCond.isnull()),
                      ["GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond"]] = [1934, "Unf", "TA", "TA"]

# house_prices_test.loc[(house_prices_test.GarageYrBlt.notnull()) & (house_prices_test.GarageCars==1) &
#                       (house_prices_test.GarageType=="Detchd") & (house_prices_test.GarageArea==360), "GarageYrBlt"].value_counts().head(3)
#
# house_prices_test.loc[(house_prices_test.GarageYrBlt.notnull()) & (house_prices_test.GarageCars==1) &
#                       (house_prices_test.GarageType=="Detchd") & (house_prices_test.GarageArea==360), "GarageFinish"].value_counts().head(3)

# Indicator flags
create_indicator_flag(df=house_prices_test,column="SaleType")

house_prices_test.loc[house_prices_test.SaleType.isnull(), "SaleType"] = house_prices_test.loc[house_prices_test.SaleType.notnull(), "SaleType"].mode().values[0]

# No columns with missing values in house_prices_test
determine_columns_with_missing_values(df=house_prices_test)

# Id - Check to make sure all rows represent a unique house
if len(house_prices_test.Id.unique()) == len(house_prices_test):
    print("All rows represnt a unique house.")
else:
    print("All rows DO NOT represent a unique house. Possible duplicate?")

### Determine types of variables - continuous, nominal (no order), ordinal (order) ###

continous_vars = ["LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
                  "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath",
                  "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
                  "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
                  "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold", "SalePrice"]

# Continous variables that could be ordinal: BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr,
# TotRmsAbvGrd, Fireplaces, GarageCars, MoSold, YrSold

ordinal_vars = ["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure",
                "BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond",
                "PoolQC"]

nominal_vars = ["MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope",
                "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",
                "Exterior2nd", "MasVnrType", "Foundation", "Heating", "CentralAir", "Electrical", "Functional",
                "GarageType", "GarageFinish", "PavedDrive", "Fence", "MiscFeature", "SaleType", "SaleCondition"]

# Nominal variables that could be ordinal: Fence

# Check levels of nominal variables (see if we can merge levels with small counts)
# house_prices_train.MSZoning.value_counts().sort_index().plot(kind='bar')

# MSZoning: group all residential into one
# Alley: Collapse Gravel and Paved into one (so alley access vs no alley access)
# LotShape: Regular vs. Irregular
# LandContour: Level vs. Non-Level
# LotConfig: Frontage on 2 or more sides (collapse FR2 and FR3)
# LandSlope: Collapse Moderate and Severe into 1 category
# Condition1 and Condition2: Collapse Artery and Feedr, RRNn and RRAn, PosN and PosA, RRNe and RRAe
# BldgType: Collapse Townhouse
# HouseStyle: Collapse 1.5 together, and 2.5 together
# RoofMatl: Collapse Wood together
# Functional: Group Maj1 and Maj2 into Maj (commented out below)

# Functional
# # Group Maj1 and Maj2 into Maj
# house_prices_train.loc[house_prices_train.Functional.isin(["Maj1", "Maj2"]), "Functional"] = "Maj"
# # Group Min1 and Min2 into Min
# house_prices_train.loc[house_prices_train.Functional.isin(["Min1", "Min2"]), "Functional"] = "Min"

# Id variable
id_var = ["Id"]

### train ###
# Indicator variables (used to denote column/row that was imputed)
indicator_vars_train = list(house_prices_train.filter(regex="_flag"))

# Check that all variables have been accounted for:
# Use function defined above

### train ###
indicator_vars_test = list(house_prices_test.filter(regex="_flag"))

# All present
all_variables_given_type(continous_vars=continous_vars, nominal_vars=nominal_vars, ordinal_vars=ordinal_vars,
                         indicator_vars=indicator_vars_train, id_var=id_var, df=house_prices_train)

### test ###

# All but SalePrice - no SalePrice in test set
all_variables_given_type(continous_vars=continous_vars, nominal_vars=nominal_vars, ordinal_vars=ordinal_vars,
                         indicator_vars=indicator_vars_test, id_var=id_var, df=house_prices_test)

#########################
### Variable Encoding ###
#########################
# Need to encode ordinal variables and one-hot encode nominal variables

### Ordinal Variables ###

# Create dict to map column None, Po, Fa, TA, Gd, Ex to numbers
none_to_ex_dict = {'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex': 5}

# Create dict for BsmtExposure
BsmtExposure_dict = {'None':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4}

# Create dict for BsmtFinType1 and BsmtFinType2
BsmtFinType_dict = {'None':0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}

# # Replace OverallQual and OverallCond values with one_to_ten_dict
# # Function to encode ordinal variables (create appropriate dict and replace values as appropriate)
# def one_to_ten_dict_apply(substted_df):
#     # Create dict to map column 1-10 to 0-9
#     one_to_ten_dict = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}
#     substted_df = substted_df.replace(one_to_ten_dict)
#     print(list(substted_df), "has been encoded")
#
# ### train ###
# one_to_ten_dict_apply(substted_df=house_prices_train[['OverallQual','OverallCond']])
#
# ### test ###
# one_to_ten_dict_apply(substted_df=house_prices_test[['OverallQual','OverallCond']])
#
# house_prices_train[['OverallQual','OverallCond']] = house_prices_train[['OverallQual','OverallCond']].replace(one_to_ten_dict)

### train ###
# Replace values with none_to_ex_dict
house_prices_train[['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC', 'KitchenQual', 'FireplaceQu',
                    'GarageQual', 'GarageCond', 'PoolQC']] = house_prices_train[['ExterQual','ExterCond','BsmtQual',
                                                                                 'BsmtCond','HeatingQC', 'KitchenQual',
                                                                                 'FireplaceQu', 'GarageQual', 'GarageCond',
                                                                                 'PoolQC']].replace(none_to_ex_dict)

# Replace values with BsmtExposure_dict
house_prices_train['BsmtExposure']=house_prices_train['BsmtExposure'].replace(BsmtExposure_dict)

# Replace values with BsmtFinType_dict
house_prices_train[['BsmtFinType1','BsmtFinType2']]=house_prices_train[['BsmtFinType1','BsmtFinType2']].replace(BsmtFinType_dict)

### test ###
# Replace values with none_to_ex_dict
house_prices_test[['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC', 'KitchenQual', 'FireplaceQu',
                    'GarageQual', 'GarageCond', 'PoolQC']] = house_prices_test[['ExterQual','ExterCond','BsmtQual',
                                                                                 'BsmtCond','HeatingQC', 'KitchenQual',
                                                                                 'FireplaceQu', 'GarageQual', 'GarageCond',
                                                                                 'PoolQC']].replace(none_to_ex_dict)

# Replace values with BsmtExposure_dict
house_prices_test['BsmtExposure']=house_prices_test['BsmtExposure'].replace(BsmtExposure_dict)

# Replace values with BsmtFinType_dict
house_prices_test[['BsmtFinType1','BsmtFinType2']]=house_prices_test[['BsmtFinType1','BsmtFinType2']].replace(BsmtFinType_dict)

### Keep copy of ordinal variables with original nominal variables for outlier analysis ###

# train
house_prices_train_with_nom = house_prices_train.copy(deep=True)

# test
house_prices_test_with_nom = house_prices_test.copy(deep=True)

### One-hot encode the nominal variables ###

# train #
# For loop to create dummy variables out of nominal variables
for var in nominal_vars:
    cat_list="var"+"_"+var
    cat_list = pd.get_dummies(data=house_prices_train[var], prefix=var)
    print(cat_list)
    df=house_prices_train.join(cat_list)
    house_prices_train=df

# Need to remove categorical_vars now since they've been turned into dummy variables
# Create list of all variables
house_prices_train_var=house_prices_train.columns.values.tolist()
# Remove categorical_vars using list comprehension
vars_to_keep=[var for var in house_prices_train_var if var not in nominal_vars]
print("Nominal variables have been one-hot encoded, and columns in nominal_vars have been removed from DataFrame.")

# Use list to get rid of nominal_vars variables
house_prices_train=house_prices_train[vars_to_keep]

# test #
# For loop to create dummy variables out of nominal variables
for var in nominal_vars:
    cat_list="var"+"_"+var
    cat_list = pd.get_dummies(data=house_prices_test[var], prefix=var)
    print(cat_list)
    df=house_prices_test.join(cat_list)
    house_prices_test=df

# Need to remove categorical_vars now since they've been turned into dummy variables
# Create list of all variables
house_prices_test_var=house_prices_test.columns.values.tolist()
# Remove categorical_vars using list comprehension
vars_to_keep=[var for var in house_prices_test_var if var not in nominal_vars]
print("Nominal variables have been one-hot encoded, and columns in nominal_vars have been removed from DataFrame.")

# Use list to get rid of nominal_vars variables
house_prices_test=house_prices_test[vars_to_keep]

### Replace all spaces with underscores ###
# train
house_prices_train.columns = house_prices_train.columns.str.replace(' ', "_")

# test
house_prices_test.columns = house_prices_test.columns.str.replace(' ', "_")

########################################
### Data Visualization/Understanding ###
########################################

# Visualize target variable (SalePrice)
# distplot can plot hist, kernel density, and rug plot at same time
# SalePrice is right-skewed (this makes sense as most people can't afford expensive home)
sns.distplot(a=house_prices_train['SalePrice'], hist=True, kde=True)

# Log transformation on SalePrice turns it into normal distribution - created LogSalePrice in Variable Creation section
sns.distplot(a=np.log(house_prices_train['SalePrice']), hist=True, kde=True)

# See correlations among continuous variables
house_prices_train_cont_ord_corr = house_prices_train[continous_vars+ordinal_vars].corr()

# Build heatmap of continuous variables with their correlations on them and visualize
sns.heatmap(house_prices_train_cont_ord_corr, cmap=sns.cm.rocket_r, annot=True, xticklabels=True, yticklabels=True)
plt.show()

# See correlations between continuous variables and SalePrice
house_prices_train_cont_ord_corr["SalePrice"].sort_values(ascending=False)

# Determine top correlations among variables (other than itself which would be the top associated variable since variable
# is 100% correlated with itself
# Create empty list to append results too (use extend to append more than one result to list)
# With for loop, find associated variable and then append variable, associated variable, and their correlation
correlations = []
for value in list(house_prices_train_cont_ord_corr):
    associated_value = abs(house_prices_train_cont_ord_corr[value]).nlargest(2)[1:2].index[0]
    correlations.extend([value, associated_value, house_prices_train_cont_ord_corr[value][associated_value]])

# Break out each 3 elements of list into its own list so DataFrame can be created
correlations = [correlations[x:x+3] for x in range(0, len(correlations), 3)]
# Build DataFrame from list of lists and name column appropriately
correlations = pd.DataFrame.from_records(correlations, columns=['variable', 'associated_variable', 'correlation'])
# Sort by highest POSITIVE correlation
correlations.sort_values(by='correlation', ascending=False)

# Function to build scatterplot, label x-axis and y-axis accordingly and correlation between the two variables
def scatter(variable, associated_variable):
    plt.scatter(x=house_prices_train[variable], y=house_prices_train[associated_variable])
    plt.xlabel(variable), plt.ylabel(associated_variable), \
    plt.legend(round(correlations.loc[(correlations.variable == variable) & (correlations.associated_variable == associated_variable)]['correlation'],3))
    plt.show()

# scatter('SalePrice', 'PoolArea')

# Jointplot builds "better" scatterplot - get regression line, confidence intervals, and distribution (along with
# kernel-density of both variables

# Build joint plot between SalePrice and GrLivArea
sns.jointplot(x="SalePrice",y="TotalBsmtSF",data=house_prices_train,kind='reg')

# Build joint plot between SalePrice and ScreenPorch - removing where ScreenPorch area is 0
sns.jointplot(x="SalePrice",y="ScreenPorch",data=house_prices_train.loc[house_prices_train.ScreenPorch!=0],kind='reg',
              xlim={house_prices_train.loc[house_prices_train.ScreenPorch!=0,"SalePrice"].min()-20000,
                    house_prices_train.loc[house_prices_train.ScreenPorch!=0,"SalePrice"].max()+20000})

# Build scatter plot to see relationships among ordinal and continuous variables
sns.boxplot(x="GarageCars",y="SalePrice",data=house_prices_train)

# Build scatter plot to for continuous variable
sns.boxplot(x="GarageArea",data=house_prices_train)

# Describe int/float columns
house_prices_train.describe()

# YearBuilt
house_prices_train.YearBuilt.value_counts().sort_values(ascending=False)
# Plot YearBuilt value counts
house_prices_train.YearBuilt.value_counts().sort_index(ascending=True).plot(kind="bar")

# YearRemodAdd
house_prices_train.YearRemodAdd.value_counts().sort_values(ascending=False)

# LotFrontage
house_prices_train.LotFrontage.value_counts().sort_index(ascending=True).plot(kind="bar")

# # Street
# house_prices_train.Street.value_counts()
#
# # Alley
# house_prices_train.Alley.value_counts()
#
# # LotShape
# house_prices_train.LotShape.value_counts()
#
# # LandContour
# house_prices_train.LandContour.value_counts().plot(kind='bar')
#
# # MSZoning - C(all)? Ask Stephen
# house_prices_train.MSZoning.value_counts().plot(kind="bar")

# # Utilities
# house_prices_train.Utilities.value_counts()
#
# # LotConfig
# house_prices_train.LotConfig.value_counts()
#
# # LandSlope
# house_prices_train.LandSlope.value_counts()
#
# # Neighborhood
# house_prices_train.Neighborhood.value_counts()
#
# # BldgType
# house_prices_train.BldgType.value_counts()
#
# # MSSubClass
# house_prices_train.MSSubClass.value_counts().plot(kind="bar")
#
# # HouseStyle
# house_prices_train.HouseStyle.value_counts()

#########################
### Variable Creation ###
#########################
# Create copy to fix SettingWithCopyWarning
house_prices_train = house_prices_train.copy()
# Log of SalePrice
house_prices_train.loc[:,'LogSalePrice'] = np.log(house_prices_train['SalePrice'])

# Create column indicating years from build date to remodel date
# house_prices_train["Built_Remod_Diff"] = house_prices_train["YearRemodAdd"] - house_prices_train["YearBuilt"]

# Season when house sold

# Age of home

# Total square foot in home

# Total number of "extras" goodies (screen porch, pool, etc.)


######################
### Model Building ###
######################
### Make copy of DataFrame (so don't have to constantly reload in DataFrame) ###
# train #
house_prices_train_copy = house_prices_train.copy(deep=True)
# test #
house_prices_test_copy = house_prices_test.copy(deep=True)

############################################
### Run Lasso Regression Model from here ###
############################################

#################
### Model One ###
#################

### train ###
# ID variable
Id = "Id"
# Target variable
SalePrice = "SalePrice"
# Create list of columns not to include in standardize
### "GarageCond", "GarageCars", "GarageQual", "PoolQC"
vars_not_to_include = indicator_vars_train + ["LogSalePrice"]
# Extend Id and Sale onto vars_not_to_include (extend is an in-place function)
vars_not_to_include.extend([Id, SalePrice])
# Get list of predictor variables
train_x = [col for col in list(house_prices_train) if col not in vars_not_to_include]
# Get predictor columns from DataFrame and set as train_x
train_x = house_prices_train[train_x]
# Set target variable - set target as LogSalePrice ##########################################
train_y = house_prices_train["SalePrice"]

# Obtain colnames of train_x for lasso_coef plot
colnames=list(train_x)

# # Standardize dependent variables (train_x) for LASSO
# Set scaler
scaler = preprocessing.MinMaxScaler()
# Compute the mean and std to be used for later scaling if needed
scaler.fit(train_x)
# Perform standardization on train_x by centering and scaling
train_x=scaler.transform(train_x)

### Determine optimal value of alpha ###
### LassoCV: coordinate descent

# Compute paths
print("Computing regularization path using the coordinate descent lasso...")
t1 = time.time()
model_lasso_cv = LassoCV(cv=20, fit_intercept=False, normalize=False, n_alphas=100, max_iter=1000, tol=0.00001).fit(train_x, train_y.values.ravel())
t_lasso_cv = time.time() - t1

# Display results
m_log_alphas = -np.log10(model_lasso_cv.alphas_)

print("Optimal value based off LassoCV:", model_lasso_cv.alpha_,"\n")

plt.figure()
#ymin, ymax = 0, 1
plt.plot(m_log_alphas, model_lasso_cv.mse_path_, ':')
plt.plot(m_log_alphas, model_lasso_cv.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model_lasso_cv.alpha_), linestyle='--', color='k',
            label='alpha: CV estimate')

plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: coordinate descent '
          '(train time: %.2fs)' % t_lasso_cv)
plt.axis('tight')
#plt.ylim(ymin, ymax)


# ### LassoLarsIC: least angle regression with BIC/AIC criterion
#
# model_bic = LassoLarsIC(criterion='bic', fit_intercept=False, normalize=False, max_iter=1000)
# t1 = time.time()
# model_bic.fit(train_x, train_y.values.ravel())
# t_bic = time.time() - t1
# alpha_bic_ = model_bic.alpha_
#
# model_aic = LassoLarsIC(criterion='aic', fit_intercept=False, normalize=False, max_iter=1000)
# model_aic.fit(train_x, train_y.values.ravel())
# alpha_aic_ = model_aic.alpha_
#
# def plot_ic_criterion(model, name, color):
#     alpha_ = model.alpha_
#     alphas_ = model.alphas_
#     criterion_ = model.criterion_
#     plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
#              linewidth=3, label='%s criterion' % name)
#     plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
#                 label='alpha: %s estimate' % name)
#     plt.xlabel('-log(alpha)')
#     plt.ylabel('criterion')
#
# plt.figure()
# plot_ic_criterion(model_aic, 'AIC', 'b')
# plot_ic_criterion(model_bic, 'BIC', 'r')
# plt.legend()
# plt.title('Information-criterion for model selection (training time %.3fs)'
#           % t_bic)
#
#
# ### LassoLarsCV: Least Angle Regression ###
# # Compute paths
# print("Computing regularization path using the Lars lasso...")
# t1 = time.time()
# model_lasso_lars_cv = LassoLarsCV(cv=20, fit_intercept=False, normalize=False, max_iter=1000).fit(train_x, train_y.values.ravel())
# t_lasso_lars_cv = time.time() - t1
#
# # Display results
# m_log_alphas = -np.log10(model_lasso_lars_cv.cv_alphas_)
#
# plt.figure()
# plt.plot(m_log_alphas, model_lasso_lars_cv.mse_path_, ':')
# plt.plot(m_log_alphas, model_lasso_lars_cv.mse_path_.mean(axis=-1), 'k',
#          label='Average across the folds', linewidth=2)
# plt.axvline(-np.log10(model_lasso_lars_cv.alpha_), linestyle='--', color='k',
#             label='alpha CV')
# plt.legend()
#
# plt.xlabel('-log(alpha)')
# plt.ylabel('Mean square error')
# plt.title('Mean square error on each fold: Lars (train time: %.2fs)'
#           % t_lasso_lars_cv)
# plt.axis('tight')
# # plt.ylim(ymin, ymax)
#
# plt.show()

### Run lasso regression with lasso_cv alpha ###
lasso = Lasso(alpha=model_lasso_cv.alpha_, fit_intercept=False, max_iter=10000, tol=0.00001)
# Second (outer) cross-validation
# Returns average loss function
cross_val_score(lasso, train_x, train_y.values.ravel(), cv=5)

# Fit the lasso model
lasso.fit(train_x, train_y.values.ravel())

# Obtain values of lasso coefficients
lasso_coef = lasso.coef_
# Create DataFrame with dependent variables and their lasso coefficients
lasso_coef_df = pd.DataFrame({'Features':colnames, 'Lasso Coefficient Value':lasso_coef.tolist()}).sort_values(by="Lasso Coefficient Value",ascending=False)
print(lasso_coef_df)

# Create plot of dependent variables and their lasso coefficients
plt.figure()
plt.plot(np.array(range(len(colnames))), np.array(lasso_coef))
plt.xticks(range(len(colnames)), list(colnames), rotation=60)
plt.margins(0.02)
plt.show()

# Predict using the lasso model
predict = lasso.predict(train_x)

# Convert train_y (SalePrice) into DataFrame
sale_price_with_predict = train_y.to_frame()
# Add predictions as column
sale_price_with_predict["Prediction"] = predict.tolist()
# See absolute value in difference between actual and predicted sales prices
sale_price_with_predict["Difference"] = np.abs(sale_price_with_predict[train_y.name] - sale_price_with_predict["Prediction"])

# Calculate root mean square error
root_mean_square_error = sqrt((sale_price_with_predict["Difference"]**2).sum()/len(sale_price_with_predict))
print("Root Mean Square Error Value:", root_mean_square_error)

# Visualize at which SalePrice the predictions are off the most/least
plt.figure()
sns.barplot(x=train_y.name,y="Difference",data=sale_price_with_predict.sort_values(by=train_y.name,ascending=True))

#####################
### Create output ###
#####################
number_of_observations = ["Number of Observations:", len(house_prices_train)]
target_variable = ["Target Variable:", train_y.name]
standardization = ["Standardization Used:", "MinMaxScaler (Normalization)"]
alpha_selection_process = ["Alpha Selection Process:", "LassoCV"]
alpha = ["Alpha value:", model_lasso_cv.alpha_]
rmse = ["Root mean square error (on train set):", root_mean_square_error]
multicollinearity = ["Remove correlated variables:", "No"]
outliers = ["Remove outliers:", "No"]
skewness = ["Fix skewness:", "No"]
feature_engineering = ["Perform feature engineering:", "No"]
space = ['','']

# Create DataFrame of model information
model_informaton = pd.DataFrame([number_of_observations,target_variable,standardization,alpha_selection_process,alpha,
                                 rmse,multicollinearity,outliers,skewness,feature_engineering,space],
                                columns=['Information', 'Value'])

# Record model number to update csv readout title
model = 'one'
### Create CSV of output ###
# Create csv with model_information
model_informaton.to_csv("lasso_model_"+model+".csv",sep=',',index=False)

# Append lasso coefficient values to csv
lasso_coef_df.to_csv("lasso_model_"+model+".csv",sep=',',mode='a',index=False)

pd.DataFrame([], columns=['', '']).to_csv("lasso_model_"+model+".csv",sep=',',mode='a',index=False)

# Append DataFrame of sale price, prediction, and difference to csv
sale_price_with_predict.to_csv("lasso_model_"+model+".csv",sep=',',mode='a',index=False)

##################
### Model Four ###
##################

### Make copy of DataFrame (so don't have to constantly reload in DataFrame) ###
# train #
house_prices_train_copy = house_prices_train.copy(deep=True)
# test #
house_prices_test_copy = house_prices_test.copy(deep=True)

# train #
house_prices_train = house_prices_train_copy.copy(deep=True)
# test #
house_prices_test = house_prices_test_copy.copy(deep=True)

### For model two, put train_y as LogSalePrice ###

# ID variable
Id = "Id"
# Target variable
SalePrice = "LogSalePrice"
# Create list of columns not to include in standardize
### "GarageCond", "GarageCars", "GarageQual", "PoolQC"
vars_not_to_include = indicator_vars_train + ["SalePrice"] + list(house_prices_train.filter(regex="Utilities_")) + \
                      list(house_prices_train.filter(regex="Street_"))
# Extend Id and Sale onto vars_not_to_include (extend is an in-place function)
vars_not_to_include.extend([Id, SalePrice])

# train
# Get list of predictor variables
train_x = [col for col in list(house_prices_train) if col not in vars_not_to_include]
# Get predictor columns from DataFrame and set as train_x
train_x = house_prices_train[train_x]

# test
# Get list of predictor variables
test_x = [col for col in list(house_prices_test) if col not in vars_not_to_include]
# Get predictor columns from DataFrame and set as test_x
test_x = house_prices_test[test_x]

# Prepare train set to only include variables present in test set (figured out on 9/23/2018, there are levels of categorical
# variables in the train set that are not present in the test set)
# Categorical variables present in train dataset but not test dataset
vars_in_train_not_in_test = list(set(list(train_x))^set(list(test_x)))

# train
train_x = train_x.drop(vars_in_train_not_in_test,axis=1,errors='ignore')

# test
test_x = test_x.drop(vars_in_train_not_in_test,axis=1,errors='ignore')

# Set target variable for train - set target as LogSalePrice #
train_y = house_prices_train["LogSalePrice"]

# Obtain colnames of train_x for lasso_coef plot
colnames=list(train_x)

# # Standardize dependent variables (train_x) for LASSO
# Set scaler
scaler = preprocessing.MinMaxScaler()
# Compute the mean and std to be used for later scaling if needed
scaler.fit(train_x)
# Perform standardization on train_x by centering and scaling
train_x=scaler.transform(train_x)

### Determine optimal value of alpha ###
### LassoCV: coordinate descent

# Compute paths
print("Computing regularization path using the coordinate descent lasso...\n")
t1 = time.time()
model_lasso_cv = LassoCV(cv=20, fit_intercept=False, normalize=False, n_alphas=100, max_iter=1000, tol=0.00001).fit(train_x, train_y.values.ravel())
t_lasso_cv = time.time() - t1

# Display results
m_log_alphas = -np.log10(model_lasso_cv.alphas_)

print("Optimal value based off LassoCV:", model_lasso_cv.alpha_,"\n")

plt.figure()
#ymin, ymax = 0, 1
plt.plot(m_log_alphas, model_lasso_cv.mse_path_, ':')
plt.plot(m_log_alphas, model_lasso_cv.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model_lasso_cv.alpha_), linestyle='--', color='k',
            label='alpha: CV estimate')

plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: coordinate descent '
          '(train time: %.2fs)' % t_lasso_cv)
plt.axis('tight')
#plt.ylim(ymin, ymax)

### Run lasso regression with lasso_cv alpha ###
lasso = Lasso(alpha=model_lasso_cv.alpha_, fit_intercept=False, max_iter=10000, tol=0.00001)
# Second (outer) cross-validation
# Returns average loss function
cross_val_score(lasso, train_x, train_y.values.ravel(), cv=5)

# Fit the lasso model
lasso.fit(train_x, train_y.values.ravel())

# Obtain values of lasso coefficients
lasso_coef = lasso.coef_
# Create DataFrame with dependent variables and their lasso coefficients
lasso_coef_df_train = pd.DataFrame({'Features':colnames, 'Lasso Coefficient Value':lasso_coef.tolist()}).sort_values(by="Lasso Coefficient Value",ascending=False)
print(lasso_coef_df_train)

# Create plot of dependent variables and their lasso coefficients
plt.figure()
plt.plot(np.array(range(len(colnames))), np.array(lasso_coef))
plt.xticks(range(len(colnames)), list(colnames), rotation=60)
plt.margins(0.02)
plt.show()

# Predict using the lasso model on the train set
predict = lasso.predict(train_x)

# Convert train_y (SalePrice) into DataFrame
sale_price_with_predict = train_y.to_frame()
# Add predictions as column
sale_price_with_predict["Prediction"] = predict.tolist()
# See absolute value in difference between actual and predicted sales prices
sale_price_with_predict["Difference"] = np.abs(sale_price_with_predict[train_y.name] - sale_price_with_predict["Prediction"])

# Calculate root mean square error
root_mean_square_error = sqrt((sale_price_with_predict["Difference"]**2).sum()/len(sale_price_with_predict))
print("Root Mean Square Error Value:", root_mean_square_error)

# Visualize at which SalePrice the predictions are off the most/least
plt.figure()
sns.barplot(x=train_y.name,y="Difference",data=sale_price_with_predict.sort_values(by=train_y.name,ascending=True))

### Predict on test set ###
# Obtain colnames of train_x for lasso_coef plot
colnames=list(test_x)

# # Standardize dependent variables (test_x) for LASSO
# Set scaler
scaler = preprocessing.MinMaxScaler()
# Compute the mean and std to be used for later scaling if needed
scaler.fit(test_x)
# Perform standardization on train_x by centering and scaling
test_x=scaler.transform(test_x)

# Predict using the lasso model on the test set
predict_test = lasso.predict(test_x)
print(predict_test)

# Create column of predicted SalePrice for test set
house_prices_test["SalePrice"] = list(np.exp(predict_test))

# Save to CSV
house_prices_test[["Id", "SalePrice"]].to_csv("house_price_test_prediction.csv",sep=",", index=False)

#####################
### Create output ###
#####################
# Record model number to update csv readout title
model = 'four'

number_of_observations = ["Number of Observations:", len(house_prices_train)]
target_variable = ["Target Variable:", train_y.name]
standardization = ["Standardization Used:", "MinMaxScaler (Normalization)"]
alpha_selection_process = ["Alpha Selection Process:", "LassoCV"]
alpha = ["Alpha value:", model_lasso_cv.alpha_]
rmse = ["Root mean square error (on train set):", root_mean_square_error]
multicollinearity = ["Remove correlated variables:", "No"]
outliers = ["Remove outliers:", "No"]
skewness = ["Fix skewness:", "No"]
feature_engineering = ["Perform feature engineering:", "No"]
space = ['','']

# Create DataFrame of model information
model_informaton = pd.DataFrame([number_of_observations,target_variable,standardization,alpha_selection_process,alpha,
                                 rmse,multicollinearity,outliers,skewness,feature_engineering,space],
                                columns=['Information', 'Value'])

### Create CSV of output ###
# Create csv with model_information
model_informaton.to_csv("lasso_model_"+model+".csv",sep=',',index=False)

# Append lasso coefficient values to csv
lasso_coef_df_train.to_csv("lasso_model_"+model+".csv",sep=',',mode='a',index=False)

# Create space
pd.DataFrame([], columns=['', '']).to_csv("lasso_model_"+model+".csv",sep=',',mode='a',index=False)

# Append DataFrame of sale price, prediction, and difference to csv
sale_price_with_predict.to_csv("lasso_model_"+model+".csv",sep=',',mode='a',index=True)

# Create space
pd.DataFrame([], columns=['', '']).to_csv("lasso_model_"+model+".csv",sep=',',mode='a',index=False)

house_prices_test[["Id", "SalePrice"]].to_csv("lasso_model_"+model+".csv",sep=',',mode='a',index=False)

#######################
### Model Two/Three ###
# Model Two contained Utilities with two categorical values - test was not able to make predictions due to only have
# one categorical value for Utilities
#######################

### Make copy of DataFrame (so don't have to constantly reload in DataFrame) ###
# train #
house_prices_train_copy = house_prices_train.copy(deep=True)
# test #
house_prices_test_copy = house_prices_test.copy(deep=True)

# train #
house_prices_train = house_prices_train_copy.copy(deep=True)
# test #
house_prices_test = house_prices_test_copy.copy(deep=True)

### For model two, put train_y as LogSalePrice ###

# ID variable
Id = "Id"
# Target variable
SalePrice = "LogSalePrice"
# Create list of columns not to include in standardize
### "GarageCond", "GarageCars", "GarageQual", "PoolQC"
vars_not_to_include = indicator_vars_train + ["SalePrice"] + list(house_prices_train.filter(regex="Utilities_"))
# Extend Id and Sale onto vars_not_to_include (extend is an in-place function)
vars_not_to_include.extend([Id, SalePrice])

# train
# Get list of predictor variables
train_x = [col for col in list(house_prices_train) if col not in vars_not_to_include]
# Get predictor columns from DataFrame and set as train_x
train_x = house_prices_train[train_x]

# test
# Get list of predictor variables
test_x = [col for col in list(house_prices_test) if col not in vars_not_to_include]
# Get predictor columns from DataFrame and set as test_x
test_x = house_prices_test[test_x]

# Prepare train set to only include variables present in test set (figured out on 9/23/2018, there are levels of categorical
# variables in the train set that are not present in the test set)
# Categorical variables present in train dataset but not test dataset
vars_in_train_not_in_test = list(set(list(train_x))^set(list(test_x)))

# train
train_x = train_x.drop(vars_in_train_not_in_test,axis=1,errors='ignore')

# test
test_x = test_x.drop(vars_in_train_not_in_test,axis=1,errors='ignore')

# Set target variable for train - set target as LogSalePrice #
train_y = house_prices_train["LogSalePrice"]

# Obtain colnames of train_x for lasso_coef plot
colnames=list(train_x)

# # Standardize dependent variables (train_x) for LASSO
# Set scaler
scaler = preprocessing.MinMaxScaler()
# Compute the mean and std to be used for later scaling if needed
scaler.fit(train_x)
# Perform standardization on train_x by centering and scaling
train_x=scaler.transform(train_x)

### Determine optimal value of alpha ###
### LassoCV: coordinate descent

# Compute paths
print("Computing regularization path using the coordinate descent lasso...\n")
t1 = time.time()
model_lasso_cv = LassoCV(cv=20, fit_intercept=False, normalize=False, n_alphas=100, max_iter=1000, tol=0.00001).fit(train_x, train_y.values.ravel())
t_lasso_cv = time.time() - t1

# Display results
m_log_alphas = -np.log10(model_lasso_cv.alphas_)

print("Optimal value based off LassoCV:", model_lasso_cv.alpha_,"\n")

plt.figure()
#ymin, ymax = 0, 1
plt.plot(m_log_alphas, model_lasso_cv.mse_path_, ':')
plt.plot(m_log_alphas, model_lasso_cv.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model_lasso_cv.alpha_), linestyle='--', color='k',
            label='alpha: CV estimate')

plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: coordinate descent '
          '(train time: %.2fs)' % t_lasso_cv)
plt.axis('tight')
#plt.ylim(ymin, ymax)

### Run lasso regression with lasso_cv alpha ###
lasso = Lasso(alpha=model_lasso_cv.alpha_, fit_intercept=False, max_iter=10000, tol=0.00001)
# Second (outer) cross-validation
# Returns average loss function
cross_val_score(lasso, train_x, train_y.values.ravel(), cv=5)

# Fit the lasso model
lasso.fit(train_x, train_y.values.ravel())

# Obtain values of lasso coefficients
lasso_coef = lasso.coef_
# Create DataFrame with dependent variables and their lasso coefficients
lasso_coef_df_train = pd.DataFrame({'Features':colnames, 'Lasso Coefficient Value':lasso_coef.tolist()}).sort_values(by="Lasso Coefficient Value",ascending=False)
print(lasso_coef_df_train)

# Create plot of dependent variables and their lasso coefficients
plt.figure()
plt.plot(np.array(range(len(colnames))), np.array(lasso_coef))
plt.xticks(range(len(colnames)), list(colnames), rotation=60)
plt.margins(0.02)
plt.show()

# Predict using the lasso model on the train set
predict = lasso.predict(train_x)

# Convert train_y (SalePrice) into DataFrame
sale_price_with_predict = train_y.to_frame()
# Add predictions as column
sale_price_with_predict["Prediction"] = predict.tolist()
# See absolute value in difference between actual and predicted sales prices
sale_price_with_predict["Difference"] = np.abs(sale_price_with_predict[train_y.name] - sale_price_with_predict["Prediction"])

# Calculate root mean square error
root_mean_square_error = sqrt((sale_price_with_predict["Difference"]**2).sum()/len(sale_price_with_predict))
print("Root Mean Square Error Value:", root_mean_square_error)

# Visualize at which SalePrice the predictions are off the most/least
plt.figure()
sns.barplot(x=train_y.name,y="Difference",data=sale_price_with_predict.sort_values(by=train_y.name,ascending=True))

### Predict on test set ###
# Obtain colnames of train_x for lasso_coef plot
colnames=list(test_x)

# # Standardize dependent variables (test_x) for LASSO
# Set scaler
scaler = preprocessing.MinMaxScaler()
# Compute the mean and std to be used for later scaling if needed
scaler.fit(test_x)
# Perform standardization on train_x by centering and scaling
test_x=scaler.transform(test_x)

# Predict using the lasso model on the test set
predict_test = lasso.predict(test_x)
print(predict_test)

# Create column of predicted SalePrice for test set
house_prices_test["SalePrice"] = list(np.exp(predict_test))

# Save to CSV
house_prices_test[["Id", "SalePrice"]].to_csv("house_price_test_prediction.csv",sep=",", index=False)

#####################
### Create output ###
#####################
# Record model number to update csv readout title
model = 'three'

number_of_observations = ["Number of Observations:", len(house_prices_train)]
target_variable = ["Target Variable:", train_y.name]
standardization = ["Standardization Used:", "MinMaxScaler (Normalization)"]
alpha_selection_process = ["Alpha Selection Process:", "LassoCV"]
alpha = ["Alpha value:", model_lasso_cv.alpha_]
rmse = ["Root mean square error (on train set):", root_mean_square_error]
multicollinearity = ["Remove correlated variables:", "No"]
outliers = ["Remove outliers:", "No"]
skewness = ["Fix skewness:", "No"]
feature_engineering = ["Perform feature engineering:", "No"]
space = ['','']

# Create DataFrame of model information
model_informaton = pd.DataFrame([number_of_observations,target_variable,standardization,alpha_selection_process,alpha,
                                 rmse,multicollinearity,outliers,skewness,feature_engineering,space],
                                columns=['Information', 'Value'])

### Create CSV of output ###
# Create csv with model_information
model_informaton.to_csv("lasso_model_"+model+".csv",sep=',',index=False)

# Append lasso coefficient values to csv
lasso_coef_df_train.to_csv("lasso_model_"+model+".csv",sep=',',mode='a',index=False)

# Create space
pd.DataFrame([], columns=['', '']).to_csv("lasso_model_"+model+".csv",sep=',',mode='a',index=False)

# Append DataFrame of sale price, prediction, and difference to csv
sale_price_with_predict.to_csv("lasso_model_"+model+".csv",sep=',',mode='a',index=True)

# Create space
pd.DataFrame([], columns=['', '']).to_csv("lasso_model_"+model+".csv",sep=',',mode='a',index=False)

house_prices_test[["Id", "SalePrice"]].to_csv("lasso_model_"+model+".csv",sep=',',mode='a',index=False)


###################
### Model Five ####
###################

### For model three, remove outliers ###

### Can make a category that all outlier points belong to ###

### Can use model fit visuals to detect outliers ### - Cook's D, DFFITs (2-11,2-14 in Statistics 2), one-class SVM model

### Outlier Visual Detection ###

# Boxplot of SalePrice - outliers are apparent (>1.5 IQR), especially two end points
plt.figure()
sns.boxplot(x=house_prices_train['SalePrice'])
# Q-Q plot of SalePrice - same two endpoints are apparent in q-q plot. Obvious skewness as well
plt.figure()
stats.probplot(x=house_prices_train['SalePrice'], plot=plt)
# Obtain skewness value
print("Skewness of SalePrice",stats.skew(a=house_prices_train['SalePrice']))
# Obtain kurtosis value
print("Kurtosis of SalePrice",stats.kurtosis(a=house_prices_train['SalePrice']))

# Boxplot of SalePrice - outliers are apparent (>1.5 IQR), especially two end points
plt.figure()
sns.boxplot(x=house_prices_train['LogSalePrice'])
# Q-Q plot of SalePrice - same two endpoints are apparent in q-q plot. Obvious skewness as well
plt.figure()
stats.probplot(x=house_prices_train['LogSalePrice'], plot=plt)
# Obtain skewness value
print("Skewness of LogSalePrice",stats.skew(a=house_prices_train['LogSalePrice']))
# Obtain kurtosis value
print("Kurtosis of LogSalePrice",stats.kurtosis(a=house_prices_train['LogSalePrice']))

#########################################################################################################################
### For visually inspecting outliers, create a column indicating if value was identifed as outlier based on that plot ###
### Those rows identified most will be removed ###
### Only removing most severe observations as outliers to start with (7/22/18) ###
#########################################################################################################################

### Understand this
sns.pairplot(data=house_prices_train[continous_vars+ordinal_vars],kind='scatter',diag_kind='hist')

# See variables with highest correlations with SalePrice
house_prices_train_cont_ord_corr["SalePrice"].sort_values(ascending=False)

# Build joint plot between SalePrice and GrLivArea - 2 houses with high GrLivArea but didn't sell for much
sns.jointplot(x="SalePrice",y="GrLivArea",data=house_prices_train,kind='reg')

# Create binary flag column indicating row has been identifed as outlier-create all zeroes and then value to 1 if identified as outlier
house_prices_train["GrLivArea_outlier"] = 0
# Put one in binary flag column indicating the row is an outlier
house_prices_train.loc[(house_prices_train.GrLivArea>4000) & (house_prices_train.SalePrice<200000), "GrLivArea_outlier"] = 1

# Build joint plot between SalePrice and GarageArea - 1 with large GarageArea that didn't sell for much
sns.jointplot(x="SalePrice",y="GarageArea",data=house_prices_train,kind='reg')

# Create binary flag column indicating row has been identifed as outlier-create all zeroes and then value to 1 if identified as outlier
house_prices_train["GarageArea_outlier"] = 0
# Put one in binary flag column indicating the row is an outlier
house_prices_train.loc[(house_prices_train.GarageArea>1200) & (house_prices_train.SalePrice<300000), "GarageArea_outlier"] = 1
# Put one in binary flag column indicating the row is an outlier
house_prices_train.loc[(house_prices_train.GarageArea<1000) & (house_prices_train.SalePrice>700000), "GarageArea_outlier"] = 1

# Build joint plot between SalePrice and TotalBsmtSF - 1 with large TotalBsmtSF that didn't sell for much
sns.jointplot(x="SalePrice",y="TotalBsmtSF",data=house_prices_train,kind='reg')

# Create binary flag column indicating row has been identifed as outlier-create all zeroes and then value to 1 if identified as outlier
house_prices_train["TotalBsmtSF_outlier"] = 0
# Put one in binary flag column indicating the row is an outlier
house_prices_train.loc[(house_prices_train.TotalBsmtSF>6000) & (house_prices_train.SalePrice<200000), "TotalBsmtSF_outlier"] = 1

### Start here post 8/5
# Build joint plot between SalePrice and 1stFlrSF - 1 with large 1stFlrSF that didn't sell for much
sns.jointplot(x="SalePrice",y="1stFlrSF",data=house_prices_train,kind='reg')

# Create binary flag column indicating row has been identifed as outlier-create all zeroes and then value to 1 if identified as outlier
house_prices_train["1stFlrSF_outlier"] = 0
# Put one in binary flag column indicating the row is an outlier
house_prices_train.loc[(house_prices_train['1stFlrSF']>4000) & (house_prices_train.SalePrice<200000), "1stFlrSF_outlier"] = 1


# Function requires less typing
def boxplot(x, y):
    sns.boxplot(x=x,y=y,data=house_prices_train)

# Boxplot between OverallQual and SalePrice - outliers present (1 for OverallQual3, 2 for QverallQual6, 1 for OQ7, 3 for OQ8, 1 for OQ9
boxplot(x="OverallQual",y="SalePrice")

# Create binary flag column indicating row has been identifed as outlier-create all zeroes and then value to 1 if identified as outlier
house_prices_train["OverallQual_outlier"] = 0
# Put one in binary flag column indicating the row is an outlier
house_prices_train.loc[(house_prices_train.OverallQual==3) & (house_prices_train.SalePrice>200000), "OverallQual_outlier"] = 1
house_prices_train.loc[(house_prices_train.OverallQual==7) & (house_prices_train.SalePrice>500000), "OverallQual_outlier"] = 1
house_prices_train.loc[(house_prices_train.OverallQual==8) & (house_prices_train.SalePrice>550000), "OverallQual_outlier"] = 1
house_prices_train.loc[(house_prices_train.OverallQual==9) & (house_prices_train.SalePrice>700000), "OverallQual_outlier"] = 1

# Boxplot between ExterQual and SalePrice - outliers present - 1 for 2, 3 for 3, 1 for 5
boxplot(x="ExterQual",y="SalePrice")

# Create binary flag column indicating row has been identifed as outlier-create all zeroes and then value to 1 if identified as outlier
house_prices_train["ExterQual_outlier"] = 0
# Put one in binary flag column indicating the row is an outlier
house_prices_train.loc[(house_prices_train.ExterQual==2) & (house_prices_train.SalePrice>190000), "ExterQual_outlier"] = 1
house_prices_train.loc[(house_prices_train.ExterQual==4) & (house_prices_train.SalePrice>500000), "ExterQual_outlier"] = 1
house_prices_train.loc[(house_prices_train.ExterQual==5) & (house_prices_train.SalePrice>700000), "ExterQual_outlier"] = 1

# Boxplot between KitchenQual and SalePrice - outliers present - 1 for 2, 3 for 3, 1 for 5
boxplot(x="KitchenQual",y="SalePrice")

# Create binary flag column indicating row has been identifed as outlier-create all zeroes and then value to 1 if identified as outlier
house_prices_train["KitchenQual_outlier"] = 0
# Put one in binary flag column indicating the row is an outlier
house_prices_train.loc[(house_prices_train.KitchenQual==3) & (house_prices_train.SalePrice>320000), "KitchenQual_outlier"] = 1
house_prices_train.loc[(house_prices_train.KitchenQual==4) & (house_prices_train.SalePrice>500000), "KitchenQual_outlier"] = 1
house_prices_train.loc[(house_prices_train.KitchenQual==5) & (house_prices_train.SalePrice>700000), "KitchenQual_outlier"] = 1


# Boxplot between GarageCars and SalePrice - outliers present - 1 for 2, 2 for 3, 1 for 4
boxplot(x="GarageCars",y="SalePrice")

# Boxplot between BsmtQual and SalePrice - outliers present - 1 for 3, 1 for 4, 2 for 5
boxplot(x="BsmtQual",y="SalePrice")

# Boxplot between FullBath and SalePrice - outliers present - 1 for 0, 3 for 1, 2 for 2
boxplot(x="FullBath",y="SalePrice")

# Boxplot between TotRmsAbvGrd and SalePrice - outliers present - 2 for 5, 1 for 6, 2 for 7, 1 for 8, 1 for 9, 2 for 10
boxplot(x="TotRmsAbvGrd",y="SalePrice")

# Boxplot between FireplaceQu and SalePrice - outliers present but variable has only ~0.51 correlation with target
boxplot(x="FireplaceQu",y="SalePrice")

### Lets check out nominal vars for outliers ###

# Boxplot between Utilities and SalePrice - two high priced houses show up again
sns.boxplot(x="Utilities",y="SalePrice",data=house_prices_train_with_nom)

# Boxplot between Street and SalePrice - two high priced houses show up again
sns.boxplot(x="Street",y="SalePrice",data=house_prices_train_with_nom)

# Boxplot between Condition2 and SalePrice - two high priced houses show up again
sns.boxplot(x="Condition2",y="SalePrice",data=house_prices_train_with_nom)

# Boxplot between Condition2 and SalePrice - two high priced houses show up again
sns.boxplot(x="RoofMatl",y="SalePrice",data=house_prices_train_with_nom)

# Boxplot between Condition2 and SalePrice - two high priced houses show up again
sns.boxplot(x="Heating",y="SalePrice",data=house_prices_train_with_nom)

# Boxplot between Functional and SalePrice - 1 in Min1, 1 in Maj2, 1 in Min2, 1 in Mod
sns.boxplot(x="Functional",y="SalePrice",data=house_prices_train_with_nom)

## Stop at Functional for now and see how model does































# Taking one level out of each nominal variable to dummy code - choose lowest level
# Create list of nominal variables that to be one hot encoded
nom_cols = ["MSSubClass", "MSZoning", "Alley", "LotShape", "LandContour", "LotConfig", "LandSlope", "Neighborhood", "Condition1",
            "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating",
            "Electrical", "GarageType", "MiscFeature", "SaleType", "SaleCondition"]
# Use for loop to see value_counts for variables
for value in nom_cols:
    print(value, '\n')
    print(house_prices_train_copy[value].value_counts().sort_values(ascending=False), '\n')





# Determine correlations among remaining garage variables (causing convergence issues)
garage_corr = house_prices_train[['GarageYrBlt', 'GarageFinish', 'GarageArea', 'GarageQual']].corr()

# Build heatmap of remaining garage variables with their correlations on them and visualize
sns.heatmap(garage_corr, cmap=sns.cm.rocket_r, annot=True)
plt.show()






### Re-run after creating columns
# Widen output to display as many columns as there are in dataset
pd.options.display.max_columns = len(list(house_prices_train))

# Increase number of rows printed out in console
pd.options.display.max_rows = len(list(house_prices_train))







### Statistical Tests ###
# Difference in Sales Price based on LotConfig? One-way ANOVA
house_prices_train.groupby("LotShape")["SalePrice"].mean()  # Are these statistically significant?

# Check assumptions for ANOVA
 # 1. Observations are independent - good data collection, check
 # 2. Errors are normally distributed - does not pass assumption
# Shapiro-Wilk test
stats.shapiro(house_prices_train["SalePrice"][house_prices_train["LotShape"] == "Reg"])

stats.probplot(house_prices_train["SalePrice"][house_prices_train["LotConfig"] == "Inside"], plot=plt)
plt.title("Q-Q Plot")

stats.levene(house_prices_train["SalePrice"][house_prices_train["LotConfig"] == "Corner"],
             house_prices_train["SalePrice"][house_prices_train["LotConfig"] == "CulDSac"],
             house_prices_train["SalePrice"][house_prices_train["LotConfig"] == "FR2"],
             house_prices_train["SalePrice"][house_prices_train["LotConfig"] == "FR3"],
             house_prices_train["SalePrice"][house_prices_train["LotConfig"] == "Inside"])

# Test for association between HouseStyle and MSSubClass (two categorical variables) - chi-square test
# Crosstab frequencies of the two groups
pd.crosstab(house_prices_train['LotShape'], house_prices_train['LandSlope'])




########################
### Memory Efficency ###
########################
# Print memory usage in gb
print("The housing dataset is", round(house_prices_train.memory_usage(index=True, deep=True).sum()/byte_to_gb, 5), "GB.")

# # Create memory efficient integer columns
# # Retrieve int64 columns and create list
# house_prices_train_int = list(house_prices_train.select_dtypes(include=['int64']))
# # Downcast int columns in created list
# house_prices_train.loc[:, house_prices_train_int] = house_prices_train.loc[:, house_prices_train_int].apply(pd.to_numeric, downcast="unsigned")
#
# # Create memory efficient float columns
# # Retrieve float64 columns and create list
# house_prices_train_float = list(house_prices_train.select_dtypes(include=['float64']))
# # Downcast float64 columns in created list
# house_prices_train.loc[:, house_prices_train_float] = house_prices_train.loc[:, house_prices_train_float].apply(pd.to_numeric, downcast="float")
#
# # Convert columns from object to category
# # Obtain object columns and convert to list
# house_prices_train_object = list(house_prices_train.select_dtypes(include=['object']))
# # Remove date columns from int - unnecessary in this case since no date columns in dataset
# # house_prices_train_object = [value for value in house_prices_train_object if not 'date' in value]
# # Convert from object to category
# for col in house_prices_train_object:
#     house_prices_train[col]=house_prices_train[col].astype('category')
#
# # Print memory usage in gb after performing memory efficiency steps
# print("The housing dataset is now", round(house_prices_train.memory_usage(index=True, deep=True).sum()/byte_to_gb, 5), "GB.")