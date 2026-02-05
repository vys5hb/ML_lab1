# %% 
# Import Libraries
import pandas as pd
import numpy as np

# Import functions from functions.py
from functions import prev, binary_encode, select_cols, split_target, standardize, train_tune_test_split, one_hot_encode

# COLLEGE COMPLETION DATASET

# Step Two:
# Write a generic question that this dataset could address
# Can we predict whether an institution is private given its institutional characteristics?

# What is an independent Business Metric for your problem? Think about the case study examples we have discussed in class.
# We can see if our predictions are correct which will save time when classifying institutions for academic research purposes

# %% 
# Data Preprocessing
# Read data in
df = pd.read_csv('cc_institution_details.csv')

# Get some basic info about the data
df.info()

# Select relevant columns
relevant_cols1 = ['chronname', 'state', 'level', 'control', 'flagship', 'hbcu', 'student_count', 'awards_per_state_value', 'exp_award_state_value', 'fte_value', 'fte_percentile']
df = select_cols(df, relevant_cols1)

# Save string labels for later
label_cols1 = ['chronname', 'state']
labels1 = select_cols(df, label_cols1)

# Binary Encode categorical variables
binary_encode(df, 'level', '4-year') # 1 if 4-year institution, 0 if 2-year institution
binary_encode(df, 'flagship', 'X') # 1 if state school, 0 if not
binary_encode(df, 'hbcu', 'X') # 1 if hbcu, 0 if not
df['control'] = (df['control'].apply(lambda x: 1 if (x == 'Private not-for-profit' or x == 'Private for-profit') else 0)) # 1 if private, 0 if public

# Standardize continuous variables
cont_vars1 = ['student_count', 'awards_per_state_value', 'exp_award_state_value', 'fte_value', 'fte_percentile']
df[cont_vars1] = standardize(df, cont_vars1)

# Ensure correct types
print(df.dtypes)

# Select target variable and drop target from training data
feature_list1 = ['level', 'flagship', 'hbcu', 'student_count', 'awards_per_state_value', 'exp_award_state_value', 'fte_value', 'fte_percentile']
X1, y1 = split_target(df, target='control', features=feature_list1)

# Calculate prevalence
print(f'The prevalence of private institutions is: {prev(y1)}')

# Train Tune Test Split
X_train1, X_tune1, X_test1, y_train1, y_tune1, y_test1 = train_tune_test_split(X1, y1)

# Step Three:

# What do your instincts tell you about the data. Can it address your problem, what areas/items are you worried about?
# Yes, I believe that this dataset holds good information to predict if an institution is private or not. This mainly comes from looking at student_count.
# Public institutions generally have larger school populations than private institutions which would be a pretty big indicator. Another feature is if a school is the flagship school of 
# their state that almost always means that the school is a public institution.
# I would be worried about exp_award_state_value because although public institutions might have more students, private institutions might get more/greater donations from their alumni
# accounting for more money spent even with less students

# %%
# JOB PLACEMENT DATASET

# Step Two:
# Write a generic question that this dataset could address
# Can we figure out if a student has been placed based off their educational achievements?

# What is an independent Business Metric for your problem? Think about the case study examples we have discussed in class.
# We can find enforce education achievements with high correlation to placement in hopes of placing more students from a given program. The independent business metric
# would be an increase in placement percentage from a educational population

# Data Preprocessing
# Read data in
url = 'https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv'
df = pd.read_csv(url)

# Get some basic info about the data
df.info()

# Select relevant columns
relevant_cols2 = ['sl_no', 'gender', 'ssc_p', 'hsc_p', 'hsc_s', 'degree_p', 'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p', 'status']
df = select_cols(df, relevant_cols2)

# Save id label for later
labels2 = select_cols(df, ['sl_no'])

# Binary Encode categorical variables
binary_encode(df, 'workex', 'Yes') # 1 if has work experience, 0 if not
binary_encode(df, 'status', 'Placed') # 1 if has a placement, 0 if not
binary_encode(df, 'gender', 'M') # 1 if male, 0 if female
binary_encode(df, 'specialisation', 'Mkt&Fin') # 1 if specialized in Marketing & Finance, 0 if specialized in Marketing & HR

# One Hot Encode categorical variables with 3+ levels
df = one_hot_encode(df, 'degree_t')
df = one_hot_encode(df, 'hsc_s')

# Standardize continuous variables
cont_vars2 = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
df[cont_vars2] = standardize(df, cont_vars2)

# Ensure correct types
print(df.dtypes)

# Select target variable and drop target from training data
feature_list2 = ['gender', 'ssc_p', 'hsc_p', 'degree_p', 'workex', 'etest_p', 'specialisation', 'mba_p', 'degree_t__Comm&Mgmt',
                'degree_t__Others', 'degree_t__Sci&Tech', 'hsc_s__Arts', 'hsc_s__Commerce', 'hsc_s__Science']
X2, y2 = split_target(df, target='status', features=feature_list2)

# Calculate prevalence
print(f'The prevalence of private institutions is: {prev(y2)}')

# Train Tune Test Split
X_train2, X_tune2, X_test2, y_train2, y_tune2, y_test2 = train_tune_test_split(X2, y2)

# Step Three:
# I think the data within this dataset can accurately predict if a student has been placed. I believe that there will be trends among degree concentrations, MBA percentages, 
# and previous work experience. I have some worries about the volatility among job placements though since there are many external factors like a candidates interviewing skills, or 
# networking connections which can greatly impact their job placements. I'm confident that my question can be reasonably answered and will fairly predict if a candidate has been placed or not.
# %%

