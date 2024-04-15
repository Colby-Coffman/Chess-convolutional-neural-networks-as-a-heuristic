# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:40:50 2023

@author: NENO
"""

import pandas as pd
import numpy as np

df = pd.read_json('data.json')
df = df.drop(["pawns", "knights","bishops", "rooks", "queens", "kings"], axis=1)


def expand_array_column(df, column_name):
    # Create a new DataFrame with the array expanded into separate columns
    expanded_df = pd.DataFrame(df[column_name].tolist(), columns=[f'{column_name}_{i}' for i in range(len(df[column_name].iloc[0]))])
    
    # Concatenate the expanded DataFrame with the original DataFrame
    df = pd.concat([df, expanded_df], axis=1)
    
    # Drop the original array column
    df.drop(column_name, axis=1, inplace=True)
    
    return df

def create_combined_and_y_df(df):
    # Get the 'y' column (if it exists)
    y_column = df['y'] if 'y' in df else None

    # Exclude the 'y' column from the DataFrame
    df = df.drop(columns='y', errors='ignore')

    # Combine all columns into a single list 'combined' column
    df['positions'] = df.apply(lambda row: row.tolist(), axis=1)

    # Re-add the 'y' column (if it exists) to the new DataFrame
    combined_and_y_df = pd.concat([df[['positions']], y_column], axis=1)

    return combined_and_y_df


def pawnsArray(arr):
    return [1 if x == 1 else -1 if x == -1 else 0 for x in arr]

def knightsArray(arr):
    return [2 if x == 2 else -2 if x == -2 else 0 for x in arr]

def bishopsArray(arr):
    return [3 if x == 3 else -3 if x == -3 else 0 for x in arr]

def rooksArray(arr):
    return [4 if x == 4 else -4 if x == -4 else 0 for x in arr]

def queensArray(arr):
    return [5 if x == 5 else -5 if x == -5 else 0 for x in arr]

def kingsArray(arr):
    return [6 if x == 6 else -6 if x == -6 else 0 for x in arr]




def combineEverything(df):
    m = len(df)
    res = []
    for i in range(0,m):
        arr = df["positions"].iloc[i]
        y = df["y"].iloc[i]
        res.append([arr, pawnsArray(arr), knightsArray(arr), bishopsArray(arr), rooksArray(arr), queensArray(arr), kingsArray(arr), y])
    
    return res
    
    
    
    

# Call the function to expand the array column
df = expand_array_column(df, 'posiciones')
df = df.drop_duplicates()
df = create_combined_and_y_df(df)







res = combineEverything(df)

data = pd.DataFrame(res)
new_column_names = ['posiciones','pawns', 'knights', 'bishops', 'rooks', 'queens', 'kings', 'y']
data.columns = new_column_names



# Define a function to reshape the rows into 8x8 matrices
def reshape_to_matrix(arr):
    return np.array(arr).reshape(8, 8)

# Apply the reshape function to each column
columns_to_reshape = ['posiciones','pawns', 'knights', 'bishops', 'rooks', 'queens', 'kings']
for col in columns_to_reshape:
    data[col] = data[col].apply(reshape_to_matrix)

data.to_json('MatrizPosiciones2.json')
