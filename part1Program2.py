# D0038E AI Project

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Files
train_file = "train-final.csv"
test_file = "test-final.csv"

# Open csv file and add some labels
def open_csv_file(input_file):
    df = pd.read_csv(input_file) # DataFrame

    column_labels = []

    # Add a number to each column
    for num in range(1, 241):
        column_labels.append("col_" + str(num))

    column_labels.append("gesture_number")
    column_labels.append("gesture_name")

    df.columns = column_labels

    return df


def center_gesture(dataFrame: pd.core.frame.DataFrame):
    df = dataFrame.copy()

    #df_first_60_columns = df.iloc[:,:60]

    # Get XYZ coords
    #x_mean_pos = df_first_60_columns.iloc[::3]
    #y_mean_pos = df_first_60_columns.iloc[1::3]
    #z_mean_pos = df_first_60_columns.iloc[2::3]
    x = df.columns[0::3][0:20]
    y = df.columns[1::3][0:20]
    z = df.columns[2::3][0:20]


    df['x_centered'] = df[x].mean(axis=1).tolist()
    df['y_centered'] = df[x].mean(axis=1).tolist()
    df['z_centered'] = df[x].mean(axis=1).tolist()

    # Center every gesture
    df[x] = df[x].sub(df['x_centered'], axis=0)
    df[y] = df[y].sub(df['y_centered'], axis=0)
    df[z] = df[z].sub(df['z_centered'], axis=0)

    return df

def scale_gesture(dataFrame: pd.core.frame.DataFrame):
    df = dataFrame.copy()

    x = df.columns[0::3][0:20]
    y = df.columns[1::3][0:20]
    z = df.columns[2::3][0:20]


    df['x_scale'] = (df[x].max(axis=1) - df[x].min(axis=1)).tolist()
    df['y_scale'] = (df[y].max(axis=1) - df[y].min(axis=1)).tolist()
    df['z_scale'] = (df[z].max(axis=1) - df[z].min(axis=1)).tolist()

    df[x] = df[x].div(df['x_scale'], axis=0)
    df[y] = df[y].div(df['y_scale'], axis=0)
    df[z] = df[z].div(df['z_scale'], axis=0)

    return df

def scale_gesture_std_pos(dataFrame: pd.core.frame.DataFrame):
    df = dataFrame.copy()

    x = df.columns[0::3][21:40]
    y = df.columns[1::3][21:40]
    z = df.columns[2::3][21:40]


    df['x_scale'] = (df[x].max(axis=1) - df[x].min(axis=1)).tolist()
    df['y_scale'] = (df[y].max(axis=1) - df[y].min(axis=1)).tolist()
    df['z_scale'] = (df[z].max(axis=1) - df[z].min(axis=1)).tolist()

    df[x] = df[x].div(df['x_scale'], axis=0)
    df[y] = df[y].div(df['y_scale'], axis=0)
    df[z] = df[z].div(df['z_scale'], axis=0)

    return df
def scale_gesture_mean_angle(dataFrame: pd.core.frame.DataFrame):
    df = dataFrame.copy()

    x = df.columns[0::3][41:60]
    y = df.columns[1::3][41:60]
    z = df.columns[2::3][41:60]


    df['x_scale'] = (df[x].max(axis=1) - df[x].min(axis=1)).tolist()
    df['y_scale'] = (df[y].max(axis=1) - df[y].min(axis=1)).tolist()
    df['z_scale'] = (df[z].max(axis=1) - df[z].min(axis=1)).tolist()

    df[x] = df[x].div(df['x_scale'], axis=0)
    df[y] = df[y].div(df['y_scale'], axis=0)
    df[z] = df[z].div(df['z_scale'], axis=0)

    return df

def scale_gesture_std_angle(dataFrame: pd.core.frame.DataFrame):
    df = dataFrame.copy()

    x = df.columns[0::3][61:80]
    y = df.columns[1::3][61:80]
    z = df.columns[2::3][61:80]


    df['x_scale'] = (df[x].max(axis=1) - df[x].min(axis=1)).tolist()
    df['y_scale'] = (df[y].max(axis=1) - df[y].min(axis=1)).tolist()
    df['z_scale'] = (df[z].max(axis=1) - df[z].min(axis=1)).tolist()

    df[x] = df[x].div(df['x_scale'], axis=0)
    df[y] = df[y].div(df['y_scale'], axis=0)
    df[z] = df[z].div(df['z_scale'], axis=0)

    return df

def remove_incomplete_rows(dataFrame: pd.core.frame.DataFrame):
    df = dataFrame.copy()

    df_dropna = df.dropna()

    return df_dropna

def replace_mean_incomplete_rows(dataFrame: pd.core.frame.DataFrame):
    df = dataFrame.copy()

    missing_values = df.isna()

    mean = df.mean()

    df = df.fillna(mean)

    return df

'''def replace_incomplete_with_mean(dataFrame: pd.core.frame.DataFrame):
    df = dataFrame.copy()
    missing_values = df.isna()

    df['last_value'] = df.iloc[:, -1]

    grouped_df = df.groupby('last_value')

    group_means = grouped_df.mean()

    group_means_dict = {}
    for group in group_means.index:
        group_means_dict[group] = group_means.loc[group]

    for index, row in df.iterrows():
        if missing_values.loc[index]:
            df.loc[index] = row.fillna(group_means_dict[row['last_value']])

    return df

def replace_incomplete_with_mean(dataFrame: pd.core.frame.DataFrame):
    df = dataFrame.copy()

    df = df.iloc[:, :240]

    missing_values = df.isna()

    df['last_value'] = df.iloc[:, -1]

    grouped_df = df.groupby('last_value')

    group_means = grouped_df.mean()

    group_means_dict = {}
    for group in group_means.index:
        group_means_dict[group] = group_means.loc[group]

  # Iterate over the DataFrame and replace the missing values with the mean of the corresponding group using the dictionary.
    for index, row in df.iterrows():
        if missing_values.loc[index].bool():
            for column in range(1, 241):
                if missing_values.loc[index, column].all():
                    df.loc[index, column] = group_means_dict[row['last_value']][column]
    
    df.drop('last_value', axis=1, inplace=True)

    return df'''


# data = open_csv_file(train_file)

# data2 = center_gesture(data)

# data3 = scale_gesture(data2)

# data4 = replace_mean_incomplete_rows(data)

# data5 = remove_incomplete_rows(data)

#data6 = replace_incomplete_with_mean(data)

#visualize_gesture_skeleton(data5)
# visualize_one_gesture_skeleton(data5, 1)



#print(open_csv_file(train_file))
     
