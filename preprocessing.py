from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

# Input Data
train_file = "train-final.csv"
test_file = "test-final.csv"


# Keeping track of the models' results more easily by creating some simple methods
class Result:
    def __init__(self):
        self.correct = 0
        self.incorrect = 0
    
    def get_percentage(self):
        return round(self.correct/(self.correct+self.incorrect)*100, 1)
    
    def print_result(self, model_name):
        print(f"Model: {model_name}; Correct: {self.correct}; Incorrect: {self.incorrect}; Accuracy: {self.get_percentage()}%")


# Generate labels for every column
def generate_column_labels():
    column_labels = []
    for type in ["pos_mean", "pos_std", "ang_mean", "ang_std"]:
        for joint_index in range(20):
            joint_name = f"joint{joint_index}"
            for dimension in ["x", "y", "z"]:
                column_labels.append(f"{joint_name}_{type}_{dimension}")

    column_labels.append("gesture_name")
    column_labels.append("gesture_number")

    return column_labels

# Scale data from 0-1
def perform_min_max_preprocess(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    minmax = min_max_scaler.fit_transform(data)

    return minmax

def perform_standard_preprocess(data):
    standard_scaler = StandardScaler()
    data = array.reshape(-1, 1)
    scaled = standard_scaler.fit_transform(data)
    return data

def remove_missing_values(data):
    data = data.dropna()

def mean_missing_values(data):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(data)
    data = imp.transform(data)
    return data


# Categorizes data by joint
def prepare_data(filename, use_preprocessing):
    # Read csv with pandas to allow easy iteration over rows, also remove broken rows
    #raw_data = pd.read_csv(filename).dropna()
    raw_data = pd.read_csv(filename)
    raw_data.columns = generate_column_labels()

    # Run preprocessing
    processed_data = raw_data
    if use_preprocessing:
        processed_data = remove_missing_values(raw_data)
        processed_data = perform_standard_preprocess(processed_data)


    return data

# Models
def test_model(train, test):
    print("################ TEST #############")
    dt_reg = DecisionTreeClassifier()
    dt_reg.fit(train, train.labels)

    prediction = dt_reg.predict(test)

    comparison = pd.DataFrame({'Real': test.labels, 'Predictions': prediction})

    print(comparison)

# Main

# Preprocessing
preprocess = True

train = prepare_data(train_file, preprocess)
test = prepare_data(test_file, preprocess)

test_model(train, test)