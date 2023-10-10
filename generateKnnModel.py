from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from part1Program import scale_gesture, center_gesture

train_file = "train-final.csv"
test_file = "test-final.csv"

# Simple class to allow cleaner looking "."-indexing
class Joint:
    def __init__(self):
        self.points = []
        self.classes = []

class TestCase:
    def __init__(self):
        self.joints = []
        self.type = []

# Code stolen and modified from jesper's part1Program.py
def generate_column_labels():
    column_labels = []
    for type in ["pos_mean", "pos_std", "ang_mean", "ang_std"]:
        for joint_index in range(20):
            joint_name = f"joint{joint_index}"
            for dimension in ["x", "y", "z"]:
                column_labels.append(f"{joint_name}_{type}_{dimension}")

    column_labels.append("gesture_number")
    column_labels.append("gesture_name")

    return column_labels

# Run 20repsej's preprocessing functions
def perform_preprocess(data):
    return scale_gesture(center_gesture(data))

# Categorizes data by joint
def categorize_data_for_training(filename):
    # Read csv with pandas to allow easy iteration over rows, also remove broken rows
    raw_data = pd.read_csv(filename).dropna()
    raw_data.columns = generate_column_labels()

    # Run preprocessing
    processed_data = perform_preprocess(raw_data)

    # Prepare the joints array (see comment below for how this is structured)
    joints = []
    for _ in range(20):
        joints.append(Joint())

    # Loop over all rows in the data, dividing up the individual joints so we get 12D vectors (mean x/y/z, std x/y/z, same for angles)
    # We will then have an array (joints) of objects {points: [[12D vector], [12D vector], ...], classes: [class, class, ...]}
    for _, row in processed_data.iterrows():
        # Loop 20 times as there are 20 joints
        for joint_index in range(20):
            # Get all 12 dimensions and put them together as an array (vector), also get class
            # There are 4 types to look at, each contains 20*3D vectors and is therefore 60 columns wide
            joint_vector = pd.concat([row[joint_index*3:joint_index*3+3], row[60+joint_index*3:60+joint_index*3+3], row[120+joint_index*3:120+joint_index*3+3], row[180+joint_index*3:180+joint_index*3+3]])
            joint_class = row.iloc[240]

            # Then add them with their respective class to the joints array
            joints[joint_index].points.append(joint_vector)
            joints[joint_index].classes.append(joint_class)

    return joints

# Categorizes data by row/gesture
def categorize_data_for_testing(filename):
    # Read csv with pandas to allow easy iteration over rows, also remove broken rows
    raw_data = pd.read_csv(filename).dropna()
    raw_data.columns = generate_column_labels()

    # Run preprocessing
    processed_data = perform_preprocess(raw_data)

    # Create a list of all test cases
    test_cases = []

    # Loop over all rows in the data, dividing up the individual joints so we get 12D vectors (mean x/y/z, std x/y/z, same for angles)
    # We will then have an array of objects {points: [[12D vector], [12D vector], ...], classes: [class, class, ...]}
    for _, row in processed_data.iterrows():
        # Create a new test case for each row, and get the correct type/class
        new_case = TestCase()
        new_case.type = row.iloc[240]

        # Loop 20 times as there are 20 joints
        for joint_index in range(20):
            # Get all 12 dimensions and put them together as an array (vector), also get class
            # There are 4 types to look at, each contains 20*3D vectors and is therefore 60 columns wide
            joint_vector = pd.concat([row[joint_index*3:joint_index*3+3], row[60+joint_index*3:60+joint_index*3+3], row[120+joint_index*3:120+joint_index*3+3], row[180+joint_index*3:180+joint_index*3+3]])

            # Then add them with their respective class to the joints array
            new_case.joints.append(joint_vector)

        test_cases.append(new_case)

    return test_cases

# Prepare data for model application
def apply_model_vote(test_case, models, joint_evals):
    # Based on unmodified accuracy of guesses
    joint_modifiers = [0.1751, 0.1638, 0.2524, 0.2919, 0.3616, 0.4407, 0.5367, 0.6629, 0.533, 0.7307, 0.2053, 0.2072, 0.2166, 0.2241, 0.2109, 0.1902, 0.2072, 0.2524, 0.29, 0.2618]

    votes = {}
    for model_index in range(len(models)):
        model = models[model_index]
        # We're doing this one case at a time, so result of .predict() will be an array of just one guess
        guess = model.predict([test_case.joints[model_index]])[0]
        
        vote = joint_modifiers[model_index]**2

        if guess in votes:
            votes[guess] += vote
        else:
            votes[guess] = vote

        # TEMP, test individual joins' accuracy in guessing
        if guess == test_case.type:
            joint_evals[model_index][0] += 1
        else:
            joint_evals[model_index][1] += 1
        
    # Answer is the most voted for type
    answer = max(votes, key=votes.get)
    return answer

def purge_joints(training_data, test_data, joints):
    joints.sort(reverse=True)

    # Purge from training data
    for joint in joints:
        training_data.pop(joint)

    # Purge from test data
    for case in test_data:
        for joint in joints:
            case.joints.pop(joint)


# Compile the data
print("Compiling data...")
training_data = categorize_data_for_training(train_file)
test_data = categorize_data_for_testing(test_file)

# Purge some joints
# purge = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# print("Purging joints...", purge)
# purge_joints(training_data, test_data, purge)

# Create 20 separate models for all joints
print("Training models (kNN)...")
kNN_models = []
joint_evals = []
for model_index in range(len(training_data)):
    joint_evals.append([0, 0])

    # Params can be changed
    kNN_model = KNeighborsClassifier(n_neighbors=1, weights='distance')
    # print(training_data[model_index].points[539], training_data[model_index].classes[539])
    kNN_model.fit(training_data[model_index].points, training_data[model_index].classes)
    kNN_models.append(kNN_model)


# Run the test data through the voting process
print("Running test (kNN)...")
correct = 0
incorrect = 0
for case in test_data:
    guess = apply_model_vote(case, kNN_models, joint_evals)
    if guess == case.type:
        correct += 1
    else:
        incorrect += 1

print(f"Correct: {correct}; Incorrect: {incorrect}; Accuracy: {round(correct/len(test_data)*100, 1)}%")
print("Joint points:")

# Joint accuracies
accuracies = []
for joint in joint_evals:
    correct, incorrect = joint
    accuracies.append(round(correct / (correct+incorrect), 4))
print(accuracies)
