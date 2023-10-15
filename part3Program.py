from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn.svm import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier


# Input files
test_file = "test-final.csv"
train_file = "train-final.csv"

# Open csv file and add some labels
def open_csv_file(input_file):
    df = pd.read_csv(input_file) # DataFrame

    column_labels = []

    # Add a number to each column
    for num in range(1, 241):
        column_labels.append("col_" + str(num))

    column_labels.append("gesture_name")
    column_labels.append("gesture_value")

    df.columns = column_labels

    # Return data and label
    return df


# Fix missing values by removing or taking mean value
def remove_missing_values(data):
    fix_data = data.dropna()
    return fix_data

#def fix_missing_values(data):
#    imp = SimpleImputer(missing_values=pd.isna(), strategy='mean')
#    imp.fit(data)
#    fix_data = imp.transform(data)
#    return fix_data

def fix_missing_values(dataFrame: pd.core.frame.DataFrame):
    df = dataFrame.copy()

    missing_values = df.isna()

    mean = df.mean()

    df = df.fillna(mean)

    return df
# Scale data using scilearn standard scaler
def scale_data(data):
    scaler = StandardScaler()
    scaler = scaler.fit(data)

    scaled = scaler.transform(data)
    return scaled

def run_PCA(data):
#    pca = PCA(n_components='mle', svd_solver='full')
    #pca = PCA()
    #data = pca.fit_transform(data)
    #return data
    transformer = PCA(n_components=30)
    X_transformed = transformer.fit_transform(data)
    return X_transformed


def run_model_test(model_name, create_new_model, train_data, test_data, train_label, test_label):
    # Create separate models for all joints
    print(f"Training models ({model_name})...")
        # Create model model of given type using inputted lambda func, then fit training data to model
    new_model = create_new_model()
    new_model.fit(train_data, train_label)

    prediction = new_model.predict(test_data)


    # Cross validation
    cross_v = cross_val_score(new_model, train_data, train_label, cv=10, scoring="accuracy")
    print("Cross Validation")
    cross_v = np.array(cross_v)
    print(np.mean(cross_v))
    #print(classification_report(test_label, prediction))
    print("Accuracy Score")
    print(accuracy_score(test_label, prediction))
    print("")

train = open_csv_file(train_file)
test = open_csv_file(test_file)



train = remove_missing_values(train)
test = remove_missing_values(test)

train_data = train.iloc[:, 0:240]
train_label = train.iloc[:, 241]
test_data = test.iloc[:, 0:240]
test_label = test.iloc[:, 241]


train_data = scale_data(train_data)
test_data = scale_data(test_data)

#train_data = run_PCA(train_data)
#test_data = run_PCA(test_data)

#print(train_label)

# Run kNN test with and without bagging
#run_model_test("kNN", lambda: KNeighborsClassifier(n_neighbors=1, weights='distance'), train_data, test_data, train_label, test_label)

# Run SVM test
#run_model_test("SVM", lambda: SVC(decision_function_shape='ovo', kernel='rbf'), train_data, test_data, train_label, test_label)
#run_model_test("SVM", lambda: SVC(decision_function_shape='ovo', kernel='linear'), train_data, test_data, train_label, test_label)
#run_model_test("SVM", lambda: SVC(decision_function_shape='ovo', kernel='poly'), train_data, test_data, train_label, test_label)

# Run Decision tree test
#run_model_test("Decision tree", lambda: DecisionTreeClassifier(), train_data, test_data, train_label, test_label)

# Run Random forest test
#run_model_test("Random forest 1", lambda: RandomForestClassifier(n_estimators=1), train_data, test_data, train_label, test_label)
#run_model_test("Random forest 10", lambda: RandomForestClassifier(n_estimators=10), train_data, test_data, train_label, test_label)

from threading import Thread
for i in range(1, 2000):
    Thread(target=run_model_test, args=(
        f"Random forest {i}",
        lambda: RandomForestClassifier(n_estimators=i),
        train_data,
        test_data,
        train_label,
        test_label
    )).start()
    #run_model_test("Random forest 100", lambda: RandomForestClassifier(n_estimators=i), train_data, test_data, train_label, test_label)

# Run MLP test

# Activation test
#run_model_test("MLP", lambda: MLPClassifier(activation="tanh", solver="lbfgs", max_iter=100, tol=.0001), train_data, test_data, train_label, test_label)
#run_model_test("MLP", lambda: MLPClassifier(activation="relu", solver="lbfgs", max_iter=100, tol=.0001), train_data, test_data, train_label, test_label)
#run_model_test("MLP", lambda: MLPClassifier(activation="identity", solver="lbfgs", max_iter=100, tol=.0001), train_data, test_data, train_label, test_label)
#run_model_test("MLP", lambda: MLPClassifier(activation="logistic", solver="lbfgs", max_iter=100, tol=.0001), train_data, test_data, train_label, test_label)

# Solver
#run_model_test("MLP", lambda: MLPClassifier(activation="tanh", solver="lbfgs", max_iter=100, tol=.0001), train_data, test_data, train_label, test_label)
#run_model_test("MLP", lambda: MLPClassifier(activation="tanh", solver="sgd", max_iter=100, tol=.0001), train_data, test_data, train_label, test_label)
#run_model_test("MLP", lambda: MLPClassifier(activation="tanh", solver="adam", max_iter=100, tol=.0001), train_data, test_data, train_label, test_label)

#
#run_model_test("MLP", lambda: MLPClassifier(activation="tanh", solver="lbfgs",hidden_layer_sizes=10, max_iter=500, tol=.0001), train_data, test_data, train_label, test_label)
#run_model_test("MLP", lambda: MLPClassifier(activation="tanh", solver="lbfgs",hidden_layer_sizes=100, max_iter=500, tol=.0001), train_data, test_data, train_label, test_label)
#run_model_test("MLP", lambda: MLPClassifier(activation="tanh", solver="lbfgs",hidden_layer_sizes=300, max_iter=500, tol=.0001), train_data, test_data, train_label, test_label)


#
# Ensembles
#

# Ada boost classifier
#run_model_test("Ada Boost", lambda: AdaBoostClassifier(n_estimators=100), train_data, test_data, train_label, test_label)

#run_model_test("kNN Bagging", lambda: BaggingClassifier(KNeighborsClassifier(n_neighbors=1, weights='distance'), max_samples=0.5, max_features=0.5), train_data, test_data, train_label, test_label)

# Random forest with Extremly Randmoized Trees
#clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

#run_model_test("ExtraTrees", lambda: ExtraTreesClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0), train_data, test_data, train_label, test_label)