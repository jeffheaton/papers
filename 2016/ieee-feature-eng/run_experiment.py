__author__ = 'jheaton'
# This Python script was used to collect the data for following paper/conference:
#
# Heaton, J. (2016, April). An Empirical Analysis of Feature Engineering for Predictive Modeling.
# In SoutheastCon 2016 (pp. 1-6). IEEE.
#
# http://www.jeffheaton.com

import math
import numpy as np
import time
import types
import codecs
import csv
import multiprocessing
from sklearn.metrics import mean_squared_error
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import rectify
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV

# Global parameters
TRAIN_SIZE = 10000
TEST_SIZE = 100
SEED = 2048
VERBOSE = 0
CYCLES = 5
FAIL_ON_NAN = False
NORMALIZE_ALL = False
DUMP_FILES = True
THREADS = multiprocessing.cpu_count()


# Define an early stopping class for the deep learning, once the validation score
# does not improve for the specified number of epochs, stop and use weights from
# the best validation score epoch.
class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if math.isnan(current_valid) and FAIL_ON_NAN:
            raise Exception("Unstable neural network. Can't validate.")
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            if VERBOSE > 0:
                print("Early stopping.")
                print("Best valid loss was {:.6f} at epoch {}.".format(
                    self.best_valid, self.best_valid_epoch))
                nn.load_params_from(self.best_weights)
            raise StopIteration()


# Another stopping class for deep learning.  If the error/loss falls
# below the specified threshold, we are done.
class AcceptLoss(object):
    def __init__(self, min=0.01):
        self.min = min

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']

        if current_valid < self.min:
            if VERBOSE > 0:
                print("Acceptable loss")
            raise StopIteration()


# Holds the data for the experiments.  This allows the data to be represented in
# the forms needed for several model types.
#
class DataHolder:
    def __init__(self, training, validation):
        self.X_train = training[0]
        self.X_validate = validation[0]
        self.y_train = training[1]
        self.y_validate = validation[1]

        # Provide normalized
        self.X_train_norm = MinMaxScaler().fit_transform(self.X_train)
        self.X_validate_norm = MinMaxScaler().fit_transform(self.X_validate)

        # Normalize all, if requested
        if NORMALIZE_ALL:
            self.X_train = StandardScaler().fit_transform(self.X_train)
            self.X_validate = StandardScaler().fit_transform(self.X_validate)

        # Format the y data for neural networks (lasange)
        self.y_train_nn = []
        self.y_validate_nn = []

        for y in self.y_train:
            self.y_train_nn.append([y])

        for y in self.y_validate:
            self.y_validate_nn.append([y])

        self.y_train_nn = np.array(self.y_train_nn, dtype=np.float32)
        self.y_validate_nn = np.array(self.y_validate_nn, dtype=np.float32)

    # Dump data to CSV files for examination.
    def dump(self, base):
        header = ",".join(["x" + str(x) for x in range(1, 1 + self.X_train.shape[1])])
        header += ","
        header += ",".join(["y" + str(x) for x in range(1, 1 + self.y_train_nn.shape[1])])

        np.savetxt(base + "_train.csv",
                   np.hstack((self.X_train, self.y_train_nn)),
                   fmt='%10.5f', delimiter=',', header=header, comments="")

        np.savetxt(base + "_validate.csv",
                   np.hstack((self.X_validate, self.y_validate_nn)),
                   fmt='%10.5f', delimiter=',', header=header, comments="")

        np.savetxt(base + "_train_norm.csv",
                   np.hstack((self.X_train_norm, self.y_train_nn)),
                   fmt='%10.5f', delimiter=',', header=header, comments="")

        np.savetxt(base + "_validate_norm.csv",
                   np.hstack((self.X_validate_norm, self.y_validate_nn)),
                   fmt='%10.5f', delimiter=',', header=header, comments="")


# Human readable time elapsed string.
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


# Generate data for the counts experiment.
def generate_data_counts(rows):
    x_array = []
    y_array = []

    for i in range(rows):
        x = [0] * 50
        y = np.random.randint(0, len(x)+1)

        remaining = y
        while remaining > 0:
            idx = np.random.randint(0, len(x) - 1)
            if x[idx] == 0:
                x[idx] = 1
                remaining -= 1

        x_array.append(x)
        y_array.append(y)

    return np.array(x_array, dtype=np.float32), np.array(y_array, dtype=np.float32)

# Generate data for the quadradic experiment, distance between polynomial roots.
def generate_data_quad(rows):
    x_array = []
    y_array = []

    while (len(x_array) < rows):
        a = float(np.random.randint(-10, 10))
        b = float(np.random.randint(-10, 10))
        c = float(np.random.randint(-10, 10))
        y = [0, 0]

        try:
            y = [
                (-b + math.sqrt((b * b) - (4 * a * c))) / (2 * a),
                (-b - math.sqrt((b * b) - (4 * a * c))) / (2 * a)]
        except (ValueError, ZeroDivisionError):
            pass

        x_array.append([a, b, c])
        y_array.append(abs(y[0] - y[1]))

    return np.array(x_array, dtype=np.float32), np.array(y_array, dtype=np.float32)

# Generate data for a BMI-like feature.
def generate_data_bmi(rows):
    x_array = []
    y_array = []

    while (len(x_array) < rows):
        m = float(np.random.randint(25, 200))
        h = float(np.random.uniform(1.5, 2.0))
        y = m / (h * h)

        x_array.append([h, m])
        y_array.append(y)

    return np.array(x_array, dtype=np.float32), np.array(y_array, dtype=np.float32)


# Generate data for the provided function, usually a lambda.
def generate_data_fn(cnt, x_low, x_high, fn):
    return lambda rows: generate_data_fn2(rows, cnt, x_low, x_high, fn)


# Used internally for generate_data_fn
def generate_data_fn2(rows, cnt, x_low, x_high, fn):
    x_array = []
    y_array = []

    while (len(x_array) < rows):
        args = []
        for i in range(cnt):
            args.append(np.random.uniform(x_low, x_high))

        try:
            y = fn(*args)
            if not math.isnan(y):
                x_array.append(args)
                y_array.append(y)
        except (ValueError, ZeroDivisionError):
            pass

    return np.array(x_array, dtype=np.float32), np.array(y_array, dtype=np.float32)


# Generate data for the ratio experiment
def generate_data_ratio(rows):
    x_array = []
    y_array = []

    while (len(x_array) < rows):
        x = [
            np.random.uniform(0, 1),
            np.random.uniform(0.01, 1)]

        try:
            y = x[0] / x[1]
            x_array.append(x)
            y_array.append(y)
        except (ValueError, ZeroDivisionError):
            pass

    return np.array(x_array, dtype=np.float32), np.array(y_array, dtype=np.float32)


# Generate data for the difference experiment
def generate_data_diff(rows):
    x_array = []
    y_array = []

    while (len(x_array) < rows):
        x = [
            np.random.uniform(0, 1),
            np.random.uniform(0.01, 1)]

        try:
            y = x[0] - x[1]
            x_array.append(x)
            y_array.append(y)
        except (ValueError, ZeroDivisionError):
            pass

    return np.array(x_array, dtype=np.float32), np.array(y_array, dtype=np.float32)


# Build a deep neural network for the experiments.
def neural_network_regression(data):
    layers0 = [('input', InputLayer),
               ('dense0', DenseLayer),
               ('dense1', DenseLayer),
               ('dense2', DenseLayer),
               ('dense3', DenseLayer),
               ('dense4', DenseLayer),
               ('output', DenseLayer)]

    net0 = NeuralNet(layers=layers0,
                     input_shape=(None, len(data.X_train[0])),
                     dense0_num_units=400,
                     dense0_nonlinearity=rectify,
                     dense1_num_units=200,
                     dense1_nonlinearity=rectify,
                     dense2_num_units=100,
                     dense2_nonlinearity=rectify,
                     dense3_num_units=50,
                     dense3_nonlinearity=rectify,
                     dense4_num_units=25,
                     dense4_nonlinearity=rectify,
                     output_num_units=1,
                     output_nonlinearity=None,

                     update=nesterov_momentum,
                     update_learning_rate=0.00001,
                     update_momentum=0.9,
                     regression=True,

                     on_epoch_finished=[
                         EarlyStopping(patience=20),
                         AcceptLoss(min=0.01)
                     ],

                     verbose=VERBOSE,
                     max_epochs=100000)

    # Provide our own validation set
    def my_split(self, X, y, eval_size):
        return data.X_train, data.X_validate, data.y_train_nn, data.y_validate_nn

    net0.train_split = types.MethodType(my_split, net0)

    return net0


# Grid-search for a SVM with good C and Gamma.
def svr_grid():
    param_grid = {
        'C': [1e-2, 1, 1e2],
        'gamma': [1e-1, 1, 1e1]

    }
    clf = GridSearchCV(SVR(kernel='rbf'), verbose=VERBOSE, n_jobs=THREADS, param_grid=param_grid)
    return clf


# Perform an experiment for a single model type.
def eval_data(writer, name, clf, data):
    model_name = clf.__class__.__name__

    X_validate = data.X_validate
    X_train = data.X_train
    y_validate = data.y_validate
    y_train = data.y_train

    if model_name == "NeuralNet":
        y_validate = data.y_validate_nn
        y_train = data.y_train_nn
    elif model_name == "SVR":
        X_validate = data.X_validate_norm
        X_train = data.X_train_norm

    cycle_list = []
    for cycle_num in range(1, CYCLES + 1):
        start_time = time.time()
        clf.fit(X_train, y_train)
        elapsed_time = hms_string(time.time() - start_time)

        pred = clf.predict(X_validate)

        # Get the validatoin score
        if np.isnan(pred).any():
            if FAIL_ON_NAN:
                raise Exception("Unstable neural network. Can't validate.")
            score = 1e5
        else:
            score = mean_squared_error(pred, y_validate)

        line = [name, model_name, score, elapsed_time]
        cycle_list.append(line)
        print("Cycle {}:{}".format(cycle_num, line))

    best_cycle = min(cycle_list, key=lambda k: k[2])
    print("{}(Best)".format(best_cycle))

    writer.writerow(best_cycle)


# Run an experiment over all model types
def run_experiment(writer, name, generate_data):
    np.random.seed(SEED)

    data = DataHolder(
        generate_data(TRAIN_SIZE),
        generate_data(TEST_SIZE))

    if DUMP_FILES:
        data.dump(name)

    # Define model types to use
    models = [
        svr_grid(),
        RandomForestRegressor(n_estimators=100),
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=0, verbose=VERBOSE),
        neural_network_regression(data)
    ]

    for model in models:
        eval_data(writer, name, model, data)

def main():
    with codecs.open("results.csv", "w", "utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(['experiment', 'model', 'error', 'elapsed'])
        start_time = time.time()
        run_experiment(writer, "counts", generate_data_counts)  #1
        run_experiment(writer, "quad", generate_data_quad)  #2
        run_experiment(writer, "sqrt", generate_data_fn(1, 1.0, 100.0 ,math.sqrt))  #3
        run_experiment(writer, "log", generate_data_fn(1, 1.0, 100.0 ,math.log))   #4
        run_experiment(writer, "pow", generate_data_fn(1, 1.0, 10.0, lambda x: x ** 2))  #5
        run_experiment(writer, "ratio", generate_data_ratio)  #6
        run_experiment(writer, "diff", generate_data_diff)  #7
        run_experiment(writer, "r_poly", generate_data_fn(1, 1.0, 10.0, lambda x: 1/( 5*x + 8 * x ** 2)))  #8
        run_experiment(writer, "poly", generate_data_fn(1, 0.0, 2.0, lambda x: 1+5*x+8*x**2))   #9
        run_experiment(writer, "r_diff", generate_data_fn(4, 1.0, 10.0, lambda a,b,c,d: ((a-b)/(c-d))))  #10

        # Others to try, not in the paper.
        #run_experiment(writer, "sum", generate_data_fn(10, 0.0, 10.0, lambda *args: np.sum(args)))
        #run_experiment(writer, "max", generate_data_fn(10, 0.0, 100.0, lambda *args: np.max(args)))
        #run_experiment(writer, "dev", generate_data_fn(10, 0.0, 100.0, lambda *args: np.std(args)))
        #run_experiment(writer, "bmi", generate_data_bmi)  #### 2

        elapsed_time = time.time() - start_time
        print("Elapsed time: {}".format(hms_string(elapsed_time)))

# Allow windows to multi-thread (unneeded on advanced OS's)
# See: https://docs.python.org/2/library/multiprocessing.html
if __name__ == '__main__':
    main()
