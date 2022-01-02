from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import numpy as np
from sklearn.utils import shuffle
import itertools
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

def models(X_train, X_test, Y_train, Y_test, models, models_str):
    for model, str_model in zip(models, models_str):
        classifier = model
        classifier.fit(X_train, Y_train)
        prediction = classifier.predict(X_test)

        print(str_model)
        print("Preceision score: ", precision_score(Y_test, prediction))
        print("Recall score: ", recall_score(Y_test, prediction))
        print("Accuracy: ", accuracy_score(Y_test, prediction))
        print("F1 score: ", f1_score(Y_test, prediction))
        print("Confusion matrix: ")
        print(confusion_matrix(Y_test, prediction))
        print()

def validate_anomaly(normal_train_set, normal_test_set, fraud_test_set, models_dict):
    for str_model, model in models_dict.items():
        classifier = model
        classifier.fit(normal_train_set)

        normal_test_predictions = classifier.predict(normal_test_set)
        fraud_test_predictions = classifier.predict(fraud_test_set)

        prediction = np.concatenate((fraud_test_predictions ,normal_test_predictions), axis=0)
        y_test = np.array([-1]*len(fraud_test_predictions)+[1]*len(normal_test_predictions))

        tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
        tn_, fp_, fn_, tp_ = tp, fn, fp, tn


        recall = tp_/(tp_+fn_)
        precision = tp_/(tp_+fp_)

        print(str_model)
        print("Precision score: ", precision)
        print("Recall score :", recall)
        print("Accuracy: ", (tp_+tn_)/(tn_+fp_+fn_+tp_))
        print("F1 score: ", (2* precision*recall)/(recall + precision))
        print()


def cross_validate(models_dict , X, y, cv, resample="normal"):
    #shuffle data
    X_shuffled, y_shuffled = shuffle(X, y)
    #divide shuffled data to N(cv) parts
    X_cv_list = np.array_split(X_shuffled, cv)
    y_cv_list = np.array_split(y_shuffled, cv)

    for str_model, model in models_dict.items():
        # scores lists
        precision = list()
        recall = list()
        f1 = list()
        accuracy = list()

        for index, (X_test,y_test) in enumerate(zip(X_cv_list, y_cv_list)):
            copied_X = X_cv_list.copy()
            copied_y = y_cv_list.copy()
            copied_X.pop(index)
            copied_y.pop(index)

            X_train = list(itertools.chain.from_iterable(copied_X))
            y_train = list(itertools.chain.from_iterable(copied_y))


            if(resample == "ros"):
                ros = RandomOverSampler(random_state=2)
                X_train, y_train = ros.fit_resample(X_train, y_train)
            elif(resample == "rus"):
                rus = RandomUnderSampler(random_state = 0)
                X_train, y_train = rus.fit_resample(X_train, y_train)
            elif(resample == "smote"):
                sm = SMOTE(random_state=2)
                X_train, y_train = sm.fit_resample(X_train, y_train)


            classifier = model
            classifier.fit(X_train, y_train)
            prediction = classifier.predict(X_test)

            precision.append(precision_score(y_test, prediction))
            recall.append(recall_score(y_test, prediction))
            accuracy.append(accuracy_score(y_test, prediction))
            f1.append(f1_score(y_test, prediction))

        print(str_model)
        print("Precision score: ", sum(precision) / len(precision))
        print("Recall score: ", sum(recall) / len(recall))
        print("Accuracy: ", sum(accuracy) / len(accuracy))
        print("F1 score: ", sum(f1) / len(f1))
        print()

