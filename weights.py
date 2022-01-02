import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from helper_functions import models, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier


weight = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
preceision = [0.9732142857142857, 0.9586776859504132, 0.92, 0.9126984126984127, 0.912, 0.8914728682170543, 0.9212598425196851, 0.8872180451127819, 0.8914728682170543, 0.90625, 0.8854961832061069, 0.8931297709923665, 0.8656716417910447, 0.8778625954198473, 0.8571428571428571, 0.8846153846153846, 0.8976377952755905, 0.849624060150376, 0.8721804511278195, 0.8666666666666667, 0.8592592592592593]
recall = [0.7517241379310344, 0.8, 0.7931034482758621, 0.7931034482758621, 0.7862068965517242, 0.7931034482758621, 0.8068965517241379, 0.8137931034482758, 0.7931034482758621, 0.8, 0.8, 0.8068965517241379, 0.8, 0.7931034482758621, 0.7862068965517242, 0.7931034482758621, 0.7862068965517242, 0.7793103448275862, 0.8, 0.8068965517241379, 0.8]
f1 = [0.8482490272373542, 0.8721804511278195, 0.851851851851852, 0.8487084870848709, 0.8444444444444443, 0.8394160583941606, 0.8602941176470588, 0.8489208633093526, 0.8394160583941606, 0.8498168498168499, 0.8405797101449277, 0.8478260869565216, 0.8315412186379928, 0.8333333333333334, 0.8201438848920864, 0.8363636363636363, 0.838235294117647, 0.8129496402877697, 0.8345323741007195, 0.8357142857142856, 0.8285714285714285]

n = 8
print(weight[n])
print(preceision[n])
print(recall[n] )
print(f1[n])


plt.plot(weight, f1, label = "F1")
plt.plot(weight, preceision, label = "precision")
plt.plot(weight, recall, label = "recall")
plt.xlabel('Weight')
# Set the y axis label of the current axis.
plt.ylabel('F1/precision/recall')
# Set a title of the current axes.
plt.title('F1/precision/recall after PCA reduction for XGBoost')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()




exit()

# Load data
fraud_data = pd.read_csv('creditcard.csv')

# Inspect data
fraude_number = len(fraud_data[fraud_data.Class == 1])

# Scale Amount
column_amount = fraud_data.Amount.values
fraud_data.Amount = RobustScaler().fit_transform(column_amount.reshape(-1,1))


# Delete Time 
fraud_data.drop('Time', axis=1, inplace=True)


# Delete duplicates
fraud_data.drop_duplicates(inplace=True)

# Divide to classification data and other attributes

# Classification data Y
Y = fraud_data['Class'].values

# Other attributes data X
data_features = list(fraud_data.columns)
data_features.remove('Class')
X = fraud_data[data_features].values

#podatke razdelimo na učno in testno množico 0.7%, 0.3% (jih premeša in poskrbi da se Y in X enako zmeasat in jih pravilno deli )
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=1)

weight = list()
preceision = list()
recall = list()
f1 = list()

for i in range(0,1001,50):
    n = i
    print(i)
    if(n == 0):
        n = 1
    xgb_model = XGBClassifier(scale_pos_weight=n, max_depth = 4, use_label_encoder=False)
    xgb_model.fit(X_train, Y_train)
    prediction_results_xgb = xgb_model.predict(X_test)
    weight.append(i)
    preceision.append(precision_score(Y_test, prediction_results_xgb))
    recall.append(recall_score(Y_test, prediction_results_xgb))
    f1.append(f1_score(Y_test, prediction_results_xgb))


print(weight)
print(preceision)
print(recall )
print(f1 )