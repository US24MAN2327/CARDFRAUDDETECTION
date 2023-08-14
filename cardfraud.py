import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
import time
import warnings
warnings.filterwarnings('ignore')

raw_data = pd.read_csv('transdata.csv')
print("There are " + str(len(raw_data)) + " records in the card fraud dataset.")
print("There are " + str(len(raw_data.columns)) + " variables.")
frequency_table = raw_data.groupby(['Time']).size().reset_index(name='Frequency')

print(frequency_table)
labels = raw_data.Class.unique()
sizes = raw_data.Class.value_counts().values
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target  Value Counts')
plt.show()

plt.hist(raw_data.Amount.values, 6, histtype='bar', facecolor='g')
plt.show()

print("Minimum amount : ", np.min(raw_data.Amount.values))
print("Maximum amount : ", np.max(raw_data.Amount.values))
print("90% of the transactions have an amount less or equal than ", np.percentile(raw_data.Amount.values, 90))


raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(raw_data.iloc[:, 1:30])
data_matrix = raw_data.values
# X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
X = data_matrix[:, 1:30]
# y: labels vector
y = data_matrix[:, 30]
# data normalization
X = normalize(X, norm="l1")
# print the shape of the features matrix and the labels vector
print('X.shape=', X.shape, 'y.shape=', y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)       
print('X_train.shape=', X_train.shape, 'Y_train.shape=', y_train.shape)
print('X_test.shape=', X_test.shape, 'Y_test.shape=', y_test.shape)


from snapml import DecisionTreeClassifier

snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=45, n_jobs=4)
w_train = compute_sample_weight('balanced', y_train)
# train a Decision Tree Classifier model using Snap ML
t0 = time.time()
snapml_dt.fit(X_train, y_train, sample_weight=w_train)
snapml_time = time.time()-t0
print(" Training time using Snap ML (s):  {0:.5f}".format(snapml_time))


from sklearn.tree import DecisionTreeClassifier

# for reproducible output across multiple function calls, set random_state to a given integer value
sklearn_dt = DecisionTreeClassifier(max_depth=4, random_state=35)

# train a Decision Tree Classifier using scikit-learn
t0 = time.time()
sklearn_dt.fit(X_train, y_train, sample_weight=w_train)
sklearn_time = time.time()-t0
print(" Training time using Sk-learn (s):  {0:.5f}".format(sklearn_time))


sklearn_pred = sklearn_dt.predict_proba(X_test)[:,1]

# evaluate the Compute Area Under the Receiver Operating Characteristic 
# Curve (ROC-AUC) score from the predictions
sklearn_roc_auc = roc_auc_score(y_test, sklearn_pred)
print(' ROC-AUC score of sk-learn : {0:.3f}'.format(sklearn_roc_auc))

# run inference and compute the probabilities of the test samples
# to belong to the class of fraudulent transactions
snapml_pred = snapml_dt.predict_proba(X_test)[:,1]

# evaluate the Compute Area Under the Receiver Operating Characteristic
# Curve (ROC-AUC) score from the prediction scores
snapml_roc_auc = roc_auc_score(y_test, snapml_pred)   
print(' ROC-AUC score of Snap ML : {0:.3f}'.format(snapml_roc_auc))



