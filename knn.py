import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_excel ('data.xls', header =None )
X= dataset.iloc[:,1:]
Y= dataset.iloc[:,0]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y , test_size=0.25, random_state=42)
neighbors = np.arange(1, 6)

from sklearn.neighbors import KNeighborsClassifier
for i, k in enumerate(neighbors):
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    print(knn.score(x_test, y_test))

classifier= KNeighborsClassifier(n_neighbors=5, p=2, metric='euclidean')
classifier.fit(x_train, y_train)
y_pred= classifier.predict(x_test)
print(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

total1 = sum(sum(cm))
Accuracy = (cm[0, 0] + cm[1, 1]) / total1
Specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
Sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
precision_positive = precision_score(y_test, y_pred, pos_label=1)
precision_negative = precision_score(y_test, y_pred, pos_label=0)
print(precision_positive, precision_negative)
print(Accuracy)
print(Specificity)
print(Sensitivity)
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

