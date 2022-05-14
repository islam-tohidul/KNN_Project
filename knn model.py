# Import libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

# Read the data
data = pd.read_csv('car.data')

print(data.head())

le = preprocessing.LabelEncoder()

# transform to numeric values
data['buying'] = le.fit_transform(data['buying'])
data['maint'] = le.fit_transform(data['maint'])
data['doors'] = le.fit_transform(data['doors'])
data['persons'] = le.fit_transform(data['persons'])
data['lug_boot'] = le.fit_transform(data['lug_boot'])
data['safety']= le.fit_transform(data['safety'])
data['class'] = le.fit_transform(data['class'])

# choose attributes & label
X = data[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']]
y = data['class']

# split into train and test
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# pick a model
model = KNeighborsClassifier(n_neighbors=9)

# train the model
model.fit(X_train, y_train)

# test the accuracy the of the model
acc = model.score(X_test, y_test)

print(acc)

names = ['acc', 'good', 'unacc', 'vgood']

# predict values using our model
predictions = model.predict(X_test)

print(predictions)

for i in range(len(predictions)):
    print('Predicted:', predictions[i], 'Data:', X_test[i], 'Actual:', y_test[i])
    print(model.kneighbors([X_test[i]], 9, True))

