# Survival Prediction in Titanic

# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import TransformerMixin

class SeriesImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        If the Series is of dtype Object, then impute with the most frequent object.
        If the Series is not of dtype Object, then impute with the mean.  

        """
    def fit(self, X, y=None):
        if   X.dtype == np.dtype('O'):
            self.fill = X.value_counts().index[0]
        else:
            self.fill = X.mean()
        return self

    def transform(self, X, y=None):
       return X.fillna(self.fill)

# Importing the training and test datasets
#X_train = pd.read_csv(filepath_or_buffer = 'Data/train.csv', header=0, usecols=['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
X_train = pd.read_csv(filepath_or_buffer = 'Data/train.csv', header='infer', usecols=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
X_test = pd.read_csv(filepath_or_buffer = 'Data/test.csv', header='infer', usecols=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
y_train = pd.read_csv(filepath_or_buffer = 'Data/train.csv', header='infer', usecols=['Survived'])
testset =  pd.read_csv(filepath_or_buffer = 'Data/test.csv')

# Dropping Cabin column by 'NaN' excess
#X_train = X_train.drop('Cabin', 1)
#X_test = X_test.drop('Cabin', 1)

# Emcoding categorical data in training set
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train.loc[:, 'Pclass'] = labelencoder_X.fit_transform(X_train.loc[:, 'Pclass'])
#X_train.loc[:, 'Name'] = labelencoder_X.fit_transform(X_train.loc[:, 'Name'])
X_train.loc[:, 'Sex'] = labelencoder_X.fit_transform(X_train.loc[:, 'Sex'])
#X_train.loc[:, 'Ticket'] = labelencoder_X.fit_transform(X_train.loc[:, 'Ticket'])
a = SeriesImputer()
a.fit(X_train.loc[:, 'Embarked'])
X_train.loc[:, 'Embarked'] = a.transform(X_train.loc[:, 'Embarked'])
X_train.loc[:, 'Embarked'] = labelencoder_X.fit_transform(X_train.loc[:, 'Embarked'])

# Encoding categorical data in test set
labelencoder_X = LabelEncoder()
X_test.loc[:, 'Pclass'] = labelencoder_X.fit_transform(X_test.loc[:, 'Pclass'])
#X_test.loc[:, 'Name'] = labelencoder_X.fit_transform(X_test.loc[:, 'Name'])
X_test.loc[:, 'Sex'] = labelencoder_X.fit_transform(X_test.loc[:, 'Sex'])
#X_test.loc[:, 'Ticket'] = labelencoder_X.fit_transform(X_test.loc[:, 'Ticket'])
a = SeriesImputer()
a.fit(X_test.loc[:, 'Embarked'])
X_test.loc[:, 'Embarked'] = a.transform(X_test.loc[:, 'Embarked'])
X_test.loc[:, 'Embarked'] = labelencoder_X.fit_transform(X_test.loc[:, 'Embarked'])

# Filling null data
X_train = X_train.values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train)
imputer = imputer.fit(X_test)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

#onehotencoder = OneHotEncoder(categorical_features = [1, 2, 3, 7, 9])
onehotencoder = OneHotEncoder(categorical_features = [0, 1, 6])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.fit_transform(X_test).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X1, X2, y1, y2 = train_test_split(X_train, y_train, test_size = 0.25, random_state = 0)

"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1 = sc.fit_transform(X1)
X2 = sc.transform(X2)
"""

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X1, y1)

# Predicting results for training set
y_pred_train = classifier.predict(X_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_train, y_pred_train)
accuracy = accuracy_score(y_train, y_pred_train)

# Predicting results for test set
y_pred_test = classifier.predict(X_test)

# Writting the predictions
submission = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission.iloc[:, 0] = testset.loc[:, 'PassengerId']
submission.iloc[:, 1] = y_pred_test
df_csv = submission.to_csv('Logistic Regression Submission.csv', index=False)

"""
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 2].min() - 1, stop = X_set[:, 2].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""