from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

#Split it in features and labels

X = iris.data
y = iris.target

#print(X,y)
print(f'Iris data: {X.shape}')
print(f'Iris target: {y.shape}')

#hours of study vs good/bad grades
#10 different students

#train with 8
#predict with the 2 remaining
# allows for determining the model accuracy
# level of accuracy

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#Test size is percentage split into the test portion

print(f'X_train Shape: {X_train.shape}')
print(f'X_train Shape: {X_test.shape}')
print(f'y_train Shape: {y_train.shape}')
print(f'y_test Shape: {y_test.shape}')