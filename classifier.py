import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("/content/MLOPS/Data/iris.csv")

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target = 'species'

X_train, X_test, y_train,y_test = train_test_split(df[features],df[target], test_size = 0.3, shuffle = True)

# step 1: initialise the model class
clf = DecisionTreeClassifier(criterion="entropy")

# step 2: train the model on training dataset
clf.fit(X_train, y_train)

# step 3: evaluate the data on testing dataset
y_pred = clf.predict(X_test)

print(f"Accuracy of the model is {accuracy_score(y_test,y_pred)*100}")
