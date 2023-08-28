
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClasssifier
from sklearn.metrics import accuracy_score

subscription_id = 'ba04a809-bb92-41ca-9b73-b1baa9000986'
resource_group = 'MLOPS'
workspace_name = 'Machine_Learning_Projects'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='Iris')
df = dataset.to_pandas_dataframe()


def test_columns():
    print(df.columns.to_list())
    assert df.columns.to_list() == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

def test_classifier_accuracy():
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'species'

    X_train, X_test, y_train,y_test = train_test_split(df[features],df[target], test_size = 0.4, shuffle = True)

    # step 1: initialise the model class
    clf = DecisionTreeClassifier(criterion="entropy")

    # step 2: train the model on training dataset
    clf.fit(X_train, y_train)

    # step 3: evaluate the data on testing dataset
    y_pred = clf.predict(X_test)
    assert accuracy_score(y_test,y_pred) > 0.90

   
    