import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

df = pd.read_csv('./data/car-purchase-decision.csv')

X = df.drop(columns=['User ID', 'Purchased'])
y = df['Purchased']

model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, 'car-purchase-recommander.joblib')

