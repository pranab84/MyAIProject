from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load sample dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict a sample
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print("Prediction:", iris.target_names[prediction[0]])