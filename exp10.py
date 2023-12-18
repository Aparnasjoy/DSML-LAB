from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_breast_cancer()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
max_depth=1
dt = DecisionTreeClassifier(max_depth=max_depth)
dt.fit(x_train,y_train)
print(dt.predict(x_test))
v = dt.predict(x_test)
result = accuracy_score(y_test, v)
report=classification_report(y_test,v)
print(result)
print(report)

# Visualize the decision tree
plt.figure(figsize=(10, 5))
plot_tree(dt, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision tree")
plt.show()
