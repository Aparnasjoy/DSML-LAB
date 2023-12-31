from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score ,classification_report
bc=load_breast_cancer()
x=bc.data
y=bc.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
lbc=DecisionTreeClassifier()
lbc.fit(x_train,y_train)
print(lbc.predict(x_test))
v=lbc.predict(x_test)
result=accuracy_score(y_test,v)
print("accuracy:",result)
result=classification_report(y_test,v)
print("classification_report:",result)
