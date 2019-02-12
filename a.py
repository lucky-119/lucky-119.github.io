import matplotlib.pyplot as plt; 
from sklearn import linear_model,tree;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import confusion_matrix, precision_score, r2_score, f1_score, accuracy_score, recall_score, mean_squared_error;
import pandas as pd;
import tensorflow
import keras;

dataset=pd.read_csv("Immunotherapy.csv");

x=dataset[["age","Time"]];
y=dataset["Result_of_Treatment"];

x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.4,random_state=0);

lr=linear_model.LinearRegression();
lr.fit(x_train,y_train);
y_pred=lr.predict(x_test);

print(lr.coef_);
print(accuracy_score(y_test,y_pred));
print(recall_score(y_test,y_pred));
print(confusion_matrix(y_test,y_pred));
print(f1_score(y_test,y_pred));
print(r2_score(y_test,y_train));
print(precision_score(y_test,y_pred));
print(pow(mean_squared_error(y_test,y_pred),0.5));

logr=linear_model.LogisticRegression();
logr.fit(x_train,y_train);
y_pred=logr.predict(x_test);

plt.scatter(x_test,y_test,color="blue");
plt.plot(x_test,y_pred,color="red");
plt.xlabel();
plt.ylabel();
plt.show();

dt=tree.DecisionTreeClassifier();
dt.fit(x_train,y_train);
y_pred=dt.predict(x_test);
