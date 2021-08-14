import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score


df = pd.read_csv('https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv')

df['Sex'] = df['Sex'].replace({'male':1,'female':0})
df['Cabin'].fillna('Missing',inplace=True)
def fst(x):
    return x[0]
df['Cabin']=df['Cabin'].map(fst)

cabin=df['Cabin'].unique()
df['Cabin']=df['Cabin'].map({k:v for v,k in enumerate(cabin,0)})


df['Age1']= df['Age']
sample=df['Age1'].dropna().sample(df['Age1'].isnull().sum(),random_state=0)
sample.index= df[df['Age1'].isnull()].index
df.loc[df['Age1'].isnull(),'Age1']=sample

df['Age']=df['Age1']
df.drop(['PassengerId','Age1','Name','Ticket','Embarked','Cabin'],axis=1,inplace=True)


X = df.drop('Survived',axis=1)
y=df['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=355)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pre = clf.predict(X_test)


sns.heatmap(confusion_matrix(y_test,y_pre),annot=True)
plt.show()
print(clf.score(X_test,y_test))
print(confusion_matrix(y_test,y_pre))


with open('model.pickle','wb') as f:
    pickle.dump(clf,f)