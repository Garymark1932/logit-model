# logit-model
already cleaned up data 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('C:/Users/user/Desktop/scm651_homework_4_universal_bank.csv')
y = df.PersonalLoan
cols = ['Age','Experience','Income','Education','CCAvg','Mortgage','Family']
X = df[cols]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
