import numpy as np
import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import linear_model, svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import os



# Cargamos y limpiamos el excel
csv = pd.read_csv('src/data/raw/heart.csv', sep=',')
df = csv[csv.Cholesterol != 0]

# Convetimos las columnas categóricas en numéricas
get_dumm_df = pd.get_dummies(df, drop_first=True)

# Dividimos los datos en train y test -> Nuestro target será 'HeartDisease': 0/1
X = np.array(get_dumm_df.drop(['HeartDisease'], axis = 1))
y = np.array(get_dumm_df['HeartDisease'])
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state = 42)

# Montamos el Pipeline y el GridSearch con el mejor modelo validado en los notebooks de prueba    
svc = Pipeline([
    ("scaler", StandardScaler()),
    ("selectkbest", SelectKBest()),
    ("svc", svm.SVC())])

svc_param = {
    "selectkbest__k": [1,2,3],
    "svc__C": np.arange(0.1, 0.9, 0.1),
    "svc__kernel": ['linear', 'poly', 'rbf']
}

gs_svm = GridSearchCV(svc,
svc_param,
cv=10,
scoring = 'recall',
n_jobs = -1,
verbose = 1)

# Entrenamos el algoritmo
gs_svm.fit(X_train, y_train)




## Esta última parte del código sólo sirve en caso de querer testear si el modelo está correctamente entrenado
# y_pred_svm = gs_svm.predict(X_test)
# print('SVM Test','\n', metrics.confusion_matrix(y_test, y_pred_svm))                                        
