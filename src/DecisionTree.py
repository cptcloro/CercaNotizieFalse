import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

def BuildReport(y_test,pred_dt):
    print('REPORT Decision Tree:')
    print(classification_report(y_test,pred_dt))

def Start(xv_train, y_train, xv_test, y_test, report):
    DT = DecisionTreeClassifier()
    DT.fit(xv_train,y_train)
    pred_dt = DT.predict(xv_test)
    DT.score(xv_test,y_test)
    print('REPORT Decision tree:')
    print(classification_report(y_test,pred_dt))
    if report == 1:
        BuildReport(y_test, pred_dt)
    return DT


#Decision Tree Classification
#DT = DecisionTreeClassifier()
#DT.fit(xv_train,y_train)
#pred_dt = DT.predict(xv_test)
#DT.score(xv_test,y_test)
#print('REPORT Decision tree:')
#print(classification_report(y_test,pred_dt))