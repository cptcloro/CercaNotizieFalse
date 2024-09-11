import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

def BuildReport(y_test,pred_lr):
    print('REPORT Regressione Logica')
    print(classification_report(y_test,pred_lr))

def Start(xv_train, y_train, xv_test, y_test, report):
    LR = LogisticRegression()
    LR.fit(xv_train,y_train)
    pred_lr = LR.predict(xv_test)
    LR.score(xv_test,y_test)
    #print(LR.score(xv_test,y_test))
    print('REPORT Regressione Logica')
    if report == 1:
        BuildReport(y_test, pred_lr)

