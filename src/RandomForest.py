import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

def BuildReport(y_test,pred_rfc):
    print('REPORT Regressione Logica')
    print(classification_report(y_test,pred_rfc))

def Start(xv_train, y_train, xv_test, y_test, report):
    RFC = RandomForestClassifier(random_state=0)
    RFC.fit(xv_train, y_train)
    pred_rfc = RFC.predict(xv_test)
    RFC.score(xv_test, y_test)
    print('REPORT Regressione Logica')
    if report == 1:
        BuildReport(y_test, pred_rfc)
    return RFC