import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

def BuildReport(y_test,pred_gbc):
    print('REPORT Gradient Boosting')
    print(classification_report(y_test,pred_gbc))

def Start(xv_train, y_train, xv_test, y_test, report):
    GBC = GradientBoostingClassifier(random_state=0)
    GBC.fit(xv_train, y_train)
    pred_gbc = GBC.predict(xv_test)
    GBC.score(xv_test, y_test)
    if report == 1:
        BuildReport(y_test, pred_gbc)
    return GBC