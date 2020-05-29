__author__ = 'Jie'
"""
This is actually the final assignment of data challenge from the online course "application of machine learning in PYTHON".
For detail, please refer to my own notebook of this course.
a problem: the test.csv can not match with the provided addresses.csv. So there is not test data for validation

We provide you with two data files for use in training and validating your models: train.csv and test.csv.
Each row in these two files corresponds to a single blight ticket, and includes information about when, why,
and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early,
on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all,
and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will
not be available at test-time, are only included in train.csv.
Note: All tickets where the violators were found not responsible are not considered during evaluation.
They are included in the training set as an additional source of data for visualization, and to enable unsupervised
and semi-supervised approaches. However, they are not included in the test set.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# load all the data
train_df=pd.read_csv('D:/python-ml/kaggles_competition/propertyFinePredict/train.csv',engine='python')
test_df=pd.read_csv('D:/python-ml/kaggles_competition/propertyFinePredict/test.csv',engine='python')
address_df=pd.read_csv('D:/python-ml/kaggles_competition/propertyFinePredict/addresses.csv',engine='python')
latlon_df=pd.read_csv('D:/python-ml/kaggles_competition/propertyFinePredict/latlons.csv',engine='python')

#################################################################################
## data clean, and manipulation
## combine the data
## some data visualization method
# train_df.info()
# train_df.describe().T
# train_df.count()
# train_df.compliance.value_counts()
train_df=train_df.dropna(subset=['compliance'])
address_latlon=pd.merge(address_df,latlon_df,how='inner',on='address')
train_df=pd.merge(train_df,address_latlon,how='inner',on='ticket_id')

#################################################################################
# check if there are any Null elements left.
miss_num=train_df.isnull().sum().sort_values(ascending=False)
count_all=train_df.isnull().count()
miss_temp=(train_df.isnull().sum()*100/count_all).sort_values(ascending=False)
miss_all=pd.concat([miss_num,miss_temp],axis=1,keys=['miss_num','miss_temp'])
#################################################################################

#################################################################################
# split the data by its types for the sake of cleaning
train_num=train_df.select_dtypes('float64')
train_obj=train_df.select_dtypes('object')
labelencoder=LabelEncoder()
labelencoder.fit(train_obj['violation_code'])
feature2=labelencoder.transform(train_obj['violation_code'])

# also, we can use pandans.get_dummies() for an one-hot transformation
feature1=pd.DataFrame(train_obj['disposition'])
feature1=pd.get_dummies(feature1)

# change all the states other than 'MI' into others
# based on their distributions
train_objC=train_obj.copy()  # to avoid the waring "SettingWithCopyWarning".
train_objC.loc[train_obj.state != 'MI','state'] ='OTHER' # No need of loop, which is time_consuming
train_obj1=pd.DataFrame(train_objC.state)
train_obj2=pd.concat([train_obj1,pd.DataFrame(feature2),feature1],axis=1)

# drop some unnecessary columns of train_num dataframe
drops=['violation_zip_code','non_us_str_code','grafitti_status','payment_amount','balance_due','admin_fee',
'state_fee','violation_street_number','mailing_address_str_number','clean_up_cost']
train_num.drop(drops,axis=1,inplace=True)

#################################################################################
# finally, get the clean data for fitting.
train_all=pd.concat([train_num,train_obj2],axis=1)
train_all=pd.get_dummies(train_all)
# print (train_all.shape)
# print (train_all.head())

#################################################################################
# start to fit the data
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
train_y=train_all['compliance']
train_X=train_all.drop(['compliance'],axis=1)
X_train,X_test,y_train,y_test=train_test_split(train_X,train_y)
params={'n_estimators':[10,20,30,40,50],'max_depth':[None,3,5,6,8,10]}
reg=RandomForestRegressor()
gridsearch=GridSearchCV(reg,param_grid=params,scoring='roc_auc',cv=10)
gridsearch.fit(X_train,y_train)

bestscore=gridsearch.best_score_
print ("the best score of cv is: {}".format(bestscore))
print ("the accuracy of train data is: {}".format(gridsearch.score(X_train,y_train)))
print ("the accuracy of test data is: {}".format(gridsearch.score(X_test,y_test)))
