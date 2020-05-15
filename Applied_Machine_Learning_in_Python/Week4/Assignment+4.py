
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend tak/
# ing a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     readonly/train.csv - the training set (all tickets issued 2004-2011)
#     readonly/test.csv - the test set (all tickets issued 2012-2016)
#     readonly/addresses.csv & readonly/latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `readonly/train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `readonly/test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32
#        
# ### Hints
# 
# * Make sure your code is working before submitting it to the autograder.
# 
# * Print out your result to see whether there is anything weird (e.g., all probabilities are the same).
# 
# * Generally the total runtime should be less than 10 mins. You should NOT use Neural Network related classifiers (e.g., MLPClassifier) in this question. 
# 
# * Try to avoid global variables. If you have other functions besides blight_model, you should move those functions inside the scope of blight_model.
# 
# * Refer to the pinned threads in Week 4's discussion forum when there is something you could not figure it out.

# In[ ]:

print('hi')


# In[4]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing 

from sklearn.preprocessing import MinMaxScaler



training = pd.read_csv('train.csv',engine = 'python')
#testing = pd.read_csv('test.csv', engine = 'python')

training = training[pd.notnull(training['compliance'])]

print('reading table done')






#df1.drop(['B', 'C'], axis=1)

X_train_p = training[['ticket_id','fine_amount','judgment_amount','admin_fee','state_fee','late_fee','discount_amount','clean_up_cost','judgment_amount']]

#X_train_p =  pd.get_dummies(X_train_p, prefix=['agency_name','violation_code','disposition'],columns=['agency_name','violation_code','disposition'])

y_train_p = training.iloc[:,33:]


print('modifiy columns done')


# Normalisation

scaler = MinMaxScaler()

X_train_p_scaled = scaler.fit_transform(X_train_p)

print('Normailisation done')









#df1.drop(['B', 'C'], axis=1)






# In[ ]:


##Feature Important


#feat_labels = ['ticket_id','fine_amount','agency_name','inspector_name','violation_code','disposition','judgment_amount','country','admin_fee','state_fee','late_fee','discount_amount','clean_up_cost','judgment_amount']
#for feature in zip(feat_labels, r.feature_importances_):
#    print(feature)





X_train, X_test, y_train, y_test = train_test_split(X_train_p, y_train_p, random_state = 0)





##Grid Search
y_train = y_train['compliance'].values.ravel()

print('y_train reshape done')

#param_grid1 = [{'C': [0.1,1,15],'gamma': [0.1, 0.5,1] }]

#r = SVC()

#grid_rdf = GridSearchCV(r, param_grid = param_grid1, n_jobs=4, scoring = 'roc_auc')

#grid_rdf.fit(X_train, y_train)

#roc_auc_result = grid_rdf.best_params_


#print('Best Parameter:',roc_auc_result )
#print('Best Auc Score:', roc_auc)





r = SVC(C=100,gamma = 0.1)

probability = r.fit(X_train, y_train).predict(X_test)

print(probability)







## AUC

#y_test = y_test['compliance'].values

#y_test = y_test.reshape(-1,1)

#fpr_rdf, tpr_fdr,_ = roc_curve(y_test, probability)

#roc_auc_fdr = auc(fpr_rdf, tpr_fdr)

#print(roc_auc_fdr)


##Metric Print
#print(y_test.shape)



#print(X_test.shape)


#print(X_train.shape)
#print(X_test.shape)

#print(probability1)
#print(probability1)
#print(y_test)


#y_test
#print(y_test.shape)
#print(probability.shape)



#print(y_test.shape)


#redo the ROC, since i removed NAN











# In[23]:

testing = pd.read_csv('test.csv', engine = 'python')


X_test_p = testing[['ticket_id','fine_amount','agency_name','inspector_name','violation_code','disposition','judgment_amount','country','admin_fee','state_fee','late_fee','discount_amount','clean_up_cost','judgment_amount']]

X_test_p =  pd.get_dummies(X_test_p, prefix=['agency_name','inspector_name','violation_code','disposition','country'],columns=['agency_name','inspector_name','violation_code','disposition','country'])


print(X_test_p.shape)
print(X_train_p.shape)



test_probability = r.fit(X_train, y_train).predict_proba(X_test_p)

print(test_probability)




# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing 





training = pd.read_csv('train.csv',engine = 'python')
#testing = pd.read_csv('test.csv', engine = 'python')




training = training[pd.notnull(training['compliance'])]








#df1.drop(['B', 'C'], axis=1)

X_train_p = training[['ticket_id','fine_amount','agency_name','inspector_name','violation_code','disposition','judgment_amount','country','admin_fee','state_fee','late_fee','discount_amount','clean_up_cost','judgment_amount']]

X_train_p =  pd.get_dummies(X_train_p, prefix=['agency_name','inspector_name','violation_code','disposition','country'],columns=['agency_name','inspector_name','violation_code','disposition','country'])

y_train_p = training.iloc[:,33:]



feat_labels = ['ticket_id','fine_amount','agency_name','inspector_name','violation_code','disposition','judgment_amount','country','admin_fee','state_fee','late_fee','discount_amount','clean_up_cost','judgment_amount']


#X_train[['agency_name','inspector_name','violator_name','violation_street_name','mailing_address_str_name','city','state','zip_code','non_us_str_code','country','ticket_issued_date','hearing_date','violation_code','violation_description','disposition','grafitti_status']] = X_train[['agency_name','inspector_name','violator_name','violation_street_name','mailing_address_str_name','city','state','zip_code','non_us_str_code','country','ticket_issued_date','hearing_date','violation_code','violation_description','disposition','grafitti_status']].stack().astype('category').unstack()



X_train, X_test, y_train, y_test = train_test_split(X_train_p, y_train_p, random_state = 0)




y_train = y_train['compliance'].values



param_grid1 = [{'n_estimators': [1,10,100],'max_features': [10,20,25,30,35]}]



r = RandomForestRegressor(random_state = 0)



grid_rdf = GridSearchCV(r, param_grid = param_grid1, scoring = 'roc_auc')


grid_rdf.fit(X_train, y_train)

roc_auc_result = grid_rdf.best_params_



print('Best Parameter:',roc_auc_result )





r = RandomForestRegressor(n_estimators = 100, max_features = 35, random_state = 0)







probability = r.fit(X_train, y_train).predict(X_test)



for feature in zip(feat_labels, r.feature_importances_):
    print(feature)

#probability1 = probability.reshape(-1,1)

#probability1 = probability1.astype(int)



y_test = y_test['compliance'].values



#y_test = y_test.as_matrix()

#y_test = y_test.reshape(-1,0)

fpr_rdf, tpr_fdr,_ = roc_curve(y_test, probability)

roc_auc_fdr = auc(fpr_rdf, tpr_fdr)

roc_auc_fdr


#print(y_test.shape)
#print(probability.shape)


#print(X_test.shape)


#print(X_train.shape)
#print(X_test.shape)

#print(probability1)
#print(probability1)
#print(y_test)


#y_test
#print(y_test.shape)
#print(probability.shape)



#print(y_test.shape)


#redo the ROC, since i removed NAN



# In[ ]:

import pandas as pd
import numpy as np

def blight_model():
    
    # Your code here
    
    return # Your answer here


# In[ ]:

blight_model()

