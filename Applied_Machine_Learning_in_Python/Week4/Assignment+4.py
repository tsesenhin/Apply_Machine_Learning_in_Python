
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

# In[35]:

#from sklearn import preprocessing
#from sklearn.svm import SVC, LinearSVC
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import roc_curve, auc
#from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.preprocessing import LabelEncoder
#import numpy as np
#import pandas as pd

#training = pd.read_csv('train.csv', engine='python',index_col = False)


#training = training[pd.notnull(training['compliance'])]


#X_train_p = training[['ticket_id', 'fine_amount', 'agency_name', 'inspector_name', 'violation_code', 'disposition',
#                          'judgment_amount', 'country', 'admin_fee', 'state_fee', 'late_fee', 'discount_amount', 'clean_up_cost', 'judgment_amount']]

#le = LabelEncoder()

#X_train_p['agency_name'] = le.fit_transform(X_train_p['agency_name'])
#X_train_p['inspector_name'] = le.fit_transform(X_train_p['inspector_name'])
#X_train_p['violation_code'] = le.fit_transform(X_train_p['violation_code'])
#X_train_p['disposition'] = le.fit_transform(X_train_p['disposition'])
#X_train_p['country'] = le.fit_transform(X_train_p['country'])
#X_train_p.head()




#y_train_p = training.iloc[:, 33:]

#X_train, X_test, y_train, y_test = train_test_split(X_train_p, y_train_p, random_state=0)

#y_train = y_train['compliance'].values

#r = RandomForestClassifier(n_estimators=100, max_features=14, random_state=0)

#probability = r.fit(X_train, y_train).predict_proba(X_test)

#y_test = y_test['compliance'].values

#y_test = y_test.reshape(-1,)









# In[46]:

#def hi():

#    testing = pd.read_csv('test.csv', engine='python',index_col = False)


#    X_test_p = testing[['ticket_id', 'fine_amount', 'agency_name', 'inspector_name', 'violation_code', 'disposition',
#                              'judgment_amount', 'country', 'admin_fee', 'state_fee', 'late_fee', 'discount_amount', 'clean_up_cost', 'judgment_amount']]

#    X_test_p['agency_name'] = le.fit_transform(X_test_p['agency_name'])
#    X_test_p['inspector_name'] = le.fit_transform(X_test_p['inspector_name'])
#    X_test_p['violation_code'] = le.fit_transform(X_test_p['violation_code'])
#    X_test_p['disposition'] = le.fit_transform(X_test_p['disposition'])
#    X_test_p['country'] = le.fit_transform(X_test_p['country'])

#    X_test_p.head()



#    r = RandomForestClassifier(n_estimators=100, max_features=14, random_state=0)

#    test_probability = r.fit(X_train, y_train).predict_proba(X_test_p)

        #test_probability = round(test_probability,8)

#   Final = pd.Series(test_probability[:,1], index = None)
    
#    X_test_p['paying_probability'] = test_probability[:,1]
    
#    Ticket_id = pd.Series(X_test_p['ticket_id'], index = None)
    
    #Ticket_id = X_test_p['ticket_id']
    
#    Final = X_test_p[['ticket_id','paying_probability']]
    

    
#    Final = X_test_p.set_index('ticket_id')
    
    
    
#    Final = Final.iloc[:,-1]

    
    #df.style.hide_index()
    
#    return Final


#hi()


# In[48]:

from sklearn import preprocessing
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

def blight_model():
    

    training = pd.read_csv('train.csv', engine='python')


    training = training[pd.notnull(training['compliance'])]


    X_train_p = training[['ticket_id', 'fine_amount', 'agency_name', 'inspector_name', 'violation_code', 'disposition',
                          'judgment_amount', 'country', 'admin_fee', 'state_fee', 'late_fee', 'discount_amount', 'clean_up_cost', 'judgment_amount']]

    le = LabelEncoder()

    X_train_p['agency_name'] = le.fit_transform(X_train_p['agency_name'])
    X_train_p['inspector_name'] = le.fit_transform(X_train_p['inspector_name'])
    X_train_p['violation_code'] = le.fit_transform(X_train_p['violation_code'])
    X_train_p['disposition'] = le.fit_transform(X_train_p['disposition'])
    X_train_p['country'] = le.fit_transform(X_train_p['country'])
    X_train_p.head()




    y_train_p = training.iloc[:, 33:]

    X_train, X_test, y_train, y_test = train_test_split(X_train_p, y_train_p, random_state=0)

    y_train = y_train['compliance'].values

    r = RandomForestClassifier(n_estimators=100, max_features=14, random_state=0)

    probability = r.fit(X_train, y_train).predict_proba(X_test)

    y_test = y_test['compliance'].values

    y_test = y_test.reshape(-1,)








    testing = pd.read_csv('test.csv', engine='python',index_col = False)


    X_test_p = testing[['ticket_id', 'fine_amount', 'agency_name', 'inspector_name', 'violation_code', 'disposition',
                              'judgment_amount', 'country', 'admin_fee', 'state_fee', 'late_fee', 'discount_amount', 'clean_up_cost', 'judgment_amount']]

    X_test_p['agency_name'] = le.fit_transform(X_test_p['agency_name'])
    X_test_p['inspector_name'] = le.fit_transform(X_test_p['inspector_name'])
    X_test_p['violation_code'] = le.fit_transform(X_test_p['violation_code'])
    X_test_p['disposition'] = le.fit_transform(X_test_p['disposition'])
    X_test_p['country'] = le.fit_transform(X_test_p['country'])

    X_test_p.head()



    r = RandomForestClassifier(n_estimators=100, max_features=14, random_state=0)

    test_probability = r.fit(X_train, y_train).predict_proba(X_test_p)

        #test_probability = round(test_probability,8)

#   Final = pd.Series(test_probability[:,1], index = None)
    
    X_test_p['paying_probability'] = test_probability[:,1]
    
#    Ticket_id = pd.Series(X_test_p['ticket_id'], index = None)
    
    #Ticket_id = X_test_p['ticket_id']
    
#    Final = X_test_p[['ticket_id','paying_probability']]
    

    
    Final = X_test_p.set_index('ticket_id')
    
    
    
    Final = Final.iloc[:,-1]

    
    #df.style.hide_index()
    
    return Final


# In[49]:

blight_model()


# In[50]:

#import numpy as np
#bm = blight_model()
#res = '{:40s}'.format('Object Type:')
#res += ['Failed: type(bm) should Series\n','Passed\n'][type(bm)==pd.Series]
#res += '{:40s}'.format('Data Shape:')
#res += ['Failed: len(bm) should be 61001\n','Passed\n'][len(bm)==61001]
#res += '{:40s}'.format('Data Values Type:')
#res += ['Failed: bm.dtype should be float\n','Passed\n'][str(bm.dtype).count('float')>0]
#res += '{:40s}'.format('Data Values Infinity:')
#res += ['Failed: values should not be infinity\n','Passed\n'][not any(np.isinf(bm))]
#res += '{:40s}'.format('Data Values NaN:')
#res += ['Failed: values should not be NaN\n','Passed\n'][not any(np.isnan(bm))]
#res += '{:40s}'.format('Data Values in [0,1] Range:')
#res += ['Failed: all values should be in [0.,1.]\n','Passed\n'][all((bm<=1.) & (bm>=0.))]
#res += '{:40s}'.format('Data Values not all 0 or 1:')
#res += ['Failed: values should be scores not predicted labels\n','Passed\n'][not all((bm.isin({0,1,0.0,1.0})))]
#res += '{:40s}'.format('Index Type:')
#res += ['Failed: type(bm.index) should be Int64Index\n','Passed\n'][type(bm.index)==pd.Int64Index]
#res += '{:40s}'.format('Index Values:')
#if bm.index.shape==(61001,):
#    res +=['Failed: index values should match test.csv\n','Passed\n'
#          ][all(pd.read_csv('test.csv',usecols=[0],index_col=0
#                           ).sort_index().index.values==bm.sort_index().index.values)]
#else:
#    res+='Failed: bm.index length should be 61001'
#res += '{:40s}'.format('Can run model twice:')
#bm2 = None
#try:
#    bm2 = blight_model()
#    res += 'Passed\n'
#except:
#    res += ['Failed: second run of blight_model() threw an Exception']
#res += '{:40s}'.format('Can run model twice with same results:')
#if not bm2 is None:
#    res += ['Failed: second run of blight_model() produced different results (this might not be a problem)\n','Passed\n'][
#        all(bm.apply(lambda x:round(x,3))==bm2.apply(lambda x:round(x,3))) and all(bm.index==bm2.index)]    
#print(res)


# In[ ]:



