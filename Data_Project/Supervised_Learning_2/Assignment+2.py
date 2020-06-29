
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.5** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# # Assignment 2
# 
# In this assignment you'll explore the relationship between model complexity and generalization performance, by adjusting key parameters of various supervised learning models. Part 1 of this assignment will look at regression and Part 2 will look at classification.
# 
# ## Part 1 - Regression

# First, run the following block to set up the variables needed for later sections.

# In[1]:

import numpy as np
np.set_printoptions(suppress = True)
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
#def part1_scatter():
#    import matplotlib.pyplot as plt
    #%matplotlib notebook
#    plt.figure()
#    plt.scatter(X_train, y_train, label='training data')
#    plt.scatter(X_test, y_test, label='test data')
#    plt.legend(loc=4);
    
    
# NOTE: Uncomment the function below to visualize the data, but be sure 
# to **re-comment it before submitting this assignment to the autograder**.   
#part1_scatter()


# In[ ]:




# ### Question 1
# 
# Write a function that fits a polynomial LinearRegression model on the *training data* `X_train` for degrees 1, 3, 6, and 9. (Use PolynomialFeatures in sklearn.preprocessing to create the polynomial features and then fit a linear regression model) For each model, find 100 predicted values over the interval x = 0 to 10 (e.g. `np.linspace(0,10,100)`) and store this in a numpy array. The first row of this array should correspond to the output from the model trained on degree 1, the second row degree 3, the third row degree 6, and the fourth row degree 9.
# 
# <img src="readonly/polynomialreg1.png" style="width: 1000px;"/>
# 
# The figure above shows the fitted models plotted on top of the original data (using `plot_one()`).
# 
# <br>
# *This function should return a numpy array with shape `(4, 100)`*

# In[2]:

def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    import decimal
    
    
    
    a = []  
    
    for i in (1,3,6,9):
    
        poly1 = PolynomialFeatures(degree = i)

        linreg = LinearRegression()

        X_poly1 = poly1.fit_transform(X_train.reshape(-1,1))
        
        

        linreg.fit(X_poly1,y_train)



        var1 = np.linspace(0,10,100)

        var2 = var1[:,np.newaxis] # = .reshape(-1,1)

        var3 = poly1.fit_transform(var2)



        prediction = linreg.predict(var3)

        prediction2 = prediction.reshape(-1,100)
    
    
        a.append(prediction2)
    
    b = np.array(a)

    c = b.reshape(4,100)
    return c

#answer_one()


# In[3]:

# feel free to use the function plot_one() to replicate the figure 
# from the prompt once you have completed question one
#def plot_one(degree_predictions):
#    import matplotlib.pyplot as plt
    #%matplotlib notebook
    #plt.figure(figsize=(10,5))
    #plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    #plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    #for i,degree in enumerate([1,3,6,9]):
    #    plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    #plt.ylim(-1,2.5)
    #plt.legend(loc=4)

#plot_one(answer_one())


# In[ ]:




# ### Question 2
# 
# Write a function that fits a polynomial LinearRegression model on the training data `X_train` for degrees 0 through 9. For each model compute the $R^2$ (coefficient of determination) regression score on the training data as well as the the test data, and return both of these arrays in a tuple.
# 
# *This function should return one tuple of numpy arrays `(r2_train, r2_test)`. Both arrays should have shape `(10,)`*

# In[4]:

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score
    
    r2_train_score = []
    r2_test_score = []
    
    for i in (0,1,2,3,4,5,6,7,8,9):
        
        poly2 = PolynomialFeatures(degree = i)
        
        linreg2 = LinearRegression()
        

        X_poly2 = poly2.fit_transform(X_train.reshape(-1,1))
        
        X_poly3 = poly2.fit_transform(X_test.reshape(-1,1))
        
        
        
        linreg3 = linreg2.fit(X_poly2,y_train)
        
        r2_train = linreg3.score(X_poly2,y_train)
        
        r2_test = linreg3.score(X_poly3,y_test)
        
        
        
        
        r2_train_score.append(r2_train)
        r2_test_score.append(r2_test)
        
        
    d = np.array(r2_train_score)
    e = np.array(r2_test_score)
        
    d = d.reshape(10,)
    e = e.reshape(10,)
   
    return(d,e)
        
 
        
        #print(linreg2.score(X_poly2,y_test)
        
        
        
        
        

    # Your code her
    #return # Your answer here
    
#answer_two()




# In[5]:

#def answer_two():
#    from sklearn.linear_model import LinearRegression
#    from sklearn.preprocessing import PolynomialFeatures
#    from sklearn.metrics.regression import r2_score
    
#    r2_train = np.zeros(10)
#    r2_test = np.zeros(10)
    
    # Your code here
#    for i in range(10):
#        poly = PolynomialFeatures(degree=i)
        
        # Train and score x_train
#        X_poly = poly.fit_transform(X_train.reshape(11,1))
#        linreg = LinearRegression().fit(X_poly, y_train)        
#        r2_train[i] = linreg.score(X_poly, y_train);
        
        # Score x_test (do not train)
#        X_test_poly = poly.fit_transform(X_test.reshape(4,1))
#        r2_test[i] = linreg.score(X_test_poly, y_test)
        
#    return (r2_train, r2_test)# Your answer here

#answer_two()


# ### Question 3
# 
# Based on the $R^2$ scores from question 2 (degree levels 0 through 9), what degree level corresponds to a model that is underfitting? What degree level corresponds to a model that is overfitting? What choice of degree level would provide a model with good generalization performance on this dataset? 
# 
# Hint: Try plotting the $R^2$ scores from question 2 to visualize the relationship between degree level and $R^2$. Remember to comment out the import matplotlib line before submission.
# 
# *This function should return one tuple with the degree values in this order: `(Underfitting, Overfitting, Good_Generalization)`. There might be multiple correct solutions, however, you only need to return one possible solution, for example, (1,2,3).* 

# In[6]:

#def answer_three():
    
#    r2_scores = answer_two()
#    df = pd.DataFrame({'training_score':r2_scores[0], 'test_score':r2_scores[1]})
#    df['diff'] = df['training_score'] - df['test_score']
    
#    df = df.sort(['diff'])
#    good_gen = df.index[0]
    
#    df = df.sort(['diff'], ascending = False)
#    overfitting = df.index[0]
    
#    df = df.sort(['training_score'])
#    underfitting = df.index[0]
    
#    return (underfitting,overfitting,good_gen)

#answer_three()


# In[43]:



def answer_three():
    
    array = answer_two()
    
    table = pd.DataFrame({'training_score': array[0], 'testing_score': array[1]})
    
    table['diff'] = table['testing_score']- table['training_score']
    
    table = round(table,1)
    
    table['good'] = table['testing_score'] == 0.9
    
    table['under'] = table['training_score'] <=0.5
    
    table['over'] = table ['training_score'] == 1.0
    


    
    
    
    table['degree'] = [0,1,2,3,4,5,6,7,8,9]
    
        
    
    table['good'] = np.where(table['testing_score'] == 0.9 ,table['degree'] , 0)
    table['under'] = np.where(table['training_score'] <= 0.5 ,table['degree'] , 0)
    table['over'] = np.where(table['training_score'] == 1.0 ,table['degree'] , 0)
    

    
    good = table.sort(['good'],ascending = False)
    under = table.sort(['under'],ascending = False)
    over = table.sort(['over'],ascending = False)
    
    
    
    good2 = int(good.iloc[0,3])
    under2 = int(under.iloc[0,4])
    over2 = int(over.iloc[0,5])
    
    
    array = (under2,over2,good2)
    
    
    
    return array# Return your answer

#answer_three()






# ### Question 4
# 
# Training models on high degree polynomial features can result in overly complex models that overfit, so we often use regularized versions of the model to constrain model complexity, as we saw with Ridge and Lasso linear regression.
# 
# For this question, train two models: a non-regularized LinearRegression model (default parameters) and a regularized Lasso Regression model (with parameters `alpha=0.01`, `max_iter=10000`) both on polynomial features of degree 12. Return the $R^2$ score for both the LinearRegression and Lasso model's test sets.
# 
# *This function should return one tuple `(LinearRegression_R2_test_score, Lasso_R2_test_score)`*

# In[8]:

def answer_four():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score
    from sklearn.linear_model import Lasso
    

# Polynominal Linear Regression    
    
    # call polynominal and linear regression models
    poly2 = PolynomialFeatures(degree = 12)
        
    linreg2 = LinearRegression()
        
    # fit and transform polynominal model to both the training and testing data
    X_poly2 = poly2.fit_transform(X_train.reshape(-1,1))
        
    X_poly3 = poly2.fit_transform(X_test.reshape(-1,1))
        
        
    # fit polynominal data to linear regression    
    linreg3 = linreg2.fit(X_poly2,y_train)
        
    r2_test_poly = linreg3.score(X_poly3,y_test)
        
    
    
    
    
    
    
    
# Lasso model    

    linlasso = Lasso(alpha = 0.01, max_iter = 10000).fit(X_poly2,y_train)
    
    r2_test_lasso = linlasso.score(X_poly3,y_test)
    
    
    
    
        

   
    return(r2_test_poly, r2_test_lasso)
        
 
        
        #print(linreg2.score(X_poly2,y_test)
        
        
        
        
        

    # Your code her
    #return # Your answer here
    
answer_four()




# In[9]:

#def answer_four():
#    from sklearn.preprocessing import PolynomialFeatures
#    from sklearn.linear_model import Lasso, LinearRegression
#    from sklearn.metrics.regression import r2_score

    # Your code here

#    return # Your answer here


# ## Part 2 - Classification
# 
# Here's an application of machine learning that could save your life! For this section of the assignment we will be working with the [UCI Mushroom Data Set](http://archive.ics.uci.edu/ml/datasets/Mushroom?ref=datanews.io) stored in `readonly/mushrooms.csv`. The data will be used to train a model to predict whether or not a mushroom is poisonous. The following attributes are provided:
# 
# *Attribute Information:*
# 
# 1. cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s 
# 2. cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s 
# 3. cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y 
# 4. bruises?: bruises=t, no=f 
# 5. odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s 
# 6. gill-attachment: attached=a, descending=d, free=f, notched=n 
# 7. gill-spacing: close=c, crowded=w, distant=d 
# 8. gill-size: broad=b, narrow=n 
# 9. gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y 
# 10. stalk-shape: enlarging=e, tapering=t 
# 11. stalk-root: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=? 
# 12. stalk-surface-above-ring: fibrous=f, scaly=y, silky=k, smooth=s 
# 13. stalk-surface-below-ring: fibrous=f, scaly=y, silky=k, smooth=s 
# 14. stalk-color-above-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y 
# 15. stalk-color-below-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y 
# 16. veil-type: partial=p, universal=u 
# 17. veil-color: brown=n, orange=o, white=w, yellow=y 
# 18. ring-number: none=n, one=o, two=t 
# 19. ring-type: cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z 
# 20. spore-print-color: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y 
# 21. population: abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y 
# 22. habitat: grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d
# 
# <br>
# 
# The data in the mushrooms dataset is currently encoded with strings. These values will need to be encoded to numeric to work with sklearn. We'll use pd.get_dummies to convert the categorical variables into indicator variables. 

# In[10]:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2



# In[11]:

print(X_train2)


# ### Question 5
# 
# Using `X_train2` and `y_train2` from the preceeding cell, train a DecisionTreeClassifier with default parameters and random_state=0. What are the 5 most important features found by the decision tree?
# 
# As a reminder, the feature names are available in the `X_train2.columns` property, and the order of the features in `X_train2.columns` matches the order of the feature importance values in the classifier's `feature_importances_` property. 
# 
# *This function should return a list of length 5 containing the feature names in descending order of importance.*
# 
# *Note: remember that you also need to set random_state in the DecisionTreeClassifier.*

# In[12]:

from sklearn.tree import DecisionTreeClassifier

def answer_five():
    
    clf = DecisionTreeClassifier(random_state= 0).fit(X_train2, y_train2)

    important_ranked = (clf.feature_importances_)

    column_ranked = (X_train2.columns)



#table = pd.DataFrame({'training_score': array[0], 'testing_score': array[1]})

    i1 = []

    y1 = []

    for i,y in zip (column_ranked,important_ranked):

        i1.append(i)
        y1.append(y)
    
    table1 = pd.DataFrame(i1,columns = ['columns_ranked'])

    table1['important_value'] = pd.DataFrame(y1)

    table2 = table1.sort('important_value', ascending = False).head(5)

    table3 = list(table2.iloc[0:5,0])

    return table3
   
    
answer_five()


# ### Question 6
# 
# For this question, we're going to use the `validation_curve` function in `sklearn.model_selection` to determine training and test scores for a Support Vector Classifier (`SVC`) with varying parameter values.  Recall that the validation_curve function, in addition to taking an initialized unfitted classifier object, takes a dataset as input and does its own internal train-test splits to compute results.
# 
# **Because creating a validation curve requires fitting multiple models, for performance reasons this question will use just a subset of the original mushroom dataset: please use the variables X_subset and y_subset as input to the validation curve function (instead of X_mush and y_mush) to reduce computation time.**
# 
# The initialized unfitted classifier object we'll be using is a Support Vector Classifier with radial basis kernel.  So your first step is to create an `SVC` object with default parameters (i.e. `kernel='rbf', C=1`) and `random_state=0`. Recall that the kernel width of the RBF kernel is controlled using the `gamma` parameter.  
# 
# With this classifier, and the dataset in X_subset, y_subset, explore the effect of `gamma` on classifier accuracy by using the `validation_curve` function to find the training and test scores for 6 values of `gamma` from `0.0001` to `10` (i.e. `np.logspace(-4,1,6)`). Recall that you can specify what scoring metric you want validation_curve to use by setting the "scoring" parameter.  In this case, we want to use "accuracy" as the scoring metric.
# 
# For each level of `gamma`, `validation_curve` will fit 3 models on different subsets of the data, returning two 6x3 (6 levels of gamma x 3 fits per level) arrays of the scores for the training and test sets.
# 
# Find the mean score across the three models for each level of `gamma` for both arrays, creating two arrays of length 6, and return a tuple with the two arrays.
# 
# e.g.
# 
# if one of your array of scores is
# 
#     array([[ 0.5,  0.4,  0.6],
#            [ 0.7,  0.8,  0.7],
#            [ 0.9,  0.8,  0.8],
#            [ 0.8,  0.7,  0.8],
#            [ 0.7,  0.6,  0.6],
#            [ 0.4,  0.6,  0.5]])
#        
# it should then become
# 
#     array([ 0.5,  0.73333333,  0.83333333,  0.76666667,  0.63333333, 0.5])
# 
# *This function should return one tuple of numpy arrays `(training_scores, test_scores)` where each array in the tuple has shape `(6,)`.*

# In[40]:

def answer_six():

    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve


    clf2 = SVC(kernel = 'rbf', C = 1, random_state = 0)
    gamma1 = np.logspace(-4,1,6)

    # the below divide (X_subset,y_subset) into 3 different models. Each model consists training and testting data.
    #Each model tests with 6 different gamma parameters

    train_scores, test_scores = validation_curve(SVC(), X_subset, y_subset, 
                                                  param_name = 'gamma',
                                                  param_range = gamma1,
                                                  scoring = 'accuracy', cv = 3)

    train_s = train_scores.mean(axis = 1)
    
    test_s = test_scores.mean(axis = 1)

  
    
    
    
    return train_s, test_s
    
answer_six()


# In[41]:

#def answer_six():
#    from sklearn.svm import SVC
#    from sklearn.model_selection import validation_curve

    # Your code here

#    return # Your answer here


# In[42]:

#def answer_six():
#    from sklearn.svm import SVC
#    from sklearn.model_selection import validation_curve

#    svc = SVC(kernel='rbf', C=1, random_state=0)
#    gamma = np.logspace(-4,1,6)
#    train_scores, test_scores = validation_curve(svc, X_subset, y_subset,
#                            param_name='gamma',
#                            param_range=gamma,
#                            scoring='accuracy')

#    scores = (train_scores.mean(axis=1), test_scores.mean(axis=1))
        
#    return scores # Your answer here


# In[137]:

#print(answer_six())



# ### Question 7
# 
# Based on the scores from question 6, what gamma value corresponds to a model that is underfitting (and has the worst test set accuracy)? What gamma value corresponds to a model that is overfitting (and has the worst test set accuracy)? What choice of gamma would be the best choice for a model with good generalization performance on this dataset (high accuracy on both training and test set)? 
# 
# Hint: Try plotting the scores from question 6 to visualize the relationship between gamma and accuracy. Remember to comment out the import matplotlib line before submission.
# 
# *This function should return one tuple with the degree values in this order: `(Underfitting, Overfitting, Good_Generalization)` Please note there is only one correct solution.*

# In[51]:



def answer_seven():
    
    array = answer_six()
    
    table = pd.DataFrame({'training_score': array[0], 'testing_score': array[1]})
    
    table['diff'] = table['testing_score']- table['training_score']
    
    table = round(table,3)
    
    table['good'] = table['testing_score'] == 0.9
    
    table['under'] = table['training_score'] <=0.5
    
    table['over'] = table ['training_score'] == 1.0
    


    
    
    
    table['gamma_degree'] = np.logspace(-4,1,6)
    
        
    


    
    good = table.sort(['good'],ascending = False)
    under = table.sort(['under'],ascending = False)
    over = table.sort(['over'],ascending = False)
    
    
    
    #good2 = int(good.iloc[0,3])
    #under2 = int(under.iloc[0,4])
    #over2 = int(over.iloc[0,5])
    
    
    #array = (under2,over2,good2)
    
    table = table.sort('diff',ascending = False)
    
    under = table.iloc[0,6]
    over = table.iloc[5,6]
    good = table.iloc[1,6]
      
    All = (under, over, good)    
    return All# Return your answer

answer_seven()






# In[ ]:

#def answer_seven():
    
    # Your code here
    
#    return # Return your answer

