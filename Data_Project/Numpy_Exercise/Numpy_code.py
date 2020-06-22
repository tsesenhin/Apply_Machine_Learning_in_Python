#Q1. Import numpy as np and print the version number.
import numpy as np

print(np.__version__)


#Create a 1D array of numbers from 0 to 9

arr = np.arange(10)

arr



# create array of 1,2,3,4,5

arr2 = np.array([1,2,3,4,5])

type(arr2)


# you can pass list, tuple or array-like object into np.array
# In here, we pass tuple into a np.array
arr3 = np.array((1,2,3,4,5))

type(arr3)


# A dimension in arrays is one level of array depth 

#0-D array
arr0 = np.array((42))

arr0.ndim


#1-D array

arr4 = np.array((42,21,43))

arr4.ndim



#2-D array
# is 2 becaz the array has x and y axis
arr5 = np.array(((1,2,3),(4,5,6)))

arr5


#3-D array
# In numpy, an array contains multiple 2-D arrays is defined as 3D array

arr6 = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[11,12,13],[14,15,16],[17,18,19]],[[11,12,13],[14,15,16],[17,18,19]]])

arr6

arr6.ndim

# go to http://jalammar.github.io/visual-numpy/ for a better explaination for 3D space


# Create a bespoken array with x-dimension
# in this case, 5 dimension, note numpy cant visualised anything beyond 3d array, even 3d array display itself is weird
 arr7 = np.array([1,2,3,4,5], ndmin = 5)

 arr7.ndim

 arr7






# Arry indexing

#get the first element from the following array:
arr8 = np.array([1,2,3,4])
print(arr8[0])


# get the second element
print(arr8[1])

# get the third and fourth element and sum them up
print(arr8[2]+arr8[3])



# Access a 2-D array
arr9 = np.array([[1,2,3,4,5],[6,7,8,9,10]])

arr9.ndim

arr9[1,2] # select row 2, and column 3, which is = 8

arr9[1,-1] # select row 2, and last column




# Access a 3-D array
arr10 = np.array([[[1,2,4],[4,5,6]],[[7,8,9],[10,11,12]]])
arr10
arr10[0,1,2]






# Slicing Array

arr11 = np.array([1,2,3,4,5,6,7])

arr11[0:3]

arr11[0:] # from index position 0 to n

arr11[:4] # from n to a value prior to index 4

arr11[-3:-1] # from 3 from the end to the 1 from the end


arr11[0::2] # intervel of 2 steps 



# slicing 2-D array

arr12 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

arr12[1, 1:4] # row 2, coloumn 1 to 3

arr12[0:2,2] # select row 1 & 2, and column 3









# Data type in Python

arr13 = np.array([1,2,3,4])

print(arr13.dtype)


arr14 = np.array(['ab','bb','vb'])


print(arr14.dtype)


# Create arrays with a defined data type

arr15 = np.array([12,22,33,3433333], dtype = 'S') # create array and format as String

arr15.dtype




# copy a arr using astype and format it as float


arr16 = np.array([1,2.4,3,4])

newarr = arr16.astype('f')

newarr



""" The Difference Between Copy and View
The main difference between a copy and a view of an array is that the copy is a new array, and the view is just a view of the original array.

The copy owns the data and any changes made to the copy will not affect original array, and any changes made to the original array will not affect the copy.

The view does not own the data and any changes made to the view will affect the original array, and any changes made to the original array will affect the view. """


#Numpy Array copied throught copy()

arr17 = np.array([1,2,3,4,5])

x = arr17.copy()

x[0]




# Create another view to display a array

arr18 = np.array([1,2,3,4,5])
x1 = arr18.view()

print(x)


# if you want to check whether a an array owns a data, u can use .base


print(x.base) # this would print None, which means it owns the data, werid but true
print(x1.base)






#Print the shape of the array


arr19 = np.array([[1,2,3,4],[5,6,7,8]])
arr19.shape



#Array Reshape using the reshape()


arr20 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

newarr1 = arr20.reshape(4,3) # 4 rows and 3 columns

newarr1

newarr2 = arr20.reshape(2,3,2) # 2 copies of 3 row by 2 columns, hence a 3D array

newarr2

print(newarr2.base) # output is a array, which means it doesnt own any data, which means newarr2 is a 'view'


# reshape with -1, i.e. let pc to decide the shape of your array

arr21 = np.array([1,2,3,4,5,6,7,8])

newarr3 = arr21.reshape(2,2,-1) # reshape to 2 copies of array containing 2 rows with X columns each (x is decided by the pc)

newarr3



flatarr = newarr3.reshape(-1) # this means flatten it to 1D array

flatarr





#Numpy array Iterating


arr22 = np.array([1,4,3,4,5,6])

for i in arr22:
    print(i)


# 2D iteration of each element (i.e. each sqaure bracket)
arr23 = np.array([[1,2,3],[4,5,6]])
for i in arr23:
    print(i)


# 2D iteration, return each value


arr24 = np.array([[1,4,9],[4,5,6]])

for x in arr24:
    for y in x:
        print(y)

# 3D Iteration of each element(i.e.each square bracket)

arr24 = np.array([[[1,2,3],[4,5,6]],[[4,4,4],[6,6,6]]])

for i in arr24:
    print(i)

arr24.ndim


# 3D iteration, return each value
for i in arr24:
    for y in i:
        for e in y:
            print(e)
        



# The usage of nditer()  = print all elements while getting rip of dimensions

# so no need to do the above iteration, just use the below
for i in np.nditer(arr24):
    print(i)



# Iteratinf with different step size

arr25 = np.array([[1,4,5,6,8,5,4,6],[5,6,7,4,2,8,8,8]])

for x in np.nditer(arr25[0:2,3::2]): # print array index 1 and 2, then start at index position 4 for each array, then do interval step of 2
    print(x)




# np.ndenumerate = print the sequence number during iteration

arr26 = np.array([1,2,4,5])

for k, x in np.ndenumerate(arr26):
    print(k,x)





""" Joining means putting contents of two or more arrays in a single array.

In SQL we join tables based on a key, whereas in NumPy we join arrays by axes.

We pass a sequence of arrays that we want to join to the concatenate() function, 

along with the axis. If axis is not explicitly passed, it is taken as 0.

 """


 # Joining two 1-D array


 arr27 = np.array([1,2,3])
 arr28 = np.array([4,5,9])

 combined = np.concatenate((arr27,arr28)) # concatenate basially = Union join in sql

 combined



 # Joining two 2D array

 arr29 = np.array([[1,5,6],[5,7,8]])

 arr30 = np.array([[1,4,2],[6,4,8]])


 combined2D = np.concatenate((arr29,arr30), axis = 1) # joining arries in x-axis direction

 combined2D




""" Stacking is same as concatenation, the only difference is that stacking is done along a new axis.

We can concatenate two 1-D arrays along the second axis which would result in putting them one over the other, ie. stacking.

We pass a sequence of arrays that we want to join to the concatenate() method along with the axis. If axis is not explicitly passed it is taken as 0."""




# Stacking is used to combined array togather and create another axis/dimension at the same time


arr31 = np.array([1,2,3])

arr32 = np.array([5,3,5])

combined_n_increase_dimen = np.stack((arr31,arr32),axis = 1) # add two 1D-array and forming a new 2D array in the x-axis direction

combined_n_increase_dimen




"""Splitting is reverse operation of Joining.

Joining merges multiple arrays into one and Splitting breaks one array into multiple.

We use array_split() for splitting arrays, we pass it the array we want to split and the number of splits. """


# Split the array in 4 parts


arr33 = np.array([1,2,3,4,45,5,7])

split1 = np.array_split(arr33,4) # array_split able to split the number into 4 parts despite even items in the array

split1 # after split, it is still 1D array

np.ndim(split1)


split1[0] # access the first array by index = 0



# Splitting 2-D array

arr34 = np.array([[1,2],[4,5],[6,4],[4,5],[7,4],[5,0],[4,6],[9,7]])


arrsplit = np.array_split(arr34,4) # splitting a 2D array into four 2x2 array, hence 3D data


arrsplit


np.ndim(arrsplit)

np.ndim(arr34)




# Split the 2-D array into three 2-D arrays along rows


arr35 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr4 = np.array_split(arr35,3,axis=1) # splitting a 3D array into three 2-D array based on column/x-axis

newarr4

arr35

np.ndim(newarr4[0])




""" You can search an array for a certain value, 
and return the indexes that get a match.

To search an array, use the where() method."""


#Searching Arrays




arr36 = np.array([1,2,3,2,3,4,3,53,4,3])


whereis4 = np.where(arr36 ==4) # return the index poisition of the search value

whereis4

# search value that is even
even = np.where(arr36 %2==0)

even

# search value that is odd

odd = np.where(arr36 %2==1)


odd




""" Sorting means putting elements in a ordered sequence.

Ordered sequence is any sequence that has an order corresponding to elements, 
like numeric or alphabetical, ascending or descending.

The NumPy ndarray object has a function called sort(), 
that will sort a specified array. """

arr37 = np.array([3,2,0,1])


sortarray = np.sort(arr37) # ascend

sortarray


sortarray1 = -np.sort(-arr37) # desend

sortarray1





""" Filtering Arrays 
Getting some elements out of an existing array
 and creating a new array out of them is called filtering.
"""


# Create an array from the elemetns on index 0 and 2

arr38 = np.array([41,42,43,44])

x3 = [True, False, True, False]



filtered = arr38[x]
filtered





# Directly arrey filtering
filter_arr = arr38>42
arr38[filter_arr]





""" what is a random number

Random number does NOT mean a different number every time. 
Random means something that can not be predicted logically.

"""



# Generate a random integer from 0 to 100
from numpy import random

x5 = random.randint(100)
x5


# generate a random float between 0 and 1

x6 = random.rand()
x6



# generate a random array

# Generate a random 1D array

x7 = random.randint(100,size =(5)) # pick 5 value between 0 to 100
x7



# Generarte a random 2D array


x8 = random.randint(100, size=(3,5)) # random 3 rows x 5 columns array
x8



# Generate a randfom array  from selected value


x9 = random.choice([3,5,7,9], size = [3,5]) # create a 3 x 5 columns array based on 3,5,7,9
x9



# adding two array together and sum it up

x6 = [1, 2, 3, 4]
y6 = [4, 5, 6, 7]
z6 = []

for i, j in zip(x6, y6):
  z6.append(i + j)
print(z6)



#create function

def function1(x,y):
    return x+y

function1(1,2)




# adding numerically of 2 array

arr39 = np.array([1,2,3,3,2,4])

arr40 = np.array([1,2,3,3,2,4])


summation = np.add(arr39,arr40) # add value of 2 arries together
summation

# np.subtract, np.multiply,np.divide, np.absolute also available in numpy

# search unique number


unique_number = np.unique(arr39)
unique_number


# Round up number
arr40 = np.around(3.133332,2) # roundup the number to 2 decimals


arr40
