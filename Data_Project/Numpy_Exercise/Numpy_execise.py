# https://www.w3resource.com/python-exercises/numpy/basic/index.php#EDITOR





""" Q3. Write a NumPy program to test whether the 
elements of a given array is zero."""


""" Q4. Write a NumPy program to test whether any of the 
elements of a given array is non-zero."""

import numpy as np

array = np.array([0,1,2,3,4,5])

for index,i in enumerate(array):
    if i >0:
        print('index position:', index, 'is bigger than zero')
    if i == 0:
        print('index position:', index, 'is equal to zero')



""" Q5. Write a NumPy program to test a given array element-wise 
for finiteness (not infinity or not a Number)."""


import numpy as np
array1 = np.array([0,1,2,3,4,5,np.nan, np.inf])
a = np.isfinite(array1) # isfinite is false when it is nan or inf




""" Q6. Write a NumPy program to test element-wise 
for positive or negative infinity."""

b = np.isinf(array1) # isinf ise used to detect infinity
b


""" Q7. Write a NumPy program to test element-wise 
for nan """

c = np.isnan(array1)
c


""" Q8. Write a NumPy program to test element-wise for complex number, real number of a given array. 
Also test whether a given number is a scalar type or not"""

array2 = np.array([1+1j, 4.6, 7])
d = np.iscomplex(array2[0]) # complex number identification
d

e = np.isreal(array2)
e

f = np.isscalar(3)
f


""" 
Scalar type in python
(int,
 float,
 complex,
 long,
 bool,
 str,
 unicode,
 buffer,
 numpy.int16,
 numpy.float16,
 numpy.int8,
 numpy.uint64,
 numpy.complex192,
 numpy.void,
 numpy.uint32,
 numpy.complex128,
 numpy.unicode_,
 numpy.uint32,
 numpy.complex64,
 numpy.string_,
 numpy.uint16,
 numpy.timedelta64,
 numpy.bool_,
 numpy.uint8,
 numpy.datetime64,
 numpy.object_,
 numpy.int64,
 numpy.float96,
 numpy.int32,
 numpy.float64,
 numpy.int32,
 numpy.float32)

 """


"""
Q.10 Write a Numpy program to create an element-wise comparsion
(greater, greater_equal, less and less_equal) of two given arrays
"""

array3 = np.array([0,1,3,4,5,6,2])

array4 = np.array([0,2,1,4,3,7,2])



Greater_than4 = np.greater(array3,array4) # is each value in array3 greater than array4?
Greater_than4


Greater_equal_than4 = np.greater_equal(array3, array4)
Greater_equal_than4


Less_than4 = np.less(array3, array4)
Less_than4

Less_equal_than4 = np.less_equal(array3,array4)
Less_equal_than4



"""Q.11 Write a Numpy program to create an element-wise comparsion
(eaual) of two given arrays
"""


equal3n4 = np.equal(array3,array4)
equal3n4



""" 
Q13.Write a NumPy program to create 
an array of 10 zeros,10 ones, 10 fives
"""


array5 = np.zeros(10)
array5

array6 = np.ones(10)
array6

array7 = np.ones(10)*5 # crate ten 5 in a array
array7



"""
Q14. Write a NumPy program to 
create an array of the integers from 30 to70. 

"""

array6 = np.arange(30, 71)
array6



"""
Q15. Write a NumPy program to 
create an array of the integers from 30 to 70 that contains even number only
"""


array7 = np.arange(30,71,2)
array7



"""
Q16. Wtie a Numpy program to create a 3 x 3 identity matrix

"""

array8 = np.array([[0,1,2],[0,1,2],[0,1,2]])

array8


"""
Q17. Write a Numpy program to generate a random number
between 1 & 0

"""

from numpy import random


array9 = random.rand() # generate a random number between 0 and 1
array9


"""
Q18. Write a Numpy program to generate an array of 15 
random number from a standard normal distribution
"""

array10 = np.random.normal(0,111,15) # normal distribution of 15 values (between 0 to 111) 
array10

"""

Q19. Write a Numpy programm to create a vector with values
ranging from 15 to 55 and print all values except the
first and last

"""


array11 = np.arange(15,56)
array11[1:-1]



"""

Q20. Write a Numpy program to create 3x4 array using 
and iterate over it

"""

array12 = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
array12

for i in np.nditer(array12):
    print(i)



""" 

Q21. Write a Numpy  program to create a vector of length 10 with values
evenly distributed between 5 and 50

"""

array13 = np.arange(5,51,5)

array13



"""

Q22. Write a numpy program to create a vector with values 
from 0 to 20 
"""

array14 = np.arange(0,21)

array14


"""
Q23. Write a Numpy program to create a vector of lenght 5 filled
with arbitraray integers from 0 - 10

"""

array15 = np.random.randint(0,11,5)
array15


""" 

Q24. Write a Numpy to multiply the values of two given vectors

"""
array16 = ([1,2,3,4])
array17 = ([1,2,3,4])


array18 = np.multiply(array16,array17)

array18


""" 
Q25. Write a Numpy program to create a 3x4 matrix
filled with values from 10 to 21
"""

array18 = random.randint(22, size=(3,4))
array18


"""
Q26. Write a Numpy program to find the number of rows and columns of a given matrix
"""

vector1 = np.size(array18, axis = 1)
vector2 = np.size(array18, axis = 2)


"""
Q27. Write a Numpy program to create a 3x3 matrix,
i.e. diagonal elements are 1, the rest are 0

"""

array19 = np.eye(3) # eye create a diagonal elements of 1, rest 0 for the matrix
array19


"""
Q28. Write a Numpy program to create a 10 x 10 matrix, 
in which the elements on theboarders will be 1, and inside zero

"""

array20 = np.ones([10,10])

array20[1:-1, 1:-1] = 0
array20



"""

Q29. Write a Numpy program to create a 5x5 zero metrix with elements
on the main disgonal equal to 1,2,3,4,5

"""

array21 = np.diag([1,2,3,4,5])
array21



"""

Q30. Write a Numpy program to create a 4 x 4 matrix in which 0
and 1 are staggered, with zeros one the main diagonal

"""
array21 = np.ones([4,4])
array21[0::2,0::2] = 0
array21[1::2,1::2] = 0
#array21[::,1::2]=0
#array21[0:3:2,1::2]=0

array21



"""  np.squeeze = it sequeezes a single dimensional array into a single number
 that represents the number of element that it contained
 
 say if you have an array of [245,1] where 1 is column and 245 is row,

 it will become the number of '245' after the np.squeeze.
 
 """
import numpy as np
a = np.array([[[0], [2], [4],[4]]])
a
a.shape

b = np.squeeze(a).shape
b



p = 11

assert p == 11, 'the numeber is not 10'

assert(isinstance(p,float)), 'this number is not a float'

assert p == int , 'this is not a float!!!'