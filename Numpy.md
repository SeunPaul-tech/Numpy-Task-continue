## ASSIGNMENT 2   



```python
## Array Creation 
```


```python
# Importing the packages
import random
import numpy as np
import pandas as pd
```


```python
## 1. Create a 2D NumPy array with 3 rows and 4 columns filled with random integers between 1 and 
## 30. 
arr_2D = np.random.randint(1,31, size=(3,4))
arr_2D
```




    array([[25, 19, 22, 15],
           [ 2, 19, 12,  7],
           [16,  1, 27, 30]])




```python
# 2. Write Python code to create a 1D NumPy array containing the first 20 even numbers.
evenNumbers = np.arange(2,41,2)
evenNumbers
```




    array([ 2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34,
           36, 38, 40])




```python
# Array Attributes   
# 3. Given a NumPy array arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10,11,12]]), write code to find 
# its shape, size, and data type. 

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10,11,12]])
arr
```




    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [10, 11, 12]])




```python
# the shape of the array
arr.shape
```




    (4, 3)




```python
# the size of the array
arr.size
```




    12




```python
# the datatype of the array
arr.dtype
```




    dtype('int32')




```python
# Array Indexing and Slicing   
# 4. Extract the fourth row and the third column from a 3x3 NumPy array arr in question 3.   
arr
```




    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [10, 11, 12]])




```python
arr[3,2]
```




    12




```python
# 5. Using slicing, reverse the elements of a 1D array arr = np.array([10, 20, 30, 40, 50]).   
arr = np.array([10, 20, 30, 40, 50])
reverse_arr = arr[::-1]
reverse_arr
```




    array([50, 40, 30, 20, 10])




```python
# Array Operations   
# 6. How would you use NumPy to perform matrix multiplication on two 2D arrays A and B? 
a = np.array([[2,3,4],[5,4,6],[6,5,4],[7,6,8]])
b = np.array([[4,5,6],[3,5,6],[8,9,10],[5,4,6]])

arrMul2D = a * b
arrMul2D
```




    array([[ 8, 15, 24],
           [15, 20, 36],
           [48, 45, 40],
           [35, 24, 48]])




```python
# 7. Create two 1D NumPy arrays of size 10. Perform element-wise addition, subtraction, and 
# multiplication. 
a = np.arange(0,10)
b = np.arange(11,21)
c = np.arange(22,32)

addition = a + b + c
substraction = c - b - a
multiplication = a * b * c

print(addition)
print(substraction)
print(multiplication)
```

    [33 36 39 42 45 48 51 54 57 60]
    [11 10  9  8  7  6  5  4  3  2]
    [   0  276  624 1050 1560 2160 2856 3654 4560 5580]
    


```python
# Mathematical Functions   
# 8. Use NumPy to calculate the mean, median, and standard deviation of a given array. 
arr = np.random.randint(3, 360, size=(3,4))

# find the mean
meanArr = arr.mean()
meanArr
```




    143.33333333333334




```python
# find the median
medianArr = np.median(arr)
medianArr
```




    163.0




```python
# find the standard deviation
stdArr = arr.std()
stdArr
```




    111.02877504903353




```python
# 9. Write a Python program to calculate the square root and exponential of all elements in a 1D NumPy array.  
arr = np.arange(0,40).reshape(5,8)
arr
```




    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13, 14, 15],
           [16, 17, 18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29, 30, 31],
           [32, 33, 34, 35, 36, 37, 38, 39]])




```python
# find the square root
sqrtArr = np.sqrt(arr)
print("Square Root :", sqrtArr)
```

    Square Root : [[0.         1.         1.41421356 1.73205081 2.         2.23606798
      2.44948974 2.64575131]
     [2.82842712 3.         3.16227766 3.31662479 3.46410162 3.60555128
      3.74165739 3.87298335]
     [4.         4.12310563 4.24264069 4.35889894 4.47213595 4.58257569
      4.69041576 4.79583152]
     [4.89897949 5.         5.09901951 5.19615242 5.29150262 5.38516481
      5.47722558 5.56776436]
     [5.65685425 5.74456265 5.83095189 5.91607978 6.         6.08276253
      6.164414   6.244998  ]]
    


```python
# find the exponential 
expArr = np.exp(arr)
print("The Exponential Value :", expArr)
```

    The Exponential Value : [[1.00000000e+00 2.71828183e+00 7.38905610e+00 2.00855369e+01
      5.45981500e+01 1.48413159e+02 4.03428793e+02 1.09663316e+03]
     [2.98095799e+03 8.10308393e+03 2.20264658e+04 5.98741417e+04
      1.62754791e+05 4.42413392e+05 1.20260428e+06 3.26901737e+06]
     [8.88611052e+06 2.41549528e+07 6.56599691e+07 1.78482301e+08
      4.85165195e+08 1.31881573e+09 3.58491285e+09 9.74480345e+09]
     [2.64891221e+10 7.20048993e+10 1.95729609e+11 5.32048241e+11
      1.44625706e+12 3.93133430e+12 1.06864746e+13 2.90488497e+13]
     [7.89629602e+13 2.14643580e+14 5.83461743e+14 1.58601345e+15
      4.31123155e+15 1.17191424e+16 3.18559318e+16 8.65934004e+16]]
    


```python
# Array Reshaping   
# 10. How can you flatten a multi-dimensional array back into a 1D array 
arr = np.random.randint(0,38, size=(2,6))
arrFlat = arr.flatten()
arrFlat
```




    array([16, 35, 32, 23,  3,  0, 37, 22, 10,  3,  1, 33])




```python
# 11. Convert a 1D NumPy array of 12 elements into a 2D array with 3 rows and 4 columns. 
arr = np.arange(0,12).reshape(4,3)
arr
```




    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11]])




```python
# Logical Operations and Filtering   
# 12. Given a NumPy array arr = np.array([10, 25, 30, 45, 50]), use logical operations to filter out all values greater than 30.
arr = np.array([10, 25, 30, 45, 50])
ValAbove30 = arr[arr>30]
print("Values Above Thirty: ", ValAbove30)
```

    Values Above Thirty:  [45 50]
    


```python

```
