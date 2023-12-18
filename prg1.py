import numpy as np
row=int(input("enter the nu.of rows"))
col=int(input("enter the no.of cols"))

print("enter the elements1:")
elements = list(map(int, input().split()))
matrix1= np.array(elements).reshape(row, col)
print(matrix1)

print("enter the elements2:")
elements = list(map(int, input().split()))
matrix2= np.array(elements).reshape(row, col)
print(matrix2)

addmatrix=np.add(matrix1,matrix2)
print("matrix addition:",addmatrix)

