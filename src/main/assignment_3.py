import numpy as np 

##question 1
def function(t,y):
    return t - y**2

t0 = 0
initial_point = 1
upper_range = 2 
iterations = 10
z = (upper_range -t0) / iterations
t = t0
y = initial_point
for i in range(iterations):
    y += z * function(t,y)
    t += z
print("%.5f" % y)
print("\n")

##question 2
t = t0
y = initial_point
for i in range(iterations):
    x1 = z * function(t,y)
    x2 = z * function(t + z/2, y + x1/2)
    x3 = z * function(t + z/2, y + x2/2)
    x4 = z * function(t + z, y + x3)
    y += (x1 + 2*x2 + 2*x3 + x4) / 6
    t += z
print("%.5f" % y)
print("\n")

##question 3
matrix = [[2,-1,1,6],
        [1,3,1,0],
        [-1,5,4,-3]]
for i in range(len(matrix)):
    max_row = i
    for j in range(i+1, len(matrix)):
        if abs(matrix[j][i]) > abs(matrix[max_row][i]):
            max_row = j
    matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
    switch = matrix[i][i]
    for j in range(i, len(matrix[i])):
        matrix[i][j] /= switch
    for j in range(len(matrix)):
        if j != i:
            det = matrix[j][i]
            for k in range(i, len(matrix[i])):
                matrix[j][k] -= det * matrix[i][k]
x = [row[-1] for row in matrix]
print(x)
print("\n")

##question 4
def lu_decomposition(matrix):
    x = len(matrix)
    L_matrix = np.zeros((x, x))
    U_matrix = np.zeros((x, x))

    for i in range(x):
        L_matrix[i][i] = 1
        for j in range(i, x):
            U_matrix[i][j] = matrix[i][j]
            for k in range(i):
                U_matrix[i][j] -= L_matrix[i][k] * U_matrix[k][j]

        for j in range(i + 1, x):
            L_matrix[j][i] = matrix[j][i]
            for k in range(i):
                L_matrix[j][i] -= L_matrix[j][k] * U_matrix[k][i]
            L_matrix[j][i] /= U_matrix[i][i]
    
    return L_matrix, U_matrix

def determinant(matrix):
    L_matrix, U_matrix = lu_decomposition(matrix)
    det_matrix = np.prod(np.diag(U_matrix), dtype = np.float64)
    return det_matrix


A_matrix = np.array([[1, 1, 0, 3],
              [2, 1, -1, 1],
              [3, -1, -1, 2],
              [-1, 2, 3, -1]])

det_matrix = determinant(A_matrix)
L_matrix, U_matrix = lu_decomposition(A_matrix)

print("%.5f" % det_matrix)
print("\n")
print(L_matrix)
print("\n")
print(U_matrix)
print("\n")

##question 5
def diagonally_dom(matrix, x) :
 for i in range(0, x) :  
  summation = 0
  for j in range(0, x) :
   summation = summation + abs(matrix[i][j]) 
  summation = summation - abs(matrix[i][i])
  if (abs(matrix[i][i]) < summation) :
   return False
 return True
x = 5
matrix = [[ 9, 0, 5, 2, 1 ],
        [ 3, 9, 1, 2, 2 ],
        [ 0, 1, 7, 2, 3 ],
        [ 4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8]]

if((diagonally_dom(matrix, x))) :
 print ("True")
 print("\n")
else :
 print ("False")
 print("\n")

##question 6
matrix = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
if not np.allclose(matrix, matrix.T):
    print("False")
else:
    eigenvalue, _ = np.linalg.eig(matrix)
    if all(eigenvalue > 0):
        print("True")
        print("\n")
    else:
        print("False")
        print("\n")

