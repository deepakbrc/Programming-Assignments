# prutor-lang:python
import math
import numpy as np


n=int(input())
A=input()
A=A[1:-1]
choice=input()
tolerance=int(input())
A=A.split()
arrL=[]
it=0
for i in range(n):
 temp=[]
 for j in range(n):
  if j==(n-1):
   t_a=A[it].split(';')
   temp.append(float(t_a[0]))
  else:
   temp.append(float(A[it]))
  it+=1
 arrL.append(temp)
arr = np.array(arrL)

# Input Converted Into proper Matrix

z0 = np.zeros(shape=(n,n))
z0[0][0] = 1
# Initial Vector z0 is assumed for Power Method

# Finding largest eigen value using Power Method
if ( choice == "L" ) :
    z1 = np.dot(arr,z0)
    max = 0
    for i in range(n) :
        if( max <= abs(z1[i][0]) ) :
            max = abs(z1[i][0]) 
            eg1 = z1[i][0]
    
    z1 = z1/eg1

    for k in range(100) :
        z2 = np.dot(arr, z1)
        max = 0
        for i in range(n) :
            if( max <= abs(z2[i][0]) ) :
                 max = abs(z2[i][0]) 
                 eg2 = z2[i][0]
        z2 = z2/eg2
        
        error = (abs(eg2 - eg1) / abs(eg2) ) *100 
        
        if (error <= tolerance) :
            break
        z1 = z2
        eg1 = eg2
    print("Largest Eigen Vector using Power Methord:", eg2)

# Finding smallest eigen value using Inverse Power Method    
if ( choice == "L" ) :
    arrInv = np.linalg.inv(arr)

    z1 = np.dot(arrInv,z0)

    max = 0
    for i in range(n) :
        if( max <= abs(z1[i][0]) ) :
            max = abs(z1[i][0]) 
            eg1 = z1[i][0]
    
    z1 = z1/eg1

    for k in range(100) :
        z2 = np.dot(arrInv, z1)
        max = 0
        for i in range(n) :
            if( max <= abs(z2[i][0]) ) :
                 max = abs(z2[i][0]) 
                 eg2 = z2[i][0]
        z2 = z2/eg2
        
        error = (abs(eg2 - eg1) / abs(eg2) ) * 100 
        
        if (error <= tolerance) :
            break
        z1 = z2
        eg1 = eg2

    print("Smalest Eigen Vector using Power Method:" , 1/eg2)
     
     
# Function for QR Decomposition by Gram-Schmidt algorithm
def gramschmidt(A):

    R = np.zeros((A.shape[1], A.shape[1]))
    Q = np.zeros(A.shape)
    for k in range(0, A.shape[1]):
        R[k, k] = np.sqrt(np.dot(A[:, k], A[:, k]))
        Q[:, k] = A[:, k]/R[k, k]
        for j in range(k+1, A.shape[1]):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] = A[:, j] - R[k, j]*Q[:, k]
    return Q, R          
  
#Finding all Eigen Values using QR decmposition 
if ( choice == "A" ) :
    Q, R = gramschmidt(arr)
    z1 = np.dot(R, Q)
    
    for k in range(100) :
        Q, R = gramschmidt(z1)
        z2 = np.dot(R, Q)
        
        maxErr = 0
        for i in range(n) :
            err = abs((z2[i][i] - z1[i][i]) / z2[i][i]) * 100
            if ( maxErr <= err ) :
                maxErr = err
        
        if ( maxErr <= tolerance) :
            break
        z1 = z2
        
    eg = []
    for k in range(n) :
        eg.append(z2[k][k])
        
    print("All Eigen Values Are : ", eg) 
        
    
    
    
            
        
    


