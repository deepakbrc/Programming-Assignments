# prutor-land:python
import math
import numpy as np
import sys
import prutorlib as pl
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
"""
INPUT FORMAT
0 .5 1 1.5 2
1 1.649 2.718 4.482 7.389
NS for  Natural Spline , NKS for not-a-knot spline
1.8
"""

# Input Should be Sorted
X = list(map(float, input().split()))
Y = list(map(float, input().split()))
n = len(X)
choice = input()
value = float(input())

# creating h array
h = np.zeros(shape=(n-1))
for i in range(n-1):
    h[i] = X[i+1] - X[i]
    
#creating divided difference array
g = np.zeros(shape=(n-2))
for i in range(n-2) :
    g[i] = 6*(( (Y[i+2]-Y[i+1])/(X[i+2]-X[i+1]) ) - ( (Y[i+1] - Y[i])/(X[i+1] - X[i]) )) 
 
#Thomas Algorithm  
def ThomasAlgo(a, b, c, d):

    nf = len(d) 
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) 
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc
    
#For Natural Spline
if( choice == "NS" ):
    b = np.zeros(shape=(n-2))
    for i in range(len(h) - 1) :
        b[i] = 2*(h[i+1]+h[i])
    
    a = np.zeros(shape=(n-3)) 
    for i in range(len(h) -2) :
        a[i] = h[i+1]
    

    
    v_s = ThomasAlgo(a, b , a, g)
    v = np.zeros(shape=(n))
    for i in range(n-2):
        v[i+1] = v_s[i]
    
    A = np.zeros(shape=(n-1))
    B = np.zeros(shape=(n-1))
    C = np.zeros(shape=(n-1))
    D = np.zeros(shape=(n-1))
    
    for i in range(n-1) :
        A[i] = v[i+1]/(6*h[i])
        B[i] = v[i]/(6*h[i])
        C[i] = ((Y[i+1]/h[i]) - ((v[i+1]*h[i])/6))   
        D[i] = ((Y[i]/h[i]) - ((v[i]*h[i])/6))
        
    print("Coefficient Matrix A : ", A) 
    print("Coefficient Matrix B : ", B)
    print("Coefficient Matrix C : ", C)
    print("Coefficient Matrix D : ", D)
 
#For not-a-knot spline   
if ( choice == "NKS" ) :
    a = np.zeros(shape=(n,n))
    b = np.zeros(shape=(n))
    for i in range(n-2):
        b[i+1] = g[i]
    
    a[0][0] = h[1]
    a[0][1] = -1*(h[1]+h[0])
    a[0][2] = h[0]
    a[n-1][n-3] = h[n-2]
    a[n-1][n-2] = -1*(h[n-2] + h[n-3]) 
    a[n-1][n-1] = h[n-3]
    
    for i in range(1, n-1):
        a[i][i-1] = h[i-1]
        a[i][i] = 2*(h[i] + h[i-1])
        a[i][i+1] = h[i]
    
    v = np.linalg.solve(a, b)
    
    A = np.zeros(shape=(n-1))
    B = np.zeros(shape=(n-1))
    C = np.zeros(shape=(n-1))
    D = np.zeros(shape=(n-1))
    
    for i in range(n-1) :
        A[i] = v[i+1]/(6*h[i])
        B[i] = v[i]/(6*h[i])
        C[i] = ((Y[i+1]/h[i]) - ((v[i+1]*h[i])/6))   
        D[i] = ((Y[i]/h[i]) - ((v[i]*h[i])/6))
        
    print("Coefficient Matrix A : ", A) 
    print("Coefficient Matrix B : ", B)
    print("Coefficient Matrix C : ", C)
    print("Coefficient Matrix D : ", D)
    
    
    
    
# For calculating interpolated value    
    
qi = 0
for i in range(n) :
    if ( value > X[i] ) :
        qi = i

if ( qi == n-1 ) :
    print("Interpolating value is outside the data range")
    
else :
    Q = A[qi]*((value - X[qi])**3) - B[qi]*((value - X[qi+1])**3) + C[qi]*(value - X[qi]) - D[qi]*(value - X[qi+1])
    print(Q)


# Function to give spline values at different x values
def Qvalue(r, A, B, C, D) :
    q_i = 0
    for i in range(n) :
        if ( r > X[i] ) :
            q_i = i
    q = A[q_i]*((r - X[q_i])**3) - B[q_i]*((r - X[q_i+1])**3) + C[q_i]*(r - X[q_i]) - D[q_i]*(r- X[q_i+1])
    return q
    
# code for prutor plot

X_spline = np.linspace(X[0], X[n-1], 100)
Y_spline = np.zeros(shape=(len(X_spline)))
for i in range(len(X_spline)):
    Y_spline[i] = Qvalue(X_spline[i], A, B, C, D)

plt.plot(X,Y,color='green',marker = 'o', label = 'Original Data', linestyle = 'None')
plt.plot(X_spline,Y_spline)
plt.legend(["Original Data" , "Cubic Spline"])
plt.grid()
plt.xlabel('x')
plt.ylabel('F(x)')
pl.prutorsaveplot(plt ,'CubicSpline.pdf')
    

    
    
    

    
    
    
    
