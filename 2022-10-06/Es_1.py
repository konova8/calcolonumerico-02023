"""1. matrici e norme """

from random import seed
import numpy as np

#help(np.linalg) # View source
#help (np.linalg.norm)
#help (np.linalg.cond)

n = 2
A = np.array([[1, 2], [0.499, 1.001]])

print ('Norme di A:')
norm1 = np.linalg.norm(A, 1)
norm2 = np.linalg.norm(A, 2)
normfro = np.linalg.norm(A, 'fro')
norminf = np.linalg.norm(A, np.inf)

print('Norma1 = ', norm1, '\n')
print('Norma2 = ', norm2, '\n')
print('Normafro = ', normfro, '\n')
print('Norma infinito = ', norminf, '\n')


cond1 = np.linalg.cond(A, 1)
cond2 = np.linalg.cond(A, 2)
condfro = np.linalg.cond(A, 'fro')
condinf = np.linalg.cond(A, np.inf)

print ('K(A)_1 = ', cond1, '\n')
print ('K(A)_2 = ', cond2, '\n')
print ('K(A)_fro =', condfro, '\n')
print ('K(A)_inf =', condinf, '\n')

x = np.ones((2,1))
b = np.dot(A,x)     # b dovrebbe essere [3, 1.5]

btilde = np.array([[3], [1.4985]])
xtilde = np.array([[2, 0.5]]).T

# Verificare che xtilde è soluzione di A xtilde = btilde
# A * xtilde = btilde
# print ('A*xtilde = ', A @ xtilde)
print ('A*xtilde = ', np.dot(A, xtilde))

deltax = x - xtilde
deltab = b - btilde

print ('delta x = ', deltax)
print ('delta b = ', deltab)


"""2. fattorizzazione lu"""

import numpy as np
import scipy
# help (scipy)
import scipy.linalg as LA
# help (scipy.linalg)
import scipy.linalg.decomp_lu as LUdec 
# help (LUdec)
# help(scipy.linalg.lu_solve )

# crazione dati e problema test
A = np.array ([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]  ])
x = np.ones((4, 1))
b = np.dot(A, x)

condA = np.linalg.cond(A)

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')


#help(LUdec.lu_factor)
lu, piv = LA.lu_factor(A)

print('lu', lu,'\n')
print('piv', piv,'\n')


# risoluzione di    Ax = b   <--->  PLUx = b 
my_x = LA.lu_solve((lu, piv), b)

print('my_x = \n', my_x)
print('norm =', LA.norm(x-my_x, 'fro'))




"""3. Choleski"""

import numpy as np
import scipy
# help (scipy)
import scipy.linalg as LA
# help (scipy.linalg)
# help (scipy.linalg.cholesky)
# help (scipy.linalg.solve)

# crazione dati e problema test
A = np.array ([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]  ], dtype=np.float64)
A = np.matmul(np.transpose(A), A)               # A^T * A è simmetrica e definita positiva
x = np.ones((4,1))
b = np.dot(A, x)

condA = np.linalg.cond(A)

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')

# decomposizione di Choleski
L = LA.cholesky(A, lower = True)
print('L:', L, '\n')

print('L * L.T =', np.dot(L, np.transpose(L)))
print('err = ', LA.norm(A-np.matmul(L, np.transpose(L)), 'fro'))

# L * y = b
# y = L^T * x  ->  L^T * x = y

y = LA.solve(L, b, lower = True)
my_x = LA.solve(L.T, y)
print('my_x = ', my_x)
print('norm =', LA.norm(x-my_x, 'fro'))



"""4. Choleski con matrice di Hilbert"""

import numpy as np
import scipy
# help (scipy)
import scipy.linalg as LA
# help (scipy.linalg)
# help (scipy.linalg.cholesky)
# help (scipy.linalg.hilbert)

# crazione dati e problema test
n = 5
A = LA.hilbert(n)
x = np.ones((n, 1))
b = np.dot(A, x)

condA = np.linalg.cond(A)

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')

# decomposizione di Choleski
L = LA.cholesky(A, lower = True)
print('L:', L, '\n')

print('L.T*L =', np.dot(L.T, L))
print('err = ', LA.norm(A-np.matmul(L, np.transpose(L)), 'fro'))

# L * y = b
# y = L^T * x  ->  L^T * x = y

y = LA.solve(L, b, lower = True)
my_x = LA.solve(L.T, y)
print('my_x = \n ', my_x)

print('norm =', LA.norm(x-my_x, 'fro'))



"""5. Choleski con matrice di matrice tridiagonale simmetrica e definita positiva """

import numpy as np
import scipy
# help (scipy)
import scipy.linalg as LA
# help (scipy.linalg)
# help (scipy.linalg.cholesky)
# help (np.diag)

# crazione dati e problema test
n = 4
A = np.zeros((n, n)) + np.diag(np.full(n, fill_value = 9)) + np.diag(np.full(n-1, fill_value = -4), k = 1) + np.diag(np.full(n-1, fill_value = -4), k = -1)
A = np.matmul(A, np.transpose(A))
x = np.ones((n, 1))
b = np.dot(A, x)

condA = np.linalg.cond(A)

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')

# decomposizione di Choleski
L = LA.cholesky(A, lower=True)
print('L:', L, '\n')

print('L*L.T =', np.dot(L, L.T))
print('err = ', LA.norm(A-np.matmul(L, np.transpose(L)), 'fro'))

# L * y = b
# y = L^T * x  ->  L^T * x = y

y = LA.solve(L, b, lower=True)
my_x = LA.solve(L.T, y)
print('my_x = \n ', my_x)

print('norm =', LA.norm(x-my_x, 'fro'))




"""6. plots """


import scipy.linalg.decomp_lu as LUdec 
import matplotlib.pyplot as plt

K_A = np.zeros((20,1))
Err = np.zeros((20,1))

for n in np.arange(10,30):
    # crazione dati e problema test
    A = np.random.randn(n, n)
    x = np.ones((n, 1))
    b = np.dot(A, x)
    
    # numero di condizione 
    K_A[n-10] = np.linalg.cond(A)
    
    # fattorizzazione 
    lu, piv = LA.lu_factor(A)
    my_x = LA.lu_solve((lu, piv), b)
    
    # errore relativo
    Err[n-10] = LA.norm(my_x - x, ord=2) / LA.norm(x, ord=2)
  
x = np.arange(10,30)

# grafico del numero di condizione vs dim
plt.plot(x, K_A)
plt.title('CONDIZIONAMENTO DI A ')
plt.xlabel('dimensione matrice: n')
plt.xticks(x)
plt.ylabel('K(A)')
plt.show()


# grafico errore in norma 2 in funzione della dimensione del sistema
plt.plot(x, Err)
plt.title('Errore relativo')
plt.xlabel('dimensione matrice: n')
plt.xticks(x)
plt.ylabel('Err = ||my_x-x||/||x||')
plt.show()