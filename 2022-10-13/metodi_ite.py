""" ** METODI ITERATIVI ** """

from itertools import count
import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt

# Somma componente per componente, non tramite prodotto di matrici
def Jacobi(A, b, x0, maxit, tol, xTrue):
    n = np.size(x0)
    ite = 0
    x = np.copy(x0)
    norma_it = 1 + tol
    relErr = np.zeros((maxit, 1))
    errIter = np.zeros((maxit, 1))
    relErr[0] = np.linalg.norm(xTrue - x0) / np.linalg.norm(xTrue)
    while (ite < maxit-1 and norma_it > tol):
        x_old = np.copy(x)
        for i in range(0, n):
            x[i] = (b[i] - sum([A[i, j] * x_old[j] for j in range(0, i)]) - sum([A[i, j] * x_old[j] for j in range(i + 1, n)])) / A[i, i]
            #x[i] = (b[i] - np.dot(A[i, 0:i], x_old[0:i]) - np.dot(A[i, i+1:n], x_old[i+1:n])) / A[i, i]
        ite = ite + 1
        norma_it = np.linalg.norm(x_old - x) / np.linalg.norm(x_old)
        relErr[ite] = np.linalg.norm(xTrue - x) / np.linalg.norm(xTrue)
        errIter[ite - 1] = norma_it
    relErr = relErr[:ite]
    errIter = errIter[:ite]  
    return [x, ite, relErr, errIter]


def GaussSeidel(A,b,x0,maxit,tol, xTrue):
    n = np.size(x0)
    ite = 0
    x = np.copy(x0)
    norma_it = 1 + tol
    relErr = np.zeros((maxit, 1))
    errIter = np.zeros((maxit, 1))
    relErr[0] = np.linalg.norm(xTrue - x0) / np.linalg.norm(xTrue)
    while (ite < maxit-1 and norma_it > tol):
        x_old = np.copy(x)
        for i in range(0, n):
            x[i] = (b[i] - sum([A[i, j] * x[j] for j in range(0, i)]) - sum([A[i, j] * x_old[j] for j in range(i + 1, n)])) / A[i, i]
            #x[i] = (b[i] - np.dot(A[i, 0:i], x[0:i]) - np.dot(A[i, i+1:n], x_old[i+1:n])) / A[i, i]
        ite = ite + 1
        norma_it = np.linalg.norm(x_old - x) / np.linalg.norm(x_old)
        relErr[ite] = np.linalg.norm(xTrue - x) / np.linalg.norm(xTrue)
        errIter[ite - 1] = norma_it
    relErr = relErr[:ite]
    errIter = errIter[:ite]
    return [x, ite, relErr, errIter]


""" **  matrice tridiagonale nxn ** """
# help(np.diag)
# help (np.eye)
# n=5
# c = np.eye(n)
# s = np.diag(np.ones(n-1)*2,k=1)
# i = np.diag(np.ones(n-1)*2,k=-1)
# print('\n c:\n',c)
# print('\n s:\n',s)
# print('\n i:\n',i)
# print('\n c+i:\n',c+i+s)

#creazione del problema test
n = 5
A = 9 * np.eye(n, n, k = 0) - 4 * np.eye(n, n, k = 1) - 4 * np.eye(n, n, k = -1)
xTrue = np.ones((n,1))
b = A @ xTrue

print('\n A:\n',A)
print('\n xTrue:\n',xTrue)
print('\n b:\n',b)


#metodi iterativi
x0 = 2 * np.ones((n,1))
maxit = 1000
tol = 1e-8

(xJacobi, kJacobi, relErrJacobi, errIterJacobi) = Jacobi(A, b, x0, maxit, tol, xTrue)
(xGS, kGS, relErrGS, errIterGS) = GaussSeidel(A, b, x0, maxit, tol, xTrue)

print('\nSoluzione calcolata da Jacobi:' )
for i in range(n):
    print('%0.2f' %xJacobi[i])

print('\nSoluzione calcolata da Gauss Seidel:' )
for i in range(n):
    print('%0.2f' %xGS[i])


# CONFRONTI

# Confronto grafico degli errori di Errore Relativo

rangeJabobi = range (0, kJacobi)
rangeGS = range(0, kGS)

plt.figure()
plt.plot(rangeJabobi, relErrJacobi, label='Jacobi', color='blue', linewidth=1, marker='o'  )
plt.plot(rangeGS, relErrGS, label='Gauss Seidel', color = 'red', linewidth=2, marker='.' )
plt.legend(loc='upper right')
plt.xlabel('iterations')
plt.ylabel('Relative Error')
plt.title('Comparison of the different algorithms')
plt.show()



#comportamento al variare di N

dim = np.arange(start=50, stop=501, step=50)

ErrRelF_J = np.zeros(np.size(dim))
ErrRelF_GS = np.zeros(np.size(dim))

ite_J = np.zeros(np.size(dim))
ite_GS = np.zeros(np.size(dim))
tol = 1e-16

i = 0


for n in dim:
    #creazione del problema test
    A = 9 * np.eye(n, n, k = 0) - 4 * np.eye(n, n, k = 1) - 4 * np.eye(n, n, k = -1)
    xTrue = np.ones((n, 1))
    b = A @ xTrue
    
    #metodi iterativi
    x0 = np.zeros((n, 1))
    x0[0] = 1
    
    (xJacobi, kJacobi, relErrJacobi, errIterJacobi) = Jacobi(A, b, x0, ite_J.size, tol, xTrue)
    (xGS, kGS, relErrGS, errIterGS) = GaussSeidel(A, b, x0, ite_GS.size, tol, xTrue)
    
    #errore relativo finale
    ErrRelF_J[i] = relErrJacobi[-1]
    ErrRelF_GS[i] = relErrGS[-1]
    
    #iterazioni
    ite_J[i] = kJacobi
    ite_GS[i]= kGS

    i = i+1
    

# Errore relativo finale dei metodi al variare della dimensione N
plt.figure()
plt.plot(dim, ErrRelF_J, label='Jacobi', color='blue', linewidth=1, marker='.')
plt.plot(dim, ErrRelF_GS, label='Gauss Seidel', color = 'red', linewidth=1, marker='.' )
plt.legend(loc='upper right')
plt.xlabel('Dimensione')
plt.ylabel('Errore Relativo')
plt.title('Errore relativo finale dei metodi al variare della dimensione N')
plt.show()

# Numero di iterazioni di entrambi i metodi al variare di N
plt.figure()
plt.plot(dim, ite_J, label='Jacobi', color='blue', linewidth=1, marker='.')
plt.plot(dim, ite_GS, label='Gauss Seidel', color = 'red', linewidth=1, marker='.' )
plt.legend(loc='upper right')
plt.xlabel('Dimensione')
plt.ylabel('Numero di iterazioni')
plt.title('Numero di iterazioni di entrambi i metodi al variare di N')
plt.show()
