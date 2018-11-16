import numpy as np 
import matplotlib.pyplot as plt 

E2 = np.loadtxt('int1E2.txt')
E4 = np.loadtxt('int1E4.txt')
E5 = np.loadtxt('int1E5.txt')
E6 = np.loadtxt('int1E6.txt')
E7 = np.loadtxt('int1E7.txt')
E8 = np.loadtxt('int1E8.txt')

E = np.array([E2, E4, E5, E6, E7, E8])
N = 6

def integral_analitica():
    return 0.5

puntos = np.int_(np.logspace(1,5,N))
diferencias = np.ones(N) 

for i in range(N):
    diferencias[i] =  (np.abs((integral_analitica()-E[i])/integral_analitica()))

plt.plot(puntos, diferencias*100)
plt.loglog()
plt.xlabel("$N_{puntos}$")
plt.ylabel("Diferencia porcentual Monte Carlo vs. Analitica")
plt.savefig("Error Porcentual Integral")
#plt.show()
