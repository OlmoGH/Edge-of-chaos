import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import scipy.io as sio
import sys
import os

def sort_eigenvalues(last_sorted, current_eigvals):
    """
    Función auxiliar para hacer el seguimiento (eigenshuffle) de los autovalores 
    a través del tiempo asociando los más cercanos.
    """
    if last_sorted is None:
        return current_eigvals
    cost_matrix = np.abs(last_sorted[:, None] - current_eigvals[None, :])
    row_index, col_index = linear_sum_assignment(cost_matrix)
    return current_eigvals[col_index]

# %%%% Carga de datos pre-inicializados desde data.mat
# Buscamos 'data.mat' en el mismo directorio que este script
directorio_actual = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
ruta_mat = os.path.join(directorio_actual, 'data.mat')

# Cargamos el archivo. 
# squeeze_me y struct_as_record nos permiten usar la sintaxis "data.W" en lugar de diccionarios anidados complejos.
mat_contents = sio.loadmat(ruta_mat, squeeze_me=True, struct_as_record=False)

# Extraemos la estructura 'data'
data = mat_contents['data']

N = 128
Nmem = 1
dt = .1
eta = .01
tau = 50
taux = 20
zeta = .01
amp = 25
inLen = 100/dt
spacing = 1000/dt
spacing0 = 200/dt
TotalSteps = spacing0 + (inLen + spacing)*Nmem
CalcEvery = 10/dt
Nsteps = TotalSteps/CalcEvery

y = np.zeros(N)
z = np.zeros(N)
x_all = np.zeros(N,TotalSteps)
H = np.sign(np.randn((N, N)))
u = H[1:Nmem,:].T/np.sqrt(N)
v = H[Nmem+1:2*Nmem,:].T/np.sqrt(N)
input1 = np.zeros(N,TotalSteps)
input2 = np.zeros(N,TotalSteps)
W_all = np.zeros(N,N,Nsteps)
input = np.zeros(N,1)
xlp = 0
B = .5*np.eye(N)
W = data.W
x = data.x

for i in range(Nmem)
    base = (i-1)*spacing + spacing0;
    input1[:,base+1:base+inLen] = repmat(u(:,i),1,inLen);
    input2[:,base+1:base+inLen] = repmat(v(:,i),1,inLen);