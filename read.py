import numpy as np
import matplotlib.pyplot as plt  
from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian
from time import time

"""

N1 = 14
q1 = 7
M1 = 10

N2 = 14
q2 = 7
M2 = 30

N3 = 14
q3 = 7
M3 = 50

N4 = 14
q4 = 7
M4 = 100



data1 = np.load("./data/DSF_{0:d}_Q_{1:.2f}_N_{2:d}.npz".format(N1, 2*q1/N1, M1))
data2 = np.load("./data/DSF_{0:d}_Q_{1:.2f}_N_{2:d}.npz".format(N2, 2*q2/N2, M2))
data3 = np.load("./data/DSF_{0:d}_Q_{1:.2f}_N_{2:d}.npz".format(N3, 2*q3/N3, M3))
data4 = np.load("./data/DSF_{0:d}_Q_{1:.2f}_N_{2:d}.npz".format(N4, 2*q4/N4, M4))

"""
n1 = 3
m1 = 3
q1n1 = 1
q1n2 = 1
dmi1 = 0.1j

n2 = 3
m2 = 3
q2n1 = 1
q2n2 = 1
dmi2 = 0.2j

n3 = 3
m3 = 3
q3n1 = 1
q3n2 = 1
dmi3 = 0.3j

data2D1 = np.load("./data/DSF_{0:d}{1:d}_Q_{2:.2f}{3:.2f},dmi={4:.2f}.npz".format(n1, m1, 2*q1n1/n1, 2*q1n2/m1, dmi1))
data2D2 = np.load("./data/DSF_{0:d}{1:d}_Q_{2:.2f}{3:.2f},dmi={4:.2f}.npz".format(n2, m2, 2*q2n1/n2, 2*q2n2/m2, dmi2))
data2D3 = np.load("./data/DSF_{0:d}{1:d}_Q_{2:.2f}{3:.2f},dmi={4:.2f}.npz".format(n3, m3, 2*q3n1/n3, 2*q3n2/m3, dmi3))

"""

"""
Sxy1 = data2D1["Sxy"]
Szz1 = data2D1["Szz"]

Sxy2 = data2D2["Sxy"]
Szz2 = data2D2["Szz"]

Sxy3 = data2D3["Sxy"]
Szz3 = data2D3["Szz"]

"""
Szz1 = data1["Szz"]
Szz2 = data2["Szz"]
Szz3 = data3["Szz"]
Szz4 = data4["Szz"]

if 2**N1 < 20000:

  Sze1 = data1["Szz_e"]
  Z1 = data1["Z"]

if 2**N2 < 20000:

  Sze2 = data4["Szz_e"]
  Z2 = data4["Z"]
"""
if 2**(2*m1*n1) < 20000:

  Sze2d1 = data2D1["Szz_e"]
  Z2d1 = data2D1["Z"]

if 2**(2*m2*n2) < 20000:
  
  Sze2d2 = data2D2["Szz_e"]
  Z2d2 = data2D2["Z"]
"""
#w1 = data1["w"]
w2 = data2["w"]
#w3 = data3["w"]

"""

w2d = data2D1["w"]



Ti = 3


T = data2D1["T"]
beta = 1/T


#plt.plot(w2d[:], 28 * Sxy1[:, Ti].real, label='Szz_r_dmi={0:.4f}_T={1:.2f}'.format(dmi1, T[Ti]))
#plt.plot(w2d[:], 28 * Sxy2[:, Ti].real, label='Szz_r_dmi={0:.4f}'.format(dmi2))
#plt.plot(w2d[:], 28 * Sxy3[:, Ti].real, label='Szz_r_dmi={0:.4f}'.format(dmi3))

plt.plot(w2d[:], 28 * Szz1[:, Ti].real, label='Szz_r_T={0:.4f},dmi={1:.2f}'.format(T[Ti], dmi1))
plt.plot(w2d[:], 28 * Szz2[:, Ti].real, label='Szz_r_T={0:.4f},dmi={1:.2f}'.format(T[Ti], dmi2))
plt.plot(w2d[:], 28 * Szz3[:, Ti].real, label='Szz_r_T={0:.4f},dmi={1:.2f}'.format(T[Ti], dmi3))

#plt.plot(w3[1:Nw3], 42 * Szz3[1:Nw3, Ti].real, label='Szz_r_T={0:.4f},N={1:d}'.format(T[Ti], M3))

#plt.plot(w2[:], - 2/(3*N1**2 * np.pi) * (Sze1[:, Ti]/Z1[Ti]).imag, label='Szz_e_T={0:.4f}'.format(T[Ti]))
#plt.plot(w2d[:], - 1/(np.pi) * (Sze2[:, Ti]/Z2[Ti]).imag, label='Szz_e_T={0:.4f}'.format(T[Ti]))

plt.legend(loc='best')
plt.show()

plt.clf()

