import os
import resource
import gc

os.environ["OMP_NUM_THREADS"] = "56"
os.environ["MKL_NUM_THREADS"] = "56"
os.environ["OPENBLAS_NUM_THREADS"] = "56"

from quspin.basis import spin_basis_general
from quspin.operators import (
        hamiltonian,
        exp_op
    )
from quspin.tools.lanczos import (
        lanczos_full,
        lanczos_iter,
        LTLM_static_iteration,
        FTLM_static_iteration,
    )
from quspin.tools.evolution import expm_multiply_parallel
    
import numpy as np
import networkx as nx  
import matplotlib.pyplot as plt  # plotting library
from time import time
import scipy as sp


ti = time()


def bootstrap_mean(O_r, Id_r, n_bootstrap=100):
    """
    Uses boostraping to esimate the error due to sampling.

    O_r: numerator
    Id_r: denominator
    n_bootstrap: bootstrap sample size

    """
    O_r = np.asarray(O_r)
    Id_r = np.asarray(Id_r)
    #
    avg = np.nanmean(O_r, axis=0) / np.nanmean(Id_r, axis=0)
    n_Id = Id_r.shape[0]
    # n_N = O_r.shape[0]
    #
    i_iter = (np.random.randint(n_Id, size=n_Id) for i in range(n_bootstrap))
    #
    bootstrap_iter = (
        np.nanmean(O_r[i, ...], axis=0) / np.nanmean(Id_r[i, ...], axis=0)
        for i in i_iter
    )
    diff_iter = ((bootstrap - avg) ** 2 for bootstrap in bootstrap_iter)
    err = np.sqrt(sum(diff_iter) / n_bootstrap)
    #
    return avg, err

def build_basis(m, n):

  ### compute basis
  m_2 = m*2
  N = m_2*n
  print("constructed hexagonal lattice with {0:d} sites.\n".format(N))

  basis = spin_basis_general(N, S = "1/2", pauli = 0)
  print("Hilbert space size: {0:d}.\n".format(basis.Ns))
  
  r = []
  
  for i in range(n):
    for j in range(m):
      r.append([(3*i + 2)/2, 3**0.5*(-i/2 + j) ])
      r.append([(3*i + 1)/2, 3**0.5*(-i/2 + j + 0.5)])
  
  return basis, r
  
def build_H(m, n, Jxy, Jz, J_nnn, J_dmi, h, basis):

  #### set up Heisenberg Hamiltonian with quspin #####

  m_2 = m*2
  N = m_2*n
  
  # set up AB sites lists
  
  A = []
  B = []
  
  for i in range(m*n):
      A.append(i*2)
      B.append(i*2 + 1)
  
  # set up spin-spin interaction lists
  
  spin_spin_xy_pn = []
  spin_spin_xy_np = []
  spin_spin_z = []
  spinz = []
  
  for i in A:
      spinz.append([h,i])
  
      spin_spin_z.append([Jz, i, (i+1)%N])
      spin_spin_xy_pn.append([Jxy/2, i, (i+1)%N])
      spin_spin_xy_np.append([Jxy/2, i, (i+1)%N])
  
      spin_spin_z.append([Jz, i, (i-1)%N])
      spin_spin_xy_pn.append([Jxy/2, i, (i-1)%N])
      spin_spin_xy_np.append([Jxy/2, i, (i-1)%N])
  
      spin_spin_z.append([Jz, i, (i+m_2+1)%N])
      spin_spin_xy_pn.append([Jxy/2, i, (i+m_2+1)%N])
      spin_spin_xy_np.append([Jxy/2, i, (i+m_2+1)%N])
  
      spin_spin_z.append([J_nnn, i, (i+2)%N])
      spin_spin_xy_pn.append([J_nnn/2 + J_dmi, i, (i+2)%N])
      spin_spin_xy_np.append([J_nnn/2 - J_dmi, i, (i+2)%N])
  
      spin_spin_z.append([J_nnn, i, (i+m_2)%N])
      spin_spin_xy_pn.append([J_nnn/2 + J_dmi, i, (i+m_2)%N])
      spin_spin_xy_np.append([J_nnn/2 - J_dmi, i, (i+m_2)%N])
  
      spin_spin_z.append([J_nnn, i, (i-m_2-2)%N])
      spin_spin_xy_pn.append([J_nnn/2 + J_dmi, i, (i-m_2-2)%N])
      spin_spin_xy_np.append([J_nnn/2 - J_dmi, i, (i-m_2-2)%N])
  
                      
  for i in B:
      spinz.append([h,i])
  
      spin_spin_z.append([J_nnn, i, (i-2)%N])
      spin_spin_xy_pn.append([J_nnn/2 + J_dmi, i, (i-2)%N])
      spin_spin_xy_np.append([J_nnn/2 - J_dmi, i, (i-2)%N])
  
      spin_spin_z.append([J_nnn, i, (i-m_2)%N])
      spin_spin_xy_pn.append([J_nnn/2 + J_dmi, i, (i-m_2)%N])
      spin_spin_xy_np.append([J_nnn/2 - J_dmi, i, (i-m_2)%N])
  
      spin_spin_z.append([J_nnn, i, (i+m_2+2)%N])
      spin_spin_xy_pn.append([J_nnn/2 + J_dmi, i, (i+m_2+2)%N])
      spin_spin_xy_np.append([J_nnn/2 - J_dmi, i, (i+m_2+2)%N])

  # define spin-spin interaction lists 
  static = [["+-", spin_spin_xy_pn], ["-+", spin_spin_xy_np], ["zz", spin_spin_z], ["z", spinz]]#
  dynamic = []
  
  ### construct Hamiltonian
  H = hamiltonian(static, dynamic, basis=basis, dtype=np.complex128, check_herm = False)
  return H

def delta(x):
  e = 0.1
  D = e/(e**2 + x**2) * 2/np.pi
  return D

def dsf_lanczos(A, B, H, E0, k_d, N_samples, w, beta, O1=None, O2=None):

  ### lanczos method

  dsfT_list = []
  dsfZ_list = []

  O1_LT = []
  Z1_LT = []
  O2_LT = []
  Z2_LT = []
  
  out = np.zeros((k_d, H.Ns), dtype=np.complex128)
  
  # calculate iterations
  for ni in range(N_samples):
      
      # generate normalized random vector
      r1 = np.random.normal(0, 1, size=H.Ns)
      r1 /= np.linalg.norm(r1)
      
      Br = B.dot(r1)
      r2 = Br/np.linalg.norm(Br)
      
      # get lanczos basis
      E1, V1, lv1 = lanczos_full(H, r1, k_d, eps=1e-8, full_ortho=True)
      E2, V2, lv2 = lanczos_full(H, r2, k_d, eps=1e-8, full_ortho=True)
      
      psi_i = np.einsum('ij,ik->jk', V1, lv1)
      psi_j = np.einsum('ij,ik->jk', V2, lv2)
      
      # shift energy to avoid overflows
      E1 -= E0
      E2 -= E0
      delta_E = E1[:, None] - E2[None, :]
      
      A_M = A.tocsr()
      B_M = B.tocsr()
      
      r_psi_i = psi_i.conj() @ r1
      psi_jB_r = psi_j.conj() @ B_M @ r1
      
      results_FT = np.zeros((len(w), len(beta)), dtype=np.complex128)
      
      for bi in range(len(beta)):
        
        psi_T = np.einsum('i,ij->ij',np.exp(-beta[bi] * E1), psi_i.conj())
        psi_iApsi_j = psi_T @ A_M @ psi_j.T 
        
        for k in range(len(w)):
          
          delta_w = delta(w[k] + delta_E)
          pAp = psi_iApsi_j * delta_w

          results_FT[k, bi] = np.vdot(r_psi_i, pAp @ psi_jB_r)
      
      # compute Id
      p = np.exp(-np.outer(beta, E1))
      c = np.einsum("j,aj,...j->a...", V1[0, :], V1, p)
      Id_T = np.squeeze(c[0, ...])
      
      # save results to a list
      dsfT_list.append(results_FT)
      dsfZ_list.append(Id_T)
      
      if O1 is not None:

        res_LT, Id_LT = LTLM_static_iteration({"O": O1}, E1, V1, lv1, beta=beta)
      
        # save results to a list
        O1_LT.append(res_LT["O"])
        Z1_LT.append(Id_LT)

      if O2 is not None:

        res_LT, Id_LT = LTLM_static_iteration({"O": O2}, E1, V1, lv1, beta=beta)
      
        # save results to a list
        O2_LT.append(res_LT["O"])
        Z2_LT.append(Id_LT)
      
      print("Peak memory used for lanczos samples {0:d}:{1:.4f} MB".format(ni+1, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
      
      del E1, V1, lv1, E2, V2, lv2, results_FT, Id_T
      gc.collect()

  if O1 is not None:
    if O2 is not None:
      return dsfT_list, dsfZ_list, O1_LT, Z1_LT, O2_LT, Z2_LT
    else: 
      return dsfT_list, dsfZ_list, O1_LT, Z1_LT
  elif O2 is not None:
      return dsfT_list, dsfZ_list, O2_LT, Z2_LT
  else:
    return dsfT_list, dsfZ_list
    
def dsf_exact(A, B, H, w, beta):

  E1, V1 = H.eigh()
  E1 -= E1[0]
  Ns = len(E1)
  
  A_M = A.tocsr()
  B_M = B.tocsr()  
  
  V2 = V1.conj().T

  pAp = (V2 @ A_M @ V1)  #(Ns, Ns)
  pBp = (V2 @ B_M @ V1)  #(Ns, Ns)

  delta_E = E1[:, None] - E1[None, :]
  
  p = np.exp(-np.outer(beta, E1))
  Z = np.einsum("j...->...", p) 
  
  results_FT = np.zeros((len(w), len(beta)), dtype=np.complex128)
  
  beta_delta_E = np.outer(beta, delta_E.ravel())  # (beta, Ns , Ns)
  beta_delta_E = np.clip(beta_delta_E, None, 700)
  exp_factor = np.exp(beta_delta_E).reshape(len(beta), Ns, Ns)
  
  for k in range(len(w)):

    denom = w[k] + delta_E + 0.05j # (Ns, Ns)
          
    pAp1 = exp_factor * pAp[None, :, :] / denom[None, :, :] 
    pABp = np.einsum('...ij,ji->...', pAp1, pBp)

    results_FT[k, :] = pABp
        
    print("Peak memory used for exact solution w={0:.1f}:{1:.4f} MB".format(w[k], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
  
  return E1, V1, results_FT, Z


### properties of system
# lattice parameters

n = 3  # number of columns of hexagons in the lattice
m = 3  # number of rows of hexagons in the lattice
m_2 = m*2

N = m_2*n  #Number of lattice sites

Jxy = -1.0
Jz = -1.0    #tunnelling matrix element
J_nnn = -0.1
J_dmi = 0.1j

h = -0.1

T_L = [0.01, 0.03, 0.05, 0.075, 0.1, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 1]
T_L = np.array(T_L)
beta_L = 1.0/(T_L + 1e-15)


N_samples = 15
k_d = 400   #Krylov dim

#b1 = [2/3, 0]
#b2 = [-1/3, 1/(3**0.5)]

b1 = [2/3, 0]
b2 = [1/3, 1/(3**0.5)]

n1 = 1
n2 = 1

Q = np.zeros(2)
Q[0] = 2*np.pi * (b1[0]*n1/n + b2[0]*n2/m)
Q[1] = 2*np.pi * (b1[1]*n1/n + b2[1]*n2/m)

w = np.linspace(0, 5, 251)

basis, r = build_basis(m, n)
H = build_H(m, n, Jxy, Jz, J_nnn, J_dmi, h, basis)

# compute eigensystem
[E0] = H.eigsh(k=1, which="SA", maxiter=1e4, return_eigenvectors=False)
#E1, V1 = H.eigh()

# build Sx, Sy, Sz operator
S1_list = []
S2_list = []
M_list = []

for i in range(N):
  S1_list.append([1/N * np.exp(1j*(Q[0]*r[i][0] + Q[1]*r[i][1])), i])
  S2_list.append([1/N * np.exp(-1j*(Q[0]*r[i][0] + Q[1]*r[i][1])), i])
  M_list.append([1/N, i])
  
Sx1 = hamiltonian([["x", S1_list]], [], basis=basis, dtype=np.complex128, check_herm = False)
Sx2 = hamiltonian([["x", S2_list]], [], basis=basis, dtype=np.complex128, check_herm = False)
  
Sy1 = hamiltonian([["y", S1_list]], [], basis=basis, dtype=np.complex128, check_herm = False)
Sy2 = hamiltonian([["y", S2_list]], [], basis=basis, dtype=np.complex128, check_herm = False)
  
Sz1 = hamiltonian([["z", S1_list]], [], basis=basis, dtype=np.complex128, check_herm = False)
Sz2 = hamiltonian([["z", S2_list]], [], basis=basis, dtype=np.complex128, check_herm = False)

M = hamiltonian([["z", M_list]], [], basis=basis, dtype=np.float64)
M2 = M**2

print("Peak memory used (MB):{0:.4f}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

o1_FT = []
do1_FT = []
o2_FT = []
do2_FT = []
o3_FT = []
do3_FT = []

S1t, z1t = dsf_lanczos(Sx1, Sx2, H, E0, k_d, N_samples, w, beta_L)
S2t, z2t = dsf_lanczos(Sy1, Sy2, H, E0, k_d, N_samples, w, beta_L)
S3t, z3t, M_LT, Z_LT, M2_LT, Z2_LT = dsf_lanczos(Sz1, Sz2, H, E0, k_d, N_samples, w, beta_L, M, M2)

print("Peak memory used (MB):{0:.4f}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))

S1t = np.array(S1t)
S2t = np.array(S2t)
S3t = np.array(S3t)

z1t = np.array(z1t)
z2t = np.array(z2t)
z3t = np.array(z3t)

for i in range(len(w)):

  a1 = []
  b1 = []
  a2 = []
  b2 = []
  a3 = []
  b3 = []

  for j in range(len(T_L)):
  
    # calculating error bars on the expectation values
    o1, do1 = bootstrap_mean(S1t[:, i, j], z1t[:, j])
    o2, do2 = bootstrap_mean(S2t[:, i, j], z2t[:, j])
    o3, do3 = bootstrap_mean(S3t[:, i, j], z3t[:, j])
  
    a1.append(o1)
    b1.append(do1)
    a2.append(o2)
    b2.append(do2)
    a3.append(o3)
    b3.append(do3)
  
  o1_FT.append(a1)
  do1_FT.append(b1)
  o2_FT.append(a2)
  do2_FT.append(b2)
  o3_FT.append(a3)
  do3_FT.append(b3)
    
  del o1, do1, o2, do2, o3, do3
  gc.collect()

o1_FT = np.array(o1_FT, dtype=np.complex128)
do1_FT = np.array(do1_FT, dtype=np.complex128)
o2_FT = np.array(o2_FT, dtype=np.complex128)
do2_FT = np.array(do2_FT, dtype=np.complex128)
o3_FT = np.array(o2_FT, dtype=np.complex128)
do3_FT = np.array(do2_FT, dtype=np.complex128)

m_LT, dm_LT = bootstrap_mean(M_LT, Z_LT)
m2_LT, dm2_LT = bootstrap_mean(M2_LT, Z2_LT)

print("Peak memory used (MB):{0:.4f}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))


fig = plt.figure()

plt.plot(w[:], (o1_FT[:, 2] + o2_FT[:, 2]).real, label='Sxy_r')
plt.plot(w[:], (o1_FT[:, 2] + o2_FT[:, 2]).imag, label='Sxy_i')

plt.plot(w[:], o3_FT[:, 2].real, label='Szz_r')
plt.plot(w[:], o3_FT[:, 2].imag, label='Szz_i')

plt.legend(loc='best')
plt.savefig('./2D_{0:d}{1:d}_Saa_T={2:.1f},Q=({3:.2f},{4:.2f})pi,dmi={5:.2f}.png'.format(n, m, T_L[2], 2*n1/n, 2*n2/m, J_dmi))
plt.clf()

plt.plot(w[:], (o1_FT[:, -1] + o2_FT[:, -1]).real, label='Sxy_r')
plt.plot(w[:], (o1_FT[:, -1] + o2_FT[:, -1]).imag, label='Sxy_i')

plt.plot(w[:], o2_FT[:, -1].real, label='Szz_r')
plt.plot(w[:], o2_FT[:, -1].imag, label='Szz_i')

plt.legend(loc='best')
plt.savefig('./2D_{0:d}{1:d}_Saa_T={2:.1f},Q=({3:.2f},{4:.2f})pi,dmi={5:.2f}.png'.format(n, m, T_L[-1], 2*n1/n, 2*n2/m, J_dmi))
plt.clf()

plt.errorbar(T_L, m_LT, dm_LT, marker=".", zorder=-1)
plt.savefig('./2D_{0:d}{1:d}_M,dmi={2:.2f}.png'.format(n, m, J_dmi))

plt.clf()

if H.Ns < 20000:

  E_e, V_e, Szz_e, Z_e = dsf_exact(Sz1, Sz2, H, w, beta_L)
  np.savez("./data/DSF_{0:d}{1:d}_Q_{2:.2f}{3:.2f},dmi={4:.2f}.npz".format(n, m, 2*n1/n, 2*n2/m, J_dmi), Sxx=o1_FT, Syy=o2_FT, Szz=o3_FT, dSxx=do1_FT, dSyy=do2_FT, dSzz=do3_FT, Szz_e=Szz_e, Z=Z_e, T=T_L, w=w, E=E_e, V=V_e, ML=m_LT, M2L=m2_LT)

else:

  np.savez("./data/DSF_{0:d}{1:d}_Q_{2:.2f}{3:.2f},dmi={4:.2f}.npz".format(n, m, 2*n1/n, 2*n2/m, J_dmi), Sxx=o1_FT, Syy=o2_FT, Szz=o3_FT, dSxx=do1_FT, dSyy=do2_FT, dSzz=do3_FT, T=T_L, w=w, ML=m_LT, M2L=m2_LT)

print("simulation took {0:.4f} sec".format(time()-ti))
