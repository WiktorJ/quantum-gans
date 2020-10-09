import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


def construct_hamiltonian(L, gzz, gx, gzxz, pbc=True):
    """
    Construct a quantum Hamiltonian as full matrix.

    Reference:
      Adam Smith, Bernhard Jobst, Andrew G. Green, Frank Pollmann
      Crossing a topological phase transition with a quantum computer
      arXiv: 1910.05351
    """
    sigma = [np.array([[1,  0 ], [0,  1]]),
             np.array([[0,  1 ], [1,  0]]),
             np.array([[0, -1j], [1j, 0]]),
             np.array([[1,  0 ], [0, -1]])]
    # Hamiltonian is actually real-valued
    H = np.zeros((2**L, 2**L))
    for i in range(1, L):
        H -= gzz * np.kron(np.identity(2**(i-1)), np.kron(sigma[3], np.kron(sigma[3], np.identity(2**(L-i-1)))))
    # external field
    for i in range(1, L+1):
        H -= gx * np.kron(np.identity(2**(i-1)), np.kron(sigma[1], np.identity(2**(L-i))))
    for i in range(1, L-1):
        H += gzxz * np.kron(np.identity(2**(i-1)), np.kron(sigma[3], np.kron(sigma[1], np.kron(sigma[3], np.identity(2**(L-i-2))))))
    if pbc:
        # periodic boundary conditions
        H -= gzz * np.kron(sigma[3], np.kron(np.identity(2**(L-2)), sigma[3]))
        H += gzxz * np.kron(sigma[3], np.kron(np.identity(2**(L-3)), np.kron(sigma[3], sigma[1])))
        H += gzxz * np.kron(sigma[1], np.kron(sigma[3], np.kron(np.identity(2**(L-3)), sigma[3])))
    return H


def thermal_state(H, beta):
    """
    Construct the thermal state of a quantum system governed by Hamltonian H at inverse temperature beta.
    """
    rho = expm(-beta*H)
    return rho/np.trace(rho)


def main():

    # number of lattice sites
    L = 9

    # example parameters along the curve (see the paper)
    g = 2
    gzz  = 2*(1 - g**2)
    gx   = (1 + g)**2
    gzxz = (g - 1)**2
    H = construct_hamiltonian(L, gzz, gx, gzxz)
    print('H:')
    print(H)
    symerr = np.linalg.norm(H - H.conj().T)
    print('symerr (should be zero):', symerr)

    λ, V = np.linalg.eigh(H)

    # show eigenvalue spectrum
    plt.plot(λ)
    plt.xlabel('i')
    plt.ylabel(r'$λ_i$')
    plt.title('eigenvalues of H')
    plt.show()

    # ground state wavefunction
    ψ = V[:, 0] / np.linalg.norm(V[:, 0])
    # consistency check
    err0 = np.linalg.norm(H @ ψ - λ[0]*ψ)
    print('err0 (should be zero):', err0)

    # ground state density matrix
    ρ0 = np.outer(ψ, ψ.conj())
    print('np.linalg.matrix_rank(ρ0) (should be 1):', np.linalg.matrix_rank(ρ0))

    # thermal density matrix
    β = 0.1
    ρβ = thermal_state(H, β)
    plt.plot(np.linalg.eigvalsh(ρβ))
    plt.xlabel('i')
    plt.ylabel(r'$λ_i$')
    plt.title('eigenvalues of ρβ')
    plt.show()


if __name__ == '__main__':
    main()