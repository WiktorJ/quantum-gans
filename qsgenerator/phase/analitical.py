import numpy as np


def construct_hamiltonian(l, g, pbc=True):
    """
    Construct a quantum Hamiltonian as full matrix.

    Reference:
      Adam Smith, Bernhard Jobst, Andrew G. Green, Frank Pollmann
      Crossing a topological phase transition with a quantum computer
      arXiv: 1910.05351
    """

    gzz = 2 * (1 - g ** 2)
    gx = (1 + g) ** 2
    gzxz = (g - 1) ** 2

    sigma = [np.array([[1, 0], [0, 1]]),
             np.array([[0, 1], [1, 0]]),
             np.array([[0, -1j], [1j, 0]]),
             np.array([[1, 0], [0, -1]])]
    # Hamiltonian is actually real-valued
    H = np.zeros((2 ** l, 2 ** l))
    for i in range(1, l):
        H -= gzz * np.kron(np.identity(2 ** (i - 1)),
                           np.kron(sigma[3], np.kron(sigma[3], np.identity(2 ** (l - i - 1)))))
    # external field
    for i in range(1, l + 1):
        H -= gx * np.kron(np.identity(2 ** (i - 1)), np.kron(sigma[1], np.identity(2 ** (l - i))))
    for i in range(1, l - 1):
        H += gzxz * np.kron(np.identity(2 ** (i - 1)),
                            np.kron(sigma[3], np.kron(sigma[1], np.kron(sigma[3], np.identity(2 ** (l - i - 2))))))
    if pbc:
        # periodic boundary conditions
        H -= gzz * np.kron(sigma[3], np.kron(np.identity(2 ** (l - 2)), sigma[3]))
        H += gzxz * np.kron(sigma[3], np.kron(np.identity(2 ** (l - 3)), np.kron(sigma[3], sigma[1])))
        H += gzxz * np.kron(sigma[1], np.kron(sigma[3], np.kron(np.identity(2 ** (l - 3)), sigma[3])))
    return H


def get_ground_state_for_g(g, size):
    H = construct_hamiltonian(size, g)

    lam, V = np.linalg.eigh(H)
    # ground state wavefunction
    return V[:, 0] / np.linalg.norm(V[:, 0])


# Methods to construct angles for U1 and U gates from the paper
def get_theta_v(g):
    return np.arcsin(np.sqrt(np.abs(g)) / np.sqrt(1 + np.abs(g)))


def get_theta_w(g):
    return np.arccos((np.sign(g) * np.sqrt(np.abs(g))) / np.sqrt(1 + np.abs(g)))


def get_theta_r(g):
    return 2 * np.arcsin(1 / np.sqrt(1 + np.abs(g)))


def get_g_parameters_provider():
    def provider(g):
        return [get_theta_v(g), get_theta_w(g), get_theta_r(g)]

    return provider
