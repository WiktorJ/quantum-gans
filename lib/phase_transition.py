import numpy as np


def construct_hamiltonian(l, gzz, gx, gzxz, pbc=True):
    """
    Construct a quantum Hamiltonian as full matrix.

    Reference:
      Adam Smith, Bernhard Jobst, Andrew G. Green, Frank Pollmann
      Crossing a topological phase transition with a quantum computer
      arXiv: 1910.05351
    """
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
