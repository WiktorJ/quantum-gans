import cirq
import numpy as np
from qsgenerator.phase.analitical import construct_hamiltonian
from cirq import GridQubit, ops
from typing import Any, Type, Callable


class PhaseTransitionFinalStateSimulator(cirq.Simulator):

    def __init__(self, g_provider: Callable, size: int, dtype: Type[np.number] = np.complex64,
                 seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None):
        super().__init__(dtype=dtype, seed=seed)
        self.size = size
        self.g_provider = g_provider

    def simulate(self, program: 'cirq.Circuit', param_resolver: 'study.ParamResolverOrSimilarType' = None,
                 qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
                 initial_state: Any = None) -> 'SimulationTrialResult':
        g = self.g_provider()

        H = construct_hamiltonian(self.size, g)

        lam, V = np.linalg.eigh(H)

        # ground state wavefunction
        psi = V[:, 0] / np.linalg.norm(V[:, 0])
        psi = np.kron([1, 0], np.kron([1, 0], psi))
        return super().simulate(program, param_resolver, qubit_order, psi)


class PhaseTransitionFinalStateSimulatorGen(cirq.Simulator):
    def __init__(self, g_provider: Callable, size: int, dtype: Type[np.number] = np.complex64,
                 seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None):
        super().__init__(dtype=dtype, seed=seed)
        self.size = size
        self.g_provider = g_provider

    def simulate(self, program: 'cirq.Circuit', param_resolver: 'study.ParamResolverOrSimilarType' = None,
                 qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
                 initial_state: Any = None) -> 'SimulationTrialResult':
        g = self.g_provider()
        label = [1, 0] if g > 0 else [0, 1]
        initial_state = [1, 0]
        for _ in range(self.size - 1):
            initial_state = np.kron([1, 0], initial_state)
        initial_state = np.kron([1, 0], np.kron(label, np.kron(initial_state, label)))
        return super().simulate(program, param_resolver, qubit_order, initial_state)


class PhaseTransitionFinalStateSimulatorPureGen(cirq.Simulator):
    def __init__(self, size: int, dtype: Type[np.number] = np.complex64,
                 seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None):
        super().__init__(dtype=dtype, seed=seed)
        self.size = size

    def simulate(self, program: 'cirq.Circuit', param_resolver: 'study.ParamResolverOrSimilarType' = None,
                 qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
                 initial_state: Any = None) -> 'SimulationTrialResult':
        label = [1, 0]
        initial_state = [1, 0]
        for _ in range(self.size - 1):
            initial_state = np.kron([1, 0], initial_state)
        initial_state = np.kron(initial_state, label)
        return super().simulate(program, param_resolver, qubit_order, initial_state)
