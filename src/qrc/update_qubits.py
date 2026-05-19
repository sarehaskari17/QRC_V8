import numpy as np

from qiskit.quantum_info import partial_trace


def update_qubits(rho, subsystem1, subsystem2, psi1, psi2):
    n_update1 = len(subsystem1)
    n_update2 = len(subsystem2)

    psi1_density = np.outer(psi1, psi1.conj())
    psi2_density = np.outer(psi2, psi2.conj())

    rho_sub1 = np.array([1])
    rho_sub2 = np.array([1])

    for _ in range(n_update1):
        rho_sub1 = np.kron(rho_sub1, psi1_density)

    for _ in range(n_update2):
        rho_sub2 = np.kron(rho_sub2, psi2_density)

    rho_tr = partial_trace(
        partial_trace(rho, subsystem1),
        subsystem2,
    )

    rho_updated = np.kron(
        np.kron(rho_tr, rho_sub2),
        rho_sub1,
    )

    rho_updated = rho_updated / (
        np.trace(rho_updated) + 1e-300
    )

    return rho_updated