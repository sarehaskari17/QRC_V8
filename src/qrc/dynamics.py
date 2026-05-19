import numpy as np

from utils import commutator, lindblad_term, psi_sk


def custom_evolution_rk4_update(t, rho_flat, param):
    n_qubits = param["n_qubits"]
    H = param["H"]
    L_mat = param["L_mat"]

    rho = rho_flat.reshape((2**n_qubits, 2**n_qubits))

    hbar = 1.0

    commutator_term = -1j / hbar * commutator(H, rho)

    lindblad_terms = np.zeros_like(rho, dtype=np.complex128)

    for i in range(n_qubits):
        lindblad_terms += lindblad_term(rho, L_mat[i])

    rho_dot = commutator_term + lindblad_terms

    return rho_dot.flatten()


def rk4_update(
    evolution_function,
    param,
    t_span,
    dt,
    y0,
    update_qubits_function=None,
):
    n_qubits = param["n_qubits"]
    n_bet = param["n_bet"]
    sk1 = param["sk1"]
    sk2 = param["sk2"]

    t0 = t_span[0]
    tn = t_span[1]

    nd = len(y0)
    nt = int(np.ceil((tn - t0) / dt)) + 1

    t = np.linspace(t0, tn, nt)
    y = np.zeros((nd, nt), dtype=np.complex128)

    y[:, 0] = y0

    for l in range(1, nt):
        k1 = dt * evolution_function(t[l - 1], y[:, l - 1], param)

        k2 = dt * evolution_function(
            t[l - 1] + dt / 2,
            y[:, l - 1] + k1 / 2,
            param,
        )

        k3 = dt * evolution_function(
            t[l - 1] + dt / 2,
            y[:, l - 1] + k2 / 2,
            param,
        )

        k4 = dt * evolution_function(
            t[l - 1] + dt,
            y[:, l - 1] + k3,
            param,
        )

        y[:, l] = y[:, l - 1] + (
            k1 + 2 * k2 + 2 * k3 + k4
        ) / 6

        if update_qubits_function is not None and l % n_bet == 0:
            rho_l = y[:, l].reshape((2**n_qubits, 2**n_qubits))

            psi1 = psi_sk(sk1[l])
            psi2 = psi_sk(sk2[l])

            rho_l = update_qubits_function(
                rho_l,
                [0],
                [0],
                psi1,
                psi2,
            )

            y[:, l] = rho_l.flatten()

    return t, y