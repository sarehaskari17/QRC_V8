import numpy as np

from qiskit.quantum_info import DensityMatrix, Operator, Pauli


def build_pauli_operators(n_qubits):
    x_ops = []
    y_ops = []
    z_ops = []

    x_mat = []
    y_mat = []
    z_mat = []

    zeros = Operator(np.zeros((2**n_qubits, 2**n_qubits)))

    for i in range(n_qubits):
        x_op = zeros + Pauli("X")(i)
        y_op = zeros + Pauli("Y")(i)
        z_op = zeros + Pauli("Z")(i)

        x_ops.append(x_op)
        y_ops.append(y_op)
        z_ops.append(z_op)

        x_mat.append(x_op.to_matrix())
        y_mat.append(y_op.to_matrix())
        z_mat.append(z_op.to_matrix())

    identity = np.identity(2**n_qubits)

    return x_ops, y_ops, z_ops, x_mat, y_mat, z_mat, identity, zeros


def initial_density_matrix(n_qubits):
    return DensityMatrix.from_label("0" * n_qubits)


def generate_input_sequences(t_eval, f0, gamma_scale, n_sequences=9):
    frequencies = np.linspace(
        f0 * gamma_scale / 1000,
        f0 * gamma_scale / 10,
        20,
    )

    sequences = []

    for _ in range(n_sequences):
        sequence = sum(
            np.sin(
                2 * np.pi * freq * t_eval
                + 2 * np.pi * np.random.rand()
            )
            for freq in frequencies
        )

        sequence = (
            (sequence - np.min(sequence))
            / (np.max(sequence) - np.min(sequence))
        )

        sequences.append(sequence)

    return sequences


def generate_random_couplings(
    n_qubits,
    n_matrices,
    coupling_strength,
):
    coupling_matrices = []

    for _ in range(n_matrices):
        matrix = np.zeros((n_qubits, n_qubits))

        for row in range(n_qubits):
            for col in range(row + 1, n_qubits):
                value = np.random.rand()
                matrix[row, col] = value
                matrix[col, row] = value

        coupling_matrices.append(
            coupling_strength * matrix
        )

    return coupling_matrices


def build_hamiltonian(
    n_qubits,
    h_z,
    j_x,
    z_ops,
    zeros,
):
    H = zeros

    for i in range(n_qubits):
        H += 0.5 * h_z * z_ops[i]

        for j in range(i):
            xx_op = zeros + Pauli("XX")(i, j)
            H += 0.5 * j_x[i][j] * xx_op

    return H.to_matrix()


def build_lindblad_operators(
    n_qubits,
    gamma,
    x_ops,
    y_ops,
    z_ops,
):
    L_ops = []
    L_mat = []

    for i in range(n_qubits):
        Y = y_ops[i]
        Z = z_ops[i]

        L = np.sqrt(gamma) * 0.5 * (Z + 1j * Y)

        L_ops.append(L)
        L_mat.append(L.to_matrix())

    return L_ops, L_mat


def initialize_reservoir_z(
    n_qubits,
    n_samples,
    n_injects,
):
    return np.zeros((n_qubits, n_samples, n_injects))


def setup_qrc_system(
    n_qubits,
    f0,
    gamma_scale,
    coupling_strength,
    h_z,
    gamma,
    t_final,
    delta_t,
    n_samples,
    n_injects,
    n_sequences=9,
    n_coupling_matrices=10,
):
    n_steps = int(np.ceil(t_final / delta_t)) + 1
    t_eval = np.linspace(0.0, t_final, n_steps)
    t_span = [0.0, t_final]

    (
        x_ops,
        y_ops,
        z_ops,
        x_mat,
        y_mat,
        z_mat,
        identity,
        zeros,
    ) = build_pauli_operators(n_qubits)

    rho_0_op = initial_density_matrix(n_qubits)
    rho_0 = rho_0_op.data

    sequences = generate_input_sequences(
        t_eval=t_eval,
        f0=f0,
        gamma_scale=gamma_scale,
        n_sequences=n_sequences,
    )

    coupling_matrices = generate_random_couplings(
        n_qubits=n_qubits,
        n_matrices=n_coupling_matrices,
        coupling_strength=coupling_strength,
    )

    J = coupling_matrices[0]
    J_x = J

    H = build_hamiltonian(
        n_qubits=n_qubits,
        h_z=h_z,
        j_x=J_x,
        z_ops=z_ops,
        zeros=zeros,
    )

    L_ops, L_mat = build_lindblad_operators(
        n_qubits=n_qubits,
        gamma=gamma,
        x_ops=x_ops,
        y_ops=y_ops,
        z_ops=z_ops,
    )

    reservoir_z = initialize_reservoir_z(
        n_qubits=n_qubits,
        n_samples=n_samples,
        n_injects=n_injects,
    )

    return {
        "x_ops": x_ops,
        "y_ops": y_ops,
        "z_ops": z_ops,
        "x_mat": x_mat,
        "y_mat": y_mat,
        "z_mat": z_mat,
        "identity": identity,
        "zeros": zeros,
        "rho_0_op": rho_0_op,
        "rho_0": rho_0,
        "t_eval": t_eval,
        "t_span": t_span,
        "sequences": sequences,
        "sk1": sequences[0],
        "sk2": sequences[1],
        "coupling_matrices": coupling_matrices,
        "J": J,
        "J_x": J_x,
        "H": H,
        "L_ops": L_ops,
        "L_mat": L_mat,
        "reservoir_z": reservoir_z,
    }
