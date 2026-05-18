#utils.py

# ============================================================================
# STANDARD LIBRARY
# ============================================================================

import csv
import heapq
import math
import os
import random as rand
from random import gauss

# ============================================================================
# NUMERICAL / SCIENTIFIC COMPUTING
# ============================================================================

import numpy as np
import scipy as sp

from numpy import linalg as LA
from numpy.linalg import inv

from scipy.integrate import solve_ivp
from scipy.linalg import schur, svd
from scipy.special import factorial, gamma

# ============================================================================
# MACHINE LEARNING
# ============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================================
# GRAPH / NETWORK TOOLS
# ============================================================================

import networkx as nx

# ============================================================================
# VISUALIZATION
# ============================================================================

import matplotlib.pyplot as plt

# ============================================================================
# QISKIT
# ============================================================================

from qiskit import QuantumCircuit, assemble

from qiskit.quantum_info import (
    DensityMatrix,
    Operator,
    Pauli,
    partial_trace,
)

from qiskit.visualization import (
    plot_bloch_vector,
    plot_histogram,
)

from qiskit_aer import Aer

from qiskit_dynamics import Signal, Solver

from qiskit_textbook.widgets import binary_widget

# ============================================================================
# QUTIP
# ============================================================================

from qutip import Qobj, partial_transpose

# ============================================================================
# LINEAR REGRESSION / MEMORY CAPACITY
# ============================================================================

def tikhonov_regularization(S, G, lambda_value):
    """
    Compute Tikhonov-regularized linear regression weights.

    Parameters
    ----------
    S : np.ndarray
        State matrix.
    G : np.ndarray
        Target/output matrix.
    lambda_value : float
        Regularization strength.

    Returns
    -------
    np.ndarray
        Regularized regression matrix.
    """
    S_T = S.T

    SS_T = np.dot(S, S_T)

    regularization_term = (
        lambda_value * np.eye(SS_T.shape[0])
    )

    inverse_term = np.linalg.inv(
        SS_T + regularization_term
    )

    A = np.dot(
        G,
        np.dot(S_T, inverse_term)
    )

    return A


# ============================================================================
# ENTANGLEMENT
# ============================================================================

def reverse_qubit_labels(qubit_list, n_qubits):
    """
    Reverse qubit labels to match QuTiP convention.

    Parameters
    ----------
    qubit_list : list[int]
        Qubit indices.
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    list[int]
        Reversed qubit indices.
    """
    reversed_labels = []

    for qubit in qubit_list:
        reversed_labels.append(
            n_qubits - 1 - qubit
        )

    return reversed_labels


def log_negativity_qt(rho, n_qubits, subsystem):
    """
    Compute logarithmic negativity using partial transpose.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix.
    n_qubits : int
        Total number of qubits.
    subsystem : list[int]
        Subsystem for partial transpose.

    Returns
    -------
    float
        Logarithmic negativity.
    """
    dims = [[2] * n_qubits, [2] * n_qubits]

    rho_qobj = Qobj(rho, dims=dims)

    subsystem = reverse_qubit_labels(
        subsystem,
        n_qubits
    )

    mask = [0] * n_qubits

    for qubit in subsystem:
        mask[qubit] = 1

    rho_transposed = partial_transpose(
        rho_qobj,
        mask
    )

    rho_np = rho_transposed.full()

    _, singular_values, _ = svd(rho_np)

    trace_norm = np.sum(
        np.abs(singular_values)
    )

    return math.log2(trace_norm)


# ============================================================================
# OPEN QUANTUM SYSTEMS
# ============================================================================

def commutator(A, B):
    """
    Compute matrix commutator [A, B].

    Parameters
    ----------
    A : np.ndarray
        Matrix A.
    B : np.ndarray
        Matrix B.

    Returns
    -------
    np.ndarray
        Commutator matrix.
    """
    return np.dot(A, B) - np.dot(B, A)


def lindblad_term(rho, L):
    """
    Compute Lindblad dissipator term.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix.
    L : np.ndarray
        Lindblad operator.

    Returns
    -------
    np.ndarray
        Lindblad contribution.
    """
    L_rho_Ld = np.dot(
        L,
        np.dot(rho, L.conj().T)
    )

    Ld_L_rho = np.dot(
        L.conj().T,
        np.dot(L, rho)
    )

    rho_Ld_L = np.dot(
        rho,
        np.dot(L.conj().T, L)
    )

    return (
        L_rho_Ld
        - 0.5 * Ld_L_rho
        - 0.5 * rho_Ld_L
    )


# ============================================================================
# INPUT STATES
# ============================================================================

def psi_sk(sk):
    """
    Generate single-qubit input state.

    Parameters
    ----------
    sk : float
        Input parameter in [0, 1].

    Returns
    -------
    np.ndarray
        Single-qubit pure state vector.
    """
    return (
        np.sqrt(1 - sk)
        * np.array((1, 0), dtype=np.complex128)
        + np.sqrt(sk)
        * np.array((0, 1), dtype=np.complex128)
    )
