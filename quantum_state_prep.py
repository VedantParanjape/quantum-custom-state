import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Statevector


def build_oracle(n, m, values):
    """
    Build an oracle circuit: |x⟩|0⟩ → |x⟩|values[x]⟩.

    n: number of address qubits (2^n entries)
    m: number of value qubits (values in [0, 2^m - 1])
    values: list of 2^n integers
    """
    x_reg = QuantumRegister(n, "x")
    val_reg = QuantumRegister(m, "v")
    qc = QuantumCircuit(x_reg, val_reg, name="oracle")

    for addr in range(2**n):
        v = values[addr]

        #Flip x qubits where addr bit is 0, so MCX activates on addr
        for k in range(n):
            if not ((addr >> k) & 1):
                qc.x(x_reg[k])

        #XOR each set bit of v into val_reg
        for j in range(m):
            if (v >> j) & 1:
                qc.mcx(list(x_reg), val_reg[j])

        #Undo the X flips
        for k in range(n):
            if not ((addr >> k) & 1):
                qc.x(x_reg[k])

    return qc


def build_state_prep_circuit(n, m, oracle_a, oracle_p):
    """
    Build the state preparation operator A.

    A|0⟩ produces a state where the flag=|1⟩ component has amplitudes
    proportional to sin(π·a_x/2^m) and phases e^{i·2π·p_x/2^m}.

    Steps:
      1. Uniform superposition H^⊗n on address register
      2. Load a_x via oracle, controlled-Ry to flag, uncompute oracle
      3. Load p_x via oracle, phase gates, uncompute oracle
    """
    x_reg = QuantumRegister(n, "x")
    val_reg = QuantumRegister(m, "v")
    flag = QuantumRegister(1, "f")
    qc = QuantumCircuit(x_reg, val_reg, flag, name="A")

    #Uniform superposition
    qc.h(x_reg)

    #Amplitude loading
    qc.compose(oracle_a, inplace=True)
    for j in range(m):
        angle = 2**(j + 1) * np.pi / 2**m
        qc.cry(angle, val_reg[j], flag[0])
    qc.compose(oracle_a.inverse(), inplace=True)

    #Phase loading
    qc.compose(oracle_p, inplace=True)
    for j in range(m):
        angle = 2 * np.pi * 2**j / 2**m
        qc.p(angle, val_reg[j])
    qc.compose(oracle_p.inverse(), inplace=True)

    return qc


def optimal_iterations(n, m, a):
    """
    Compute the optimal number of Grover iterations for amplitude
    amplification.

    Success probability: p = (1/2^n) Σ_x sin²(π·a_x/2^m)
    Optimal iterations:  k ≈ round((π/(2·arcsin(√p)) - 1) / 2)
    """
    p = sum(np.sin(np.pi * ax / 2**m)**2 for ax in a) / 2**n
    if p <= 0 or p >= 1:
        return 0
    theta = np.arcsin(np.sqrt(p))
    k = int(round((np.pi / (2 * theta) - 1) / 2))
    return max(0, k)


def build_amplified_circuit(n, m, oracle_a, oracle_p, num_iterations):
    x_reg = QuantumRegister(n, "x")
    val_reg = QuantumRegister(m, "v")
    flag = QuantumRegister(1, "f")
    qc = QuantumCircuit(x_reg, val_reg, flag)

    total = n + m + 1
    all_qubits = list(range(total))

    A = build_state_prep_circuit(n, m, oracle_a, oracle_p)

    #Apply A
    qc.compose(A, qubits=all_qubits, inplace=True)

    if num_iterations > 0:
        # S_f: phase oracle marking flag=|1⟩ via Z gate
        oracle_sf = QuantumCircuit(total)
        oracle_sf.z(total - 1)  # flag is the last qubit

        grover_op = GroverOperator(
            oracle=oracle_sf,
            state_preparation=A,
        )

        for _ in range(num_iterations):
            qc.compose(grover_op, qubits=all_qubits, inplace=True)

    return qc


def compute_expected_state(n, m, a, p):
    """
    Classically compute the expected target state with sin encoding.

    Amplitudes: sin(π·a_x / 2^m)
    Phases:     e^{i·2π·p_x / 2^m}
    """
    state = np.zeros(2**n, dtype=complex)
    for x in range(2**n):
        amp = np.sin(np.pi * a[x] / 2**m)
        phase = np.exp(1j * 2 * np.pi * p[x] / 2**m)
        state[x] = amp * phase
    norm = np.linalg.norm(state)
    if norm > 1e-10:
        state /= norm
    return state


def compute_target_state(n, m, a, p):
    """
    Classically compute the assignment's target state (linear encoding).

    Amplitudes: a_x
    Phases:     e^{i·2π·p_x / 2^m}
    """
    state = np.zeros(2**n, dtype=complex)
    for x in range(2**n):
        phase = np.exp(1j * 2 * np.pi * p[x] / 2**m)
        state[x] = a[x] * phase
    norm = np.linalg.norm(state)
    if norm > 1e-10:
        state /= norm
    return state


def state_fidelity(state1, state2):
    """Compute fidelity |⟨ψ1|ψ2⟩|² between two normalized states."""
    return abs(np.dot(state1.conj(), state2))**2


def prepare_state(n, m, a, p):
    """
    Build oracles, construct circuit with amplitude
    amplification, simulate, and return the prepared state.

    Returns the normalized statevector extracted from the flag=|1⟩,
    val_reg=|0⟩ subspace.
    """
    oracle_a = build_oracle(n, m, a)
    oracle_p = build_oracle(n, m, p)

    num_iter = optimal_iterations(n, m, a)
    qc = build_amplified_circuit(n, m, oracle_a, oracle_p, num_iter)

    sv = Statevector(qc)
    data = np.array(sv)

    #Extract flag=1, val_reg=0 component
    #Qubit ordering: x_reg (n) | val_reg (m) | flag (1)
    #Index = x + val * 2^n + flag_bit * 2^(n+m)
    flag_offset = 2**(n + m)
    state = np.array([data[x + flag_offset] for x in range(2**n)])

    norm = np.linalg.norm(state)
    if norm > 1e-10:
        state /= norm

    return state