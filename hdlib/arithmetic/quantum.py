"""Quantum implementation of the MAP arithmetic operators."""

from math import atan2, sqrt, ceil, log2, pi
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from mthree import M3Mitigation
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import DiagonalGate, QFT
from qiskit.quantum_info import Statevector
from qiskit.providers.backend import Backend
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Sampler


def phase_oracle_gate(vec_bipolar: np.ndarray, label: str="O_v") -> DiagonalGate:
    """Creates an efficient diagonal unitary gate that imparts phases based on a bipolar vector.

    This function is a core component for encoding classical bipolar vectors into the phase
    of a quantum state. A `DiagonalGate` is highly efficient as it corresponds to a
    diagonal matrix, which only requires single-qubit Z rotations to implement.

    Parameters
    ----------
    vec_bipolar : numpy.ndarray
        A classical vector containing only -1 and +1 values.
    label : str, default "O_v"
        An optional label for the created Qiskit gate.

    Returns
    -------
    qiskit.circuit.library.DiagonalGate
        A unitary gate that, when applied to a state in the computational basis,
        multiplies the amplitude of each basis state `|i>` by `vec_bipolar[i]`.

    Raises
    ------
    ValueError
        If the input `vec_bipolar` contains values other than -1 or +1.
    """

    vec = np.asarray(vec_bipolar)

    if not np.all(np.isin(vec, [-1, 1])):
        raise ValueError("Bipolar vector must contain only -1 or +1.")

    # Convert to complex diagonal entries
    gate = DiagonalGate(vec.tolist())
    gate.label = label

    return gate

def bind(circuits: List[QuantumCircuit]) -> QuantumCircuit:
    """Applies a sequence of quantum circuits to perform binding.
    This function only accepts a list of QuantumCircuit objects as input.

    It assumes all inputs logically operate on the same number of qubits.
    """

    if not circuits:
        raise ValueError("Input list for bind cannot be empty.")

    # Infer the number of qubits from the first circuit in the list.
    num_qubits = circuits[0].num_qubits
    qc = QuantumCircuit(num_qubits, name="Phase_Bind_Op")

    # Sequentially compose each circuit.
    for circuit in circuits:
        # This check is for robustness, though the type hint should prevent incorrect types.
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("All items in the bind list must be QuantumCircuit objects.")

        if circuit.num_qubits != num_qubits:
            raise ValueError("All circuits in the bind list must have the same number of qubits.")

        qc.compose(circuit, inplace=True)

    return qc

def bundle(
    unitary_circuits: List[QuantumCircuit], 
    weights: List[float], 
    oaa_rounds: int=1,
    optimize_rounds: bool=False,
    classical_computation: bool=False,
    probabilistic: bool=False,
    probabilistic_rounds: int=1
) -> Tuple[QuantumCircuit, QuantumRegister, QuantumRegister]:
    """High-level function for quantum bundling using LCU and OAA.

    This function encapsulates the two-stage process of quantum bundling. First, it
    constructs a Linear Combination of Unitaries (LCU) circuit to prepare a state
    that is a weighted superposition of the states prepared by each input circuit.
    Second, it applies Oblivious Amplitude Amplification (OAA) to amplify the
    component of this state corresponding to the successful preparation, effectively
    creating the bundled prototype.

    Parameters
    ----------
    unitary_circuits : list[qiskit.QuantumCircuit]
        A list of quantum circuits, each preparing a state to be bundled.
    weights : list[float]
        A list of weights corresponding to each circuit for the LCU superposition.
    oaa_rounds : int, default 1
        The number of amplification rounds for the OAA algorithm.
    optimize_rounds : bool, default False
        Search for the optimal number of rounds if `optimize_rounds=True and oaa_rounds>1`.
    classical_computation : bool, default False
        If enabled, it performs the bundling classically, without performing LCU+OAA.
    probabilistic : bool, default False
        If enabled, it performs a probabilistic LCU by controlling only one unitary at a time.
        It prevents the explosion in circuit's depth but heavily relies on `probabilistic_rounds`.
    probabilistic_rounds : int, default 1
        The number of rounds for the probabilistic LCU.

    Returns
    -------
    tuple[qiskit.QuantumCircuit, qiskit.QuantumRegister, qiskit.QuantumRegister]
        A tuple containing:
        - The final quantum circuit that prepares the bundled state.
        - The quantum register for the LCU ancilla qubits.
        - The quantum register for the system qubits.
    """

    def build_lcu_from_unitaries(
        unitary_circuits: List[QuantumCircuit], 
        weights: List[float],
        probabilistic: bool=False,
        rounds: int=1,
        seed: int=42
    ) -> Tuple[QuantumCircuit, QuantumRegister, QuantumRegister]:
        """Builds a Linear Combination of Unitaries (LCU) operator from a list of circuits.

        This function constructs the core operator for the LCU algorithm. It prepares a
        state `|psi>` such that `|psi> = A |0...0>`, where `A` is the LCU operator.
        The state `|psi>` is a superposition of states, where each component corresponds
        to one of the input unitary circuits acting on the system qubits, controlled by
        the state of an ancilla register.

        The operator A is constructed as the sandwich A = (V_inv @ I) * C-U * (V @ I).

        It can optionally build a shallow probabilistic LCU operator by controlling 
        only one unitary at a time. This prevents the explosion of the circuit's depth, 
        but it's probabilistic and heavily depends on the number of rounds.
        """

        K = len(unitary_circuits)

        if K == 0:
            raise ValueError("List of unitary circuits cannot be empty.")

        if probabilistic:
            rng = np.random.default_rng(seed)

            n_sys = unitary_circuits[0].num_qubits

            # Single ancilla for probabilistic control
            anc = QuantumRegister(1, "anc")  
            sys_reg = QuantumRegister(n_sys, "sys")
            circ = QuantumCircuit(anc, sys_reg, name="Shallow_LCU")

            for r in range(rounds):
                # Pick one unitary according to weights
                idx = rng.choice(K, p=np.array(weights)/np.sum(weights))
                unitary = unitary_circuits[idx]

                # Convert to single-qubit controlled gate (controlled on ancilla)
                cu_gate = unitary.to_gate(label=f"U_{idx}").control(1)
                circ.h(anc)
                circ.append(cu_gate, [anc[0]] + sys_reg[:])
                circ.h(anc)

            return circ, anc, sys_reg

        n = unitary_circuits[0].num_qubits
        m = int(ceil(log2(K))) or 1

        anc = QuantumRegister(m, "anc")
        sys = QuantumRegister(n, "sys")
        circ = QuantumCircuit(anc, sys, name="A_from_U")

        # Create the V (prepare) operator as a separate circuit
        amp_anc = np.zeros(2**m, dtype=float)
        amp_anc[:K] = np.sqrt(np.asarray(weights, dtype=float))

        # We build V on its own qubits to easily invert it
        v_circuit = QuantumCircuit(m, name="V")
        prepare_real_state(v_circuit, v_circuit.qubits, amp_anc)
        v_gate = v_circuit.to_gate()
        v_inv_gate = v_circuit.inverse().to_gate(label="V_inv")

        # Assemble the full LCU operator: A = (V_inv) * (C-U) * (V)
        # 1. Apply V to the ancilla register
        circ.append(v_gate, anc)

        # 1b. Prepare the system in the uniform superposition |+..._>
        # This ensures the unitaries (oracles) are applied to the |+> state, not the |0> state.
        circ.h(sys)

        # 2. Apply the controlled-Unitaries (C-U)
        for k, unitary in enumerate(unitary_circuits):
            if unitary.num_qubits != n:
                raise ValueError("All unitary circuits must act on the same number of qubits.")

            controlled_unitary_gate = unitary.to_gate(label=f"U_{k}").control(num_ctrl_qubits=m, ctrl_state=k)
            circ.append(controlled_unitary_gate, anc[:] + sys[:])

        # 3. Apply V_inv to the ancilla register
        circ.append(v_inv_gate, anc)

        return circ, anc, sys

    def choose_oaa_rounds(lcu_circ, anc_reg, sys_reg, p_target=0.99, max_rounds=100):
        """Given an LCU circuit `lcu_circ` and its anc/sys registers, estimate initial success amplitude and 
        choose number of OAA rounds to reach target ancilla success probability p_target.
        """

        # Simulate once to get the initial proto amplitude
        sv = Statevector.from_instruction(lcu_circ)
        proto = extract_system_state_when_anc_zero(sv, anc_reg, sys_reg)

        # Amplitude (not squared)
        alpha = np.linalg.norm(proto)
        p0_initial = alpha**2

        if p0_initial >= p_target:
            return 0, p0_initial, alpha

        # If alpha is zero, no amount of rounds will help (degenerate)
        if alpha <= 0.0:
            return 0, p0_initial, alpha

        theta = np.arcsin(min(1.0, alpha))

        # Analytic r
        r_est = int(np.floor(np.pi/(4*theta) - 0.5))
        r_est = max(0, r_est)

        # Clamp
        r = min(r_est, max_rounds)

        # Optionally test nearby r to pick best under rounding
        best_r, best_p = 0, p0_initial

        for rr in range(max(0, r-2), min(max_rounds, r+3)+1):
            # Skip round 0, we already have its probability
            if rr == 0:
                continue

            # Build OAA with rr rounds using your build_oaa_circuit helper
            full_circ = build_oaa_circuit(lcu_circ, anc_reg, sys_reg, rounds=rr)
            sv_rr = Statevector.from_instruction(full_circ)
            proto_rr = extract_system_state_when_anc_zero(sv_rr, anc_reg, sys_reg)
            p0_rr = np.linalg.norm(proto_rr)**2

            if p0_rr > best_p:
                best_p = p0_rr
                best_r = rr

            if best_p >= p_target:
                break

        #print(f"[OAA] Best rounds: {best_r}; Best prob: {best_p:.4f}; Initial alpha: {alpha:.4f}")
        return best_r, best_p, alpha

    def build_oaa_circuit(lcu_operator_circ: QuantumCircuit, anc_qubits: QuantumRegister, sys_qubits: QuantumRegister, rounds: int=1) -> QuantumCircuit:
        """Builds the full Oblivious Amplitude Amplification (OAA) circuit.

        OAA is a variant of Grover's algorithm used to amplify the probability of a
        desired "good" state. In the LCU context, the "good" state is the one where
        the ancilla register is in the |0...0> state. This circuit applies the OAA
        operator `Q = -A S_0 A^-1 S_psi` for a specified number of rounds to amplify
        the probability of measuring the ancilla as all zeros.

        Parameters
        ----------
        lcu_operator_circ : qiskit.QuantumCircuit
            The LCU circuit `A` that prepares the initial state.
        anc_qubits : qiskit.QuantumRegister
            The register of ancilla qubits used in the LCU operator.
        sys_qubits : qiskit.QuantumRegister
            The register of system qubits.
        rounds : int, default 1
            The number of amplification rounds to apply.

        Returns
        -------
        qiskit.QuantumCircuit
            The complete quantum circuit that first applies `A` and then `Q` for the
            specified number of `rounds`.
        """

        def reflection_about_zero(qc, qubits):
            """Implements the S0 = I - 2|0><0| operator on the given list/register of qubits.
            """

            if not qubits:
                return

            qc.x(qubits)

            if len(qubits) == 1:
                qc.z(qubits[0])

            else:
                qc.h(qubits[-1])
                qc.mcx(list(qubits[:-1]), qubits[-1])
                qc.h(qubits[-1])

            qc.x(qubits)

        if rounds == 0:
            # No OAA is applied here
            return lcu_operator_circ

        circ = QuantumCircuit(anc_qubits, sys_qubits, name="OAA_Prototype")

        A_gate = lcu_operator_circ.to_gate(label="A")
        A_inv_gate = lcu_operator_circ.inverse().to_gate(label="A_inv")

        # Start with A
        circ.append(A_gate, circ.qubits)

        # Apply OAA rounds
        for _ in range(rounds):
            # Reflection on ancilla being |0>
            reflection_about_zero(circ, anc_qubits)

            # A_inverse
            circ.append(A_inv_gate, circ.qubits)

            # Reflection about the initial state |0...0> on all qubits of circ
            reflection_about_zero(circ, circ.qubits)

            # A
            circ.append(A_gate, circ.qubits)

        return circ

    if classical_computation:
        # Recover the original bipolar vectors from each feature circuit
        vectors = [get_classical_vector_from_oracle_circuit(circ) for circ in unitary_circuits]

        # Element-wise sum (keep magnitude and sign)
        vector_bundled = np.sum(vectors, axis=0)

        # We want to encode both the sign and relative magnitude of each component into a quantum oracle.
        # Directly using DiagonalGate requires unit-modulus complex numbers, so we need to convert our vector into phases on the complex unit circle.
        # Compute the normalization factor: root-mean-square (RMS) of the vector.
        # This ensures that the typical amplitude of each component is ~1 without letting very large or very small components dominate excessively.
        rms = np.linalg.norm(vector_bundled) / np.sqrt(len(vector_bundled))

        # Map each component to a complex phase using e^(i * pi * x / RMS)
        # - The sign of the original component is preserved in the phase (positive -> 0, negative -> pi);
        # - The relative magnitude of each component is approximately preserved in the phase;
        # - The resulting complex number all have unit modulus (required for DiagonalGate).
        phases = np.exp(1j * np.pi * vector_bundled / rms)

        # Build the diagonal gate with these phases
        oracle_gate = DiagonalGate(phases.tolist())
        oracle_gate.label = "O_bundle"

        # Build the circuit
        n_sys = int(log2(len(vector_bundled)))
        sys_reg = QuantumRegister(n_sys, "sys")
        bundling_circuit = QuantumCircuit(sys_reg, name="Hybrid_Prototype")
        bundling_circuit.append(oracle_gate, sys_reg)

        return bundling_circuit, None, sys_reg

    # 1. Build the LCU operator from the provided unitaries
    lcu_op, anc, sys = build_lcu_from_unitaries(
        unitary_circuits,
        weights,
        probabilistic=probabilistic,
        rounds=probabilistic_rounds
    )

    if optimize_rounds and oaa_rounds > 1:
        # 2. Estimate a proper number of OAA rounds adaptively (optional)
        oaa_rounds, _, _ = choose_oaa_rounds(lcu_op, anc, sys, p_target=0.98, max_rounds=oaa_rounds)

    # 3. Build the full OAA circuit to amplify the bundled state
    bundling_circuit = build_oaa_circuit(lcu_op, anc, sys, rounds=oaa_rounds)

    return bundling_circuit, anc, sys

def permute(num_qubits: int, shift: int) -> Gate:
    """Creates a synthesizable circuit gate that implements a cyclic permutation.

    This function implements a cyclic shift on the computational basis states using
    the Quantum Fourier Transform (QFT). The algorithm leverages the property that a
    cyclic shift in the time/computational domain is equivalent to a linear phase

    shift in the frequency/Fourier domain. This method is highly efficient and
    decomposes into standard gates, making it suitable for any backend.

    Parameters
    ----------
    num_qubits : int
        The number of qubits in the register to be permuted. The dimension is 2**num_qubits.
    shift : int
        The number of positions to cyclically shift the basis states.

    Returns
    -------
    qiskit.circuit.Gate
        A Qiskit gate that implements the specified cyclic permutation.
    """

    qc = QuantumCircuit(num_qubits, name=f"Perm(>>{shift})")
    D = 2**num_qubits

    # A shift of 0 is just an identity operation.
    if shift == 0:
        return qc.to_gate()

    # 1. Go to the Fourier basis
    qc.append(QFT(num_qubits, inverse=True, do_swaps=True).to_gate(), range(num_qubits))

    # 2. Apply the phase shifts
    for j in range(num_qubits):
        angle = (-2 * pi * shift / D) * (2**j)

        if abs(angle) > 1e-12:
            qc.p(angle, j)

    # 3. Return to the computational basis
    qc.append(QFT(num_qubits, do_swaps=True).to_gate(), range(num_qubits))

    return qc.to_gate()

def prepare_real_state(qc: QuantumCircuit, qubits: List[Qubit], amplitudes: np.ndarray) -> None:
    """Efficiently prepares a quantum state with real-valued amplitudes.

    This function uses a recursive decomposition method to prepare an arbitrary quantum
    state with only real amplitudes. It is more efficient than Qiskit's generic
    `initialize` for this specific case, as it uses a sequence of controlled-Y
    rotations. The input amplitudes are automatically normalized.

    Parameters
    ----------
    qc : qiskit.QuantumCircuit
        The quantum circuit where the state preparation will be applied.
    qubits : list[qiskit.circuit.Qubit]
        The list of qubits that will hold the prepared state.
    amplitudes : numpy.ndarray
        A NumPy array of real-valued amplitudes for the desired quantum state.
        The length must be 2**len(qubits).

    Raises
    ------
    ValueError
        If the length of the `amplitudes` array does not match the number of qubits.
    """

    a = np.array(amplitudes, dtype=float)
    norm = np.linalg.norm(a)

    if norm < 1e-12:
        return

    a = a / norm
    n = len(qubits)

    if len(a) != 2**n:
        raise ValueError("Amplitudes length must match qubit register size.")

    def _prep(qc, qlist, amps):
        if not qlist:
            return

        if len(qlist) == 1:
            # If the second amplitude is non-negligible, apply the rotation.
            if abs(amps[1]) > 1e-12:
                theta = 2 * atan2(amps[1], amps[0])
                qc.ry(theta, qlist[0])

            return

        half = len(amps) // 2
        norm0 = np.linalg.norm(amps[:half])
        norm1 = np.linalg.norm(amps[half:])

        # Rotate last qubit to prepare the branch weights
        if (norm0 + norm1) > 1e-12:
            theta = 0.0
            if norm0 > 1e-12 or norm1 > 1e-12:
                theta = 2 * atan2(norm1, norm0)

            qc.ry(theta, qlist[-1])

        # Prepare lower-level amplitudes conditioned on the last qubit
        if norm0 > 1e-12:
            _prep(qc, qlist[:-1], amps[:half] / norm0)

        if norm1 > 1e-12:
            qc.x(qlist[-1])
            _prep(qc, qlist[:-1], amps[half:] / norm1)
            qc.x(qlist[-1])

    _prep(qc, list(qubits), a)

def extract_system_state_when_anc_zero(statevector: Statevector, anc_reg: QuantumRegister, sys_reg: QuantumRegister) -> np.ndarray:
    """Extracts system state amplitudes corresponding to the ancilla being in the |0...0> state.

    This is a classical post-processing function used in simulation. Given the final
    statevector of an LCU/OAA circuit, it filters for the component where the
    ancilla was successfully measured as all zeros and returns the corresponding
    unnormalized state of the system qubits.

    Parameters
    ----------
    statevector : qiskit.quantum_info.Statevector
        The final statevector of the combined system (ancilla + system).
    anc_reg : qiskit.QuantumRegister
        The ancilla register.
    sys_reg : qiskit.QuantumRegister
        The system register.

    Returns
    -------
    numpy.ndarray
        A complex-valued NumPy array representing the unnormalized state of the
        system qubits, conditioned on the ancilla being |0...0>.
    """

    if anc_reg is None:
        # If `anc_reg` is None, it means no ancillas were used (it comes from the bundling using `classical_computation=True`).
        # The statevector is already the pure system state.
        return np.asarray(statevector.data)

    num_anc = len(anc_reg)
    num_sys = len(sys_reg)

    if num_sys == 0:
        return np.array([1.0], dtype=complex)

    sv = np.asarray(statevector.data)

    # For Qiskit's little-endian ordering where anc are the least-significant bits,
    # the index in the full vector that corresponds to system basis index `s` and anc==0 is:
    # full_index = (s << num_anc)
    proto = np.zeros(2**num_sys, dtype=complex)

    for s in range(2**num_sys):
        idx = s << num_anc
        proto[s] = sv[idx]

    return proto

def apply_negative_phase(circuit: QuantumCircuit) -> QuantumCircuit:
    """Applies a global phase of pi to a circuit.
    If applied before LCU+OAA, it has the same effect of performing the element-wise subtraction.

    Parameters
    ----------
    circuit : QuantumCircuit
        The input circuit.

    Returns
    -------
    QuantumCircuit
        The output flipped circuit.
    """

    phased_circ = QuantumCircuit(*circuit.qregs, name=f"{circuit.name}_phased")
    phased_circ.global_phase = np.pi
    phased_circ.append(circuit.to_gate(), circuit.qubits)

    return phased_circ

def run_hadamard_test(
    state_left_circ: QuantumCircuit,
    state_right_circ: QuantumCircuit,
    backend: Backend,
    shots: int=1024,
    seed: int=42,
    sampler: Optional[Sampler]=None
) -> Tuple[float, dict]: 
    """Performs a Hadamard test to measure the real part of the inner product
    between two quantum states: Re(<L|R>).

    The states |L> and |R> are prepared by state_left_circ and state_right, respectively.
    This circuit measures P(0) - P(1), which equals Re(<L|R>).

    Parameters
    ----------
    state_left_circ : qiskit.QuantumCircuit
        The circuit that prepares the first quantum state |L> (the "target").
        In the reasoning test, this has n_sys qubits.
    state_right_circ : qiskit.QuantumCircuit
        The circuit that prepares the second quantum state |R> (the "query").
    backend : qiskit.providers.backend.Backend
        The Qiskit backend (simulator or real hardware) on which to run the test.
    shots : int, default 1024
        The number of times to run the circuit to estimate probabilities.
    seed : int, default 42
        A seed for reproducibility of simulation and transpilation.
    sampler : qiskit_ibm_runtime.Sampler, optional
        An optional, pre-configured Sampler object for efficient hardware execution
        within a Session.

    Returns
    -------
    tuple[float, dict]
        A tuple containing:
        - The calculated similarity (Re(<L|R>) = P(0) - P(1)).
        - A dictionary of the measurement counts ('0' and '1').

    Raises
    ------
    ValueError
        - If qubit dimensions are incompatible (n_total < n_sys).
        - If a Sampler is not provided for hardware execution.
    """

    def get_measured_physical_qubits(transpiled_circuit: QuantumCircuit, measured_register: ClassicalRegister) -> list[int]:
        """Returns the list of physical qubits that correspond to the measured classical bits.
        """

        physical_qubits = list()

        for creg_index in range(len(measured_register)):
            # Find the classical bit object
            cbit = measured_register[creg_index]

            # Get the qubit that is measured into this classical bit
            qbit = transpiled_circuit.find_bit(cbit)

            # The index of the physical qubit in the transpiled circuit
            physical_qubits.append(qbit.index)

        return physical_qubits

    def mitigate_counts(counts, backend, shots, measured_qubits):
        """Apply readout error mitigation using mthree to a single-qubit measurement.
        """

        # Initialize mitigator from backend
        mit = M3Mitigation(backend)
        mit.cals_from_system(qubits=measured_qubits)

        # Apply correction to get mitigated probabilities
        probs = mit.apply_correction(counts, qubits=measured_qubits)

        # Return mitigated probabilities as counts
        return {"0": int(round(probs.get("0", 0) * shots)), "1": int(round(probs.get("1", 0) * shots))}

    is_simulated = isinstance(backend, AerSimulator)
    n_total = state_right_circ.num_qubits

    # Total qubits (sys + anc) from larger circuit
    n_sys = state_left_circ.num_qubits

    # System-only qubits from smaller circuit
    if n_total < n_sys:
        raise ValueError(f"Right circuit qubits ({n_total}) < Left circuit qubits ({n_sys}).")

    num_anc_pad = n_total - n_sys

    # We want to compute Re(<psi_L_padded | psi_R>)
    # |psi_L_padded> = (state_left_circ @ I_anc) |0>
    # |psi_R> = state_right_circ |0>
    v_l_padded_gate = state_left_circ.to_gate(label="Prep_L_Pad")
    v_r_gate = state_right_circ.to_gate(label="Prep_R")

    anc = QuantumRegister(1, "anc_had")
    sys = QuantumRegister(n_total, "sys")
    creg = ClassicalRegister(1, "c_had")
    qc = QuantumCircuit(anc, sys, creg)

    # 1. Start ancilla in |+>
    qc.h(anc)

    # 2. Apply C-V_L_padded
    # State is 1/sqrt(2) * (|0>|0> + |1>|psi_L_padded>)
    system_qubits_on_sys = sys[num_anc_pad:]
    qc.append(v_l_padded_gate.control(1), [anc[0]] + system_qubits_on_sys)

    # 3. Apply X to ancilla
    # State is 1/sqrt(2) * (|1>|0> + |0>|psi_L_padded>)
    qc.x(anc)

    # 4. Apply C-V_R
    # State is 1/sqrt(2) * (|1>|psi_R> + |0>|psi_L_padded>)
    qc.append(v_r_gate.control(1), anc[:] + sys[:])

    # 5. End with H on ancilla (measures in X basis)
    qc.h(anc)
    qc.measure(anc, creg)

    # Report circuit metrics
    #print(f"Hadamard Test metrics: {get_circuit_metrics(qc, n_sys, backend, optimization_level=3)}")

    if is_simulated:
        tqc = transpile(qc, backend)
        counts = backend.run(tqc, shots=shots, seed_simulator=seed).result().get_counts()

    else:
        if not sampler:
            raise ValueError("A Sampler object must be provided for hardware execution.")

        tqc = transpile(qc, backend, optimization_level=3)

        job = sampler.run([tqc], shots=shots)
        result = job.result()

        bit_array_data = result[0].data.c_had

        counts_1 = np.count_nonzero(bit_array_data.array == 1)
        counts_0 = shots - counts_1
        counts = {"0": counts_0, "1": counts_1}

        # Automatically detect measured qubits
        measured_qubits = get_measured_physical_qubits(tqc, creg)

        # Apply readout error mitigation
        counts = mitigate_counts(counts, backend, shots, measured_qubits)

    # Calculation
    # Re(<psi_L|psi_R>) = P(0) - P(1)
    p0 = counts.get("0", 0) / shots
    p1 = counts.get("1", 0) / shots
    similarity = p0 - p1

    return similarity, counts

def get_classical_vector_from_oracle_circuit(circuit: QuantumCircuit) -> np.ndarray:
    """Extracts the classical bipolar vector from a quantum oracle circuit.

    This function traverses the given circuit, applying the effect of DiagonalGates (interpreted as ±1 phases) and
    any permutation gates encoded as 'Perm(>>k)'. The resulting vector is rolled according to all permutations.

    Warning: it does not work on the QuantumCircuit resulting from the fully quantum LCU.

    Parameters
    ----------
    circuit : QuantumCircuit
        A Qiskit quantum circuit containing one DiagonalGate (the phase oracle) and optional permutation gates 'Perm(>>k)'.

    Returns
    -------
    np.ndarray
        The recovered classical bipolar vector of dtype int.
    """

    diag_gate = None
    total_shift = 0

    # Traverse the circuit instructions
    for inst, _, _ in circuit.data:
        if isinstance(inst, DiagonalGate) and diag_gate is None:
            diag_gate = inst

        elif inst.name.startswith("Perm(>>"):
            shift = int(inst.name.split(">>")[1].split(")")[0])
            total_shift += shift

    if diag_gate is None:
        raise ValueError("Circuit contains no DiagonalGate")

    # Recover the bipolar vector from the first DiagonalGate only
    diag = np.array(diag_gate.params, dtype=complex)
    vec = np.sign(np.real(diag))

    if total_shift != 0:
        # Apply the total shift (permutation)
        vec = np.roll(vec, total_shift)

    return vec.astype(int)

def statevector_to_bipolar(statevector_data: np.ndarray) -> np.ndarray:
    """Extracts a classical bipolar vector from the phases of a quantum statevector.

    This function provides a method to decode a quantum state back into a classical HDC vector. 
    It assumes the information is encoded in the sign of the real part of the amplitudes, 
    mapping positive signs to +1 and negative signs to -1.

    Parameters
    ----------
    statevector_data : numpy.ndarray
        A complex-valued NumPy array representing the amplitudes of a quantum state.

    Returns
    -------
    numpy.ndarray
        The corresponding classical bipolar vector of integers (+1 or -1).
    """

    vec = np.sign(np.real(statevector_data))
    vec[vec == 0] = 1

    return vec.astype(int)

def get_circuit_metrics(circuit: QuantumCircuit, num_system_qubits: int, backend: Backend, optimization_level: int=3) -> Dict[str, int]:
    """Analyzes a quantum circuit for key computational expense metrics.

    This function transpiles the circuit to a specified basis gate set
    to accurately report its depth and CNOT count.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        The circuit to analyze.
    num_system_qubits : int
        The number of qubits in the circuit dedicated to the "system".
        The remaining qubits are assumed to be ancillas.
    backend : qiskit.providers.backend.Backend
        The backend for which to transpile the circuit.
    optimization_level : int, default 1
        The optimization level for the transpiler (0-3).

    Returns
    -------
    dict[str, int]
        A dictionary containing the following metrics:
        - "num_qubits_total": Total number of qubits in the circuit;
        - "num_qubits_system": The provided number of system_qubits;
        - "num_qubits_ancilla": Calculated number of ancilla qubits;
        - "depth": The depth of the transpiled circuit;
        - "cnot_count": The number of CNOT (cx) gates in the transpiled circuit.

    Raises
    ------
    ValueError
        If num_system_qubits is larger than the total qubits in the circuit.
    """

    num_qubits_total = circuit.num_qubits

    if num_system_qubits > num_qubits_total:
        raise ValueError(f"num_system_qubits ({num_system_qubits}) cannot be larger than total circuit qubits ({num_qubits_total}).")

    num_qubits_ancilla = num_qubits_total - num_system_qubits

    # Transpile the circuit to break down high-level gates
    t_circ = transpile(circuit, backend, optimization_level=optimization_level)

    # Get metrics from the transpiled circuit
    depth = t_circ.depth()
    ops_count = t_circ.count_ops()
    cnot_count = ops_count.get('cx', 0)

    return {
        "num_qubits_total": num_qubits_total,
        "num_qubits_system": num_system_qubits,
        "num_qubits_ancilla": num_qubits_ancilla,
        "depth": depth,
        "cnot_count": cnot_count,
        "ops_count": ops_count
    }
