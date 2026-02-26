"""Quantum implementation of the MAP arithmetic operators."""

import re
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


def statevector_to_bipolar(circuit: QuantumCircuit) -> np.ndarray:
    """Extracts a classical bipolar vector from the phases of a quantum statevector.

    This function provides a method to decode a quantum state back into a classical vector. 
    It assumes the information is encoded in the sign of the real part of the amplitudes, 
    mapping positive signs to +1 and negative signs to -1.

    Automatically detects if the data is in standard (0/pi) or symmetric (+/- delta) encoding
    and rotates if necessary.

    Parameters
    ----------
    circuit : QuantumCircuit
        A quantum circuit to simulate and retrieve the classical bipolar vector from.

    Returns
    -------
    numpy.ndarray
        The corresponding classical bipolar vector of integers (+1 or -1).
    """

    statevector = Statevector.from_instruction(circuit.decompose())
    statevector_data = np.asarray(statevector.data)

    # Heuristic: determine encoding based on the presence of negative real components.
    # Standard encoding (0/pi): has amplitudes ~ +1 and ~ -1. Min real < -0.5.
    # Symmetric encoding (+/- delta): has amplitudes e^(+id) and e^(-id).
    # For small delta, real part is cos(d) ~ 1 (always positive).
    min_real = np.min(np.real(statevector_data))

    # Adapt
    data = statevector_data

    # If all real parts are non-negative, the information must be in the phase.
    # Rotate by -90 degrees to project phase (imag) onto real axis for decoding.
    if min_real > -1e-5:
        data = statevector_data * -1j

    # Decode
    reals = np.real(data)
    tolerance = 1e-9

    vec = np.ones(len(reals), dtype=int)

    # Positive real +1, negative real -1
    vec[reals < -tolerance] = -1

    return vec.astype(int)

def encode(vec_bipolar: np.ndarray, label: str="O_v") -> QuantumCircuit:
    """Creates a circuit containing a diagonal phase oracle.
    This function is a core component for encoding classical bipolar vectors into the phase of a quantum state.

    Parameters
    ----------
    vec_bipolar : numpy.ndarray
        A classical vector containing only -1 and +1 values.
    label : str, default "O_v"
        An optional label for the created Qiskit gate.

    Returns
    -------
    qiskit.QuantumCircuit
        A quantum circuit containing the diagonal gate.

    Raises
    ------
    ValueError
        If the input `vec_bipolar` contains values other than -1 or +1.
    """

    vec = np.asarray(vec_bipolar)

    if not np.all(np.isin(vec, [-1, 1])):
        raise ValueError("Bipolar vector must contain only -1 or +1.")

    num_qubits = int(ceil(log2(len(vec))))

    # Pad vector if necessary to match 2^N
    if len(vec) < 2**num_qubits:
        padding = np.ones(2**num_qubits - len(vec))
        vec = np.concatenate([vec, padding])

    # Convert to complex diagonal entries
    gate = DiagonalGate(vec.tolist())
    gate.label = label

    qc = QuantumCircuit(num_qubits, name=label)
    qc.append(gate, range(num_qubits))

    return qc

def bind(circuits: List[QuantumCircuit]) -> QuantumCircuit:
    """Applies a sequence of quantum circuits to perform binding.
    This function only accepts a list of QuantumCircuit objects as input.

    It assumes all inputs logically operate on the same number of qubits.

    Warning: Composability limit!
    Trying to bind two vectors that are already symmetric bundles would fail.

    Parameters
    ----------
    circuits : list[QuantumCircuit]
        List of feature circuits to bind.

    Returns
    -------
    QuantumCircuit
        A state preparation circuit for Bind.
    """

    if not circuits:
        raise ValueError("Input list for bind cannot be empty.")

    # Infer the number of qubits from the first circuit in the list.
    num_qubits = circuits[0].num_qubits
    qc = QuantumCircuit(num_qubits, name="Bind_Op")

    # Sequentially compose each circuit.
    for circuit in circuits:
        # This check is for robustness, though the type hint should prevent incorrect types.
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("All items in the bind list must be QuantumCircuit objects.")

        if circuit.num_qubits != num_qubits:
            raise ValueError("All circuits in the bind list must have the same number of qubits.")

        qc.compose(circuit, inplace=True)

    return qc

def bundle(circuits: list[QuantumCircuit], method: str="incremental_delta", fixed_delta: float=0.1) -> QuantumCircuit:
    """Bundles circuits symbolically using Phase Accumulation.

    This function constructs a new circuit that represents the 'Bundle' (Sum) of the input circuits.
    It uses a 'Sandwich' logic to inject scaled phases into the correct basis states defined by the circuit structure.

    Key Features:
    1. Handles Binding: Accumulates raw phases from all DiagonalGates first (XOR logic), then maps to symmetric.
    2. Handles Permutation: Wraps the phase injection between Structure and InverseStructure.
    3. Symmetric Encoding: Maps binary +1/-1 to symmetric phases (+pi/2, -pi/2) to preserve Majority Rule direction.

    Parameters:
    -----------
    circuits : list[QuantumCircuit]
        List of feature circuits to bundle.
    method : str
        "classical": Perform the bundling classically, no quantum operations involved;
        "average": Scales phases by 1/N (Exact arithmetic mean);
        "incremental_delta": Scales by `fixed_delta` (Streaming accumulation).
    fixed_delta : float
        The scaling factor for incremental mode.

    Returns:
    --------
    QuantumCircuit
        A state preparation circuit for the Bundle.
    """

    if not circuits:
        raise ValueError("Circuit list cannot be empty")

    if method == "classical":
        # Recover the original bipolar vectors from each feature circuit
        vectors = [statevector_to_bipolar(circ) for circ in circuits]

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
        qc = QuantumCircuit(sys_reg, name="Hybrid_Prototype")
        qc.append(oracle_gate, sys_reg)

        return qc

    def get_indices(qubits):
        return [input_circ.find_bit(q).index for q in qubits]

    N = circuits[0].num_qubits
    M = len(circuits)

    qc = QuantumCircuit(N, name="Bundle_Op")
    qc.h(range(N))

    scale = (1.0 / M) if method == "average" else fixed_delta

    for i, input_circ in enumerate(circuits):
        # Accumulate raw phases (binding logic)
        term_raw_angles = np.zeros(2**N)
        post_structure_ops = list()
        found_any_diagonal = False

        for instr in input_circ.data:   
            op, qargs, cargs = instr.operation, instr.qubits, instr.clbits

            if isinstance(op, DiagonalGate):
                found_any_diagonal = True
                diag_complex = np.array(op.params, dtype=complex)
                angles = np.angle(diag_complex)
                term_raw_angles += angles

            else:
                post_structure_ops.append((op, qargs))

        if not found_any_diagonal:
            continue

        # Normalize to symmetric domain (composability)
        # We need to map whatever the input is to a "vote" of +/- 1.

        # Resolve binding (XOR)
        # cos(sum) is +1 for 0/2pi, -1 for pi
        # If input was already symmetric small angles, sum is small, cos is +1
        # This preserves the sign of small inputs too

        # We need to detect the sign of small angles
        # We use a hybrid check on the Net Angle `theta`:
        # If cos(theta) < -0.5  -> it's pi-like -> vote -1
        # Else if sin(theta) < -1e-5 -> it's neg-delta -> vote -1
        # Else -> vote +1
        net_complex = np.exp(1j * term_raw_angles)
        votes = np.ones(2**N)

        # Detect pi-like (standard negative)
        votes[np.real(net_complex) < -0.1] = -1

        # Detect negative-delta (symmetric negative)
        # Only check this if not pi-like (to avoid boundary issues)
        mask_small_angle = np.real(net_complex) > 0.1
        votes[mask_small_angle & (np.imag(net_complex) < -1e-9)] = -1

        # Scale & inject
        # Now we have a clean +/- 1 vote vector
        # Map to Symmetric Target (+pi/2, -pi/2) for the new bundle
        symmetric_target = votes * (np.pi / 2)

        scaled_phases = symmetric_target * scale
        new_diag_entries = np.exp(1j * scaled_phases)
        scaled_diagonal_op = DiagonalGate(new_diag_entries.tolist())

        # Sandwich
        for op, qargs in post_structure_ops:
            qc.append(op, get_indices(qargs))

        qc.append(scaled_diagonal_op, range(N))

        for op, qargs in reversed(post_structure_ops):
            try:
                inv_op = op.inverse()

            except:
                inv_op = op

            qc.append(inv_op, get_indices(qargs))

    return qc

def permute(qc: QuantumCircuit, num_qubits: int, shift: int=0) -> QuantumCircuit:
    """Creates a synthesizable circuit gate that implements a cyclic permutation.

    This function implements a cyclic shift on the computational basis states using
    the Quantum Fourier Transform (QFT). The algorithm leverages the property that a
    cyclic shift in the time/computational domain is equivalent to a linear phase
    shift in the frequency/Fourier domain.

    Sequence: QFT -> PhaseShift -> IQFT

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to apply the cyclic shift to.
    num_qubits : int
        The number of qubits in the register to be permuted. The dimension is 2**num_qubits.
    shift : int, default 0
        The number of positions to cyclically shift the basis states.

    Returns
    -------
    QuantumCircuit
        A state preparation circuit for Permute.
    """

    if qc is None:
        # Create a new circuit representing just the permutation operation
        # if no quantum circuit is provided
        qc = QuantumCircuit(num_qubits, name=f"Perm(>>{shift})")

    if shift == 0:
        return qc

    D = 2**num_qubits
    
    # 1. Forward QFT (Time -> Frequency)
    qc.append(QFT(num_qubits, do_swaps=True), range(num_qubits))

    # 2. Phase Gradients
    for j in range(num_qubits):
        # Phase(k) = exp(2*pi*i * s * k / D)
        angle = (2 * np.pi * shift / D) * (2**j)

        if abs(angle) > 1e-12:
            qc.p(angle, j)

    # 3. Inverse QFT (Frequency -> Time)
    qc.append(QFT(num_qubits, inverse=True, do_swaps=True), range(num_qubits))

    return qc

def apply_negative_phase(circuit: QuantumCircuit) -> QuantumCircuit:
    """Applies a global phase of pi to a circuit.
    If applied before bundling, it has the same effect of performing the element-wise subtraction.

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
    try:
        v_l_padded_gate = state_left_circ.to_gate(label="Prep_L_Pad")

    except:
        t_left = transpile(state_left_circ, basis_gates=["u", "cx"], optimization_level=0)
        v_l_padded_gate = t_left.to_gate(label="Prep_L")

    try:
        v_r_gate = state_right_circ.to_gate(label="Prep_R")

    except:
        t_right = transpile(state_right_circ, basis_gates=["u", "cx"], optimization_level=0)
        v_r_gate = t_right.to_gate(label="Prep_L")

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

def calibrate_shift_direction(dimensionality: int):
    """Determines if Quantum Shift +1 corresponds to numpy.roll +1 or -1.
    This aligns the classical ground truth with Qiskit's QFT definition.

    Parameters
    ----------
    dimensionality : int
        Classical vector dimensionality.

    Returns
    -------
    int
        The direction of the cyclic shift.
    """

    probe = [-1] * dimensionality
    probe[0] = 1

    # Encode the probe vector
    qc = encode(probe)
    N = qc.num_qubits

    # Apply a permutation
    qc = permute(qc, N, shift=1)

    # Transpile needed for QFT decomposition
    gate_qc = transpile(qc, basis_gates=['u', 'cx'], optimization_level=0)

    test_qc = QuantumCircuit(qc.num_qubits)
    test_qc.h(range(qc.num_qubits))
    test_qc.append(gate_qc.to_gate(), range(qc.num_qubits))

    sv = Statevector.from_instruction(test_qc)

    # Extract signal location (Real part is roughly +1 at the shifted index)
    # We ignore global phase here because we just want the index
    mags = np.abs(np.real(sv.data))
    peak_idx = np.argmax(mags)

    if peak_idx == 1:
        return 1

    else:
        return -1
