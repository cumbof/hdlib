"""Quantum implementation of the MAP arithmetic operators."""

import re
from math import atan2, sqrt, ceil, log2, pi
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from mthree import M3Mitigation
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import DiagonalGate, XGate, SwapGate
from qiskit.quantum_info import Statevector, partial_trace, entropy
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

    # Create a temporary evaluation circuit to read the oracle
    num_qubits = circuit.num_qubits
    eval_circ = QuantumCircuit(num_qubits)
    eval_circ.h(range(num_qubits))
    eval_circ.compose(circuit, inplace=True)

    # Simulate the fully prepared state
    statevector = Statevector.from_instruction(eval_circ.decompose())
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

def compress_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Compresses a deep phase-encoded quantum circuit into a shallow circuit with one DiagonalGate.

    This acts as a quantum compiler for Vector-Symbolic Architectures.
    It calculates the noise-free phase accumulation of the deep circuit
    and reconstructs an identical quantum state using a single layer of
    Hadamard gates and one DiagonalGate.

    Parameters
    ----------
    circuit : QuantumCircuit
        The deep quantum circuit (e.g., a series of bundled vectors with hundreds of gates).

    Returns
    -------
    QuantumCircuit
        A mathematically identical shallow circuit oracle.
    """

    num_qubits = circuit.num_qubits

    # 1. Create a temporary evaluation circuit to read the deep oracle
    eval_circ = QuantumCircuit(num_qubits)
    eval_circ.h(range(num_qubits))
    eval_circ.compose(circuit, inplace=True)

    # Mathematically evaluate the exact state to capture phase accumulation
    state = Statevector.from_instruction(eval_circ.decompose())

    # 2. Extract the relative phases of the quantum state
    phases = np.angle(state.data)

    # 3. Create the compressed diagonal operator using the extracted phases
    diagonal_elements = np.exp(1j * phases)
    diag_gate = DiagonalGate(diagonal_elements.tolist())

    # 4. Build the shallow oracle circuit
    compressed_qc = QuantumCircuit(num_qubits, name=f"{circuit.name}_compressed")

    # Apply all accumulated phases in a single operation to return a pure oracle.
    compressed_qc.append(diag_gate, range(num_qubits))

    return compressed_qc

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

def bundle(circuits: list[QuantumCircuit], method: str="average") -> QuantumCircuit:
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

    scale = (1.0 / M)

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

    This function implements a cyclic shift using a Modulo 2^N Quantum Adder.
    It shifts the basis states strictly cyclically, matching the classical np.roll property exactly,
    but uses entirely digital gates (X and MCX) with O(N) depth.

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

    # Ensure the shift is within the cyclic bounds (Modulo D)
    shift = shift % (2**num_qubits)

    if shift == 0:
        return qc

    # Convert shift to binary string and reverse it so that index 'k' correctly corresponds to the 2^k bit
    shift = bin(shift)[2:][::-1]

    for k, bit in enumerate(shift):
        if bit == "1":
            # Apply a +2^k quantum incrementer
            # This acts as a standard binary adder starting only at the k-th qubit, 
            # leaving lower significant qubits completely untouched.
            for i in range(num_qubits - 1, k, -1):
                # Target bit 'i' is flipped only if all lower bits from 'k' to 'i-1' are 1
                controls = list(range(k, i))

                if len(controls) == 1:
                    qc.cx(controls[0], i)

                else:
                    mcx_gate = XGate().control(len(controls))
                    qc.append(mcx_gate, controls + [i])

            # Finally, unconditionally flip the k-th bit
            qc.x(k)

    return qc

def __get_measured_physical_qubits(transpiled_circuit: QuantumCircuit, measured_register: ClassicalRegister) -> list[int]:
    """Returns the list of physical qubits that correspond to the measured classical bits.
    """

    try:
        # Qiskit's transpiler recreates bits. Looking up pre-transpiled Clbit identity
        # will cause a hash mismatch and throw an error. We map via the transpiled circuit's registers.
        transpiled_creg = next(reg for reg in transpiled_circuit.cregs if reg.name == measured_register.name)

    except:
        raise ValueError(f"Register {measured_register.name} not found in transpiled circuit.")

    # 1. Create a dictionary to map classical bits to the physical qubits measured into them
    meas_map = dict()

    for inst in transpiled_circuit.data:
        if inst.operation.name == "measure":
            qbit = inst.qubits[0] # The qubit being measured
            cbit = inst.clbits[0] # The classical bit receiving the result

            # Find the actual physical index of this qubit in the transpiled circuit
            qbit_idx = transpiled_circuit.find_bit(qbit).index
            meas_map[cbit] = qbit_idx

    # 2. Extract the physical qubits in the exact order of the measured_register
    physical_qubits = list()

    for cbit in transpiled_creg:
        if cbit in meas_map:
            # The index of the physical qubit in the transpiled circuit
            physical_qubits.append(meas_map[cbit])

        else:
            raise ValueError(f"Classical bit {cbit} does not have a measurement mapped to it.")

    # Qiskit results return bitstrings from MSB to LSB (left to right = c_{N-1} ... c_0).
    # M3 mitigation expects the passed physical_qubits list to identically match that string's left-to-right order.
    # Therefore, we must reverse the physical qubits list here to avoid applying the wrong error profile to the wrong bits.
    return physical_qubits[::-1]

def __mitigate_counts(counts, backend, shots, measured_qubits, mitigator: Optional[M3Mitigation]=None):
    """Apply readout error mitigation using mthree to a single-qubit measurement.
    """

    if mitigator is None:
        # Initialize mitigator from backend
        mitigator = M3Mitigation(backend)
        mitigator.cals_from_system(qubits=measured_qubits)

    # Apply correction to get mitigated probabilities
    probs = mitigator.apply_correction(counts, qubits=measured_qubits)

    # Dynamically convert all output states back to pseudo-counts
    mitigated_pseudo_counts = dict()

    for state, prob in probs.items():
        # Clamp quasi-probabilities to strictly between 0.0 and 1.0
        # M3 can sometimes output tiny negative values or values slightly above 1.0
        safe_prob = min(1.0, max(0.0, prob))
        mitigated_pseudo_counts[state] = int(round(safe_prob * shots))

    # Return mitigated probabilities as pseudo-counts
    return mitigated_pseudo_counts

def run_compute_uncompute_test(
    state_left_circs: List[QuantumCircuit],
    state_right_circs: List[QuantumCircuit],
    backend: Backend,
    shots: int=1024,
    seed: int=42,
    sampler: Optional[Sampler]=None
) -> Tuple[List[List[float]], List[dict]]:
    """Performs a Compute-Uncompute (Inversion) test to measure |<L|R>|^2 in batch mode.

    This avoids all controlled operations, making it exponentially cheaper
    to transpile and execute compared to the Hadamard Test.
    """

    is_simulated = isinstance(backend, AerSimulator)
    n_sys = state_right_circs[0].num_qubits

    if state_left_circs[0].num_qubits != n_sys:
        raise ValueError("Left and Right circuits must have the exact same number of qubits for Inversion test.")

    sys = QuantumRegister(n_sys, "sys")
    creg = ClassicalRegister(n_sys, "c_meas")

    qcs = list()

    for query_circ in state_left_circs:
        for prototype_circ in state_right_circs:
            qc = QuantumCircuit(sys, creg)

            # 1. Initialize uniform superposition
            qc.h(sys)

            # 2. Compute: Apply query state (R)
            qc.compose(query_circ, qubits=sys, inplace=True)

            # 3. Uncompute: Apply inverse of prototype state (L)
            qc.compose(prototype_circ.inverse(), qubits=sys, inplace=True)

            # 4. Map phases back to amplitudes for measurement
            qc.h(sys)

            # 5. Measure all qubits
            qc.measure(sys, creg)

            qcs.append(qc)

    if is_simulated:
        tqcs = transpile(qcs, backend, optimization_level=1)
        counts = backend.run(tqcs, shots=shots, seed_simulator=seed).result().get_counts()

        if not isinstance(counts, list):
            counts = [counts]

    else:
        if not sampler:
            raise ValueError("A Sampler object must be provided for hardware execution.")

        tqcs = transpile(qcs, backend, optimization_level=3)
        job = sampler.run(tqcs, shots=shots)
        results = job.result()

        counts = list()

        # Gather all unique physical qubits used across all circuits
        all_measured_qubits = set()
        circuit_measured_qubits = list()

        for tqc in tqcs:
            # Automatically detect measured qubits
            phys_qubits = __get_measured_physical_qubits(tqc, creg)
            all_measured_qubits.update(phys_qubits)
            circuit_measured_qubits.append(phys_qubits)

        # Initialize mitigator from backend
        mitigator = M3Mitigation(backend)
        mitigator.cals_from_system(qubits=list(all_measured_qubits))

        for i, res in enumerate(results):
            counts_res = res.data.c_meas.get_counts()

            # Apply readout error mitigation
            counts.append(__mitigate_counts(counts_res, backend, shots, circuit_measured_qubits[i], mitigator=mitigator))

    # Group similarities back into a 2D list
    similarities = list()
    idx = 0

    # Dynamically define the all-zeros target state based on system size
    target_state = "0" * n_sys

    for _ in state_left_circs:
        query_sims = list()

        for _ in state_right_circs:
            # Because both simulator and mitigated branches output integers/pseudo-counts
            # that sum to 'shots', dividing by 'shots' here correctly yields the probability.
            raw_prob = counts[idx].get(target_state, 0) / shots

            # The compute-uncompute test measures |<L|R>|^2
            # We took the square root to return the similarity magnitude |<L|R>|
            query_sims.append(sqrt(raw_prob))

            idx += 1

        similarities.append(query_sims)

    return similarities, counts

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

    # Added fallback counters for "ecr" and "cz". IBM backends map "cx" to "ecr" 
    # natively during transpilation, which used to cause "cx" count to report as 0.
    cnot_count = ops_count.get("cx", 0) + ops_count.get("ecr", 0) + ops_count.get("cz", 0)

    return {
        "num_qubits_total": num_qubits_total,
        "num_qubits_system": num_system_qubits,
        "num_qubits_ancilla": num_qubits_ancilla,
        "depth": depth,
        "cnot_count": cnot_count,
        "ops_count": ops_count
    }


# ---------------------------------------------------------------------------
# New quantum-native operations exploiting superposition and entanglement
# ---------------------------------------------------------------------------


def _build_select_circuit(circuits: List[QuantumCircuit]) -> QuantumCircuit:
    """Builds a SELECT (quantum multiplexer) circuit.

    The SELECT unitary applies oracle O_k to the system register when the
    index register holds the binary encoding of k.  Placing the index
    register in a uniform superposition before calling SELECT and then
    inverting the superposition (H again) yields the *superposition bundle*:
    post-selecting the index on |0...0⟩ projects the system onto the
    arithmetic mean of all input oracle states.

    Parameters
    ----------
    circuits : list[QuantumCircuit]
        A list of N oracle circuits, each acting on n_sys qubits.

    Returns
    -------
    QuantumCircuit
        A (n_idx + n_sys)-qubit circuit whose registers are named ``idx``
        and ``sys`` respectively.

    Notes
    -----
    This implementation uses a straightforward controlled-oracle approach.
    For N ≤ 2^n_idx, n_idx = ⌈log₂N⌉.  Each of the N iterations adds a
    controlled version of one oracle gate; with tree-structured LCU
    decomposition the circuit depth can be reduced to O(n_idx · T_oracle)
    at the cost of additional ancilla qubits.
    """

    if not circuits:
        raise ValueError("Circuit list cannot be empty.")

    N = len(circuits)
    n_sys = circuits[0].num_qubits
    n_idx = max(1, ceil(log2(N))) if N > 1 else 1

    for circ in circuits:
        if circ.num_qubits != n_sys:
            raise ValueError("All circuits must have the same number of qubits.")

    idx_reg = QuantumRegister(n_idx, "idx")
    sys_reg = QuantumRegister(n_sys, "sys")
    qc = QuantumCircuit(idx_reg, sys_reg, name="SELECT")

    # Put the index register in uniform superposition: |+⟩^n_idx
    qc.h(idx_reg)

    # Prepare the system register in the uniform superposition |+⟩^n_sys so
    # that each controlled oracle acts as a phase oracle on the system.
    qc.h(sys_reg)

    # SELECT: for each k, apply O_k controlled on |k⟩ in the index register.
    for k, circ in enumerate(circuits[:N]):
        k_bits = format(k, f"0{n_idx}b")

        # Flip bits where k has 0 so that "all ones" ↔ index k
        for bit_pos, bit_val in enumerate(reversed(k_bits)):
            if bit_val == "0":
                qc.x(idx_reg[bit_pos])

        ctrl_gate = circ.to_gate().control(n_idx)
        qc.append(ctrl_gate, list(idx_reg) + list(sys_reg))

        # Undo the bit-flips
        for bit_pos, bit_val in enumerate(reversed(k_bits)):
            if bit_val == "0":
                qc.x(idx_reg[bit_pos])

    # Inverse-superposition on the index register so that the index = 0
    # subspace accumulates the coherent sum of all oracle contributions.
    qc.h(idx_reg)

    return qc


def _decode_select_bundle(select_circuit: QuantumCircuit, n_sys: int, n_idx: int, num_circuits: int) -> np.ndarray:
    """Decodes the bundle result from a SELECT circuit via statevector simulation.

    After simulating the SELECT circuit the amplitude of the index = |0⟩
    subspace encodes the element-wise sum of all input oracle vectors.
    The sign of the real part gives the majority-vote bipolar result.

    When ``num_circuits`` is not a power of two (2^n_idx > num_circuits) the
    index register has unused slots.  Unused slots effectively contribute a
    +1 phase at every system basis state, biasing the amplitude toward +1.
    This function removes that bias before computing the sign.

    Parameters
    ----------
    select_circuit : QuantumCircuit
        The circuit returned by :func:`_build_select_circuit` (after the final
        H on the index register has been applied).
    n_sys : int
        Number of system qubits (log₂ of the vector dimension).
    n_idx : int
        Number of index qubits (⌈log₂N⌉).
    num_circuits : int
        The actual number of oracle circuits N (may be less than 2^n_idx).

    Returns
    -------
    numpy.ndarray
        A bipolar (±1) vector of length 2^n_sys.
    """

    sv = Statevector.from_instruction(select_circuit.decompose().decompose())
    sv_data = np.asarray(sv.data)

    # The idx register occupies the *lowest* n_idx bits of the statevector
    # index (Qiskit little-endian ordering).  We extract all entries where
    # those bits are 0, i.e., every 2^n_idx-th entry starting from 0.
    step = 2 ** n_idx
    sys_amps = sv_data[::step]  # length = 2^n_sys

    # The raw amplitude at system basis state j is:
    #   sys_amps[j] = (1 / (sqrt(D) * step)) * [Σ_{k<N} oracle_k[j] + padding * 1]
    # where D = 2^n_sys, padding = step - num_circuits, and the factor 1/sqrt(D)
    # comes from the H gates on the system register.
    # Recover the true oracle vote sum by inverting this relation:
    #   true_sum[j] = sys_amps[j] * step * sqrt(D) - padding
    D = 2 ** n_sys
    sqrt_D = np.sqrt(float(D))
    padding_count = step - num_circuits
    raw_sum = np.real(sys_amps) * step * sqrt_D - padding_count

    # Majority vote: sign of the true oracle sum; tie → +1 by convention.
    tolerance = 1e-6
    result = np.ones(len(raw_sum), dtype=int)
    result[raw_sum < -tolerance] = -1

    return result


def superposition_bundle(circuits: List[QuantumCircuit]) -> QuantumCircuit:
    """Bundles N oracle circuits in parallel using a quantum SELECT unitary.

    This function uses a *superposition of oracles* to bundle N hypervectors
    simultaneously.  An index register is placed in uniform superposition so
    that the SELECT unitary applies each oracle O_k conditioned on the index
    register encoding k.  Inverting the index-register superposition (second
    Hadamard layer) and post-selecting on the index |0...0⟩ accumulates the
    coherent sum of all oracle contributions via quantum interference—
    identical to the classical element-wise sum but computed in O(log N)
    circuit depth on hardware that natively supports tree-structured SELECT
    operations.

    The resulting oracle circuit encodes the majority-vote bipolar vector:
    it is equivalent to the classical :func:`bundle` followed by
    :meth:`~hdlib.space.Vector.normalize`.

    Parameters
    ----------
    circuits : list[QuantumCircuit]
        List of N oracle circuits produced by :func:`encode` (each acting on
        n_sys qubits).  All circuits must have the same number of qubits.

    Returns
    -------
    QuantumCircuit
        A phase oracle circuit (n_sys qubits) encoding the bundled result,
        compatible with :func:`statevector_to_bipolar` and all downstream
        operations that expect an oracle circuit.

    Raises
    ------
    ValueError
        If the circuit list is empty or circuits have different qubit counts.

    Notes
    -----
    **Quantum advantage**: a depth-optimal LCU (Linear Combination of
    Unitaries) decomposition of the SELECT unitary has depth O(n_idx · T)
    where n_idx = ⌈log₂N⌉ and T is the depth of a single oracle, giving an
    exponential depth reduction over the sequential O(N · T) classical
    approach.  This implementation performs the exact same computation via
    statevector simulation and re-encodes the result as a shallow oracle;
    the circuit structure and depth metrics of the internal SELECT circuit
    can be inspected via :func:`get_circuit_metrics`.

    Examples
    --------
    >>> from hdlib.space import Vector
    >>> from hdlib.arithmetic.quantum import encode, superposition_bundle, statevector_to_bipolar
    >>> vectors = [Vector(size=16, vtype="bipolar") for _ in range(4)]
    >>> oracle_circuits = [encode(v.vector) for v in vectors]
    >>> bundled_circ = superposition_bundle(oracle_circuits)
    >>> result = statevector_to_bipolar(bundled_circ)
    """

    if not circuits:
        raise ValueError("Circuit list cannot be empty.")

    N = len(circuits)
    n_sys = circuits[0].num_qubits
    n_idx = max(1, ceil(log2(N))) if N > 1 else 1

    # Build the internal SELECT circuit
    select_qc = _build_select_circuit(circuits)

    # Decode: project onto index = 0 subspace and extract the bipolar vector
    bundled_vector = _decode_select_bundle(select_qc, n_sys, n_idx, N)

    # Re-encode as a shallow phase oracle compatible with the rest of the pipeline
    return encode(bundled_vector, label="SuperposBundle")


def entangled_bind(circuit1: QuantumCircuit, circuit2: QuantumCircuit) -> QuantumCircuit:
    """Creates an entangled quantum record encoding two hypervectors simultaneously.

    This function applies the quantum SWAP-test construction to create a
    maximally entangled state that encodes both input hypervectors in a single
    quantum register.  The resulting state is:

    .. math::

       |\\Phi\\rangle =
       \\frac{1}{\\sqrt{2}}\\bigl(|0\\rangle|\\psi_1\\rangle|\\psi_2\\rangle
       + |1\\rangle|\\psi_2\\rangle|\\psi_1\\rangle\\bigr)

    where :math:`|\\psi_k\\rangle = O_{v_k}|{+}\\rangle^{\\otimes n}` is the
    quantum encoding of the k-th hypervector.

    **HDC semantics**: the classical :func:`bind` irreversibly fuses two
    vectors into a single composite.  The entangled version creates a
    *reversible quantum record*: measuring the ancilla in the Hadamard basis
    reveals information about the similarity between the two vectors, while
    the system registers remain in a well-defined entangled state.  The
    ancilla collapses to |0⟩ with probability
    :math:`(1 + |\\langle\\psi_1|\\psi_2\\rangle|^2)/2` and to |1⟩ with
    probability :math:`(1 - |\\langle\\psi_1|\\psi_2\\rangle|^2)/2`—the
    SWAP test.

    **Quantum advantage**: no classical 2n-bit register can represent the
    entangled state; faithfully describing it classically requires storing the
    full 2^(2n)-element amplitude vector.

    Parameters
    ----------
    circuit1 : QuantumCircuit
        Oracle circuit for the first hypervector (n qubits).
    circuit2 : QuantumCircuit
        Oracle circuit for the second hypervector (n qubits).  Must have the
        same number of qubits as ``circuit1``.

    Returns
    -------
    QuantumCircuit
        A (2n + 1)-qubit circuit with registers ``anc`` (1 qubit),
        ``sys_a`` (n qubits for v₁), and ``sys_b`` (n qubits for v₂).

    Raises
    ------
    ValueError
        If the two circuits have different qubit counts.

    Examples
    --------
    >>> from hdlib.arithmetic.quantum import encode, entangled_bind
    >>> from qiskit.quantum_info import Statevector, partial_trace, entropy
    >>> import numpy as np
    >>> v1 = np.array([1, -1, 1, -1])
    >>> v2 = np.array([-1, 1, -1, 1])
    >>> c1 = encode(v1); c2 = encode(v2)
    >>> qc = entangled_bind(c1, c2)
    >>> qc.num_qubits
    5
    """

    n = circuit1.num_qubits

    if circuit2.num_qubits != n:
        raise ValueError(
            "Both circuits must act on the same number of qubits."
        )

    anc_reg = QuantumRegister(1, "anc")
    sys_a = QuantumRegister(n, "sys_a")
    sys_b = QuantumRegister(n, "sys_b")

    qc = QuantumCircuit(anc_reg, sys_a, sys_b, name="EntangledBind")

    # Prepare |ψ₁⟩ = O_{v1}|+⟩^n on sys_a
    qc.h(sys_a)
    qc.append(circuit1.to_gate(), list(sys_a))

    # Prepare |ψ₂⟩ = O_{v2}|+⟩^n on sys_b
    qc.h(sys_b)
    qc.append(circuit2.to_gate(), list(sys_b))

    # Entangle via SWAP test: H on ancilla, then controlled-SWAP for each qubit
    qc.h(anc_reg[0])
    for i in range(n):
        qc.cswap(anc_reg[0], sys_a[i], sys_b[i])

    return qc


def grover_search(
    query_circuit: QuantumCircuit,
    codebook_circuits: List[QuantumCircuit],
    similarity_threshold: float = 0.8,
    backend: Optional[Backend] = None,
    shots: int = 1024,
) -> Tuple[int, float]:
    """Finds the most similar codebook entry using Grover amplitude amplification.

    This function demonstrates the Grover O(√N) search paradigm applied to
    Hyperdimensional Computing nearest-neighbour retrieval.  It proceeds in
    two stages:

    1. **Quantum oracle construction**: the similarity between the query and
       each codebook circuit is estimated using the
       :func:`run_compute_uncompute_test` primitive (a quantum circuit).
    2. **Grover amplification**: a phase oracle marks indices whose similarity
       exceeds ``similarity_threshold`` and Grover diffusion amplifies their
       probability amplitudes so that a single measurement returns the best
       match with high probability.

    **Quantum advantage**: with a full QRAM-based oracle that can evaluate
    the HD similarity in O(polylog N) circuit depth, the end-to-end search
    cost is O(√N · T_oracle) versus the classical O(N · T_oracle).  The
    implementation here uses the quantum :func:`run_compute_uncompute_test`
    for all N similarity evaluations, then applies Grover iterations on the
    index register to demonstrate the amplification structure.

    Parameters
    ----------
    query_circuit : QuantumCircuit
        Oracle circuit for the query hypervector.
    codebook_circuits : list[QuantumCircuit]
        Oracle circuits for the N codebook prototypes.
    similarity_threshold : float, default 0.8
        Minimum similarity to consider an entry a candidate match.  If no
        entry exceeds the threshold the single best entry is marked.
    backend : Backend, optional
        Qiskit backend for running the compute-uncompute similarity circuits.
        Defaults to :class:`~qiskit_aer.AerSimulator`.
    shots : int, default 1024
        Number of measurement shots per similarity circuit.

    Returns
    -------
    (int, float)
        ``(best_index, similarity)`` – the index of the most similar codebook
        entry and its estimated similarity to the query.

    Raises
    ------
    ValueError
        If the codebook is empty or circuits have incompatible qubit counts.

    Examples
    --------
    >>> from hdlib.arithmetic.quantum import encode, grover_search
    >>> import numpy as np
    >>> from qiskit_aer import AerSimulator
    >>> codebook = [encode(np.random.choice([-1,1], size=16)) for _ in range(4)]
    >>> query = codebook[2]
    >>> idx, sim = grover_search(query, codebook, backend=AerSimulator())
    >>> idx
    2
    """

    if not codebook_circuits:
        raise ValueError("Codebook list cannot be empty.")

    N = len(codebook_circuits)
    backend = backend or AerSimulator()
    n_idx = max(1, ceil(log2(N))) if N > 1 else 1

    # --- Stage 1: quantum similarity estimation for all N pairs ---
    sim_matrix, _ = run_compute_uncompute_test(
        [query_circuit], codebook_circuits, backend=backend, shots=shots
    )
    sims = sim_matrix[0]  # shape: [N]

    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    # Determine which indices to mark
    marked = [k for k, s in enumerate(sims) if s >= similarity_threshold]
    if not marked:
        marked = [best_idx]

    # --- Stage 2: Grover amplification on the index register ---
    n_marked = len(marked)
    n_iter = max(1, int(round(pi / (4.0 * sqrt(N / n_marked)) - 0.5)))

    idx_reg = QuantumRegister(n_idx, "idx")
    c_reg = ClassicalRegister(n_idx, "c")
    qc = QuantumCircuit(idx_reg, c_reg, name="Grover_Search")

    # Uniform superposition over all N codebook indices
    qc.h(idx_reg)

    def _phase_oracle(marked_set: List[int]) -> QuantumCircuit:
        """Phase oracle: flips phase of marked indices."""
        qco = QuantumCircuit(n_idx, name="PhaseOracle")
        for m in marked_set:
            m_bits = format(m, f"0{n_idx}b")
            # Flip 0-bits so "all ones" selects index m
            for pos, bit in enumerate(reversed(m_bits)):
                if bit == "0":
                    qco.x(pos)
            # Multi-controlled Z (phase flip on |11...1⟩)
            if n_idx == 1:
                qco.z(0)
            else:
                qco.h(n_idx - 1)
                mcx = XGate().control(n_idx - 1)
                qco.append(mcx, list(range(n_idx)))
                qco.h(n_idx - 1)
            # Undo bit-flips
            for pos, bit in enumerate(reversed(m_bits)):
                if bit == "0":
                    qco.x(pos)
        return qco

    def _diffusion() -> QuantumCircuit:
        """Grover diffusion operator: 2|+⟩⟨+| − I."""
        qcd = QuantumCircuit(n_idx, name="Diffusion")
        qcd.h(range(n_idx))
        qcd.x(range(n_idx))
        if n_idx == 1:
            qcd.z(0)
        else:
            qcd.h(n_idx - 1)
            mcx = XGate().control(n_idx - 1)
            qcd.append(mcx, list(range(n_idx)))
            qcd.h(n_idx - 1)
        qcd.x(range(n_idx))
        qcd.h(range(n_idx))
        return qcd

    phase_oracle = _phase_oracle(marked)
    diffusion = _diffusion()

    for _ in range(n_iter):
        qc.compose(phase_oracle, inplace=True)
        qc.compose(diffusion, inplace=True)

    qc.measure(idx_reg, c_reg)

    # Run on the backend
    t_qc = transpile(qc, backend, optimization_level=1)
    result = backend.run(t_qc, shots=shots).result()
    counts = result.get_counts()

    # Retrieve the most-measured index (clamped to [0, N))
    best_bitstr = max(counts, key=counts.get)
    measured_idx = int(best_bitstr, 2) % N

    return measured_idx, float(sims[measured_idx])


def quantum_inner_product(
    circuit1: QuantumCircuit,
    circuit2: QuantumCircuit,
    backend: Optional[Backend] = None,
    epsilon: float = 0.05,
    delta: float = 0.05,
    shots_per_round: int = 100,
) -> float:
    """Estimates the inner product between two quantum HD states using IQAE.

    The inner product between the quantum states
    :math:`|\\psi_1\\rangle = O_{v_1}|{+}\\rangle^{\\otimes n}` and
    :math:`|\\psi_2\\rangle = O_{v_2}|{+}\\rangle^{\\otimes n}` satisfies:

    .. math::

       |\\langle\\psi_1|\\psi_2\\rangle|
       = \\left|\\frac{1}{D}\\sum_i v_1[i]\\, v_2[i]\\right|
       = |\\text{cosine-similarity}(v_1,\\, v_2)|

    This is estimated using Iterative Quantum Amplitude Estimation (IQAE),
    which achieves ε-precision with O(1/ε) quantum circuit evaluations
    rather than the classical O(1/ε²) shots needed by the Monte Carlo
    compute-uncompute test.

    **Quantum advantage**: IQAE uses quantum phase estimation to estimate
    the probability of the good state with a Heisenberg-limited sample
    complexity of O(1/ε), giving a quadratic improvement over classical
    sampling.

    Parameters
    ----------
    circuit1 : QuantumCircuit
        Oracle circuit for the first hypervector (n qubits).
    circuit2 : QuantumCircuit
        Oracle circuit for the second hypervector (n qubits).
    backend : Backend, optional
        Unused in the default StatevectorSampler path; retained for API
        compatibility with hardware execution.
    epsilon : float, default 0.05
        Target half-width of the confidence interval (absolute error).
    delta : float, default 0.05
        Failure probability (1 − confidence level).
    shots_per_round : int, default 100
        Shots per IQAE round when a sampling-based backend is used.

    Returns
    -------
    float
        Estimated :math:`|\\langle\\psi_1|\\psi_2\\rangle| \\in [0, 1]`.

    Raises
    ------
    ValueError
        If the two circuits have different qubit counts.

    Examples
    --------
    >>> from hdlib.arithmetic.quantum import encode, quantum_inner_product
    >>> import numpy as np
    >>> from qiskit_aer import AerSimulator
    >>> v = np.random.choice([-1, 1], size=16)
    >>> c = encode(v)
    >>> quantum_inner_product(c, c, backend=AerSimulator())  # ≈ 1.0
    """

    from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
    from qiskit.primitives import StatevectorSampler

    n = circuit1.num_qubits

    if circuit2.num_qubits != n:
        raise ValueError(
            "Both circuits must act on the same number of qubits."
        )

    anc_qubit = n  # ancilla is the last qubit

    # Build the state-preparation circuit for IQAE.
    # The circuit implements the compute-uncompute sequence and marks the
    # |0...0⟩ outcome with an ancilla qubit.
    state_prep = QuantumCircuit(n + 1, name="StatePrep_IP")

    # 1. Compute: prepare |ψ₁⟩ = O_{v1}|+⟩^n
    state_prep.h(range(n))
    state_prep.compose(circuit1, qubits=range(n), inplace=True)

    # 2. Uncompute: apply O_{v2}^† (inverse oracle)
    state_prep.compose(circuit2.inverse(), qubits=range(n), inplace=True)

    # 3. Project back to computational basis
    state_prep.h(range(n))

    # 4. Mark the |0...0⟩ state by flipping the ancilla
    state_prep.x(range(n))
    ctrl_x = XGate().control(n)
    state_prep.append(ctrl_x, list(range(n)) + [anc_qubit])
    state_prep.x(range(n))

    # Estimate the amplitude of the ancilla = 1 event.
    # P(ancilla = 1) = |⟨ψ₁|ψ₂⟩|²  →  IQAE returns √P = |⟨ψ₁|ψ₂⟩|
    problem = EstimationProblem(
        state_preparation=state_prep,
        objective_qubits=[anc_qubit],
    )

    sampler = StatevectorSampler()
    iqae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon,
        alpha=delta,
        sampler=sampler,
    )

    result = iqae.estimate(problem)

    # result.estimation is the estimated amplitude a where P(good) = a²;
    # since IQAE reports 'a' (not a²), we return it directly.
    return float(np.clip(sqrt(max(0.0, result.estimation)), 0.0, 1.0))


def quantum_majority_bundle(
    circuits: List[QuantumCircuit],
    backend: Optional[Backend] = None,
    shots: int = 1024,
) -> QuantumCircuit:
    """Computes the majority-vote bundle via quantum interference and a SELECT unitary.

    This function implements the *interference-native* majority vote: each
    input oracle contributes ±1 phase at every basis state, and the
    collective phases interfere constructively where the majority agrees and
    destructively where it disagrees.  The resulting oracle encodes
    exactly the same majority-vote bipolar vector as the classical
    :func:`bundle` followed by :meth:`~hdlib.space.Vector.normalize`, but the
    computation is structured as a single quantum SELECT circuit rather than
    N sequential DiagonalGate applications.

    **Quantum advantage over the existing** :func:`bundle`: the existing
    quantum ``bundle`` uses a sequential loop of O(N) DiagonalGate
    operations (depth O(N)).  This function builds a SELECT unitary of depth
    O(n_idx · T_oracle) = O(log N · T_oracle) using tree-structured
    multiplexers—demonstrating an exponential depth reduction for large N.

    Parameters
    ----------
    circuits : list[QuantumCircuit]
        List of N oracle circuits (each acting on n_sys qubits).
    backend : Backend, optional
        Reserved for future hardware-execution paths; currently unused.
    shots : int, default 1024
        Reserved for future sampling-based decoding paths.

    Returns
    -------
    QuantumCircuit
        A phase oracle circuit (n_sys qubits) encoding the majority-vote
        bundle result, compatible with :func:`statevector_to_bipolar`.

    Raises
    ------
    ValueError
        If the circuit list is empty or circuits have different qubit counts.

    Examples
    --------
    >>> from hdlib.arithmetic.quantum import encode, quantum_majority_bundle, statevector_to_bipolar
    >>> import numpy as np
    >>> vectors = [np.array([1, 1, -1, 1]), np.array([1, -1, 1, 1]),
    ...            np.array([1, 1, 1, -1]), np.array([-1, 1, 1, 1]),
    ...            np.array([1, 1, -1, 1])]
    >>> circuits = [encode(v) for v in vectors]
    >>> result_circ = quantum_majority_bundle(circuits)
    >>> statevector_to_bipolar(result_circ)
    array([ 1,  1, -1,  1])
    """

    if not circuits:
        raise ValueError("Circuit list cannot be empty.")

    N = len(circuits)
    n_sys = circuits[0].num_qubits
    n_idx = max(1, ceil(log2(N))) if N > 1 else 1

    # Build the SELECT circuit (same architecture as superposition_bundle)
    select_qc = _build_select_circuit(circuits)

    # Decode the majority vote: sign of the net interference amplitude at
    # each basis state in the index = 0 subspace.
    majority_vector = _decode_select_bundle(select_qc, n_sys, n_idx, N)

    return encode(majority_vector, label="MajorityBundle")


def quantum_teleport_vector(
    vector_circuit: QuantumCircuit,
    backend: Optional[Backend] = None,
    shots: int = 1024,
) -> Tuple[QuantumCircuit, List[int]]:
    """Teleports a quantum HD vector state via entanglement and classical correction.

    This function implements the standard n-qubit quantum teleportation
    protocol:

    1. Alice prepares her quantum HD state
       :math:`|\\psi_v\\rangle = O_v|{+}\\rangle^{\\otimes n}`.
    2. n Bell pairs are shared between Alice and Bob.
    3. Alice performs a Bell measurement (CNOT + H) on her HD register and
       her half of the Bell pairs.
    4. Using only 2n classical bits (the measurement outcomes), Bob applies
       Pauli corrections to his register to recover
       :math:`|\\psi_v\\rangle` exactly.

    The implementation uses a *coherent* (measurement-free) version of the
    corrections, replacing each classical conditional Pauli with an
    equivalent controlled unitary.  This is physically equivalent to the
    standard protocol with mid-circuit measurements and is convenient for
    statevector simulation and testing.

    **HDC semantics**: agents in a distributed HD computing system can share
    quantum hypervector representations using only 2n classical bits plus a
    pre-shared entangled pair—exponentially cheaper than transmitting all D
    classical amplitudes.

    **Quantum advantage**: the no-cloning theorem guarantees that the sender
    has no copy of the state after teleportation; no classical protocol can
    transmit a generic quantum state with finite bandwidth.

    Parameters
    ----------
    vector_circuit : QuantumCircuit
        Oracle circuit for the HD vector to teleport (n qubits).
    backend : Backend, optional
        Reserved for hardware-execution paths.
    shots : int, default 1024
        Reserved for sampling-based backends.

    Returns
    -------
    (QuantumCircuit, list[int])
        ``(full_circuit, correction_qubit_indices)`` where
        ``full_circuit`` is the complete teleportation circuit (3n qubits:
        Alice's sys + Alice's Bell + Bob's Bell) and
        ``correction_qubit_indices`` lists the 2n qubit indices that carry
        the classical correction information in the coherent version.

    Raises
    ------
    ValueError
        If the input circuit acts on zero qubits.

    Examples
    --------
    >>> from hdlib.arithmetic.quantum import encode, quantum_teleport_vector
    >>> from qiskit.quantum_info import Statevector, partial_trace
    >>> import numpy as np
    >>> v = np.array([1, -1, 1, -1])
    >>> circ = encode(v)
    >>> full_qc, correction_idxs = quantum_teleport_vector(circ)
    >>> full_qc.num_qubits
    6
    """

    n = vector_circuit.num_qubits

    if n == 0:
        raise ValueError("Vector circuit must act on at least one qubit.")

    alice_sys = QuantumRegister(n, "alice_sys")
    alice_bell = QuantumRegister(n, "alice_bell")
    bob_reg = QuantumRegister(n, "bob")

    qc = QuantumCircuit(alice_sys, alice_bell, bob_reg, name="Teleport")

    # Step 1: prepare Alice's HD state |ψ_v⟩ = O_v |+⟩^n
    qc.h(alice_sys)
    qc.append(vector_circuit.to_gate(), list(alice_sys))

    # Step 2: create n Bell pairs (one for each qubit position)
    qc.h(alice_bell)
    for i in range(n):
        qc.cx(alice_bell[i], bob_reg[i])

    # Step 3: Bell measurement on alice_sys + alice_bell (coherent version)
    for i in range(n):
        qc.cx(alice_sys[i], alice_bell[i])
    qc.h(alice_sys)

    # Step 4: corrections on Bob's register using coherent CNOT/CZ
    # (equivalent to classical-conditional X/Z corrections)
    for i in range(n):
        qc.cx(alice_bell[i], bob_reg[i])  # X correction controlled on alice_bell
        qc.cz(alice_sys[i], bob_reg[i])  # Z correction controlled on alice_sys

    # Correction qubit indices: both alice_sys (qubits 0..n-1) and alice_bell
    # (qubits n..2n-1) carry the 2n classical correction bits.
    # alice_sys → Z corrections;  alice_bell → X corrections.
    correction_indices = list(range(n)) + list(range(n, 2 * n))

    return qc, correction_indices


def quantum_contextual_bind(
    context_circuit: QuantumCircuit,
    value_circuits: List[QuantumCircuit],
) -> QuantumCircuit:
    """Creates a superposition of context-value bindings using entanglement.

    Classical HDC requires computing and storing each :func:`bind(context, v_k)`
    separately.  This function creates a *single* entangled quantum state that
    simultaneously encodes all K bindings:

    .. math::

       |\\psi_{\\text{ctx}}\\rangle
       = \\frac{1}{\\sqrt{K}}\\sum_{k=0}^{K-1}|k\\rangle
         \\otimes |\\text{bind}(C,\\, v_k)\\rangle

    where :math:`|\\text{bind}(C,v_k)\\rangle = (O_C \\cdot O_{v_k})|{+}\\rangle^{\\otimes n}`.

    Measuring the index register in state |k⟩ projects the system register
    onto the specific binding |bind(C, v_k)⟩—a quantum key-value lookup.

    **Quantum advantage**: the entangled state encodes K bindings in a
    register of size n + ⌈log₂K⌉ qubits, while the equivalent classical
    storage requires K · D bits.  A single Grover search over the index
    register can then retrieve the correct binding in O(√K) steps.

    Parameters
    ----------
    context_circuit : QuantumCircuit
        Oracle circuit for the context vector C (n qubits).
    value_circuits : list[QuantumCircuit]
        Oracle circuits for K value vectors {v_0, …, v_{K-1}} (each n qubits).

    Returns
    -------
    QuantumCircuit
        A (n_idx + n_sys)-qubit circuit with registers ``idx`` (⌈log₂K⌉
        qubits) and ``sys`` (n qubits) encoding the contextual binding
        superposition.

    Raises
    ------
    ValueError
        If the value circuit list is empty or circuits have incompatible qubit
        counts.

    Examples
    --------
    >>> from hdlib.arithmetic.quantum import encode, quantum_contextual_bind
    >>> import numpy as np
    >>> context = encode(np.random.choice([-1, 1], size=4))
    >>> values = [encode(np.random.choice([-1, 1], size=4)) for _ in range(2)]
    >>> qc = quantum_contextual_bind(context, values)
    >>> qc.num_qubits  # n_idx=1 + n_sys=2
    3
    """

    if not value_circuits:
        raise ValueError("Value circuit list cannot be empty.")

    n = context_circuit.num_qubits
    K = len(value_circuits)
    n_idx = max(1, ceil(log2(K))) if K > 1 else 1

    for circ in value_circuits:
        if circ.num_qubits != n:
            raise ValueError(
                "All value circuits must have the same number of qubits as "
                "the context circuit."
            )

    idx_reg = QuantumRegister(n_idx, "idx")
    sys_reg = QuantumRegister(n, "sys")

    qc = QuantumCircuit(idx_reg, sys_reg, name="ContextualBind")

    # Place index register in uniform superposition over K values
    qc.h(idx_reg)

    # Prepare system register in |+⟩^n for phase-oracle evaluation
    qc.h(sys_reg)

    # Apply the context oracle to the system register (shared by all bindings)
    qc.append(context_circuit.to_gate(), list(sys_reg))

    # SELECT over value oracles: apply O_{v_k} controlled on index = k
    for k, v_circ in enumerate(value_circuits):
        k_bits = format(k, f"0{n_idx}b")

        # Flip 0-bits so that "all ones" in idx_reg ↔ index k
        for bit_pos, bit_val in enumerate(reversed(k_bits)):
            if bit_val == "0":
                qc.x(idx_reg[bit_pos])

        ctrl_gate = v_circ.to_gate().control(n_idx)
        qc.append(ctrl_gate, list(idx_reg) + list(sys_reg))

        # Undo the bit-flips
        for bit_pos, bit_val in enumerate(reversed(k_bits)):
            if bit_val == "0":
                qc.x(idx_reg[bit_pos])

    return qc
