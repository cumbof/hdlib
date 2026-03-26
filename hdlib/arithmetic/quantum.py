"""Quantum implementation of the MAP arithmetic operators."""

import re
from math import atan2, sqrt, ceil, log2, pi
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from mthree import M3Mitigation
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import DiagonalGate, XGate
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

def compress_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Compresses a deep phase-encoded quantum circuit into a shallow circuit with one DiagonalGate.

    This acts as a quantum compiler for Vector-Symbolic Architectures.
    It calculates the noise-free phase accumulation of the deep circuit
    and reconstructs an identical quantum state using a single layer of
    Hadamard gates and one DiagonalGate.

    Parameters
    ----------
    circuit : QuantumCircuit
        The deep quantum circuit (e.g., a series of bundled veectors with hundreds of gates).

    Returns
    -------
    QuantumCircuit
        A mathematically identical shallow circuit.
    """

    num_qubits = circuit.num_qubits

    # 1. Mathematically evaluate the exact state of the deep circuit
    # This captures the pure quantum phase accumulation without noise
    state = Statevector.from_instruction(circuit.decompose())

    # 2. Extract the relative phases of the quantum state
    # Since vectors are encoded in the phase of a uniform superposition,
    # the amplitudes are all 1/sqrt(2^N), so we only need the angles.
    phases = np.angle(state.data)

    # 3. Create the compressed diagonal operator using the extracted phases
    diagonal_elements = np.exp(1j * phases)
    diag_gate = DiagonalGate(diagonal_elements.tolist())

    # 4. Build the shallow circuit
    compressed_qc = QuantumCircuit(num_qubits, name=f"{circuit.name}_compressed")

    # Initialize the uniform superposition
    compressed_qc.h(range(num_qubits))

    # Apply all accumulated phases in a single operation
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

def negate_circuits(circuits: List[QuantumCircuit]) -> List[QuantumCircuit]:
    """Flips the bipolar phase of the circuits for subtraction.
    Multiplying the complex eigenvalues of the DiagonalGate by -1 reflects the vector.

    Parameters
    ----------
    circuits : list
        The input circuits.

    Returns
    -------
    list
        List of phase-flipped circuits.
    """

    negated = list()

    for circuit in circuits:
        neg_circuit = QuantumCircuit(*circuit.qregs, name=f"{circuit.name}_neg")

        for instr in circuit.data:
            if isinstance(instr.operation, DiagonalGate):
                # -1 inverts the bipolar phases
                new_phases = np.array(instr.operation.params, dtype=complex) * -1.0
                neg_circuit.append(DiagonalGate(new_phases.tolist()), instr.qubits)

            else:
                neg_circuit.append(instr)

        negated.append(neg_circuit)

    return negated

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
