"""Analogical Reasoning Test for Classical and Quantum HDC Models.
This script tests the ability to solve the analogy: "What is the Dollar of Mexico?"
The logic is based on the official hdlib documentation: https://github.com/cumbof/hdlib/wiki/Examples#what-is-the-dollar-of-mexico
"""

import numpy as np
from math import log2, sqrt, ceil
from typing import Dict, Optional

from hdlib.space import Space
from hdlib.vector import Vector
from hdlib.arithmetic import (
    bundle as classical_bundle,
    bind as classical_bind
)

from hdlib.arithmetic.quantum import (
    phase_oracle_gate,
    run_hadamard_test,
    bundle as quantum_bundle,
    bind as quantum_bind,
)

from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import (
    QiskitRuntimeService, 
    Session, 
    Sampler, 
    SamplerOptions
)


# Configuration for Hardware (Optional)
CHANNEL = "IBM-CHANNEL"
INSTANCE = "IBM-INSTANCE"
BACKEND = "IBM-BACKEND"
API_KEY = "YOUR-API-KEY"

def run_classical_reasoning(dim: int, concepts: Dict[str, np.ndarray]):
    """Executes the analogical reasoning task using the classical HDC model."""
    print(f"\n--- Running Classical Reasoning (Dimension: {dim}) ---")

    space = Space(size=dim, vtype="bipolar")
    for name, vec_data in concepts.items():
        space.insert(Vector(name=name, vector=vec_data))

    # 1. Define feature and entity vectors from the space
    usa = space.get(names=["USA"])[0]
    dol = space.get(names=["DOL"])[0]
    wdc = space.get(names=["WDC"])[0]
    mex = space.get(names=["MEX"])[0]
    pes = space.get(names=["PES"])[0]
    mxc = space.get(names=["MXC"])[0]
    nam = space.get(names=["NAM"])[0]
    mon = space.get(names=["MON"])[0]
    cap = space.get(names=["CAP"])[0]

    # 2. Form composite concepts for USA and Mexico as per documentation
    ustates_nam = classical_bind(nam, usa)
    ustates_cap = classical_bind(cap, wdc)
    ustates_mon = classical_bind(mon, dol)
    ustates = classical_bundle(classical_bundle(ustates_nam, ustates_cap), ustates_mon)

    mexico_nam = classical_bind(nam, mex)
    mexico_cap = classical_bind(cap, mxc)
    mexico_mon = classical_bind(mon, pes)
    mexico_data = classical_bundle(classical_bundle(mexico_nam, mexico_cap), mexico_mon)

    # 3. Create the mapping vector F_UM
    f_um = classical_bind(ustates, mexico_data)

    # 4. Formulate the query
    guess_pes = classical_bind(dol, f_um)

    distances = dict()

    codebook_items = ["PES", "MXC", "MEX", "DOL", "WDC", "USA"]
    for name in codebook_items:
        distances[name] = guess_pes.dist(space.get(names=[name])[0])
        print(f"  - Distance with {name}: {distances[name]:.4f}")

    # 5. Search for the answer
    closest_match = min(distances, key=distances.get)

    print(f"Classical Query: 'What is the Dollar of Mexico?'")
    print(f"Classical Answer: {closest_match}")
    print(f"Correct Answer: PES")
    print(f"Result: {'SUCCESS' if closest_match == 'PES' else 'FAILURE'}")

def run_quantum_reasoning(
    dim: int, 
    concepts: Dict[str, np.ndarray],
    backend: Backend, 
    oaa_rounds: int=1, 
    shots: int=10000, 
    seed: int=42,
    sampler: Optional[Sampler]=None
):
    """Executes the analogical reasoning task using the Quantum HDC model.
    
    This function translates the classical reasoning query into a fully quantum
    circuit by applying the distributive property of 'bind' over 'bundle'.

    Classical Query: 
      Query = bind(DOL, bind(USTATES, MEXICO))
      where USTATES = bundle(U_nam, U_cap, U_mon)
      and   MEXICO  = bundle(M_nam, M_cap, M_mon)

    Quantum Translation:
      Query_State = bundle([
          bind(DOL, U_nam, M_nam),
          bind(DOL, U_nam, M_cap),
          bind(DOL, U_nam, M_mon),
          bind(DOL, U_cap, M_nam),
          ...
          bind(DOL, U_mon, M_mon)
      ])
    """

    print(f"\n--- Running Quantum Reasoning (Dimension: {dim}) ---")

    # 1. --- Setup ---
    n = int(ceil(log2(dim)))
    target_dim = 2**n

    if dim != target_dim:
        print(f"  Note: Padding concepts from {dim} to {target_dim} (for {n} qubits).")

    # 2. --- Helper Functions ---
    def get_padded_vec(vec_name: str) -> np.ndarray:
        """Pads or truncates a classical vector to match the 2**n dimension."""
        classical_vec = concepts[vec_name]

        if len(classical_vec) == target_dim:
            return classical_vec

        if len(classical_vec) < target_dim:
            # Pad with +1 (neutral element for bipolar binding)
            return np.pad(classical_vec, (0, target_dim - len(classical_vec)), 'constant', constant_values=1)

        # If classical_vec is > target_dim, truncate it
        return classical_vec[:target_dim]

    def get_oracle_circuit(vec_name: str) -> QuantumCircuit:
        """Creates a circuit containing only the phase oracle for a concept."""
        vec = get_padded_vec(vec_name)
        gate = phase_oracle_gate(vec, label=f"O_{vec_name}")
        qc = QuantumCircuit(n, name=vec_name)
        qc.append(gate, range(n))
        return qc

    def get_state_prep_circuit(op_circuit: QuantumCircuit) -> QuantumCircuit:
        """Creates a state preparation circuit |v> = O_v |+> from an oracle circuit O_v."""
        n_q = op_circuit.num_qubits
        # Ensure we are using the same register as the operator
        qc = QuantumCircuit(*op_circuit.qregs, name=f"Prep_{op_circuit.name}")
        qc.h(range(n_q)) # Start in uniform superposition
        qc.append(op_circuit.to_gate(), range(n_q)) # Apply the operator
        return qc

    # 3. --- Define Base Operators ---
    print("1. Creating base oracle circuits for all concepts...")

    # Get base oracle circuits for all atomic concepts
    # These are the *operators* U_v, not the states
    C = dict()
    all_concepts = ["USA", "DOL", "WDC", "MEX", "PES", "MXC", "NAM", "MON", "CAP"]
    for name in all_concepts:
        if name not in concepts:
            raise ValueError(f"Concept vector for '{name}' not found in input.")
        C[name] = get_oracle_circuit(name)

    # 4. --- Define Composite Operators ---
    print("2. Defining composite operators for query...")

    # USTATES component operators:
    ustates_ops = [
        quantum_bind([C["NAM"], C["USA"]]), # ustates_nam
        quantum_bind([C["CAP"], C["WDC"]]), # ustates_cap
        quantum_bind([C["MON"], C["DOL"]])  # ustates_mon
    ]

    # MEXICO component operators:
    mexico_ops = [
        quantum_bind([C["NAM"], C["MEX"]]), # mexico_nam
        quantum_bind([C["CAP"], C["MXC"]]), # mexico_cap
        quantum_bind([C["MON"], C["PES"]])  # mexico_mon
    ]

    # 5. --- Build Final Query State ---
    print(f"3. Building final query state (Bundle of 9 terms - {oaa_rounds} OAA rounds)...")

    # Build the 9 terms for the final bundle
    # Query = bundle( [ bind(DOL, U_i, M_j) for U_i in ustates_ops for M_j in mexico_ops ] )
    query_component_ops = list()
    for u_op in ustates_ops:
        for m_op in mexico_ops:
            # This is O_dol * O_ustates_comp * O_mexico_comp
            term_op = quantum_bind([C["DOL"], u_op, m_op])
            query_component_ops.append(term_op)

    # Now, bundle these 9 operators to create the final query state
    # The 'bundle' function returns a state preparation circuit
    query_circ, _, _ = quantum_bundle(
        unitary_circuits=query_component_ops,
        weights=[1/9] * 9, # Equal weights
        oaa_rounds=oaa_rounds
    )
    query_circ.name = "QUERY_Guess"

    # 6. --- Build Codebook Target States ---
    print("4. Building target state circuits for codebook...")

    codebook_circuits = dict()
    codebook_items = ["PES", "MXC", "MEX", "DOL", "WDC", "USA"]

    for name in codebook_items:
        # Get the base operator circuit
        op_circ = C[name]
        # Create the state prep circuit |v> = O_v |+>
        codebook_circuits[name] = get_state_prep_circuit(op_circ)

    # 7. --- Run Hadamard Tests ---
    print(f"5. Running Hadamard tests ({shots} shots each)...")

    similarities = dict()

    for name, target_circ in codebook_circuits.items():
        print(f"  - Comparing Query with {name}...")

        # state_left_circ = the small circuit (system only)
        # state_right = the big circuit (system + ancillas)
        similarity, counts = run_hadamard_test(
            target_circ,     # The simple |v> = O_v |+> state (n_sys qubits)
            query_circ,          # The complex bundled state (n_total qubits)
            backend,
            shots=shots,
            seed=seed,
            sampler=sampler
        )

        similarities[name] = similarity
        print(f"    Similarity with {name}: {similarity:.4f} (Counts: {counts})")

    # 8. --- Report Results ---
    # In classical HDC, we find the minimum distance.
    # In this quantum analogue, we find the maximum similarity (fidelity = sqrt(inner_product)).
    closest_match = max(similarities, key=similarities.get)

    print("\n--- Quantum Reasoning Results ---")
    print(f"Quantum Query: 'What is the Dollar of Mexico?'")
    print(f"Quantum Answer (Max Similarity): {closest_match}")
    print(f"Correct Answer: PES")
    print(f"Result: {'SUCCESS' if closest_match == 'PES' else 'FAILURE'}")

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    CONCEPT_NAMES = [
        "NAM", "MON", "CAP", 
        "USA", "DOL", "WDC", 
        "MEX", "PES", "MXC"
    ]

    CONCEPTS_10000D = {name: np.random.choice([-1, 1], size=10000) for name in CONCEPT_NAMES}
    CONCEPTS_16D = {name: np.random.choice([-1, 1], size=16) for name in CONCEPT_NAMES}

    # --- PART 1: CLASSICAL BASELINE ---
    run_classical_reasoning(10000, CONCEPTS_10000D)
    run_classical_reasoning(16, CONCEPTS_16D)

    # --- PART 2: QUANTUM EXPERIMENT - SIMULATION (NOISE-FREE) ---
    # Simulation (noise-free)
    run_quantum_reasoning(16, CONCEPTS_16D, AerSimulator(), oaa_rounds=6, shots=10000, seed=seed, sampler=None)

    # --- PART 3: QUANTUM EXPERIMENT - SIMULATION (WITH NOISE MODEL) ---
    # Initialize a temporary service connection
    # Always use "ibm_quantum_platform" to fetch the backend properties
    noise_service = QiskitRuntimeService(channel="ibm_quantum_platform", token=API_KEY)

    # Retrieve a backend
    # We only need its noise model
    backend_for_noise = noise_service.backend(BACKEND)

    # Finally, define the noise model
    noise_model = NoiseModel.from_backend(backend_for_noise)

    # Simulation (with noise model)
    run_quantum_reasoning(16, CONCEPTS_16D, AerSimulator(noise_model=noise_model), oaa_rounds=6, shots=10000, seed=seed, sampler=None)

    # --- PART 4: QUANTUM EXPERIMENT - HARDWARE ---
    # Initialize a quantum runtime service for a specific IBM QC channel, instance, and backend
    service = QiskitRuntimeService(channel=CHANNEL, token=API_KEY, instance=INSTANCE)

    backend = service.backend(BACKEND)

    with Session(backend=backend) as session:
        # Set up error mitigation options. T-REx is enabled by default at `resilience_level=1`
        options = SamplerOptions()
        options.resilience_level = 1

        sampler = Sampler(mode=session, options=options)

        # Hardware
        run_quantum_reasoning(16, CONCEPTS_16D, backend, oaa_rounds=6, shots=10000, seed=seed, sampler=sampler)
