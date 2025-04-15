"""
Frauchiger-Renner Paradox Simulation

This simulation demonstrates how the FR paradox emerges and how contextual certainty resolves it.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Statevector, partial_trace
import numpy as np
import matplotlib.pyplot as plt

class QuantumAgent:
    """Represents a quantum agent who can perform measurements and make inferences."""
    
    def __init__(self, name):
        self.name = name
        self.observations = {}
        self.inferences = {}
        self.statevector = None
        self.reduced_density_matrix = None
    
    def observe_subsystem(self, statevector, qubit_indices, system_size):
        """
        Computes the reduced density matrix for the agent's subsystem and calculates measurement probabilities.
        
        Args:
            statevector: Full system state vector
            qubit_indices: Indices of qubits the agent has access to
            system_size: Total number of qubits
            
        Returns:
            Dictionary of measurement probabilities
        """
        # Calculate which qubits to trace out
        qubits_to_trace = [i for i in range(system_size) if i not in qubit_indices]
        
        # Partial trace to get the reduced state
        self.reduced_density_matrix = partial_trace(statevector, qubits_to_trace)
        
        # Calculate the probability of measurement outcomes
        # For a single qubit, diagonal elements are probabilities of |0⟩ and |1⟩
        if len(qubit_indices) == 1:
            p0 = np.real(self.reduced_density_matrix.data[0, 0])
            p1 = np.real(self.reduced_density_matrix.data[1, 1])
            return {"0": p0, "1": p1}
        
        return self.reduced_density_matrix
    
    def measure_in_bell_basis(self, statevector, qubit_indices, system_size):
        """
        Simulates measurement in the Bell basis by analyzing the quantum state.
        Returns probabilities for each Bell state outcome.
        
        Uses rigorous calculation of Tr(ρ * Π_BellState) for each Bell state projector.
        """
        # Get reduced density matrix for the two-qubit subsystem
        qubits_to_trace = [i for i in range(system_size) if i not in qubit_indices]
        subsystem = partial_trace(statevector, qubits_to_trace)
        self.reduced_density_matrix = subsystem
        
        # Bell state projectors
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        # |Φ-⟩ = (|00⟩ - |11⟩)/√2
        # |Ψ+⟩ = (|01⟩ + |10⟩)/√2
        # |Ψ-⟩ = (|01⟩ - |10⟩)/√2
        
        # Construct the Bell state projectors as numpy arrays
        # |Φ+⟩⟨Φ+| = (|00⟩ + |11⟩)(⟨00| + ⟨11|)/2
        phi_plus_projector = np.zeros((4, 4), dtype=complex)
        phi_plus_projector[0, 0] = 0.5  # |00⟩⟨00|
        phi_plus_projector[0, 3] = 0.5  # |00⟩⟨11|
        phi_plus_projector[3, 0] = 0.5  # |11⟩⟨00|
        phi_plus_projector[3, 3] = 0.5  # |11⟩⟨11|
        
        # |Φ-⟩⟨Φ-| = (|00⟩ - |11⟩)(⟨00| - ⟨11|)/2
        phi_minus_projector = np.zeros((4, 4), dtype=complex)
        phi_minus_projector[0, 0] = 0.5   # |00⟩⟨00|
        phi_minus_projector[0, 3] = -0.5  # -|00⟩⟨11|
        phi_minus_projector[3, 0] = -0.5  # -|11⟩⟨00|
        phi_minus_projector[3, 3] = 0.5   # |11⟩⟨11|
        
        # |Ψ+⟩⟨Ψ+| = (|01⟩ + |10⟩)(⟨01| + ⟨10|)/2
        psi_plus_projector = np.zeros((4, 4), dtype=complex)
        psi_plus_projector[1, 1] = 0.5  # |01⟩⟨01|
        psi_plus_projector[1, 2] = 0.5  # |01⟩⟨10|
        psi_plus_projector[2, 1] = 0.5  # |10⟩⟨01|
        psi_plus_projector[2, 2] = 0.5  # |10⟩⟨10|
        
        # |Ψ-⟩⟨Ψ-| = (|01⟩ - |10⟩)(⟨01| - ⟨10|)/2
        psi_minus_projector = np.zeros((4, 4), dtype=complex)
        psi_minus_projector[1, 1] = 0.5   # |01⟩⟨01|
        psi_minus_projector[1, 2] = -0.5  # -|01⟩⟨10|
        psi_minus_projector[2, 1] = -0.5  # -|10⟩⟨01|
        psi_minus_projector[2, 2] = 0.5   # |10⟩⟨10|
        
        # Calculate probabilities using Tr(ρ * Π_Bell) for each Bell state
        rho_matrix = subsystem.data
        bell_probs = {
            "Φ+": np.real(np.trace(np.matmul(rho_matrix, phi_plus_projector))),
            "Φ-": np.real(np.trace(np.matmul(rho_matrix, phi_minus_projector))),
            "Ψ+": np.real(np.trace(np.matmul(rho_matrix, psi_plus_projector))),
            "Ψ-": np.real(np.trace(np.matmul(rho_matrix, psi_minus_projector)))
        }
        
        # Normalize in case of small numerical errors
        total_prob = sum(bell_probs.values())
        if abs(total_prob - 1.0) > 1e-10:  # If probabilities don't sum to 1 (within numerical precision)
            for state in bell_probs:
                bell_probs[state] /= total_prob
                
        return bell_probs
    
    def make_inference(self, about_agent, inference_key, inference_value, confidence=1.0):
        """Agent makes an inference about another agent's observations or system state."""
        if about_agent not in self.inferences:
            self.inferences[about_agent] = {}
        self.inferences[about_agent][inference_key] = (inference_value, confidence)
    
    def check_consistency(self, other_agent):
        """Checks if this agent's inferences are consistent with another agent's observations."""
        if other_agent.name not in self.inferences:
            return True, []  # No inferences about this agent
        
        inconsistencies = []
        for key, (inferred_value, confidence) in self.inferences[other_agent.name].items():
            if key in other_agent.observations and other_agent.observations[key] != inferred_value:
                inconsistencies.append({
                    "key": key,
                    "inferred": inferred_value,
                    "actual": other_agent.observations[key],
                    "confidence": confidence
                })
        
        return (len(inconsistencies) == 0, inconsistencies)
    
    def __str__(self):
        output = f"Agent: {self.name}\n"
        output += "  Observations:\n"
        for key, value in self.observations.items():
            output += f"    {key}: {value}\n"
        
        output += "  Inferences:\n"
        for agent, inferences in self.inferences.items():
            output += f"    About {agent}:\n"
            for key, (value, confidence) in inferences.items():
                output += f"      {key}: {value} (confidence: {confidence:.2f})\n"
        
        return output


def simulate_fr_paradox():
    """
    Simulates the Frauchiger-Renner paradox with agents making observations and inferences.
    This version calculates explicit probabilities at each step and clearly marks
    conditional path analysis.
    """
    
    print("\n========== DYNAMIC FRAUCHIGER-RENNER PARADOX SIMULATION ==========\n")
    print("This simulation demonstrates how agents reach contradictory conclusions")
    print("when applying quantum theory from different perspectives.\n")
    
    # Create our quantum agents
    alice = QuantumAgent("Alice")
    bob = QuantumAgent("Bob")
    charlie = QuantumAgent("Charlie")
    debbie = QuantumAgent("Debbie")
    
    # Step 1: Initialize the quantum circuit
    q_system = QuantumRegister(1, 'system')
    q_alice = QuantumRegister(1, 'alice_memory')
    q_charlie = QuantumRegister(1, 'charlie_memory')
    q_comm = QuantumRegister(1, 'communication')
    
    qc = QuantumCircuit(q_system, q_alice, q_charlie, q_comm)
    
    # Step 2: Initial state preparation
    qc.h(q_system[0])  # Put system in |+⟩ state
    qc.barrier()
    
    # Execute circuit to get state after initialization
    sim = AerSimulator()
    qc_with_save = qc.copy()
    qc_with_save.save_statevector('init_state')
    transpiled_qc = transpile(qc_with_save, sim)
    result = sim.run(transpiled_qc).result()
    state_init = result.data()['init_state']
    
    print("Initial state prepared. The system is in |+⟩ state.\n")
    
    # Step 3: Alice's measurement interaction
    print("----- Alice's Perspective -----")
    # Alice's unitary interaction
    qc_alice = qc.copy()
    qc_alice.cx(q_system[0], q_alice[0])  # Record measurement in Alice's memory
    qc_alice.barrier()
    
    qc_alice_with_save = qc_alice.copy()
    qc_alice_with_save.save_statevector('alice_state')
    transpiled_qc_alice = transpile(qc_alice_with_save, sim)
    result_alice = sim.run(transpiled_qc_alice).result()
    state_after_alice = result_alice.data()['alice_state']
    
    # Calculate Alice's outcome probabilities based on the state *before* assumption
    alice_probs = alice.observe_subsystem(state_after_alice, [1], 4)  # q_alice is at index 1
    print(f"Analysis: Quantum mechanics predicts Alice (qubit 1) observes:")
    print(f"  Outcome '0' with probability: {alice_probs['0']:.4f}")
    print(f"  Outcome '1' with probability: {alice_probs['1']:.4f}")
    
    # State the assumption for this analysis path
    assumed_alice_outcome = "1"
    print(f"\n*** For demonstrating the paradox's logic, we proceed CONDITIONALLY on Alice observing '{assumed_alice_outcome}' ***")
    print(f"*** This is a valid path that quantum mechanics predicts occurs with probability {alice_probs[assumed_alice_outcome]:.4f} ***\n")
    
    alice.observations["direct"] = assumed_alice_outcome
    print(f"Alice (conditionally) observes outcome: {alice.observations['direct']}")
    
    # Alice's inference about the system state
    alice.make_inference("System", "state", "|1⟩" if alice.observations["direct"] == "1" else "|0⟩")
    print(f"Alice (conditionally) concludes: System is in state {alice.inferences['System']['state'][0]}")
    print()
    
    # Communication setup
    qc_alice.h(q_comm[0])  # Put communication channel in superposition
    qc_alice.cx(q_alice[0], q_comm[0])  # Entangle Alice's memory with communication
    qc_alice.barrier()
    
    qc_comm_with_save = qc_alice.copy()
    qc_comm_with_save.save_statevector('comm_state')
    transpiled_qc_comm = transpile(qc_comm_with_save, sim)
    result_comm = sim.run(transpiled_qc_comm).result()
    state_after_comm = result_comm.data()['comm_state']
    
    print("Communication channel established between labs.\n")
    
    # Step 5: Charlie's measurement interaction
    print("----- Charlie's Perspective -----")
    qc_charlie = qc_alice.copy()
    qc_charlie.cx(q_comm[0], q_charlie[0])  # Record in Charlie's memory
    qc_charlie.barrier()
    
    # Fixed version - explicitly save the statevector
    qc_charlie_with_save = qc_charlie.copy()
    qc_charlie_with_save.save_statevector('charlie_state')
    transpiled_qc_charlie = transpile(qc_charlie_with_save, sim)
    result_charlie = sim.run(transpiled_qc_charlie).result()
    state_after_charlie = result_charlie.data()['charlie_state']
    
    # Calculate Charlie's outcome probabilities
    charlie_probs = charlie.observe_subsystem(state_after_charlie, [2], 4)  # q_charlie is at index 2
    print(f"Analysis: Quantum mechanics predicts Charlie (qubit 2) observes:")
    print(f"  Outcome '0' with probability: {charlie_probs['0']:.4f}")
    print(f"  Outcome '1' with probability: {charlie_probs['1']:.4f}")
    
    # State the assumption for this analysis path
    assumed_charlie_outcome = "1"
    print(f"\n*** For demonstrating the paradox's logic, we proceed CONDITIONALLY on Charlie observing '{assumed_charlie_outcome}' ***")
    print(f"*** This is a valid path that quantum mechanics predicts occurs with probability {charlie_probs[assumed_charlie_outcome]:.4f} ***\n")
    
    charlie.observations["direct"] = assumed_charlie_outcome
    print(f"Charlie (conditionally) observes outcome: {charlie.observations['direct']}")
    
    # Based on protocol design, Charlie infers Alice's result
    charlie.make_inference(alice.name, "direct", charlie.observations["direct"])
    print(f"Charlie infers (conditionally): Alice must have measured {charlie.inferences[alice.name]['direct'][0]}")
    
    # And transitively, Charlie infers the system state
    charlie.make_inference("System", "state", "|1⟩" if charlie.observations["direct"] == "1" else "|0⟩")
    print(f"Charlie concludes (conditionally): System must be in state {charlie.inferences['System']['state'][0]}")
    print()
    
    # Step 6: Bob's measurement analysis
    print("----- Bob's Perspective -----")
    # Bob analyzes the state *after* Charlie's interaction but *before* his own measurement
    bob_bell_probs = bob.measure_in_bell_basis(state_after_charlie, [0, 1], 4)  # System and Alice's memory
    print("Bob's Bell measurement probabilities (based on analysis of the quantum state):")
    for state, prob in sorted(bob_bell_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  |{state}⟩: {prob:.4f}")
    
    # Find which Bell state has highest probability
    most_likely_bell = max(bob_bell_probs, key=bob_bell_probs.get)
    max_prob = bob_bell_probs[most_likely_bell]
    
    print(f"\nQuantum mechanics predicts Bob most likely observes Bell state |{most_likely_bell}⟩ with probability {max_prob:.4f}")
    
    # For demonstration, analyze consequences of Bob observing Φ+
    assumed_bob_outcome = "Φ+"
    bob.observations["bell"] = assumed_bob_outcome
    print(f"\n*** Analyzing the consequences if Bob observes Bell state |{bob.observations['bell']}⟩ ***\n")
    
    # Bob's inference about Alice and system
    if bob.observations["bell"] == "Φ+":
        bob.make_inference(alice.name, "definite_outcome", False, confidence=0.99)
        bob.make_inference("System", "definite_state", False, confidence=0.99)
        print("Bob concludes (from quantum analysis): Alice CANNOT have a definite measurement outcome.")
        print("Bob concludes (from quantum analysis): System CANNOT be in a definite state |0⟩ or |1⟩.")
        print("This is because |Φ+⟩ represents an entangled superposition (|00⟩ + |11⟩)/√2 of Alice+System.")
    print()
    
    # Step 7: Debbie's measurement analysis
    print("----- Debbie's Perspective -----")
    # Debbie analyzes the state after Charlie's interaction
    debbie_bell_probs = debbie.measure_in_bell_basis(state_after_charlie, [2, 3], 4)  # Charlie and comm
    print("Debbie's Bell measurement probabilities (based on analysis of the quantum state):")
    for state, prob in sorted(debbie_bell_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"  |{state}⟩: {prob:.4f}")
    
    # Find which Bell state has highest probability
    most_likely_bell_debbie = max(debbie_bell_probs, key=debbie_bell_probs.get)
    max_prob_debbie = debbie_bell_probs[most_likely_bell_debbie]
    
    print(f"\nQuantum mechanics predicts Debbie most likely observes Bell state |{most_likely_bell_debbie}⟩ with probability {max_prob_debbie:.4f}")
    
    # For demonstration, analyze consequences of Debbie observing Φ+
    assumed_debbie_outcome = "Φ+"
    debbie.observations["bell"] = assumed_debbie_outcome
    print(f"\n*** Analyzing the consequences if Debbie observes Bell state |{debbie.observations['bell']}⟩ ***\n")
    
    # Debbie's inference about Charlie
    if debbie.observations["bell"] == "Φ+":
        debbie.make_inference(charlie.name, "definite_outcome", False, confidence=0.99)
        print("Debbie concludes (from quantum analysis): Charlie CANNOT have a definite measurement outcome.")
        print("This is because |Φ+⟩ represents an entangled superposition (|00⟩ + |11⟩)/√2 of Charlie+Communication qubits.")
    print()
    
    # Step 8: Check for contradictions
    print("===== CHECKING FOR CONTRADICTIONS (Conditional Path Analysis) =====")
    
    # Compare Bob's inference about Alice with Alice's conditional observation
    bob_alice_consistent, inconsistencies = bob.check_consistency(alice)
    print(f"Bob's inference about Alice vs. Alice's conditional observation ('{alice.observations['direct']}'): Consistent = {bob_alice_consistent}")
    if not bob_alice_consistent:
        print("  Inconsistencies found:")
        for i in inconsistencies:
            # Check the specific inference about 'definite_outcome'
            if i['key'] == 'definite_outcome' and i['inferred'] is False:
                print(f"  -> Bob inferred Alice's outcome is NOT definite, but we followed the path where Alice conditionally observed '{alice.observations['direct']}'.")
                print(f"  -> PARADOXICAL ELEMENT: These cannot both be true if measurement outcomes are universal facts!")
            else:
                print(f"    Bob inferred Alice's {i['key']} is {i['inferred']}, but Alice (conditionally) observed {i['actual']}")
    
    # Compare Charlie's inference about the system with Bob's inference
    charlie_system = charlie.inferences.get("System", {}).get("state", (None, 0))[0]
    bob_system = bob.inferences.get("System", {}).get("definite_state", (None, 0))[0]
    system_consistent = True  # Assume consistent unless proven otherwise
    
    if charlie_system in ["|0⟩", "|1⟩"] and bob_system is False:
        system_consistent = False
        print(f"Charlie's conditional conclusion vs Bob's conclusion about System: Consistent = {system_consistent}")
        print(f"  -> Charlie (conditionally) concluded: System is in definite state {charlie_system}")
        print(f"  -> Bob concluded (from analysis): System CANNOT be in a definite state.")
        print(f"  -> PARADOXICAL ELEMENT: These cannot both be true if system states are universal facts!")
    else:
        print(f"Charlie's conditional conclusion vs Bob's conclusion about System: Consistent = {system_consistent}")
    
    # Compare Debbie's inference about Charlie with Charlie's observation
    debbie_charlie_consistent, inconsistencies = debbie.check_consistency(charlie)
    print(f"Debbie's inference about Charlie vs. Charlie's conditional observation ('{charlie.observations['direct']}'): Consistent = {debbie_charlie_consistent}")
    if not debbie_charlie_consistent:
        print("  Inconsistencies found:")
        for i in inconsistencies:
            if i['key'] == 'definite_outcome' and i['inferred'] is False:
                print(f"  -> Debbie inferred Charlie's outcome is NOT definite, but we followed the path where Charlie conditionally observed '{charlie.observations['direct']}'.")
                print(f"  -> PARADOXICAL ELEMENT: These cannot both be true if measurement outcomes are universal facts!")
            else:
                print(f"    Debbie inferred Charlie's {i['key']} is {i['inferred']}, but Charlie (conditionally) observed {i['actual']}")
    
    # The FR paradox emerges when we try to combine these perspectives
    print("\n===== THE FRAUCHIGER-RENNER PARADOX EMERGES =====")
    print("The paradox occurs when we try to combine all perspectives into a single framework.")
    print("We demonstrated this by following a specific conditional path where:")
    print(f"1. Alice (conditionally) measures '{alice.observations['direct']}', leading her to conclude the system is in state {alice.inferences['System']['state'][0]}")
    print(f"2. Charlie (conditionally) measures '{charlie.observations['direct']}', confirming Alice's measurement")
    print(f"3. Bob measures the Bell state |{bob.observations['bell']}⟩, which implies Alice CANNOT have a definite result")
    print(f"4. Debbie measures the Bell state |{debbie.observations['bell']}⟩, which implies Charlie CANNOT have a definite result")
    print()
    print("These perspectives CANNOT be reconciled if we assume facts about measurement outcomes")
    print("are universal and observer-independent. This creates a direct logical contradiction.")
    print()
    print("IMPORTANT: We didn't need to 'collapse' the quantum state to show this paradox.")
    print("Instead, we analyzed what quantum mechanics itself predicts about the probabilities")
    print("and the logical consequences that follow if certain outcomes occur.")
    
    print("\n===== CONTEXTUAL CERTAINTY RESOLUTION =====")
    print("Contextual Certainty resolves the paradox by recognizing:")
    print("- Alice's certainty about the system is only valid in Alice's context")
    print("- Bob's certainty about Alice+system is only valid in Bob's context")
    print("- Charlie's certainty about his measurement is only valid in Charlie's context")
    print("- Debbie's certainty about Charlie+comm is only valid in Debbie's context")
    
    print("\nThese agents' observations are internally consistent when interpreted")
    print("as context-dependent facts rather than universal, observer-independent facts.")
    print("Quantum theory correctly predicts all observations, but their interpretation requires")
    print("abandoning the assumption that measurement outcomes are universal facts.")
    
    # Return all agents for further analysis
    return alice, bob, charlie, debbie, state_after_charlie


if __name__ == "__main__":
    print("Running Frauchiger-Renner Paradox Simulation...")
    print("(This demonstrates the necessity of contextual certainty in quantum mechanics)")
    print()
    
    try:
        alice, bob, charlie, debbie, final_state = simulate_fr_paradox()
        
        # Optionally display full agent states
        print("\n===== AGENT STATES (Conditional Path Analysis) =====")
        print(alice)
        print(bob)
        print(charlie)
        print(debbie)
        
        print("\nSimulation completed successfully!")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()