# Import necessary libraries
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import time 
import math 
from math import log2, ceil
import random 

def diffusion_operator(n):
    diffuser = QuantumCircuit(n)
    diffuser.h(range(n))
    diffuser.x(range(n))
    diffuser.h(n - 1)
    diffuser.mcx(list(range(n - 1)), n - 1)
    diffuser.h(n - 1)
    diffuser.x(range(n))
    diffuser.h(range(n))
    return diffuser

# Set up the quantum circuit
dataset_size = 10000000  # You can make this dynamic, max ~256–512 for performance
num_qubits = ceil(log2(dataset_size))
search_space_size = 2 ** num_qubits  # Grover needs powers of 2

# Optimal number of Grover iterations
optimal_iterations = math.floor((math.pi / 4) * math.sqrt(search_space_size))

# Generate dataset
dataset = list(range(dataset_size))
random.shuffle(dataset)

# Random target
target = random.choice(dataset)
target_bin = format(target, f'0{num_qubits}b')

print(f"Target: {target} (binary: {target_bin})")

grover_circuit = QuantumCircuit(num_qubits, num_qubits)

# Step 1: Initialization - Apply Hadamard gates to all qubits
grover_circuit.h(range(num_qubits))

# Step 2: Build the oracle
oracle = QuantumCircuit(num_qubits)

for i in range(num_qubits):
    if target_bin[i] == '0':
        oracle.x(i)

# oracle.cz(0, num_qubits - 1)
oracle.h(num_qubits - 1)
oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
oracle.h(num_qubits - 1)

# # Apply X gates to flip the bits corresponding to the desired item
# for i in range(num_qubits):
#     if target_bin[i] == '0':
#         oracle.x(i)

# # Apply the oracle by converting it to a gate and applying it to the circuit
# grover_circuit.append(oracle.to_gate(), range(num_qubits))

# # Step 3: Construct the diffusion operator
# grover_circuit.h(range(num_qubits))
# grover_circuit.z(range(num_qubits))
# grover_circuit.cz(0, num_qubits-1)
# grover_circuit.h(range(num_qubits))

# Step 4: Run the algorithm iteratively
# num_iterations = 2  # Number of iterations to run the algorithm
# for _ in range(num_iterations):
#     grover_circuit.append(oracle.to_gate(), range(num_qubits))
#     grover_circuit.h(range(num_qubits))
#     grover_circuit.z(range(num_qubits))
#     grover_circuit.cz(0, num_qubits-1)
#     grover_circuit.h(range(num_qubits))
grover_circuit.append(oracle.to_gate(), range(num_qubits))
grover_circuit.append(diffusion_operator(num_qubits).to_gate(), range(num_qubits))

# Step 5: Measure the results
grover_circuit.measure(range(num_qubits), range(num_qubits))

# For execution
simulator = AerSimulator()
compiled_circuit = transpile(grover_circuit, simulator)

measured_decimal = -1

start = time.time()
# while measured_decimal != target:
sim_result = simulator.run(compiled_circuit, shots=optimal_iterations).result()
counts = sim_result.get_counts()
# measured = list(counts.keys())[0]
measured = max(counts, key=counts.get)
measured_decimal = int(measured, 2)
end = time.time()

# Output
# print(counts)
print(f"Grover Measured Result: {measured_decimal}")
print(f"Grover Execution Time: {end - start:.6f} seconds")

start = time.time()
for i in range(dataset_size):
    if dataset[i] == target:
        found = True
        break
end = time.time()

print(f"Linear Measured Result: {measured_decimal}")
print(f"Linear Execution Time: {end - start:.6f} seconds")

# def diffusion_operator(n):
#     diffuser = QuantumCircuit(n)
#     diffuser.h(range(n))
#     diffuser.x(range(n))
#     diffuser.h(n - 1)
#     diffuser.mcx(list(range(n - 1)), n - 1)
#     diffuser.h(n - 1)
#     diffuser.x(range(n))
#     diffuser.h(range(n))
#     return diffuser

