import time
from qiskit_aer import Aer
from qiskit_algorithms import Grover
from qiskit.circuit.library import PhaseOracle
from qiskit.utils import QuantumInstance

def build_oracle(dataset, target):
    # Create a Boolean expression for the oracle
    index = dataset.index(target)
    bin_index = format(index, f'0{len(dataset).bit_length()}b')
    expr = " & ".join([f"{'' if bit == '1' else '~'}x{i}" for i, bit in enumerate(bin_index)])
    return expr

def grover_search(dataset, target):
    if target not in dataset:
        return {
            "found": False,
            "time": 0.0
        }

    start = time.time()

    oracle_expr = build_oracle(dataset, target)
    oracle = PhaseOracle(oracle_expr)

    backend = Aer.get_backend("qasm_simulator")
    quantum_instance = QuantumInstance(backend, shots=1024)

    grover = Grover(oracle=oracle, quantum_instance=quantum_instance)
    result = grover.run()

    end = time.time()

    found = result.top_measurement is not None

    return {
        "found": found,
        "time": round(end - start, 6)
    }
