import qucumber
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram, plot_bloch_multivector

def qft_rotations(circuit, n):
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cu1(np.pi/2**(n-qubit), qubit, n)
    qft_rotations(circuit, n)

def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n):
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

n_qubits = 4
qc = QuantumCircuit(n_qubits)
qft(qc,n_qubits)
print(qc)

backend = Aer.get_backend('statevector_simulator')
result = execute(qc, backend=backend).result()    
statevector = result.get_statevector(qc)
print("State vector: {}".format(statevector))
np.savetxt("phi_fourier.txt", statevector, fmt='%1.5f %1.5f')