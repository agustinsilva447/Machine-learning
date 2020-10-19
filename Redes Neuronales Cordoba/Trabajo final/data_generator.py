import qucumber
import numpy as np
import matplotlib.pyplot as plt
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
qc = QuantumCircuit(n_qubits,n_qubits)
qc.h(0)
qc.h(1)
qft(qc,n_qubits)

backend = Aer.get_backend('statevector_simulator')
result = execute(qc, backend=backend).result()    
statevector = result.get_statevector(qc)
print("State vector: {}".format(statevector.real))

np.savetxt("phi_fourier.txt", statevector, fmt='%1.5f %1.5f')
fig1 = plt.bar(np.arange(np.power(2,n_qubits)),np.power(statevector.real,2))
plt.xlabel("Probabilities")
plt.savefig("state.jpg")

qc.measure(np.arange(n_qubits), np.arange(n_qubits)) 
num_medidas = 1000
backend = Aer.get_backend('qasm_simulator')
result = execute(qc, backend=backend, shots=num_medidas).result()
counts = result.get_counts(qc)
print("Counts: {}".format(counts))

mediciones = np.zeros([num_medidas,1])
num_medidas = 0
for medida, cantidad in counts.items():
    mediciones[num_medidas:num_medidas + cantidad] = medida
    num_medidas += cantidad
np.random.shuffle(mediciones)

np.savetxt("data_fourier.txt", mediciones, fmt='%04i')

fin = open("data_fourier.txt", "rt")
data = fin.read()
data = data.replace('0', '0 ')
data = data.replace('1', '1 ')
fin.close()
fin = open("data_fourier.txt", "wt")
fin.write(data)
fin.close()

fig2 = plot_histogram(counts)
fig2.savefig("data.jpg")

print(qc)