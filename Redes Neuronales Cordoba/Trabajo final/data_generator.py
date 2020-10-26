import numpy as np
import matplotlib.pyplot as plt

from qiskit import IBMQ
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
from qiskit.providers.ibmq import least_busy

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

n_qubits = 8
qc = QuantumCircuit(n_qubits,n_qubits)
qft(qc,n_qubits)

backend = Aer.get_backend('statevector_simulator')
result = execute(qc, backend=backend).result()    
statevector = result.get_statevector(qc)
print("State vector: {}".format(statevector.real))

np.savetxt("phi_fourier.txt", statevector, fmt='%1.5f %1.5f')
fig1 = plt.bar(np.arange(np.power(2,n_qubits)),np.power(statevector.real,2))
plt.xlabel("Probabilities")
plt.ylim(0, 0.01)
plt.savefig("state.jpg")

qc.measure(np.arange(n_qubits), np.arange(n_qubits)) 
num_medidas = 1500

backend1 = Aer.get_backend('qasm_simulator')
result = execute(qc, backend=backend1, shots=num_medidas).result()
counts = result.get_counts(qc)
print("Counts: {}".format(counts))

"""
IBMQ.save_account('bc0221c2c435408a718744a0e3388325cd4241375e15beefb8909a759a28201db7c6a425e8d07e09e54105e373aceccb175fa8eb46700b4db88ac50dbdc6fa84')
provider = IBMQ.load_account()
backend2 = least_busy(provider.backends(filters=lambda b: b.configuration().n_qubits >= n_qubits and not b.configuration().simulator and b.status().operational==True))
result = execute(qc, backend=backend2, shots=num_medidas).result()
counts = result.get_counts(qc)
print("Counts: {}".format(counts))
"""

mediciones = np.zeros([num_medidas,1])
num_medidas = 0
for medida, cantidad in counts.items():
    mediciones[num_medidas:num_medidas + cantidad] = medida
    num_medidas += cantidad
np.random.shuffle(mediciones)
np.savetxt("data_fourier.txt", mediciones, fmt='%0{}i'.format(n_qubits))

fin = open("data_fourier.txt", "rt")
data = fin.read()
data = data.replace('0', '0 ')
data = data.replace('1', '1 ')
fin.close()
fin = open("data_fourier.txt", "wt")
fin.write(data)
fin.close()

result = counts.values()
data1 = list(result) 
numpyArray = np.array(data1) / num_medidas

if (np.arange(np.power(2,n_qubits)).shape == numpyArray.shape):
    plt.close()
    fig2 = plt.bar(np.arange(np.power(2,n_qubits)),numpyArray)
    plt.xlabel("Probabilities")
    plt.ylim(0, 0.01)
    plt.savefig("state_hist.jpg")
    print(qc)
else:
    print("\nHay estados con que no tienen mediciones :(. Repetir ejecución!")