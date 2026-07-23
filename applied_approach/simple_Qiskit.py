from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

qreg = QuantumRegister(1, "q")
creg = ClassicalRegister(1, "c")

circuit = QuantumCircuit(qreg, creg)
circuit.x(qreg[0])  # NOT
circuit.measure(qreg, creg)  # Measurement

print("Circuit:")
print(circuit.draw())

backend = AerSimulator()
job = backend.run(circuit, shots=10)
result = job.result()

print(result.get_counts())
