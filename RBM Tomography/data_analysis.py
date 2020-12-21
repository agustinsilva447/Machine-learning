import numpy as np
import matplotlib.pyplot as plt

from qucumber.nn_states import PositiveWaveFunction
from qucumber.callbacks import MetricEvaluator
import qucumber.utils.training_statistics as ts
import qucumber.utils.data as data
import qucumber

psi_path = "phi_fourier.txt"
train_path = "data_fourier.txt"
train_data, true_psi = data.load_data(train_path, psi_path)

nv = train_data.shape[-1]
nh = nv * 10
nn_state = PositiveWaveFunction(num_visible=nv, num_hidden=nh, gpu=False)

def psi_coefficient(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][0] / norm

pbs = 100
nbs = pbs
epochs = 500
lr = 0.005
k = 10
period = 50
space = nn_state.generate_hilbert_space()

callbacks = [
    MetricEvaluator(
        period,
        {"Fidelity": ts.fidelity, "KL": ts.KL, "A_Ψrbm_0": psi_coefficient},
        target=true_psi,
        verbose=True,
        space=space,
        A=1.0,
    )
]

nn_state.fit(
    train_data,
    epochs=epochs,
    pos_batch_size=pbs,
    neg_batch_size=nbs,
    lr=lr,
    k=k,
    callbacks=callbacks,
    time=True,
)

final_state_vector = np.array(nn_state.psi(space)[0] / nn_state.compute_normalization(space).sqrt_())
np.savetxt("phi_fourier_RBM.txt", final_state_vector, fmt='%1.5f')
print("Estado cuántico final:", final_state_vector)

plt.close()
fig1 = plt.bar(np.arange(np.power(2,nv)),np.power(final_state_vector.real,2))
plt.xlabel("Probabilities")
plt.ylim(0, 0.01)
plt.savefig("state_RBM.jpg")

"""
fidelities = callbacks[0].Fidelity
KLs = callbacks[0]["KL"]
coeffs = callbacks[0]["A_Ψrbm_"]
epoch = np.arange(period, epochs + 1, period)

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 3))
ax = axs[0]
ax.plot(epoch, fidelities, "o", color="C0", markeredgecolor="black")
ax.set_ylabel("Fidelity")
ax.set_xlabel("Epoch")

ax = axs[1]
ax.plot(epoch, KLs, "o", color="C1", markeredgecolor="black")
ax.set_ylabel("KL Divergence")
ax.set_xlabel("Epoch")

ax = axs[2]
ax.plot(epoch, coeffs, "o", color="C2", markeredgecolor="black")
ax.set_ylabel("A_RBM[0]")
ax.set_xlabel("Epoch")

plt.savefig("RBM_analysis.jpg")
nn_state.save("saved_params.pt")
"""