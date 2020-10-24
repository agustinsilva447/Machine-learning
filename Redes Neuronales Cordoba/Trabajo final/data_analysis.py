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

epochs = 500
pbs = 100
nbs = pbs
lr = 0.0075
k = 10

def psi_coefficient0(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][0] / norm
def psi_coefficient1(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][1] / norm
def psi_coefficient2(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][2] / norm
def psi_coefficient3(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][3] / norm
def psi_coefficient4(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][4] / norm
def psi_coefficient5(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][5] / norm
def psi_coefficient6(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][6] / norm
def psi_coefficient7(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][7] / norm
def psi_coefficient8(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][8] / norm
def psi_coefficient9(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][9] / norm
def psi_coefficient10(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][10] / norm
def psi_coefficient11(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][11] / norm
def psi_coefficient12(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][12] / norm
def psi_coefficient13(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][13] / norm
def psi_coefficient14(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][14] / norm
def psi_coefficient15(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][15] / norm
def psi_coefficient16(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][16] / norm
def psi_coefficient17(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][17] / norm
def psi_coefficient18(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][18] / norm
def psi_coefficient19(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][19] / norm
def psi_coefficient20(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][20] / norm
def psi_coefficient21(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][21] / norm
def psi_coefficient22(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][22] / norm
def psi_coefficient23(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][23] / norm
def psi_coefficient24(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][24] / norm
def psi_coefficient25(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][25] / norm
def psi_coefficient26(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][26] / norm
def psi_coefficient27(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][27] / norm
def psi_coefficient28(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][28] / norm
def psi_coefficient29(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][29] / norm
def psi_coefficient30(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][30] / norm
def psi_coefficient31(nn_state, space, A, **kwargs):
    norm = nn_state.compute_normalization(space).sqrt_()
    return A * nn_state.psi(space)[0][31] / norm

period = 10
space = nn_state.generate_hilbert_space()

callbacks = [
    MetricEvaluator(
        period,
        {"Fidelity": ts.fidelity, "KL": ts.KL, 
        "\nA_Ψrbm_0": psi_coefficient0, "A_Ψrbm_1": psi_coefficient1, "A_Ψrbm_2": psi_coefficient2, "A_Ψrbm_3": psi_coefficient3, 
        "\nA_Ψrbm_4": psi_coefficient4, "A_Ψrbm_5": psi_coefficient5, "A_Ψrbm_6": psi_coefficient6, "A_Ψrbm_7": psi_coefficient7, 
        "\nA_Ψrbm_8": psi_coefficient8, "A_Ψrbm_9": psi_coefficient9, "A_Ψrbm_10": psi_coefficient10, "A_Ψrbm_11": psi_coefficient11, 
        "\nA_Ψrbm_12": psi_coefficient12, "A_Ψrbm_13": psi_coefficient13, "A_Ψrbm_14": psi_coefficient14, "A_Ψrbm_15": psi_coefficient15,
        "\nA_Ψrbm_16": psi_coefficient16, "A_Ψrbm_17": psi_coefficient17, "A_Ψrbm_18": psi_coefficient18, "A_Ψrbm_19": psi_coefficient19, 
        "\nA_Ψrbm_20": psi_coefficient20, "A_Ψrbm_21": psi_coefficient21, "A_Ψrbm_22": psi_coefficient22, "A_Ψrbm_23": psi_coefficient23, 
        "\nA_Ψrbm_24": psi_coefficient24, "A_Ψrbm_25": psi_coefficient25, "A_Ψrbm_26": psi_coefficient26, "A_Ψrbm_27": psi_coefficient27, 
        "\nA_Ψrbm_28": psi_coefficient28, "A_Ψrbm_29": psi_coefficient29, "A_Ψrbm_30": psi_coefficient30, "A_Ψrbm_31": psi_coefficient31},
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

"""
# Note that the key given to the *MetricEvaluator* must be
# what comes after callbacks[0].
fidelities = callbacks[0].Fidelity

# Alternatively, we can use the usual dictionary/list subsripting
# syntax. This is useful in cases where the name of the
# metric contains special characters or spaces.
KLs = callbacks[0]["KL"]
coeffs = callbacks[0]["\nA_Ψrbm_0"]
epoch = np.arange(period, epochs + 1, period)

# Plotting
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

final_state_vector = np.array([callbacks[0]["\nA_Ψrbm_0"][-1],
                                callbacks[0]["A_Ψrbm_1"][-1],
                                callbacks[0]["A_Ψrbm_2"][-1],
                                callbacks[0]["A_Ψrbm_3"][-1],
                                callbacks[0]["\nA_Ψrbm_4"][-1],
                                callbacks[0]["A_Ψrbm_5"][-1],
                                callbacks[0]["A_Ψrbm_6"][-1],
                                callbacks[0]["A_Ψrbm_7"][-1],
                                callbacks[0]["\nA_Ψrbm_8"][-1],
                                callbacks[0]["A_Ψrbm_9"][-1],
                                callbacks[0]["A_Ψrbm_10"][-1],
                                callbacks[0]["A_Ψrbm_11"][-1],
                                callbacks[0]["\nA_Ψrbm_12"][-1],
                                callbacks[0]["A_Ψrbm_13"][-1],
                                callbacks[0]["A_Ψrbm_14"][-1],
                                callbacks[0]["A_Ψrbm_15"][-1],
                                callbacks[0]["\nA_Ψrbm_16"][-1],
                                callbacks[0]["A_Ψrbm_17"][-1],
                                callbacks[0]["A_Ψrbm_18"][-1],
                                callbacks[0]["A_Ψrbm_19"][-1],
                                callbacks[0]["\nA_Ψrbm_20"][-1],
                                callbacks[0]["A_Ψrbm_21"][-1],
                                callbacks[0]["A_Ψrbm_22"][-1],
                                callbacks[0]["A_Ψrbm_23"][-1],
                                callbacks[0]["\nA_Ψrbm_24"][-1],
                                callbacks[0]["A_Ψrbm_25"][-1],
                                callbacks[0]["A_Ψrbm_26"][-1],
                                callbacks[0]["A_Ψrbm_27"][-1],
                                callbacks[0]["\nA_Ψrbm_28"][-1],
                                callbacks[0]["A_Ψrbm_29"][-1],
                                callbacks[0]["A_Ψrbm_30"][-1],
                                callbacks[0]["A_Ψrbm_31"][-1]])

np.savetxt("phi_fourier_RBM.txt", final_state_vector, fmt='%1.5f')

plt.close()
fig1 = plt.bar(np.arange(np.power(2,nv)),np.power(final_state_vector.real,2))
plt.xlabel("Probabilities")
plt.ylim(0, 0.05)
plt.savefig("state_RBM.jpg")

print("Estado cuántico final:", final_state_vector)