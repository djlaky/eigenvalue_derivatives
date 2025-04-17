import numpy as np
import copy


test_size = 5

eps = 0.0001

A = np.random.rand(test_size, test_size)
A_psd = np.dot(A, A.transpose())  # Positive semidefinte matrix A_psd

A_psd_inv = np.linalg.inv(A_psd)

trace_A = np.trace(A_psd)
trace_Ainv = np.trace(A_psd_inv)

# Vectors and values for the original matrix
vals_psd, vecs_psd = np.linalg.eig(A_psd)
min_eig = min(vals_psd)

cond = np.linalg.cond(A_psd)

residuals_trace = []
residuals_trace_inv = []

# perturb each direction
for i in range(test_size):
    for j in range(test_size):
        A_psd_new = copy.deepcopy(A_psd)
        A_psd_new[i, j] += eps

        A_psd_new_inv = np.linalg.inv(A_psd_new)

        trace_new = np.trace(A_psd_new)
        trace_new_inv = np.trace(A_psd_new_inv)

        residuals_trace.append((trace_A - trace_new) / eps)
        residuals_trace_inv.append((trace_Ainv - trace_new_inv) / eps)

import matplotlib.pyplot as plt

data = []
for i in range(len(residuals_trace)):
    if np.sign(np.array(residuals_trace[i])) == -np.sign(np.array(residuals_trace_inv[i])):
        data.append(1)
    elif min(np.array(residuals_trace[i]), 0) == 0 and min(np.array(residuals_trace_inv[i]), 0) < 0:
        data.append(0)
    elif min(np.array(residuals_trace[i]), 0) == 0 and max(np.array(residuals_trace_inv[i]), 0) > 0:
        data.append(2)
    else:
        data.append(3)

# plt.plot(range(len(residuals_trace)), residuals_trace, color='black', label='Difference from trace to trace')
# plt.plot(range(len(residuals_trace)), residuals_trace_inv, color='green', label='Difference from trace of inv to trace of inv')
plt.scatter(range(len(residuals_trace)), data)
plt.ylabel("0 for no change and inverse increases, 1 for matched, 2 for no change but inverse decreases", fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()

residuals_trace = []
residuals_trace_inv = []

# perturb each direction
for i in range(test_size):
    for j in range(test_size):
        A_psd_new = copy.deepcopy(A_psd)
        A_psd_new[i, j] += -eps

        A_psd_new_inv = np.linalg.inv(A_psd_new)

        trace_new = np.trace(A_psd_new)
        trace_new_inv = np.trace(A_psd_new_inv)

        residuals_trace.append((trace_A - trace_new) / -eps)
        residuals_trace_inv.append((trace_Ainv - trace_new_inv) / -eps)


data = []
for i in range(len(residuals_trace)):
    if np.sign(np.array(residuals_trace[i])) == -np.sign(np.array(residuals_trace_inv[i])):
        data.append(1)
    elif min(np.array(residuals_trace[i]), 0) == 0 and min(np.array(residuals_trace_inv[i]), 0) < 0:
        data.append(0)
    elif min(np.array(residuals_trace[i]), 0) == 0 and max(np.array(residuals_trace_inv[i]), 0) > 0:
        data.append(2)
    else:
        data.append(3)

# plt.plot(range(len(residuals_trace)), residuals_trace, color='black', label='Difference from trace to trace')
# plt.plot(range(len(residuals_trace)), residuals_trace_inv, color='green', label='Difference from trace of inv to trace of inv')
plt.scatter(range(len(residuals_trace)), data)
plt.ylabel("0 for no change and inverse increases, 1 for matched, 2 for no change but inverse decreases", fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()