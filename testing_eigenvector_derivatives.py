import numpy as np
import copy


def get_eigenvalue_and_vector(M, option='min'):
    vals, vecs = np.linalg.eig(M)
    min_eig_loc = np.argmin(vals)
    max_eig_loc = np.argmax(vals)

    min_eig_vec = np.array([vecs[:, min_eig_loc]])  # Make this a matrix so transpose makes sense
    max_eig_vec = np.array([vecs[:, max_eig_loc]])  # Make this a matrix so transpose makes sense

    if option == 'min':
        return vals[min_eig_loc], min_eig_vec
    elif option == 'max':
        return vals[max_eig_loc], max_eig_vec
    elif option == 'both':
        return vals[min_eig_loc], min_eig_vec, vals[max_eig_loc], max_eig_vec

test_size = 3

eps = 0.00001

np.random.seed(1022)

A = np.random.rand(test_size, test_size)
A_psd = np.dot(A, A.transpose())  # Positive semidefinte matrix A_psd

A_psd_inv = np.linalg.pinv(A_psd)

print(A_psd)

# Vectors and values for the original matrix
min_eig, min_eig_vec, max_eig, max_eig_vec = get_eigenvalue_and_vector(A_psd, "both")

all_eig_vals, all_eig_vecs = np.linalg.eig(A_psd)
min_eig_loc = np.argmin(all_eig_vals)
max_eig_loc = np.argmax(all_eig_vals)

cond = np.linalg.cond(A_psd)

residuals_det = []
residuals_eig = []
residuals_k = []

# perturb each direction
for i in range(test_size):
    for j in range(test_size):
        for k in range(test_size):
            for l in range(test_size):
                # Make copies of A_psd to ensure
                # value and not reference.....
                A_psd_new_1 = copy.deepcopy(A_psd)
                A_psd_new_2 = copy.deepcopy(A_psd)
                A_psd_new_3 = copy.deepcopy(A_psd)
                A_psd_new_4 = copy.deepcopy(A_psd)

                # Need 4 perturbations to cover the
                # formula H[i, j] = [(A + eps (both))
                # + (A +/- eps one each)
                # + (A -/+ eps one each)
                # + (A - eps (both))] / (4*eps**2)
                A_psd_new_1[i, j] += eps
                # A_psd_new_1[k, l] += eps

                A_psd_new_2[i, j] += eps
                A_psd_new_2[k, l] += -eps

                A_psd_new_3[i, j] += -eps
                A_psd_new_3[k, l] += eps

                A_psd_new_4[i, j] += -eps
                A_psd_new_4[k, l] += -eps

                # Calculate log-determinant for finite difference (D-opt)
                _, det1 = np.linalg.slogdet(A_psd_new_1)
                _, det2 = np.linalg.slogdet(A_psd_new_2)
                _, det3 = np.linalg.slogdet(A_psd_new_3)
                _, det4 = np.linalg.slogdet(A_psd_new_4)

                # Calculate condition numbers for finite difference (ME-opt)
                cond1 = np.linalg.cond(A_psd_new_1)
                cond2 = np.linalg.cond(A_psd_new_2)
                cond3 = np.linalg.cond(A_psd_new_3)
                cond4 = np.linalg.cond(A_psd_new_4)

                # Calculate eigenvalues and vectors (E-opt)
                min_eig1, min_eig_vec1 = get_eigenvalue_and_vector(A_psd_new_1, "min")
                min_eig2, _ = get_eigenvalue_and_vector(A_psd_new_2, "min")
                min_eig3, _ = get_eigenvalue_and_vector(A_psd_new_3, "min")
                min_eig4, _ = get_eigenvalue_and_vector(A_psd_new_4, "min")

                # Checking eigenvector derivatives
                exact_eig = np.zeros(test_size)
                for curr_eig in range(len(all_eig_vals)):
                    if curr_eig == min_eig_loc:
                        continue
                    exact_eig += (min_eig_vec[0, i] * all_eig_vecs[curr_eig, j]) * all_eig_vecs[curr_eig, :] / (min_eig - all_eig_vals[curr_eig])
                
                print("(i, j): ({}, {})".format(i, j))
                print("Exact eigenvector derivative: ")
                print(exact_eig)
                print("\nF.D. eigenvector derivative: ")
                print((min_eig_vec1 - min_eig_vec) / eps)
                print("~~~~~~~~~~~")


import matplotlib.pyplot as plt

print(residuals_det)
print(residuals_eig)
print(residuals_k)

plt.plot(range(len(residuals_det)), np.log(abs(np.array(residuals_det))), color='orange', label='Difference from \'exact\' to F.D. (log det)')
plt.plot(range(len(residuals_det)), np.log(abs(np.array(residuals_eig))), color='black', label='Difference from \'exact\' to F.D. (Min Eig)')
plt.plot(range(len(residuals_det)), np.log(abs(np.array(residuals_k))), color='green', label='Difference from \'exact\' to F.D. (Condition Number)')
print(np.log(abs(np.array(residuals_det))))
print(np.log(abs(np.array(residuals_eig))))
print(np.log(abs(np.array(residuals_k))))
plt.ylabel("log-10(relative error)", fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()