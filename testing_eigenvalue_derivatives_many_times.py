import numpy as np
import copy


# Set up stat trackers for the 100 random samples
means_eig = []
means_k = []
means_logk = []
means_a = []
means_d = []

stds_eig = []
stds_k = []
stds_logk = []
stds_a = []
stds_d = []

for sizes in range(9):
    # Specify what dimension matrix (test_size x test_size)
    test_size = sizes + 2

    # Specify step length    
    eps = 0.0001

    # Set up stat trackers for the 100 random samples
    means_eig_int = []
    means_k_int = []
    means_logk_int = []
    means_a_int = []
    means_d_int = []

    # Generate random samples 100 times
    for sample in range(100):
        A = np.random.rand(test_size, test_size)
        A_psd = np.dot(A, A.transpose()) + (test_size * 2) * np.ones(test_size)  # Positive semidefinte matrix A_psd

        # A_psd = A

        # Vectors and values for the original matrix
        vals_psd, vecs_psd = np.linalg.eig(A_psd)
        min_eig = min(vals_psd)

        det = np.log(np.linalg.det(A_psd))

        cond = np.linalg.cond(A_psd)

        trace_inv = np.trace(np.linalg.inv(A_psd))

        residuals = []
        residuals_k = []
        residuals_logk = []
        residuals_a = []
        residuals_d = []

        # perturb each direction
        for i in range(test_size):
            for j in range(test_size):
                A_psd_new = copy.deepcopy(A_psd)
                A_psd_new[i, j] += eps

                # Calculate eigenvalues
                vals, vecs = np.linalg.eig(A_psd_new)
                min_eig_loc = np.argmin(vals)

                min_eig_vec = np.array([vecs[:, min_eig_loc]])  # Make this a matrix so transpose makes sense

                # Calculate the inverse and trace
                A_psd_new_inv = np.linalg.inv(A_psd_new)

                # Calculate the change in minimum eigenvalue
                change_min_eig = vals[min_eig_loc] - min_eig
                
                # Calculate the derivative matrix
                dEigdM = min_eig_vec * np.transpose(min_eig_vec)

                # Compare the FD value versus the "exact derivative"
                residuals.append((dEigdM[i, j] - (change_min_eig / eps)) / abs(dEigdM[i, j]))

                # Test the condition number changes
                max_eig_loc = np.argmax(vals)
                max_eig_vec = np.array([vecs[:, max_eig_loc]])  # Make this a matrix so transpose makes sense

                # new_cond = abs(vals[max_eig_loc] / vals[min_eig_loc])
                new_cond = np.linalg.cond(A_psd_new)

                diff_conds = new_cond - cond
                diff_logconds = np.log(new_cond) - np.log(cond)

                # Calculating the condition number change formula
                dEigmaxdM = max_eig_vec * np.transpose(max_eig_vec)
                dKdM = 1 / vals[min_eig_loc] * (dEigmaxdM - cond * dEigdM)
                dlnKdM = 1 / vals[max_eig_loc] * dEigmaxdM - 1 / vals[min_eig_loc] * dEigdM

                # Compare the FD value versus the "exact derivative"
                residuals_k.append((dKdM[i, j] - diff_conds / eps) / abs(dKdM[i, j]))
                residuals_logk.append((dlnKdM[i, j] - diff_logconds / eps) / abs(dlnKdM[i, j]))

                # Calculating the change in trace
                change_trace = np.trace(A_psd_new_inv) - trace_inv

                # Calculating the derivative value
                dAdM = -A_psd_new_inv @ A_psd_new_inv

                # compare the FD versus "exact derivative" for trace
                residuals_a.append((dAdM[i, j] - change_trace / eps) / abs(dAdM[i, j]))

                # New determinant
                new_det = np.log(np.linalg.det(A_psd_new))
                diff_dets = new_det - det

                # Calculating determinant change formula
                dDdM = A_psd_new_inv
                residuals_d.append((dDdM[i, j] - diff_dets / eps) / abs(dDdM[i, j]))
            
        # Calculate statistics on residual vectors
        residuals = np.log10(abs(np.array(residuals)))
        residuals_k = np.log10(abs(np.array(residuals_k)))
        residuals_logk = np.log10(abs(np.array(residuals_logk)))
        residuals_a = np.log10(abs(np.array(residuals_a)))
        residuals_d = np.log10(abs(np.array(residuals_d)))

        means_eig_int.append(np.mean(residuals))
        means_k_int.append(np.mean(residuals_k))
        means_logk_int.append(np.mean(residuals_logk))
        means_a_int.append(np.mean(residuals_a))
        means_d_int.append(np.mean(residuals_d))

    means_eig.append(np.mean(np.array(means_eig_int)))
    means_k.append(np.mean(np.array(means_k_int)))
    means_logk.append(np.mean(np.array(means_logk_int)))
    means_a.append(np.mean(np.array(means_a_int)))
    means_d.append(np.mean(np.array(means_d_int)))
    print("Test_size: ")
    print(test_size)
    print("Means length: ")
    print(len(means_eig_int))

    stds_eig.append(np.std(residuals))
    stds_k.append(np.std(residuals_k))
    stds_logk.append(np.std(residuals_logk))
    stds_a.append(np.std(residuals_a))
    stds_d.append(np.std(residuals_d))

import matplotlib.pyplot as plt

plt.errorbar(range(2, len(means_eig) + 2), means_eig, yerr=stds_eig, capsize=3, color='black', fmt='o', label='Difference from \'exact\' to F.D. (Min Eig)')
plt.errorbar(range(2, len(means_k) + 2), means_k, yerr=stds_k, capsize=3, color='green', fmt='o', label='Difference from \'exact\' to F.D. (Condition Number)')
plt.errorbar(range(2, len(means_k) + 2), means_logk, yerr=stds_logk, capsize=3, color='purple', fmt='o', label='Difference from \'exact\' to F.D. (Log Condition Number)')
plt.errorbar(range(2, len(means_a) + 2), means_a, yerr=stds_a, capsize=3, color='orange', fmt='o', label='Difference from \'exact\' to F.D. (A-opt)')
plt.errorbar(range(2, len(means_d) + 2), means_d, yerr=stds_d, capsize=3, color='blue', fmt='o', label='Difference from \'exact\' to F.D. (D-opt)')

plt.ylabel("log-10(relative error)", fontsize=20)
plt.xlabel("Square dimension of Matrices", fontsize=20)
plt.legend()
plt.tight_layout()
plt.savefig("testing_first_derivatives_many_times.png", format="png", dpi=600)
plt.show()