import numpy as np
import copy


# Set up stat trackers for the 100 random samples
means_eig = []
means_k = []

stds_eig = []
stds_k = []

for sizes in range(9):
    # Specify what dimension matrix (test_size x test_size)
    test_size = sizes + 2

    # Specify step length    
    eps = 0.0001

    # Set up stat trackers for the 100 random samples
    means_eig_int = []
    means_k_int = []

    # Generate random samples 100 times
    for sample in range(10):
        A = np.random.rand(test_size, test_size)
        A_psd = np.dot(A, A.transpose())  # Positive semidefinte matrix A_psd

        # A_psd = A

        # Vectors and values for the original matrix
        vals_psd, vecs_psd = np.linalg.eig(A_psd)
        min_eig = min(vals_psd)

        cond = np.linalg.cond(A_psd)

        residuals = []
        residuals_k = []

        # perturb each direction
        for i in range(test_size):
            for j in range(test_size):
                A_psd_new = copy.deepcopy(A_psd)
                A_psd_new[i, j] += eps

                # Calculate eigenvalues
                vals, vecs = np.linalg.eig(A_psd_new)
                min_eig_loc = np.argmin(vals)

                min_eig_vec = np.array([vecs[:, min_eig_loc]])  # Make this a matrix so transpose makes sense

                # Calculate the change in minimum eigenvalue
                change_min_eig = vals[min_eig_loc] - min_eig
                
                # Calculate the derivative matrix
                dEigdM = min_eig_vec * np.transpose(min_eig_vec)

                # Compare the FD value versus the "exact derivate"
                residuals.append((dEigdM[i, j] - (change_min_eig / eps)) / abs(dEigdM[i, j]))

                # Test the condition number changes
                max_eig_loc = np.argmax(vals)
                max_eig_vec = np.array([vecs[:, max_eig_loc]])  # Make this a matrix so transpose makes sense

                # new_cond = abs(vals[max_eig_loc] / vals[min_eig_loc])
                new_cond = np.linalg.cond(A_psd_new)

                diff_conds = new_cond - cond

                # Calculating the condition number change formula
                dEigmaxdM = max_eig_vec * np.transpose(max_eig_vec)
                dKdM = 1 / vals[min_eig_loc] * (dEigmaxdM - cond * dEigdM)

                # Compare the FD value versus the "exact derivative"
                residuals_k.append((dKdM[i, j] - diff_conds / eps) / abs(dKdM[i, j]))
            
        # Calculate statistics on residual vectors
        residuals = np.log10(abs(np.array(residuals)))
        residuals_k = np.log10(abs(np.array(residuals_k)))

        means_eig_int.append(np.mean(residuals))
        means_k_int.append(np.mean(residuals_k))

    means_eig.append(np.mean(np.array(means_eig_int)))
    means_k.append(np.mean(np.array(means_k_int)))
    print("Test_size: ")
    print(test_size)
    print("Means length: ")
    print(len(means_eig_int))

    stds_eig.append(np.std(residuals))
    stds_k.append(np.std(residuals_k))

import matplotlib.pyplot as plt

plt.errorbar(range(len(means_eig)), means_eig, yerr=stds_eig, color='black', fmt='o', label='Difference from \'exact\' to F.D. (Min Eig)')
plt.errorbar(range(len(means_k)), means_k, yerr=stds_k, color='green', fmt='o', label='Difference from \'exact\' to F.D. (Condition Number)')
plt.ylabel("log-10(relative error)", fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()