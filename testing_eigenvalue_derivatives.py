import numpy as np
import copy


test_size = 10

eps = 0.0001

A = np.random.rand(test_size, test_size)
A_psd = np.dot(A, A.transpose())  # Positive semidefinte matrix A_psd

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

        print("Original Eigenvalues and vectors: ")
        print(vals_psd)
        print(vecs_psd)
        print("New Eigenvalues and vectors: ")
        print(vals)
        print(vecs)

        print("Minimum vector: ")
        print(min_eig_vec)

        # Compare the FD value versus the "exact derivate"
        print("dEigDM component: ")
        print(dEigdM[i, j])
        print("Change in minimum eigenvalue, finite difference: ")
        print(change_min_eig / eps)
        print("Overall difference between both figures: ")
        print(dEigdM[i, j] - (change_min_eig / eps))

        residuals.append((dEigdM[i, j] - (change_min_eig / eps)) / abs(dEigdM[i, j]))

        # Test the condition number changes
        max_eig_loc = np.argmax(vals)
        max_eig_vec = np.array([vecs[:, max_eig_loc]])  # Make this a matrix so transpose makes sense

        new_cond = vals[max_eig_loc] / vals[min_eig_loc]

        diff_conds = new_cond - cond

        # Calculating the condition number change formula
        dEigmaxdM = max_eig_vec * np.transpose(max_eig_vec)
        dKdM = 1 / vals[min_eig_loc] * (dEigmaxdM - cond * dEigdM)

        # Compare the FD value versus the "exact derivative"
        print("dKdM component: ")
        print(dKdM[i, j])
        print("Change in condition number, finite difference: ")
        print(diff_conds / eps)
        print("Overall difference between both figures: ")
        print((dKdM[i, j] - diff_conds / eps) / abs(dKdM[i, j]))

        residuals_k.append((dKdM[i, j] - diff_conds / eps) / abs(dKdM[i, j]))


import matplotlib.pyplot as plt


plt.plot(range(len(residuals)), np.log(abs(np.array(residuals))), color='black', label='Difference from \'exact\' to F.D. (Min Eig)')
plt.plot(range(len(residuals)), np.log(abs(np.array(residuals_k))), color='green', label='Difference from \'exact\' to F.D. (Condition Number)')
print(np.log(abs(np.array(residuals))))
print(np.log(abs(np.array(residuals_k))))
plt.ylabel("log-10(relative error)", fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()