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

test_size = 2

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

print(A_psd_inv)

residuals_det = []
residuals_eig = []
residuals_k = []

count = 0

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
                A_psd_new_1[k, l] += eps

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

                print(det1, det2, det3, det4)

                # Calculate condition numbers for finite difference (ME-opt)
                cond1 = np.linalg.cond(A_psd_new_1)
                cond2 = np.linalg.cond(A_psd_new_2)
                cond3 = np.linalg.cond(A_psd_new_3)
                cond4 = np.linalg.cond(A_psd_new_4)

                print(cond1, cond2, cond3, cond4)

                # Calculate eigenvalues and vectors (E-opt)
                min_eig1, _ = get_eigenvalue_and_vector(A_psd_new_1, "min")
                print(_)
                min_eig2, _ = get_eigenvalue_and_vector(A_psd_new_2, "min")
                print(_)
                min_eig3, _ = get_eigenvalue_and_vector(A_psd_new_3, "min")
                print(_)
                min_eig4, _ = get_eigenvalue_and_vector(A_psd_new_4, "min")
                print(_)

                print(min_eig1, min_eig2, min_eig3, min_eig4)


                #####################################
                # Calculating the FD approximations 
                hess_det = (det1 - det2 - det3 + det4) / (4 * eps**2)
                hess_cond = (cond1 - cond2 - cond3 + cond4) / (4 * eps**2)
                hess_eig = (min_eig1 - min_eig2 - min_eig3 + min_eig4) / (4 * eps**2)

                #      End FD approximations
                #####################################

                
                #####################################
                # Calculating exact Hessian value 

                # Determinant formula comes from 1/2 * (dMinv/dM + dMinv^T/dM)
                # which is -1/4*(Minv[i, k]Minv[l, j] + Minv[i, l]Minv[k, j]
                #                + Minv[j, k]Minv[l, i] + Minv[j, l]Minv[k, i])
                # https://www.et.byu.edu/~vps/ME505/AAEM/V5-07.pdf
                exact_det = -1/4*(A_psd_inv[i, k] * A_psd_inv[l, j] +
                                  A_psd_inv[i, l] * A_psd_inv[k, j] +
                                  A_psd_inv[j, k] * A_psd_inv[l, i] +
                                  A_psd_inv[j, l] * A_psd_inv[k, i]
                )

                # Fixed the formula? Not sure why there's a 
                # transpose on the inner elements.
                exact_det = -(A_psd_inv[i, l] * A_psd_inv[k, j])

                # Eigenvalue formula comes from:
                # https://cs.nyu.edu/~overton/papers/pdffiles/eighess.pdf
                # Which was rederived with help from Liviu
                # 2 * sum[s != 0]((v_0^T dM/dMij v_s v_0^T dM/dMkl v_s) / (eig_val_0 - eig_val_s))
                # Which when reduced will be...
                # 2 * sum[s != 0]((v_0[i] * v_s[j] * v_0[k] * v_s[l]]) / (eig_val_0 - eig_val_s))
                # Where the sum is over all other eigenvalues
                # Proof (apparently) chapter 2 section 6 of Kato's
                # Linear perturbation theory book
                exact_eig = 0
                for curr_eig in range(len(all_eig_vals)):
                    if curr_eig == min_eig_loc:
                        continue
                    # exact_eig += 2 * (min_eig_vec[0, i] * 
                    #                   all_eig_vecs[curr_eig, j] * 
                    #                   min_eig_vec[0, k] * 
                    #                   all_eig_vecs[curr_eig, l]) / (min_eig - all_eig_vals[curr_eig])
                    # exact_eig += 1 * (min_eig_vec[0, j] * 
                    #                   all_eig_vecs[curr_eig, i] * 
                    #                   min_eig_vec[0, l] * 
                    #                   all_eig_vecs[curr_eig, k]) / (min_eig - all_eig_vals[curr_eig])
                    exact_eig += 1 * (min_eig_vec[0, i] * 
                                      all_eig_vecs[j, curr_eig] * 
                                      min_eig_vec[0, k] * 
                                      all_eig_vecs[l, curr_eig]) / (min_eig - all_eig_vals[curr_eig])
                    exact_eig += 1 * (min_eig_vec[0, i] * 
                                      all_eig_vecs[j, curr_eig] * 
                                      min_eig_vec[0, k] * 
                                      all_eig_vecs[l, curr_eig]) / (min_eig - all_eig_vals[curr_eig])
                    # exact_eig += 1 * (min_eig_vec[0, j] * 
                    #                   all_eig_vecs[curr_eig, i] * 
                    #                   min_eig_vec[0, l] * 
                    #                   all_eig_vecs[curr_eig, k]) / (min_eig - all_eig_vals[curr_eig])
                
                # Condition number formula was derived by myself
                # using product and chain rule and using the
                # definition above for second-order eigenvalue derivatives
                # Formula is very long, so I won't include it here
                
                # Term 1 is 1 / eig_0 ** 2 * deig_0/dMkl * deig_N/dMij
                cond_term_1 = 1 / (min_eig ** 2) * (min_eig_vec[0, l] * min_eig_vec[0, k]) * (max_eig_vec[0, j] * max_eig_vec[0, i])
                
                # Term 2 is 1 / eig_0 * second_deriv_max_eig
                exact_max_eig = 0
                for curr_eig in range(len(all_eig_vals)):
                    if curr_eig == max_eig_loc:
                        continue
                    exact_max_eig += 2 * (max_eig_vec[0, j] * 
                                      all_eig_vecs[i, curr_eig] * 
                                      max_eig_vec[0, l] * 
                                      all_eig_vecs[k, curr_eig]) / (max_eig - all_eig_vals[curr_eig])

                cond_term_2 = 1 / min_eig * exact_max_eig
                
                # Term 3 is 1 / eig_0 * dcond/dMkl * deig_0/dMij
                cond_term_3 = 1 / min_eig * (1 / min_eig * 
                                             (max_eig_vec[0, l] * max_eig_vec[0, k] - 
                                              cond * min_eig_vec[0, l] * min_eig_vec[0, k])) * (min_eig_vec[0, j] * min_eig_vec[0, i])
                
                # Term 4 is cond / eig_0 ** 2 * deig_0/dMkl * deig_0/dMij
                cond_term_4 = cond / (min_eig ** 2) * (min_eig_vec[0, l] * min_eig_vec[0, k]) * (min_eig_vec[0, j] * min_eig_vec[0, i])

                # Term 5 is cond / eig_0 * second_deriv_min_eig <-- already computed as "exact_eig"
                cond_term_5 = cond / min_eig * exact_eig

                # Combine everything at the end
                exact_cond = -cond_term_1 + cond_term_2 - cond_term_3 + cond_term_4 - cond_term_5

                # End exact Hessian value computation
                ######################################

                #####################################
                # Calculating the residuals!

                print("\n\nIteration ({}, {}, {}, {})".format(i, j, k, l))
                print("Or: {}\n\n".format(count))

                print(A_psd_new_1 - A_psd)
                print(A_psd_new_2 - A_psd)
                print(A_psd_new_3 - A_psd)
                print(A_psd_new_4 - A_psd)

                print("Determinant: ")
                print(hess_det)
                print(exact_det)
                print("~~~~~~~~~~~~~")

                print("Min eig: ")
                print(hess_eig)
                print(exact_eig)
                print("~~~~~~~~~~~~~")

                print("Condition number: ")
                print(hess_cond)
                print(exact_cond)
                print("~~~~~~~~~~~~~")


                residuals_det.append((exact_det - hess_det) / abs(exact_det))
                residuals_eig.append((exact_eig - hess_eig) / abs(exact_eig))
                residuals_k.append((exact_cond - hess_cond) / abs(exact_cond))

                # Maybe add printing tests
                # Compare the FD value versus the "exact derivative"
                # print("dKdM component: ")
                # print(dKdM[i, j])
                # print("Change in condition number, finite difference: ")
                # print(diff_conds / eps)
                # print("Overall difference between both figures: ")
                # print((dKdM[i, j] - diff_conds / eps) / abs(dKdM[i, j]))

                # End residual computation!!!
                ######################################
                
                count += 1


import matplotlib.pyplot as plt#

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