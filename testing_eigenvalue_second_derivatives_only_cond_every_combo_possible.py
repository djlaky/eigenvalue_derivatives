import numpy as np
import copy
import itertools
import time as t

# from testing_eigenvalue_derivatives_many_times import residuals_logk


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
#cond = max_eig / min_eig

permute = list(itertools.permutations([0, 1, 2, 3]))

# (0, 1, 2, 3), (1, 0, 3, 2), (0, 1, 3, 2), (1, 0, 2, 3)
permute1 = [(0, 1, 2, 3), ]
permute2 = [(1, 0, 3, 2), ]
permute3 = [(0, 1, 3, 2), ]
permute4 = [(1, 0, 2, 3), ]
small_list = [1, ]

residuals_det = []
residuals_eig = []
residuals_k = []
residuals_logk = []
store_orders = []
i_def = 0
count = 0
curr_percent = 0

total_num = len(permute) ** 4

t_begin = t.time()

# perturb each direction
for order1 in permute:  # Second Derivative one, one (max eig)
    for order2 in permute:  # Second Derivative one, two (max eig)
        for order3 in permute:  # Second Derivative two, one (min eig)
            for order4 in permute:  # Second Derivative two, two (min eig)
                for order5 in small_list:  # First Derivative one (cond term 1)
                    for order6 in small_list:  # First Derivative two (cond term 3)
                        for order7 in small_list:  # First Derivative three (cond term 4)
                            residuals_k.append([])
                            residuals_logk.append([])
                            store_orders.append([])
                            store_orders[-1].append(order1)
                            store_orders[-1].append(order2)
                            store_orders[-1].append(order3)
                            store_orders[-1].append(order4)
                            store_orders[-1].append(order5)
                            store_orders[-1].append(order6)
                            store_orders[-1].append(order7)
                            last_percent = curr_percent
                            curr_percent = np.floor(count / total_num * 1000)
                            if last_percent != curr_percent:
                                print("Current Iteration: {}".format(count))
                                it_left = total_num - count
                                print("Iterations Left: {}".format(it_left))
                                print("Current Percentage: {:.1f}".format(curr_percent / 10.0))
                                time_passed = t.time() - t_begin
                                if curr_percent != 0:
                                    time_left = time_passed / (count + 1e-6) * it_left
                                    print("Approximate time remaining: {:.2f}s".format(time_left))
                            for i in range(test_size):
                                for j in range(test_size):
                                    for k in range(test_size):
                                        for l in range(test_size):
                                            # Set the current order quadruple
                                            curr_quad = (i, j, k, l)
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

                                            # Calculate condition numbers for finite difference (ME-opt)
                                            cond1 = np.linalg.cond(A_psd_new_1)
                                            cond2 = np.linalg.cond(A_psd_new_2)
                                            cond3 = np.linalg.cond(A_psd_new_3)
                                            cond4 = np.linalg.cond(A_psd_new_4)

                                            # Calculate eigenvalues and vectors (E-opt)
                                            min_eig1, _ = get_eigenvalue_and_vector(A_psd_new_1, "min")
                                            min_eig2, _ = get_eigenvalue_and_vector(A_psd_new_2, "min")
                                            min_eig3, _ = get_eigenvalue_and_vector(A_psd_new_3, "min")
                                            min_eig4, _ = get_eigenvalue_and_vector(A_psd_new_4, "min")

                                            #####################################
                                            # Calculating the FD approximations
                                            hess_cond = (cond1 - cond2 - cond3 + cond4) / (4 * eps**2)
                                            hess_log_cond = (
                                                                    np.log(cond1) - np.log(cond2) - np.log(
                                                                cond3) + np.log(cond4)
                                                            ) / (4 * eps ** 2)

                                            #      End FD approximations
                                            #####################################


                                            #####################################
                                            # Calculating exact Hessian value

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
                                                # Originally thought it would be this based on
                                                # Conventions for eigenvector derivatives
                                                # However it changed to a transposed version...
                                                # exact_eig += 1 * (min_eig_vec[0, j] *
                                                                  # all_eig_vecs[i, curr_eig] *
                                                                  # min_eig_vec[0, l] *
                                                                  # all_eig_vecs[k, curr_eig]) / (min_eig - all_eig_vals[curr_eig])
                                                # exact_eig += 1 * (min_eig_vec[0, i] *
                                                                  # all_eig_vecs[j, curr_eig] *
                                                                  # min_eig_vec[0, k] *
                                                                  # all_eig_vecs[l, curr_eig]) / (min_eig - all_eig_vals[curr_eig])
                                                # exact_eig += 1 * (min_eig_vec[0, i] *
                                                #                   all_eig_vecs[j, curr_eig] *
                                                #                   min_eig_vec[0, l] *
                                                #                   all_eig_vecs[k, curr_eig]) / (min_eig - all_eig_vals[curr_eig])
                                                # exact_eig += 1 * (min_eig_vec[0, k] *
                                                #                   all_eig_vecs[i, curr_eig] *
                                                #                   min_eig_vec[0, j] *
                                                #                   all_eig_vecs[l, curr_eig]) / (min_eig - all_eig_vals[curr_eig])
                                                # Testing changing the minimum eigenvalue derivative as well.
                                                exact_eig += 1 * (min_eig_vec[0, curr_quad[order3[0]]] *
                                                                  all_eig_vecs[curr_quad[order3[1]], curr_eig] *
                                                                  min_eig_vec[0, curr_quad[order3[2]]] *
                                                                  all_eig_vecs[curr_quad[order3[3]], curr_eig]) / (min_eig - all_eig_vals[curr_eig])
                                                exact_eig += 1 * (min_eig_vec[0, curr_quad[order4[0]]] *
                                                                  all_eig_vecs[curr_quad[order4[1]], curr_eig] *
                                                                  min_eig_vec[0, curr_quad[order4[2]]] *
                                                                  all_eig_vecs[curr_quad[order4[3]], curr_eig]) / (min_eig - all_eig_vals[curr_eig])


                                            # Condition number formula was derived by myself
                                            # using product and chain rule and using the
                                            # definition above for second-order eigenvalue derivatives
                                            # Formula is very long, so I won't include it here

                                            # Term 1 is 1 / eig_0 ** 2 * deig_0/dMkl * deig_N/dMij
                                            cond_term_1 = 1 / (min_eig ** 2) * (min_eig_vec[0, l] * min_eig_vec[0, k]) * (max_eig_vec[0, j] * max_eig_vec[0, i])
                                            # cond_term_1 = 1 / (min_eig ** 2) * (
                                            #             min_eig_vec[0, curr_quad[order5[0]]] * min_eig_vec[0, curr_quad[order5[1]]]) * (
                                            #                           max_eig_vec[0, curr_quad[order5[2]]] * max_eig_vec[0, curr_quad[order5[3]]])

                                            # Term 2 is 1 / eig_0 * second_deriv_max_eig
                                            exact_max_eig = 0
                                            for curr_eig in range(len(all_eig_vals)):
                                                if curr_eig == max_eig_loc:
                                                    continue
                                                # Trying the old transpose for Condition number?
                                                # exact_max_eig += 1 * (max_eig_vec[0, j] *
                                                                  # all_eig_vecs[i, curr_eig] *
                                                                  # max_eig_vec[0, l] *
                                                                  # all_eig_vecs[k, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                                                # exact_max_eig += 1 * (max_eig_vec[0, i] *
                                                                  # all_eig_vecs[j, curr_eig] *
                                                                  # max_eig_vec[0, k] *
                                                                  # all_eig_vecs[l, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                                                exact_max_eig += 1 * (max_eig_vec[0, curr_quad[order1[0]]] *
                                                                  all_eig_vecs[curr_quad[order1[1]], curr_eig] *
                                                                  max_eig_vec[0, curr_quad[order1[2]]] *
                                                                  all_eig_vecs[curr_quad[order1[3]], curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                                                exact_max_eig += 1 * (max_eig_vec[0, curr_quad[order2[0]]] *
                                                                  all_eig_vecs[curr_quad[order2[1]], curr_eig] *
                                                                  max_eig_vec[0, curr_quad[order2[2]]] *
                                                                  all_eig_vecs[curr_quad[order2[3]], curr_eig]) / (max_eig - all_eig_vals[curr_eig])

                                            cond_term_2 = 1 / min_eig * exact_max_eig

                                            # New term 3
                                            cond_term_3 = 1 / (min_eig ** 2) * (
                                                        max_eig_vec[0, l] * max_eig_vec[0, k]) * (
                                                                      min_eig_vec[0, j] * min_eig_vec[0, i])
                                            # cond_term_3 = 1 / (min_eig ** 2) * (
                                            #         max_eig_vec[0, curr_quad[order6[0]]] * max_eig_vec[0, curr_quad[order6[1]]]) * (
                                            #                       min_eig_vec[0, curr_quad[order6[2]]] * min_eig_vec[0, curr_quad[order6[3]]])

                                            # New term 4
                                            cond_term_4 = 2 * max_eig / (min_eig ** 3) * (
                                                        min_eig_vec[0, l] * min_eig_vec[0, k]) * (
                                                                      min_eig_vec[0, j] * min_eig_vec[0, i])
                                            # cond_term_4 = 2 * max_eig / (min_eig ** 3) * (
                                            #         min_eig_vec[0, curr_quad[order7[0]]] * min_eig_vec[0, curr_quad[order6[1]]]) * (
                                            #                       min_eig_vec[0, curr_quad[order7[2]]] * min_eig_vec[0, curr_quad[order7[3]]])


                                            # Term 5 is cond / eig_0 * second_deriv_min_eig <-- already computed as "exact_eig"
                                            cond_term_5 = cond / min_eig * exact_eig

                                            # Combine everything at the end
                                            exact_cond = -cond_term_1 + cond_term_2 - cond_term_3 + cond_term_4 - cond_term_5

                                            # Computing log condition number exact
                                            log_cond_term_1 = 1 / max_eig * exact_max_eig
                                            log_cond_term_2 = (
                                                    1
                                                    / (max_eig ** 2)
                                                    * (max_eig_vec[0, l] * max_eig_vec[0, k])
                                                    * (max_eig_vec[0, j] * max_eig_vec[0, i])
                                            )
                                            log_cond_term_3 = 1 / min_eig * exact_eig
                                            log_cond_term_4 = (
                                                    1
                                                    / (min_eig ** 2)
                                                    * (min_eig_vec[0, l] * min_eig_vec[0, k])
                                                    * (min_eig_vec[0, j] * min_eig_vec[0, i])
                                            )
                                            exact_log_cond = (
                                                    log_cond_term_1
                                                    - log_cond_term_2
                                                    - log_cond_term_3
                                                    + log_cond_term_4
                                            )

                                            # End exact Hessian value computation
                                            ######################################

                                            #####################################
                                            # Calculating the residuals!
                                            bad_quads = [(0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)]
                                            if curr_quad in bad_quads:
                                                residuals_k[-1].append((exact_cond - hess_cond) / abs(exact_cond))
                                                residuals_logk[-1].append((exact_log_cond - hess_log_cond) / abs(log_cond_term_1))

                                            # End residual computation!!!
                                            ######################################

                            count += 1


import matplotlib.pyplot as plt

min_sum = 0
min_max = 0
i_min_sum = 0
i_min_max = 0

print(count)

for i in range(len(residuals_k)):
    curr_sum = sum(np.log(abs(np.array(residuals_k[i]))))
    curr_max = max(np.log(abs(np.array(residuals_k[i]))))
    if curr_sum < min_sum:
        min_sum = curr_sum
        i_min_sum = i
    if curr_max < min_max:
        min_max = curr_max
        i_min_max = i

print("Min sum is: {:.4f}".format(min_sum))
print("Min sum orders are: ")
print(store_orders[i_min_sum])
print("Residuals from min: ")
print(np.log(abs(np.array(residuals_k[i_min_sum]))))
print("Current best is: ")
print(store_orders[i_def])
print(np.log(abs(np.array(residuals_k[i_def]))))
print("Best mins is: ")
print(np.log(abs(np.array(residuals_k[i_min_max]))))
print(store_orders[i_min_max])

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

min_sum = 0
min_max = 0
i_min_sum = 0
i_min_max = 0
for i in range(len(residuals_k)):
    curr_sum = sum(np.log(abs(np.array(residuals_logk[i]))))
    curr_max = max(np.log(abs(np.array(residuals_logk[i]))))
    if curr_sum < min_sum:
        min_sum = curr_sum
        i_min_sum = i
    if curr_max < min_max:
        min_max = curr_max
        i_min_max = i

print("Min sum is: {:.4f}".format(min_sum))
print("Min sum orders are: ")
print(store_orders[i_min_sum])
print("Residuals from min: ")
print(np.log(abs(np.array(residuals_logk[i_min_sum]))))
print("Current best is: ")
print(store_orders[i_def])
print(np.log(abs(np.array(residuals_logk[i_def]))))
print("Best mins is: ")
print(np.log(abs(np.array(residuals_logk[i_min_max]))))
print(store_orders[i_min_max])