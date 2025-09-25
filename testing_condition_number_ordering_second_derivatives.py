import numpy as np
import copy
import itertools


def get_eigenvalue_and_vector(M, option='min'):
    vals, vecs = np.linalg.eig(M)
    min_eig_loc = np.argmin(vals)
    max_eig_loc = np.argmax(vals)

    min_eig_vec = np.array(
        [vecs[:, min_eig_loc]]
    )  # Make this a matrix so transpose makes sense
    max_eig_vec = np.array(
        [vecs[:, max_eig_loc]]
    )  # Make this a matrix so transpose makes sense

    if option == 'min':
        return vals[min_eig_loc], min_eig_vec
    elif option == 'max':
        return vals[max_eig_loc], max_eig_vec
    elif option == 'both':
        return vals[min_eig_loc], min_eig_vec, vals[max_eig_loc], max_eig_vec


permute = list(itertools.permutations([0, 1, 2, 3]))

test_size = 2

eps = 0.00001

np.random.seed(1022)

A = np.random.rand(test_size, test_size)
A_psd = np.dot(A, A.transpose())  # Positive semidefinte matrix A_psd

# Simple PSD matrix
# A_psd = np.array([[3, 1], [1, 1]], dtype=float)

A_psd_inv = np.linalg.pinv(A_psd)
A_psd_inv_sq = A_psd_inv @ A_psd_inv

print(A_psd)

# Vectors and values for the original matrix
min_eig, min_eig_vec, max_eig, max_eig_vec = get_eigenvalue_and_vector(A_psd, "both")

all_eig_vals, all_eig_vecs = np.linalg.eig(A_psd)
min_eig_loc = np.argmin(all_eig_vals)
max_eig_loc = np.argmax(all_eig_vals)

print(all_eig_vals)
print(all_eig_vecs)
print(min_eig_loc)
print(max_eig_loc)

print(all_eig_vecs.T @ all_eig_vecs)

cond = np.linalg.cond(A_psd)

trace_inv = np.trace(A_psd_inv)

print(A_psd_inv)

residuals_det = []
residuals_eig = []
residuals_eig_max = []
residuals_k = []
residuals_logk = []
residuals_trace = []

save_log_cond_residuals = [[],] * test_size**4
its = [[],] * test_size**4
abs_residuals_logcond = [0,] * test_size**4

bad_its = []

count = 0
count_bad_quads = 0

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

                print(A_psd_new_4)
                print(eps)

                # Need 4 perturbations to cover the
                # formula H[i, j] = [(A + eps (both))
                # + (A +/- eps one each)
                # + (A -/+ eps one each)
                # + (A - eps (both))] / (4*eps**2)
                A_psd_new_1[i, j] = A_psd_new_1[i, j] + eps
                A_psd_new_1[k, l] += eps

                print(A_psd_new_1)

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

                print("Cond with conds")
                print(cond1, cond2, cond3, cond4)

                # Calculate eigenvalues and vectors (E-opt)
                min_eig1, _, max_eig1, _ = get_eigenvalue_and_vector(
                    A_psd_new_1, "both"
                )
                print(_)
                min_eig2, _, max_eig2, _ = get_eigenvalue_and_vector(
                    A_psd_new_2, "both"
                )
                print(_)
                min_eig3, _, max_eig3, _ = get_eigenvalue_and_vector(
                    A_psd_new_3, "both"
                )
                print(_)
                min_eig4, _, max_eig4, _ = get_eigenvalue_and_vector(
                    A_psd_new_4, "both"
                )
                print(_)

                print(min_eig1, min_eig2, min_eig3, min_eig4)

                print("Cond with eigs")
                print(
                    max_eig1 / min_eig1,
                    max_eig2 / min_eig2,
                    max_eig3 / min_eig3,
                    max_eig4 / min_eig4,
                )

                # Calculate trace of inverse for finite difference (ME-opt)
                inv1 = np.linalg.inv(A_psd_new_1)
                trace1 = np.trace(inv1)
                inv2 = np.linalg.inv(A_psd_new_2)
                trace2 = np.trace(inv2)
                inv3 = np.linalg.inv(A_psd_new_3)
                trace3 = np.trace(inv3)
                inv4 = np.linalg.inv(A_psd_new_4)
                trace4 = np.trace(inv4)

                # REMOVE?
                cond1 = max_eig1 / min_eig1
                cond2 = max_eig2 / min_eig2
                cond3 = max_eig3 / min_eig3
                cond4 = max_eig4 / min_eig4
                # REMOVE?

                #####################################
                # Calculating the FD approximations
                hess_det = (det1 - det2 - det3 + det4) / (4 * eps**2)
                hess_cond = (cond1 - cond2 - cond3 + cond4) / (4 * eps**2)
                hess_log_cond = (
                    np.log(cond1) - np.log(cond2) - np.log(cond3) + np.log(cond4)
                ) / (4 * eps**2)
                hess_eig = (min_eig1 - min_eig2 - min_eig3 + min_eig4) / (4 * eps**2)
                hess_eig_max = (max_eig1 - max_eig2 - max_eig3 + max_eig4) / (4 * eps**2)
                hess_trace_inv = (trace1 - trace2 - trace3 + trace4) / (4 * eps**2)

                #      End FD approximations
                #####################################

                #####################################
                # Calculating exact Hessian value

                # Determinant formula comes from 1/2 * (dMinv/dM + dMinv^T/dM)
                # which is -1/4*(Minv[i, k]Minv[l, j] + Minv[i, l]Minv[k, j]
                #                + Minv[j, k]Minv[l, i] + Minv[j, l]Minv[k, i])
                # https://www.et.byu.edu/~vps/ME505/AAEM/V5-07.pdf
                exact_det = (
                    -1
                    / 4
                    * (
                        A_psd_inv[i, k] * A_psd_inv[l, j]
                        + A_psd_inv[i, l] * A_psd_inv[k, j]
                        + A_psd_inv[j, k] * A_psd_inv[l, i]
                        + A_psd_inv[j, l] * A_psd_inv[k, i]
                    )
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
                    exact_eig += (
                        1
                        * (
                            min_eig_vec[0, i]
                            * all_eig_vecs[j, curr_eig]
                            * min_eig_vec[0, l]
                            * all_eig_vecs[k, curr_eig]
                        )
                        / (min_eig - all_eig_vals[curr_eig])
                    )
                    exact_eig += (
                        1
                        * (
                            min_eig_vec[0, k]
                            * all_eig_vecs[i, curr_eig]
                            * min_eig_vec[0, j]
                            * all_eig_vecs[l, curr_eig]
                        )
                        / (min_eig - all_eig_vals[curr_eig])
                    )
                    # exact_eig += (
                    #         2
                    #         * (
                    #                 min_eig_vec[0, l]
                    #                 * all_eig_vecs[k, curr_eig]
                    #                 * min_eig_vec[0, j]
                    #                 * all_eig_vecs[i, curr_eig]
                    #         )
                    #         / (min_eig - all_eig_vals[curr_eig])
                    # )

                # Condition number formula was derived by myself
                # using product and chain rule and using the
                # definition above for second-order eigenvalue derivatives
                # Formula is very long, so I won't include it here

                # Term 1 is 1 / eig_0 ** 2 * deig_0/dMkl * deig_N/dMij
                cond_term_1 = (
                    1
                    / (min_eig**2)
                    * (min_eig_vec[0, l] * min_eig_vec[0, k])
                    * (max_eig_vec[0, j] * max_eig_vec[0, i])
                )

                # Term 2 is 1 / eig_0 * second_deriv_max_eig
                exact_max_eig = 0
                for curr_eig in range(len(all_eig_vals)):
                    if curr_eig == max_eig_loc:
                        continue
                    # Trying the old transpose for Condition number?
                    # exact_max_eig += 1 * (max_eig_vec[0, j] *
                    #                   all_eig_vecs[i, curr_eig] *
                    #                   max_eig_vec[0, l] *
                    #                   all_eig_vecs[k, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                    # exact_max_eig += 1 * (max_eig_vec[0, i] *
                    #                   all_eig_vecs[j, curr_eig] *
                    #                   max_eig_vec[0, k] *
                    #                   all_eig_vecs[l, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                    # exact_max_eig += 1 * (max_eig_vec[0, j] *
                    #                   all_eig_vecs[i, curr_eig] *
                    #                   max_eig_vec[0, l] *
                    #                   all_eig_vecs[k, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                    # exact_max_eig += 1 * (max_eig_vec[0, i] *
                    #                   all_eig_vecs[j, curr_eig] *
                    #                   max_eig_vec[0, k] *
                    #                   all_eig_vecs[l, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                    # Trying some random stuff
                    # exact_max_eig += 1 * (max_eig_vec[0, j] *
                    #                       all_eig_vecs[j, curr_eig] *
                    #                       max_eig_vec[0, l] *
                    #                       all_eig_vecs[k, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                    # exact_max_eig += 1 * (max_eig_vec[0, i] *
                    #                       all_eig_vecs[i, curr_eig] *
                    #                       max_eig_vec[0, k] *
                    #                       all_eig_vecs[l, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                    # Somehow the derivative is not correct, maybe?
                    exact_max_eig += (
                        1
                        * (
                            max_eig_vec[0, i]
                            * all_eig_vecs[j, curr_eig]
                            * max_eig_vec[0, l]
                            * all_eig_vecs[k, curr_eig]
                        )
                        / (max_eig - all_eig_vals[curr_eig])
                    )
                    exact_max_eig += (
                        1
                        * (
                            max_eig_vec[0, k]
                            * all_eig_vecs[i, curr_eig]
                            * max_eig_vec[0, j]
                            * all_eig_vecs[l, curr_eig]
                        )
                        / (max_eig - all_eig_vals[curr_eig])
                    )
                    # exact_max_eig += (
                    #     2
                    #     * (
                    #         max_eig_vec[0, i]
                    #         * all_eig_vecs[j, curr_eig]
                    #         * max_eig_vec[0, k]
                    #         * all_eig_vecs[l, curr_eig]
                    #     )
                    #     / (max_eig - all_eig_vals[curr_eig])
                    # )

                cond_term_2 = 1 / min_eig * exact_max_eig

                # Term 3 is 1 / eig_0 * dcond/dMkl * deig_0/dMij
                # cond_term_3 = 1 / min_eig * (1 / min_eig *
                #                              (max_eig_vec[0, l] * max_eig_vec[0, k] -
                #                               cond * min_eig_vec[0, l] * min_eig_vec[0, k])) * (min_eig_vec[0, j] * min_eig_vec[0, i])
                # Trying new term 3, 4, 5
                # New term 3
                cond_term_3 = (
                    1
                    / (min_eig**2)
                    * (max_eig_vec[0, l] * max_eig_vec[0, k])
                    * (min_eig_vec[0, j] * min_eig_vec[0, i])
                )

                # Term 4 is cond / eig_0 ** 2 * deig_0/dMkl * deig_0/dMij
                # cond_term_4 = cond / (min_eig ** 2) * (min_eig_vec[0, l] * min_eig_vec[0, k]) * (min_eig_vec[0, j] * min_eig_vec[0, i])
                # New term 4
                cond_term_4 = (
                    2
                    * max_eig
                    / (min_eig**3)
                    * (min_eig_vec[0, l] * min_eig_vec[0, k])
                    * (min_eig_vec[0, j] * min_eig_vec[0, i])
                )

                # Term 5 is cond / eig_0 * second_deriv_min_eig <-- already computed as "exact_eig"
                # cond_term_5 = cond / min_eig * exact_eig
                # New term 5
                cond_term_5 = max_eig / (min_eig**2) * exact_eig

                # Combine everything at the end
                # exact_cond = cond_term_1 + cond_term_2 + cond_term_3 + cond_term_4 + cond_term_5
                exact_cond = (
                    -cond_term_1 + cond_term_2 - cond_term_3 + cond_term_4 - cond_term_5
                )

                # Computing log condition number exact
                log_cond_term_1 = 1 / max_eig * exact_max_eig
                log_cond_term_2 = (
                    1
                    / (max_eig**2)
                    * (max_eig_vec[0, l] * max_eig_vec[0, k])
                    * (max_eig_vec[0, j] * max_eig_vec[0, i])
                )
                log_cond_term_3 = 1 / min_eig * exact_eig
                log_cond_term_4 = (
                    1
                    / (min_eig**2)
                    * (min_eig_vec[0, l] * min_eig_vec[0, k])
                    * (min_eig_vec[0, j] * min_eig_vec[0, i])
                )
                exact_log_cond = (
                    log_cond_term_1
                    - log_cond_term_2
                    - log_cond_term_3
                    + log_cond_term_4
                )

                # Computing exact inverse trace
                exact_trace_inv = (
                    A_psd_inv[i, l] * A_psd_inv_sq[k, j]
                    + A_psd_inv_sq[i, l] * A_psd_inv[k, j]
                )

                # End exact Hessian value computation
                ######################################

                #####################################
                # Calculating the residuals!

                print("\n\nIteration ({}, {}, {}, {})".format(i, j, k, l))
                print("Or: {}\n\n".format(count))

                if np.log(np.abs((exact_cond - hess_cond) / exact_cond)) > -10:
                    bad_its.append((i, j, k, l))

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

                print("Max eig: ")
                print(hess_eig_max)
                print(exact_max_eig)
                print("~~~~~~~~~~~~~")

                print("Condition number: ")
                print(hess_cond)
                print(exact_cond)
                print("~~~~~~~~~~~~~")
                print("Condiditon Number Term 1: ")
                print(cond_term_1)
                print("Condiditon Number Term 2: ")
                print(cond_term_2)

                # diff = np.abs(exact_cond - hess_cond)
                #
                # curr_quad = (i, j, k, l)
                # bad_quads = [(0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)]
                # if curr_quad in bad_quads:
                #     for order1 in permute:
                #         for order2 in permute:
                #             exact_max_eig_recalc = 0
                #             for curr_eig in range(len(all_eig_vals)):
                #                 if curr_eig == max_eig_loc:
                #                     continue
                #                 # exact_max_eig += 1 * (max_eig_vec[0, j] *
                #                 #                   all_eig_vecs[i, curr_eig] *
                #                 #                   max_eig_vec[0, l] *
                #                 #                   all_eig_vecs[k, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                #                 # exact_max_eig += 1 * (max_eig_vec[0, i] *
                #                 #                   all_eig_vecs[j, curr_eig] *
                #                 #                   max_eig_vec[0, k] *
                #                 #                   all_eig_vecs[l, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                #                 exact_max_eig_recalc += (
                #                     1
                #                     * (
                #                         max_eig_vec[0, curr_quad[order1[0]]]
                #                         * all_eig_vecs[curr_quad[order1[1]], curr_eig]
                #                         * max_eig_vec[0, curr_quad[order1[2]]]
                #                         * all_eig_vecs[curr_quad[order1[3]], curr_eig]
                #                     )
                #                     / (max_eig - all_eig_vals[curr_eig])
                #                 )
                #                 exact_max_eig_recalc += (
                #                     1
                #                     * (
                #                         max_eig_vec[0, curr_quad[order2[0]]]
                #                         * all_eig_vecs[curr_quad[order2[1]], curr_eig]
                #                         * max_eig_vec[0, curr_quad[order2[2]]]
                #                         * all_eig_vecs[curr_quad[order2[3]], curr_eig]
                #                     )
                #                     / (max_eig - all_eig_vals[curr_eig])
                #                 )
                #             cond_term_2_recalc = 1 / min_eig * exact_max_eig_recalc
                #     count_bad_quads += 1
                print("Condiditon Number Term 3: ")
                print(cond_term_3)
                print("Condiditon Number Term 4: ")
                print(cond_term_4)
                print("Condiditon Number Term 5: ")
                print(cond_term_5)
                print("~~~~~~~~~~~~~")

                print("Log condition number: ")
                print(hess_log_cond)
                print(exact_log_cond)
                print("~~~~~~~~~~~~~")
                print("Log Condiditon Number Term 1: ")
                print(log_cond_term_1)
                print("Log Condiditon Number Term 2: ")
                print(log_cond_term_2)
                print("Log Condiditon Number Term 3: ")
                print(log_cond_term_3)
                print("Log Condiditon Number Term 4: ")
                print(log_cond_term_4)
                print("~~~~~~~~~~~~~")
                curr_quad = (i, j, k, l)
                bad_quads = [(0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)]
                if curr_quad in bad_quads and False:
                    for order1 in permute:
                        print(order1)
                        log_cond_term_2_other = (
                                1
                                / (max_eig ** 2)
                                * (max_eig_vec[0, curr_quad[order1[0]]] * max_eig_vec[0, curr_quad[order1[1]]])
                                * (max_eig_vec[0, curr_quad[order1[2]]] * max_eig_vec[0, curr_quad[order1[3]]])
                        )
                        print(max_eig)
                        print(max_eig_vec[0, curr_quad[order1[0]]], max_eig_vec[0, curr_quad[order1[1]]], max_eig_vec[0, curr_quad[order1[2]]], max_eig_vec[0, curr_quad[order1[3]]])
                        print(log_cond_term_2_other)
                        for order2 in permute:
                            print(order1)
                            print(order2)
                            # Recalculate the second derivative term for the maximum eigenvalue
                            exact_max_eig_other = 0
                            for curr_eig in range(len(all_eig_vals)):
                                if curr_eig == max_eig_loc:
                                    continue
                                # Trying the old transpose for Condition number?
                                # exact_max_eig += 1 * (max_eig_vec[0, j] *
                                #                   all_eig_vecs[i, curr_eig] *
                                #                   max_eig_vec[0, l] *
                                #                   all_eig_vecs[k, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                                # exact_max_eig += 1 * (max_eig_vec[0, i] *
                                #                   all_eig_vecs[j, curr_eig] *
                                #                   max_eig_vec[0, k] *
                                #                   all_eig_vecs[l, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                                # exact_max_eig += 1 * (max_eig_vec[0, j] *
                                #                   all_eig_vecs[i, curr_eig] *
                                #                   max_eig_vec[0, l] *
                                #                   all_eig_vecs[k, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                                # exact_max_eig += 1 * (max_eig_vec[0, i] *
                                #                   all_eig_vecs[j, curr_eig] *
                                #                   max_eig_vec[0, k] *
                                #                   all_eig_vecs[l, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                                # Trying some random stuff
                                # exact_max_eig += 1 * (max_eig_vec[0, j] *
                                #                       all_eig_vecs[j, curr_eig] *
                                #                       max_eig_vec[0, l] *
                                #                       all_eig_vecs[k, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                                # exact_max_eig += 1 * (max_eig_vec[0, i] *
                                #                       all_eig_vecs[i, curr_eig] *
                                #                       max_eig_vec[0, k] *
                                #                       all_eig_vecs[l, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                                # Somehow the derivative is not correct, maybe?
                                exact_max_eig_other += (
                                        1
                                        * (
                                                max_eig_vec[0, curr_quad[order1[0]]]
                                                * all_eig_vecs[curr_quad[order1[1]], curr_eig]
                                                * max_eig_vec[0, curr_quad[order1[2]]]
                                                * all_eig_vecs[curr_quad[order1[3]], curr_eig]
                                        )
                                        / (max_eig - all_eig_vals[curr_eig])
                                )
                                exact_max_eig_other += (
                                        1
                                        * (
                                                max_eig_vec[0, curr_quad[order2[0]]]
                                                * all_eig_vecs[curr_quad[order2[1]], curr_eig]
                                                * max_eig_vec[0, curr_quad[order2[2]]]
                                                * all_eig_vecs[curr_quad[order2[3]], curr_eig]
                                        )
                                        / (max_eig - all_eig_vals[curr_eig])
                                )
                            print(1 / max_eig * exact_max_eig_other)

                #     for order1 in permute:
                #         for order2 in permute:
                #             exact_max_eig_recalc = 0
                #             for curr_eig in range(len(all_eig_vals)):
                #                 if curr_eig == max_eig_loc:
                #                     continue
                #                 # exact_max_eig += 1 * (max_eig_vec[0, j] *
                #                 #                   all_eig_vecs[i, curr_eig] *
                #                 #                   max_eig_vec[0, l] *
                #                 #                   all_eig_vecs[k, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                #                 # exact_max_eig += 1 * (max_eig_vec[0, i] *
                #                 #                   all_eig_vecs[j, curr_eig] *
                #                 #                   max_eig_vec[0, k] *
                #                 #                   all_eig_vecs[l, curr_eig]) / (max_eig - all_eig_vals[curr_eig])
                #                 exact_max_eig_recalc += (
                #                         1
                #                         * (
                #                                 max_eig_vec[0, curr_quad[order1[0]]]
                #                                 * all_eig_vecs[curr_quad[order1[1]], curr_eig]
                #                                 * max_eig_vec[0, curr_quad[order1[2]]]
                #                                 * all_eig_vecs[curr_quad[order1[3]], curr_eig]
                #                         )
                #                         / (max_eig - all_eig_vals[curr_eig])
                #                 )
                #                 exact_max_eig_recalc += (
                #                         1
                #                         * (
                #                                 max_eig_vec[0, curr_quad[order2[0]]]
                #                                 * all_eig_vecs[curr_quad[order2[1]], curr_eig]
                #                                 * max_eig_vec[0, curr_quad[order2[2]]]
                #                                 * all_eig_vecs[curr_quad[order2[3]], curr_eig]
                #                         )
                #                         / (max_eig - all_eig_vals[curr_eig])
                #                 )
                #             cond_term_2_recalc = 1 / min_eig * exact_max_eig_recalc
                #     count_bad_quads += 1

                print("Trace inverse (A-opt): ")
                print(hess_trace_inv)
                print(exact_trace_inv)
                print("~~~~~~~~~~~~~")

                residuals_det.append((exact_det - hess_det) / abs(exact_det))
                residuals_eig.append((exact_eig - hess_eig) / abs(exact_eig))
                residuals_eig_max.append((exact_max_eig - hess_eig_max) / abs(exact_max_eig))
                residuals_k.append((exact_cond - hess_cond) / abs(exact_cond))
                residuals_logk.append(
                    (exact_log_cond - hess_log_cond) / abs(exact_log_cond)
                )
                residuals_trace.append(
                    (exact_trace_inv - hess_trace_inv) / abs(exact_trace_inv)
                )

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

                save_log_cond_residuals[count] = [log_cond_term_1, log_cond_term_2, log_cond_term_3, log_cond_term_4]
                its[count] = [i, j, k, l]
                abs_residuals_logcond[count] = exact_log_cond - hess_log_cond

                if i == 1 and j == 0 and k == 1 and l == 0:
                    lots_of_stuff = [0,] * 24 * 24
                    count_cond_log = 0
                    for order1 in permute:
                        for order2 in permute:
                            exact_max_eig_ = 0
                            exact_eig_ = 0
                            for curr_eig in range(len(all_eig_vals)):
                                if curr_eig == max_eig_loc:
                                    continue
                                exact_max_eig_ += 1 * (max_eig_vec[0, curr_quad[order1[0]]] *
                                                      all_eig_vecs[curr_quad[order1[1]], curr_eig] *
                                                      max_eig_vec[0, curr_quad[order1[2]]] *
                                                      all_eig_vecs[curr_quad[order1[3]], curr_eig]) / (
                                                             max_eig - all_eig_vals[curr_eig])
                                exact_max_eig_ += 1 * (max_eig_vec[0, curr_quad[order2[0]]] *
                                                      all_eig_vecs[curr_quad[order2[1]], curr_eig] *
                                                      max_eig_vec[0, curr_quad[order2[2]]] *
                                                      all_eig_vecs[curr_quad[order2[3]], curr_eig]) / (
                                                             max_eig - all_eig_vals[curr_eig])
                            for curr_eig in range(len(all_eig_vals)):
                                if curr_eig == min_eig_loc:
                                    continue
                                exact_eig_ += 1 * (min_eig_vec[0, curr_quad[order1[0]]] *
                                                  all_eig_vecs[curr_quad[order1[1]], curr_eig] *
                                                  min_eig_vec[0, curr_quad[order1[2]]] *
                                                  all_eig_vecs[curr_quad[order1[3]], curr_eig]) / (
                                                         min_eig - all_eig_vals[curr_eig])
                                exact_eig_ += 1 * (min_eig_vec[0, curr_quad[order2[0]]] *
                                                  all_eig_vecs[curr_quad[order2[1]], curr_eig] *
                                                  min_eig_vec[0, curr_quad[order2[2]]] *
                                                  all_eig_vecs[curr_quad[order2[3]], curr_eig]) / (
                                                         min_eig - all_eig_vals[curr_eig])

                            lots_of_stuff[count_cond_log] = [order1, order2, exact_max_eig_ / max_eig - exact_max_eig / max_eig - exact_eig_ / min_eig + exact_eig / min_eig,]
                            count_cond_log += 1


                count += 1



print(A_psd)
print(all_eig_vals, all_eig_vecs)

# print(bad_its)
# print(len(bad_its))
#
# print("overlap")
# print(bad_quad_maybe_fix)

import matplotlib.pyplot as plt

plt.plot(
    range(len(residuals_det)),
    np.log(abs(np.array(residuals_det))),
    color='orange',
    label='Difference from \'exact\' to F.D. (log det)',
)
plt.plot(
    range(len(residuals_det)),
    np.log(abs(np.array(residuals_eig))),
    color='black',
    label='Difference from \'exact\' to F.D. (Min Eig)',
)
plt.plot(
    range(len(residuals_det)),
    np.log(abs(np.array(residuals_eig_max))),
    color='pink',
    label='Difference from \'exact\' to F.D. (Max Eig)',
)
plt.plot(
    range(len(residuals_det)),
    np.log(abs(np.array(residuals_k))),
    color='green',
    label='Difference from \'exact\' to F.D. (Condition Number)',
)
plt.plot(
    range(len(residuals_det)),
    np.log(abs(np.array(residuals_logk))),
    color='purple',
    label='Difference from \'exact\' to F.D. (Log Condition Number)',
)
plt.plot(
    range(len(residuals_det)),
    np.log(abs(np.array(residuals_trace))),
    color='blue',
    label='Difference from \'exact\' to F.D. (A-opt)',
)
# print(np.log(abs(np.array(residuals_det))))
# print(np.log(abs(np.array(residuals_eig))))
# print(np.log(abs(np.array(residuals_k))))
# print(np.log(abs(np.array(residuals_trace))))
plt.ylabel("log-10(relative error)", fontsize=20)
plt.legend()
plt.tight_layout()
plt.show()

# np.set_printoptions(legacy='1.25')
# for i in range(len(save_log_cond_residuals)):
#     print(its[i])
#     print(abs_residuals_logcond[i])
#     print(save_log_cond_residuals[i])
#
# for j in range(len(lots_of_stuff)):
#     print(lots_of_stuff[j])