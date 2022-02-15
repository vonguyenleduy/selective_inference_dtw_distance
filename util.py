import numpy as np
from mpmath import mp

mp.dps = 500

error_threshold = 8


def dtw(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.Inf

    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (s[i - 1] - t[j - 1])**2
            last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = cost + last_min

    return dtw_matrix


def traceback(dtw_matrix):
    i, j = np.array(dtw_matrix.shape) - 2

    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((dtw_matrix[i, j], dtw_matrix[i, j + 1], dtw_matrix[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)

    return np.array(p), np.array(q)


def construct_Omega(n, m):
    idx_matrix = np.identity(n)
    Omega = None
    for i in range(n):
        temp_vec = None
        for j in range(n):
            if idx_matrix[i][j] == 1.0:
                if j == 0:
                    temp_vec = np.ones((m, 1))
                else:
                    temp_vec = np.hstack((temp_vec, np.ones((m, 1))))
            else:
                if j == 0:
                    temp_vec = np.zeros((m, 1))
                else:
                    temp_vec = np.hstack((temp_vec, np.zeros((m, 1))))

        temp_vec = np.hstack((temp_vec, -np.identity(m)))

        if i == 0:
            Omega = temp_vec.copy()
        else:
            Omega = np.vstack((Omega, temp_vec))

    return Omega


def compute_a_b(data, eta, dim_data):
    sq_norm = (np.linalg.norm(eta))**2

    e1 = np.identity(dim_data) - (np.dot(eta, eta.T))/sq_norm
    a = np.dot(e1, data)

    b = eta/sq_norm

    return a, b


def construct_para_pairwise_distance_table(Omega, a, b, n, m):

    para_pairwise_distance_table = np.zeros((n + 1, m + 1), dtype=object)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            e_1 = i - 1
            e_2 = j - 1
            unit_vector = np.zeros((n * m, 1))
            unit_vector[e_1 * m + e_2][0] = 1.0

            temp = np.dot(unit_vector.T, Omega)
            e_1 = np.dot(temp, a)[0][0]
            e_2 = np.dot(temp, b)[0][0]

            const_coef = e_1 ** 2
            lin_coef = 2 * e_1 * e_2
            quad_coef = e_2 ** 2

            para_pairwise_distance_table[i][j] = [const_coef, lin_coef, quad_coef]

    return para_pairwise_distance_table


def dp_selection_event(dtw_matrix, para_pairwise_distance_table, n, m):

    # u_0 + u_1 * z + u_2 * z^2 <= 0
    list_se_dp_u_0 = []
    list_se_dp_u_1 = []
    list_se_dp_u_2 = []

    para_dtw_matrix = np.zeros((n + 1, m + 1), dtype=object)
    para_dtw_matrix[0][0] = [0, 0, 0]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_const = para_pairwise_distance_table[i][j][0]
            cost_lin = para_pairwise_distance_table[i][j][1]
            cost_quad = para_pairwise_distance_table[i][j][2]

            min_idx = np.argmin([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            temp_min_const = 0
            temp_min_lin = 0
            temp_min_quad = 0

            # Update para_dtw_matrix
            if min_idx == 0:
                temp_min_const = para_dtw_matrix[i - 1][j][0]
                temp_min_lin = para_dtw_matrix[i - 1][j][1]
                temp_min_quad = para_dtw_matrix[i - 1][j][2]
            elif min_idx == 1:
                temp_min_const = para_dtw_matrix[i][j - 1][0]
                temp_min_lin = para_dtw_matrix[i][j - 1][1]
                temp_min_quad = para_dtw_matrix[i][j - 1][2]
            else:
                temp_min_const = para_dtw_matrix[i - 1][j - 1][0]
                temp_min_lin = para_dtw_matrix[i - 1][j - 1][1]
                temp_min_quad = para_dtw_matrix[i - 1][j - 1][2]

            para_dtw_matrix[i][j] = [cost_const + temp_min_const, cost_lin + temp_min_lin, cost_quad + temp_min_quad]

            if (i == 1) or (j == 1):
                continue

            # DP selection event
            candidate_0_const = para_dtw_matrix[i - 1][j][0]
            candidate_0_lin = para_dtw_matrix[i - 1][j][1]
            candidate_0_quad = para_dtw_matrix[i - 1][j][2]

            candidate_1_const = para_dtw_matrix[i][j - 1][0]
            candidate_1_lin = para_dtw_matrix[i][j - 1][1]
            candidate_1_quad = para_dtw_matrix[i][j - 1][2]

            candidate_2_const = para_dtw_matrix[i - 1][j - 1][0]
            candidate_2_lin = para_dtw_matrix[i - 1][j - 1][1]
            candidate_2_quad = para_dtw_matrix[i - 1][j - 1][2]

            if min_idx == 0:
                list_se_dp_u_0.append(candidate_0_const - candidate_1_const)
                list_se_dp_u_0.append(candidate_0_const - candidate_2_const)
                list_se_dp_u_1.append(candidate_0_lin - candidate_1_lin)
                list_se_dp_u_1.append(candidate_0_lin - candidate_2_lin)
                list_se_dp_u_2.append(candidate_0_quad - candidate_1_quad)
                list_se_dp_u_2.append(candidate_0_quad - candidate_2_quad)

            elif min_idx == 1:
                list_se_dp_u_0.append(candidate_1_const - candidate_0_const)
                list_se_dp_u_0.append(candidate_1_const - candidate_2_const)
                list_se_dp_u_1.append(candidate_1_lin - candidate_0_lin)
                list_se_dp_u_1.append(candidate_1_lin - candidate_2_lin)
                list_se_dp_u_2.append(candidate_1_quad - candidate_0_quad)
                list_se_dp_u_2.append(candidate_1_quad - candidate_2_quad)

            else:
                list_se_dp_u_0.append(candidate_2_const - candidate_0_const)
                list_se_dp_u_0.append(candidate_2_const - candidate_1_const)
                list_se_dp_u_1.append(candidate_2_lin - candidate_0_lin)
                list_se_dp_u_1.append(candidate_2_lin - candidate_1_lin)
                list_se_dp_u_2.append(candidate_2_quad - candidate_0_quad)
                list_se_dp_u_2.append(candidate_2_quad - candidate_1_quad)

    return para_dtw_matrix, list_se_dp_u_0, list_se_dp_u_1, list_se_dp_u_2


def quadratic_solver(a, b, c):
    delta = b**2 - 4*a*c

    delta = np.around(delta, 3)

    if delta <= 0:
        return None, None

    sqrt_delta = np.sqrt(delta)
    x_1 = (-b - sqrt_delta) / (2*a)
    x_2 = (-b + sqrt_delta) / (2*a)

    if x_1 <= x_2:
        return x_1, x_2
    else:
        return x_2, x_1


def find_list_min_function(list_candidate):
    list_min_funct = []

    curr_min_funct = [np.Inf, np.NINF, np.Inf]

    for element in list_candidate:
        if np.around(element[2] - curr_min_funct[2], error_threshold) == 0:
            if np.around(element[1] - curr_min_funct[1], error_threshold) == 0:
                if element[0] < curr_min_funct[0]:
                    curr_min_funct = element

            elif element[1] > curr_min_funct[1]:
                curr_min_funct = element

        elif element[2] < curr_min_funct[2]:
            curr_min_funct = element

    list_min_funct.append(curr_min_funct)
    list_candidate.remove(curr_min_funct)

    curr_z = np.NINF

    while len(list_candidate) != 0:
        update_list_candidate = []

        next_z = np.Inf
        next_min_funct = None

        for each_element in list_candidate:
            a = np.around(curr_min_funct[2] - each_element[2], error_threshold)
            b = np.around(curr_min_funct[1] - each_element[1], error_threshold)
            c = np.around(curr_min_funct[0] - each_element[0], error_threshold)

            if a == 0:
                if b == 0:
                    if c > 0:
                        continue

                    continue
                else:
                    z_intersect = - c / b

                    if np.around(z_intersect - curr_z, 8) > 0:
                        update_list_candidate.append(each_element)

                        if np.around(next_z - z_intersect, 8) > 0:
                            next_z = z_intersect
                            next_min_funct = each_element
                        elif np.around(next_z - z_intersect, 8) == 0:
                            if each_element[1] < next_min_funct[1]:
                                next_z = z_intersect
                                next_min_funct = each_element

            else:
                x_1, x_2 = quadratic_solver(a, b, c)

                if (x_1 is None) and (x_2 is None):
                    continue
                else:
                    z_intersect = None
                    if x_2 <= curr_z:
                        continue
                    elif x_1 <= curr_z < x_2:
                        z_intersect = x_2
                    elif curr_z < x_1:
                        z_intersect = x_1

                    update_list_candidate.append(each_element)

                    if np.around(next_z - z_intersect, error_threshold) > 0:
                        next_z = z_intersect
                        next_min_funct = each_element

                    elif np.around(next_z - z_intersect, error_threshold) == 0:
                        if each_element[2] == next_min_funct[2]:
                            if each_element[1] == next_min_funct[1]:
                                if each_element[0] < next_min_funct[0]:
                                    next_min_funct = each_element

                            elif each_element[1] > next_min_funct[1]:
                                next_min_funct = each_element

                        elif each_element[2] < next_min_funct[2]:
                            next_min_funct = each_element

        if len(update_list_candidate) > 0:
            update_list_candidate.remove(next_min_funct)

            curr_z = next_z
            curr_min_funct = next_min_funct
            list_min_funct.append(curr_min_funct)

        list_candidate = update_list_candidate

    return list_min_funct


def parametric_dp(para_pairwise_distance_table, n, m):
    para_dtw_matrix = np.zeros((n + 1, m + 1), dtype=object)

    for i in range(n + 1):
        for j in range(m + 1):
            para_dtw_matrix[i, j] = [[np.Inf, np.Inf, np.Inf]]

    para_dtw_matrix[0][0] = [[0, 0, 0]]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_const = para_pairwise_distance_table[i][j][0]
            cost_lin = para_pairwise_distance_table[i][j][1]
            cost_quad = para_pairwise_distance_table[i][j][2]

            list_candidate = []

            for element in para_dtw_matrix[i - 1, j]:
                if element == [np.Inf, np.Inf, np.Inf]:
                    continue

                list_candidate.append([
                    np.around(element[0] + cost_const, error_threshold),
                    np.around(element[1] + cost_lin, error_threshold),
                    np.around(element[2] + cost_quad, error_threshold)])

            for element in para_dtw_matrix[i, j - 1]:
                if element == [np.Inf, np.Inf, np.Inf]:
                    continue

                list_candidate.append([
                    np.around(element[0] + cost_const, error_threshold),
                    np.around(element[1] + cost_lin, error_threshold),
                    np.around(element[2] + cost_quad, error_threshold)])

            for element in para_dtw_matrix[i - 1, j - 1]:
                if element == [np.Inf, np.Inf, np.Inf]:
                    continue

                list_candidate.append([
                    np.around(element[0] + cost_const, error_threshold),
                    np.around(element[1] + cost_lin, error_threshold),
                    np.around(element[2] + cost_quad, error_threshold)])

            list_min_function = find_list_min_function(list_candidate.copy())

            para_dtw_matrix[i, j] = list_min_function

    return para_dtw_matrix



def sign_selection_event(nu, Theta, a, b):
    # u_0 + u_1 z
    list_se_sign_u_0 = []
    list_se_sign_u_1 = []

    list_const = (nu * np.dot(Theta, a)).flatten()
    list_lin = (nu * np.dot(Theta, b)).flatten()

    for i in range(len(nu)):
        if nu[i][0] == 0:
            continue

        list_se_sign_u_0.append(- list_const[i])
        list_se_sign_u_1.append(- list_lin[i])

    return list_se_sign_u_0, list_se_sign_u_1


def intersect(range_1, range_2):
    lower = max(range_1[0], range_2[0])
    upper = min(range_1[1], range_2[1])

    if upper < lower:
        return []
    else:
        return [lower, upper]


def intersect_interval(initial_range, list_2_range):
    if len(initial_range) == 0:
        return []

    final_list = [initial_range]

    for each_2_range in list_2_range:

        lower_range = [np.NINF, each_2_range[0]]
        upper_range = [each_2_range[1], np.Inf]

        new_final_list = []

        for each_1_range in final_list:
            local_range_1 = intersect(each_1_range, lower_range)
            local_range_2 = intersect(each_1_range, upper_range)

            if len(local_range_1) > 0:
                new_final_list.append(local_range_1)

            if len(local_range_2) > 0:
                new_final_list.append(local_range_2)

        final_list = new_final_list

    return final_list


def find_truncation_interval_dp(list_se_dp_u_0, list_se_dp_u_1, list_se_dp_u_2):
    list_se_dp_u_0 = np.around(list_se_dp_u_0, 10)
    list_se_dp_u_1 = np.around(list_se_dp_u_1, 10)
    list_se_dp_u_2 = np.around(list_se_dp_u_2, 10)

    L_prime = np.NINF
    U_prime = np.Inf

    L_tilde = np.NINF
    U_tilde = np.Inf

    list_2_interval = []

    for i in range(len(list_se_dp_u_0)):
        c = list_se_dp_u_0[i]
        b = list_se_dp_u_1[i]
        a = list_se_dp_u_2[i]

        if a == 0:
            if b == 0:
                if c > 0:
                    print('Error a = 0, b = 0, c > 0')

            elif b < 0:
                temporal_lower_bound = - c / b
                L_prime = max(L_prime, temporal_lower_bound)

            elif b > 0:
                temporal_upper_bound = - c / b
                U_prime = min(U_prime, temporal_upper_bound)
        else:
            delta = b ** 2 - 4 * a * c

            delta = np.around(delta, 10)

            if delta == 0:
                continue
            elif delta < 0:
                continue
            elif delta > 0:
                if a > 0:
                    x_lower = (-b - np.sqrt(delta)) / (2 * a)
                    x_upper = (-b + np.sqrt(delta)) / (2 * a)

                    if x_lower > x_upper:
                        print('x_lower > x_upper')

                    L_tilde = max(L_tilde, x_lower)
                    U_tilde = min(U_tilde, x_upper)

                else:
                    x_1 = (-b - np.sqrt(delta)) / (2 * a)
                    x_2 = (-b + np.sqrt(delta)) / (2 * a)

                    x_low = min(x_1, x_2)
                    x_up = max(x_1, x_2)
                    list_2_interval.append([x_low, x_up])

    list_1_interval = [intersect([L_prime, U_prime], [L_tilde, U_tilde])]

    return list_1_interval, list_2_interval


def find_truncation_interval_sign(list_1, list_2):
    Vminus, Vplus = np.NINF, np.Inf

    for i in range(len(list_1)):
        left = list_2[i]
        right = - list_1[i]

        left = np.around(left, 5)
        right = np.around(right, 5)

        if left == 0:
            if right < 0:
                print('error 1')

            continue

        temp = right / left
        if left > 0:
            Vplus = min(Vplus, temp)
        else:
            Vminus = max(Vminus, temp)

    return Vminus, Vplus


def pivot_with_constructed_interval(z_interval, eta, etaTy, cov, tn_mu):
    tn_sigma = np.sqrt(np.dot(np.dot(eta.T, cov), eta))[0][0]
    numerator = 0
    denominator = 0

    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]

        denominator = denominator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

        if etaTy >= ar:
            numerator = numerator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
        elif (etaTy >= al) and (etaTy < ar):
            numerator = numerator + mp.ncdf((etaTy - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

    if denominator != 0:
        return float(numerator/denominator)
    else:
        return None







