import numpy as np

import util


def run():
    n = 5
    m = 5

    true_x = np.ones(n)
    true_y = np.ones(m) + 0

    x_obs = true_x + np.random.normal(loc=0, scale=1, size=n)
    y_obs = true_y + np.random.normal(loc=0, scale=1, size=m)

    data_obs = np.vstack((np.reshape(x_obs, (n, 1)), np.reshape(y_obs, (m, 1)))).copy()
    true_data = np.vstack((np.reshape(true_x, (n, 1)), np.reshape(true_y, (m, 1)))).copy()

    # Obtain dtw matrix and path
    dtw_matrix = util.dtw(x_obs, y_obs)
    dtw_path = util.traceback(dtw_matrix)

    # Obtain Theta
    Omega = util.construct_Omega(n, m)
    Theta = np.sign(np.dot(Omega, data_obs)) * Omega

    # Construct eta
    nu = np.zeros((n * m, 1))

    for i in range(len(dtw_path[0])):
        e_1 = dtw_path[0][i]
        e_2 = dtw_path[1][i]
        nu[e_1 * m + e_2][0] = 1.0

    eta = (np.dot(nu.T, Theta)).T

    # etaTdata
    etaTdata = np.dot(eta.T, data_obs)[0][0]

    # Construct a_line and b_line: y(z) = a + bz
    a, b = util.compute_a_b(data_obs, eta, n + m)

    # Construct para_pairwise_distance_table: const + lin * z + quad * z^2
    para_pairwise_distance_table = util.construct_para_pairwise_distance_table(Omega, a, b, n, m)

    # DP selection event
    para_dtw_matrix = util.parametric_dp(para_pairwise_distance_table, n, m)

    list_final_min_cost_function = para_dtw_matrix[n][m]

    list_cost = []
    for element in list_final_min_cost_function:
        cost = element[0] + element[1] * etaTdata + element[2] * etaTdata * etaTdata
        list_cost.append(cost)

    optimal_cost_function = list_final_min_cost_function[np.argmin(list_cost)]

    # Sign selection event
    list_se_sign_u_0, list_se_sign_u_1 = util.sign_selection_event(nu, Theta, a, b)

    # DP event interval
    list_se_dp_u_0 = []
    list_se_dp_u_1 = []
    list_se_dp_u_2 = []

    for element in list_final_min_cost_function:
        if element != optimal_cost_function:
            list_se_dp_u_0.append(optimal_cost_function[0] - element[0])
            list_se_dp_u_1.append(optimal_cost_function[1] - element[1])
            list_se_dp_u_2.append(optimal_cost_function[2] - element[2])

    list_1_interval, list_2_interval = util.find_truncation_interval_dp(list_se_dp_u_0, list_se_dp_u_1, list_se_dp_u_2)

    # Sign event interval
    Vminus_sign, Vplus_sign = util.find_truncation_interval_sign(list_se_sign_u_0, list_se_sign_u_1)

    final_list_interval = util.intersect_interval(
        util.intersect(list_1_interval[0], [Vminus_sign, Vplus_sign]),
        list_2_interval)

    cov = np.identity(n + m)
    p_value = 1 - util.pivot_with_constructed_interval(final_list_interval, eta, etaTdata, cov, 0)

    print('p-value:', p_value)


if __name__ == '__main__':
    run()