import numpy as np

def compare_data_to_dist(x, mu_1=5, mu_2=7, sd_1=3, sd_2=3):
    ll_1 = 0
    ll_2 = 0
    for i in x:
        ll_1 += np.log(np.linalg.norm.pdf(i, mu_1, sd_1))
        ll_2 += np.log(np.linalg.norm.pdf(i, mu_2, sd_2))

    print("The LL of of x for mu = %d and sd = %d is: %.4f" % (mu_1, sd_1, ll_1))
    print("The LL of of x for mu = %d and sd = %d is: %.4f" % (mu_2, sd_2, ll_2))

x = [4, 5, 7, 8, 8, 9, 10, 5, 2, 3, 5, 4, 8, 9]
compare_data_to_dist(x)