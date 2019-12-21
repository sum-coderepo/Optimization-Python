import matplotlib.pyplot as plt
import numpy as np
def plot_ll(x):
    plt.figure(figsize=(5,8))
    plt.title("Maximim Likelihood Functions")
    plt.xlabel("Mean Estimate")
    plt.ylabel("Log Likelihood")
    plt.ylim(-40, -30)
    plt.xlim(0, 12)
mu_set = np.linspace(0, 16, 1000)
sd_set = [.5, 1, 1.5, 2.5, 3, 3.5]
max_val = max_val_location = None

for i in sd_set:
    ll_array = []

    for j in mu_set:
        temp_mm = 0

        for k in x:
            temp_mm += np.log(norm.pdf(k, j, i)) # The LL function
        ll_array.append(temp_mm)

        if (max_val is None):
            max_val = max(ll_array)
elif max(ll_array) > max_val:
max_val = max(ll_array)
max_val_location = j

# Plot the results
plt.plot(mu_set, ll_array, label="sd: %.1f" % i)

print "The max LL for sd %.2f is %.2f" % (i, max(ll_array))
plt.axvline(x=max_val_location, color='black', ls='-.')
plt.legend(loc='lower left')
plot_ll(x);
