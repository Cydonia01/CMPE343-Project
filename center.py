import numpy as np
from numpy.random import default_rng

mu = np.array([20.0, 0.3, 0.8])
Sigma = np.array([[4.0, 0.5, 0.2],
                  [0.5, 0.7, 0.2],
                  [0.2, 0.2, 0.1]])

def g(x):
    x1, x2, x3 = x
    return 0.1*(x1**2) + 12.5*(x2**2) - 7.5*(x3**2)

# set seed
rng = default_rng(12345)

# Sample sizes for the experiments
sample_sizes = [50, 100, 1000, 10000]

# Sample sizes
n0 = 10000
n1 = 50

# Generate large sample
X_large = rng.multivariate_normal(mu, Sigma, n0)
g_vals_large = np.array([g(x) for x in X_large])
g0 = g_vals_large.mean()
S0_sq = g_vals_large.var(ddof=1)  # Sample variance with ddof=1

# Generate small sample
X_small = rng.multivariate_normal(mu, Sigma, n1)
g_vals_small = np.array([g(x) for x in X_small])
g1 = g_vals_small.mean()
S1_sq = g_vals_small.var(ddof=1)

# test statistic
Z = (g0 - g1) / np.sqrt(S0_sq/n0 + S1_sq/n1)

# Decision rule:
alpha = 0.05
z_crit = 1.96

print("g0 estimate:", g0)
print("g1 estimate:", g1)
print("Z statistic:", Z)

if abs(Z) > z_crit:
    print("Reject H0: g0 and g1 differ significantly.")
else:
    print("Fail to reject H0: No significant difference between g0 and g1.")

for n in sample_sizes:
    # Draw n samples from the multivariate normal
    X = rng.multivariate_normal(mu, Sigma, n)
    
    # Compute g(x) for each sample
    g_vals = np.array([g(x) for x in X])
    
    # Compute mean and sample std
    g_mean = g_vals.mean()
    g_std = g_vals.std(ddof=1)  # sample standard deviation with ddof=1
    
    # 95% confidence interval
    margin = z_crit * g_std / np.sqrt(n)
    ci_lower = g_mean - margin
    ci_upper = g_mean + margin
    
    print(f"n={n}:")
    print(f"  Estimate of E[g(x)]: {g_mean:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n")
