"""Compulsions as an epistemic trap: policy-induced non-identifiability of control.

Package layout
--------------
hazard      Analytic core: lambda_k = q (1-p)^k, likelihood, Fisher information.
grid        Grid-based Bayesian posterior over (q, p).
metrics     Entropy / correlation / information-gain / identifiability metrics.
policies    Belief-to-action (number of preventive repetitions k) policies.
simulate    Episode-level generative simulation and experiment drivers.
pomdp       Rigoux et al. (2024) POMDP replication (clean/dirty x cook/wash/check).
plotstyle   Shared publication-quality matplotlib style.
"""

__version__ = "0.1.0"
