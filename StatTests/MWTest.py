import jax
import jax.numpy as jnp
from jax import lax

##############################
# JIT–compatible rankdata without dynamic slicing
##############################
def rankdata(a):
    """
    Assign ranks to the 1D array `a`, averaging ranks for tied values.
    This version avoids jnp.unique and dynamic slicing, and is JIT–compilable.

    Parameters:
       a: 1D array.

    Returns:
       ranks: an array of the same shape as a with the rank (starting at 1) for each element.
    """
    n = a.shape[0]
    # Get the indices that would sort the array.
    order = jnp.argsort(a)
    # Sort the array.
    a_sorted = a[order]
    # Preliminary ranks (1-indexed) in sorted order.
    ranks_sorted = jnp.arange(1, n + 1, dtype=jnp.float32)
    # Precompute cumulative sum of the preliminary ranks.
    cum_ranks = jnp.cumsum(ranks_sorted)
    # Initialize output array (for sorted ranks).
    sorted_out = jnp.zeros(n, dtype=jnp.float32)

    # The outer loop over groups: state=(i, sorted_out)
    def outer_cond(state):
        i, _ = state
        return i < n

    def outer_body(state):
        i, sorted_out = state

        # Inner loop: find group_end such that all elements from i to group_end-1 are tied.
        def inner_cond(j):
            # Continue while j < n and a_sorted[j] == a_sorted[i]
            return (j < n) & (a_sorted[j] == a_sorted[i])

        def inner_body(j):
            return j + 1

        group_end = lax.while_loop(inner_cond, inner_body, i)
        t = group_end - i  # number of tied elements (an integer scalar)

        # Compute the sum over this tie group from the cumulative sum.
        group_sum = lax.cond(
            i > 0,
            lambda _: cum_ranks[group_end - 1] - cum_ranks[i - 1],
            lambda _: cum_ranks[group_end - 1],
            operand=None,
        )
        avg_rank = group_sum / t

        # Inner loop to update sorted_out at indices i to group_end-1 with avg_rank.
        def update_body(j, out):
            idx = i + j
            return out.at[idx].set(avg_rank)

        sorted_out = lax.fori_loop(0, t, update_body, sorted_out)
        return (group_end, sorted_out)

    _, final_sorted = lax.while_loop(outer_cond, outer_body, (0, sorted_out))
    # Scatter the sorted ranks back into the original order.
    final_ranks = jnp.zeros_like(final_sorted)
    final_ranks = final_ranks.at[order].set(final_sorted)
    return final_ranks


##############################
# Helper: Tie Correction Term for the Rank–Sum Test
##############################
def tie_term(a):
    """
    Compute the tie–term ∑(t^3 - t) from the 1D array a.
    This is used to correct the variance of the Mann–Whitney U statistic.
    This version avoids jnp.unique.

    Parameters:
       a: 1D array.

    Returns:
       tie_total: a scalar equal to ∑(t^3 - t), where t is the number of tied values in each group.
    """
    n = a.shape[0]
    a_sorted = jnp.sort(a)

    def cond_fun(state):
        i, total = state
        return i < n

    def body_fun(state):
        i, total = state

        def inner_cond(j):
            return (j < n) & (a_sorted[j] == a_sorted[i])

        group_end = lax.while_loop(inner_cond, lambda j: j + 1, i)
        t = group_end - i
        total = total + (t**3 - t)
        return (group_end, total)

    _, total = lax.while_loop(cond_fun, body_fun, (0, 0.0))
    return total


##############################
# Standard Normal CDF
##############################
def norm_cdf(x):
    return 0.5 * (1.0 + lax.erf(x / jnp.sqrt(2.0)))


##############################
# Wilcoxon Rank–Sum Test (Mann–Whitney U Test)
##############################
def wilcoxon_rank_sum_test(x, y):
    """
    Wilcoxon rank–sum test (Mann–Whitney U test) for independent samples.

    Parameters:
       x: 1D array of sample values (group 1).
       y: 1D array of sample values (group 2).

    Returns:
       U: The Mann–Whitney U statistic for group x.
       p_value: Two–tailed p–value (using a normal approximation).
    """
    n1 = x.shape[0]
    n2 = y.shape[0]
    N = n1 + n2
    combined = jnp.concatenate([x, y])
    # Rank the combined data.
    ranks = rankdata(combined)
    # Sum of ranks for group x.
    R1 = jnp.sum(ranks[:n1])
    U = R1 - n1 * (n1 + 1) / 2.0

    # Under H0, the mean of U is:
    mu_U = n1 * n2 / 2.0
    # Tie correction using the original combined data.
    t_term = tie_term(combined)
    sigma_U = jnp.sqrt(n1 * n2 / 12.0 * ((N + 1) - t_term / (N * (N - 1))))

    # Apply continuity correction.
    z = (U - mu_U + 0.5) / sigma_U
    p_value = 2 * (1.0 - norm_cdf(jnp.abs(z)))
    return U, p_value


##############################
# Wilcoxon Signed–Rank Test for Paired Data
##############################
def wilcoxon_signed_rank_test(x, y):
    """
    Wilcoxon signed–rank test for paired samples.

    Parameters:
       x: 1D array of sample values for condition 1.
       y: 1D array of sample values for condition 2.

    Returns:
       W: Sum of ranks for positive differences.
       p_value: Two–tailed p–value (using a normal approximation).

    Note:
       This implementation does not specially handle zeros (ties with zero difference).
    """
    # Compute paired differences.
    d = x - y
    abs_d = jnp.abs(d)
    # Rank the absolute differences.
    ranks = rankdata(abs_d)
    # Sum the ranks corresponding to positive differences.
    W = jnp.sum(jnp.where(d > 0, ranks, 0.0))
    n = x.shape[0]
    mu_W = n * (n + 1) / 4.0
    sigma_W = jnp.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu_W - 0.5) / sigma_W
    p_value = 2 * (1.0 - norm_cdf(jnp.abs(z)))
    return W, p_value


##############################
# JIT–Compilable Wilcoxon Test Interface
##############################
@jax.jit
def wilcoxon_test(x, y, paired=False):
    """
    Perform a Wilcoxon test.

    Parameters:
       x: 1D array of sample values from the first distribution.
       y: 1D array of sample values from the second distribution.
       paired: If True, perform the Wilcoxon signed–rank test (paired data);
               if False, perform the Wilcoxon rank–sum (Mann–Whitney U) test (independent samples).

    Returns:
       statistic: For independent samples, the Mann–Whitney U statistic for x;
                  for paired samples, the sum of ranks for positive differences.
       p_value: Two–tailed p–value based on a normal approximation.
    """
    return lax.cond(
        paired,
        lambda _: wilcoxon_signed_rank_test(x, y),
        lambda _: wilcoxon_rank_sum_test(x, y),
        operand=None,
    )
