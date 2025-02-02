import jax.numpy as jnp
import jax.scipy.special as jsp_special

def t_test(x, y, equal_var=True):
    """
    Perform a two-sample t-test on the samples x and y.

    Parameters:
      x (array): 1D array of sample values from the first distribution.
      y (array): 1D array of sample values from the second distribution.
      equal_var (bool): If True, perform Student's t test (assumes equal variances).
                        If False, perform Welch's t test (does not assume equal variances).

    Returns:
      t_stat (float): The computed t-statistic.
      p_value (float): The two-tailed p-value.

    For the Student's t test (equal_var=True), the t statistic is given by
         t = (mean1 - mean2) / sqrt(pooled_variance*(1/n1 + 1/n2))
    with pooled variance computed as:
         pooled_var = (((n1 - 1)*var1 + (n2 - 1)*var2) / (n1 + n2 - 2))
    and degrees of freedom (df) equal to (n1 + n2 - 2).

    For Welch's t test (equal_var=False), the t statistic is computed as
         t = (mean1 - mean2) / sqrt(var1/n1 + var2/n2)
    with degrees of freedom approximated by the Welch–Satterthwaite equation:
         df = (var1/n1 + var2/n2)^2 / [ (var1/n1)^2/(n1-1) + (var2/n2)^2/(n2-1) ]

    In both cases, the two-tailed p-value is computed via the regularized incomplete beta function:
         p = I_{df/(df+t_stat^2)}(df/2, 1/2)
    """
    n1 = x.shape[0]
    n2 = y.shape[0]
    mean1 = jnp.mean(x)
    mean2 = jnp.mean(y)

    # Compute sample variances with Bessel's correction.
    var1 = jnp.var(x, ddof=1)
    var2 = jnp.var(y, ddof=1)

    if equal_var:
        # Student's t test with pooled variance.
        df = n1 + n2 - 2
        pooled_var = (((n1 - 1) * var1) + ((n2 - 1) * var2)) / df
        t_stat = (mean1 - mean2) / jnp.sqrt(pooled_var * (1.0 / n1 + 1.0 / n2))
    else:
        # Welch's t test.
        t_stat = (mean1 - mean2) / jnp.sqrt(var1 / n1 + var2 / n2)
        numerator = (var1 / n1 + var2 / n2) ** 2
        denominator = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        df = numerator / denominator

    # Compute the two-tailed p-value.
    # The p-value is given by:
    #    p = I_{df/(df+t^2)}(df/2, 1/2)
    # where I_x(a,b) is the regularized incomplete beta function.
    p_value = jsp_special.betainc(df / 2.0, 0.5, df / (df + t_stat**2))

    return t_stat, p_value
