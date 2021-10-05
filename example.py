"""Minimal NTK example."""
from jax import random
from jax import numpy as jnp
from jax.experimental import stax
from fast_finite_width_ntk import empirical

key1, key2, key3 = random.split(random.PRNGKey(1), 3)
x1 = random.normal(key1, (6, 8, 8, 3))
x2 = random.normal(key2, (3, 8, 8, 3))

# A vanilla CNN.
init_fn, f = stax.serial(
    stax.Conv(32, (3, 3)),
    stax.Relu,
    stax.Conv(32, (3, 3)),
    stax.Relu,
    stax.Conv(32, (3, 3)),
    stax.Flatten,
    stax.Dense(10)
)

_, params = init_fn(key3, x1.shape)
kwargs = dict(
    f=f,
    trace_axes=(),
    vmap_axes=0,
)


# Default, baseline Jacobian contraction.
jacobian_contraction = empirical.empirical_ntk_fn(
    **kwargs,
    implementation=empirical.NtkImplementation.JACOBIAN_CONTRACTION)

# (6, 3, 10, 10) full `np.ndarray` test-train NTK
ntk_jc = jacobian_contraction(x2, x1, params)


# NTK-vector products-based implementation.
ntk_vector_products = empirical.empirical_ntk_fn(
    **kwargs,
    implementation=empirical.NtkImplementation.NTK_VECTOR_PRODUCTS)

ntk_vp = ntk_vector_products(x2, x1, params)


# Structured derivatives-based implementation.
structured_derivatives = empirical.empirical_ntk_fn(
    **kwargs,
    implementation=empirical.NtkImplementation.STRUCTURED_DERIVATIVES)

ntk_sd = structured_derivatives(x2, x1, params)


# Auto-FLOPs-selecting implementation. Doesn't work correctly on CPU/GPU.
auto = empirical.empirical_ntk_fn(
    **kwargs,
    implementation=empirical.NtkImplementation.AUTO)

ntk_auto = auto(x2, x1, params)


# Check that implementations match
for ntk1 in [ntk_jc, ntk_vp, ntk_sd, ntk_auto]:
    for ntk2 in [ntk_jc, ntk_vp, ntk_sd, ntk_auto]:
        diff = jnp.max(jnp.abs(ntk1 - ntk2))
        print(f'NTK implementation diff {diff}.')
        assert diff < 1e-4

print('All NTK implementations match.')
