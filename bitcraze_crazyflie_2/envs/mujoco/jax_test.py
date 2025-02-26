import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'METAL')

A = jnp.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]])
B = jnp.linalg.inv(A)
print(B)