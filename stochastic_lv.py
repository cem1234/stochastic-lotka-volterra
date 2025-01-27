import jax
import jax.numpy as jnp
from diffrax import (
    diffeqsolve, ODETerm, MultiTerm, MultiBrownianPath,
    Heun, SaveAt, PIDController
)


# -------------------------
# 1) Problem Setup
# -------------------------

# Domain/grid parameters
Lx, Ly = 1.0, 1.0    # domain size in x and y
Nx, Ny = 32, 32      # number of grid points in x, y
dx = Lx / Nx
dy = Ly / Ny

# Reaction-diffusion parameters
Du = 0.01             # Diffusivity for u
Dv = 0.01             # Diffusivity for v
Ds_b = 0.02           # Diffusivity for sigma_beta
Ds_d = 0.02           # Diffusivity for sigma_delta

alpha = 1.0           # Prey growth
beta_mean = 1.0       # Mean beta (overline{beta})
delta_mean = 1.0      # Mean delta (overline{delta})
gamma = 1.0           # Predator death rate

# Time parameters
t0, t1 = 0.0, 2.0     # start, end times

# Initial condition arrays (shape = (4, Nx, Ny)):
#   state[0] = u
#   state[1] = v
#   state[2] = sigma_beta
#   state[3] = sigma_delta
u0 = 1.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (Nx, Ny))
v0 = 0.5 + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (Nx, Ny))
sb0 = 0.2 + 0.0 * jnp.zeros((Nx, Ny))   # start sigma_beta nearly uniform
sd0 = 0.2 + 0.0 * jnp.zeros((Nx, Ny))   # start sigma_delta nearly uniform

# Combine into one initial state
Y0 = jnp.stack([u0, v0, sb0, sd0], axis=0)  # shape (4, Nx, Ny)


# -------------------------
# 2) Periodic Laplacian
# -------------------------
def laplacian_2d(u):
    """
    Compute the 2D Laplacian with periodic boundary conditions.
    u is shape (Nx, Ny).
    Returns lap(u) of shape (Nx, Ny).
    """
    # Roll up/down/left/right to implement periodic BC
    u_xp = jnp.roll(u, shift=-1, axis=0)
    u_xm = jnp.roll(u, shift=+1, axis=0)
    u_yp = jnp.roll(u, shift=-1, axis=1)
    u_ym = jnp.roll(u, shift=+1, axis=1)
    return (u_xp + u_xm + u_yp + u_ym - 4.0*u) / (dx*dy)  # or dx^2 if dx=dy


# -------------------------
# 3) Drift Function (f)
# -------------------------
def drift_fn(t, Y):
    """
    The 'drift' (deterministic) part of dY/dt = f(t,Y) + ...
    Here Y has shape (4, Nx, Ny):
       Y[0] = u
       Y[1] = v
       Y[2] = sigma_beta
       Y[3] = sigma_delta
    """
    u, v, sb, sd = Y

    # Compute Laplacians
    lap_u  = jax.vmap(laplacian_2d)(u[None])[0]   # shape (Nx, Ny)
    lap_v  = jax.vmap(laplacian_2d)(v[None])[0]
    lap_sb = jax.vmap(laplacian_2d)(sb[None])[0]
    lap_sd = jax.vmap(laplacian_2d)(sd[None])[0]

    # Reaction terms for u, v
    # Noise terms (beta, delta) *will* come via separate diffusion function
    # but here we keep the *mean* parts in the drift.
    du = Du * lap_u + alpha*u - beta_mean * u * v
    dv = Dv * lap_v + delta_mean*u*v - gamma*v

    # Diffusion for sigma_beta, sigma_delta
    dsb = Ds_b * lap_sb
    dsd = Ds_d * lap_sd

    return jnp.stack([du, dv, dsb, dsd], axis=0)


# -------------------------
# 4) Diffusion Function (g)
# -------------------------
def diffusion_fn(t, Y):
    """
    The 'diffusion' (stochastic) part of dY = ... + g(t, Y) dW.
    We'll produce a shape (state_shape, noise_dim) array
    for each noise dimension.

    We assume 2 independent Wiener processes:
      W1(t) for the beta noise,
      W2(t) for the delta noise.

    => noise_dim = 2
    => g(t, Y) has shape (4, Nx, Ny, 2).
    """
    u, v, sb, sd = Y

    # The local amplitude for beta-noise is sb(x,y,t)*u*v
    # The local amplitude for delta-noise is sd(x,y,t)*u*v
    # Because we are applying global dW_1(t), dW_2(t), each cell uses
    # the same dW_1(t) (and dW_2(t)) but scaled by the local amplitude.
    # So for dimension #0 (beta noise):
    g_beta = sb * u * v  # shape (Nx, Ny)

    # For dimension #1 (delta noise):
    g_delta = sd * u * v # shape (Nx, Ny)

    # No direct noise on sigma_beta, sigma_delta themselves in this example.
    # So the last two components (for Y[2], Y[3]) are zero for both noise dims.
    zeros = jnp.zeros_like(sb)

    # Stack into shape (4, Nx, Ny, 2)
    #   dimension ordering: [component of Y, Nx, Ny, noise_dim]
    g_out = jnp.stack([
        jnp.stack([g_beta, g_delta], axis=-1),  # for u
        jnp.stack([0.0*g_beta, 0.0*g_delta], axis=-1),  # for v? Or also for v?
        jnp.stack([zeros, zeros], axis=-1),     # sigma_beta
        jnp.stack([zeros, zeros], axis=-1)      # sigma_delta
    ], axis=0)
    return g_out


# -------------------------
# 5) Wrap into an SDE
# -------------------------
# In Diffrax, we combine the drift and diffusion into separate "Term" objects
# and then combine them into a MultiTerm if we have multiple Wiener processes.
# However, we have a single 'diffusion_fn' returning a shape [..., 2].
# We can use 'MultiTerm' with one "SDETerm" that has 'diffusion' = diffusion_fn
# and specify the dimension = 2 for the Brownian motion.

from diffrax import SDETerm

sde_term = SDETerm(drift=drift_fn, diffusion=diffusion_fn)

# We'll combine into a MultiTerm for convenience.
# But here we have just one SDETerm (which covers 2 noise dims):
terms = MultiTerm(sde_term)


# -------------------------
# 6) Initial Brownian Paths
# -------------------------
# We want 2D noise => dimension=2
bm = MultiBrownianPath(
    shape=(2,),  # we have two independent Wiener processes
    key=jax.random.PRNGKey(42),
)


# -------------------------
# 7) Solve
# -------------------------
def flatten_fn(Y):
    return Y.reshape(-1)

def unflatten_fn(y_flat):
    return y_flat.reshape(4, Nx, Ny)

# We'll do the SDE solve in *flattened* form. Diffrax requires that
# the state be a 1D array for classical solvers. We'll wrap/unwrap inside.

def sde_drift(t, y):
    # unflatten
    Y = unflatten_fn(y)
    return flatten_fn(drift_fn(t, Y))

def sde_diffusion(t, y):
    Y = unflatten_fn(y)
    g = diffusion_fn(t, Y)  # shape (4, Nx, Ny, 2)
    # flatten each noise dimension separately
    # so final shape => (state_size, 2)
    g_flat = jnp.stack([
        flatten_fn(g[..., 0]),
        flatten_fn(g[..., 1])
    ], axis=-1)
    return g_flat

# Now build a "multi-term" where each dimension is accounted for:
term = SDETerm(drift=sde_drift, diffusion=sde_diffusion)
sde = MultiTerm(term)

# We pick an SDE solver, e.g. Heun (a stochastic version of the midpoint method)
solver = Heun()

# Wrap initial condition
y0 = flatten_fn(Y0)

# Solve
saveat = SaveAt(ts=jnp.linspace(t0, t1, 21))  # save ~21 frames
stepsize_controller = PIDController(rtol=1e-3, atol=1e-3)  # or set fixed step

solution = diffeqsolve(
    sde,
    solver=solver,
    t0=t0,
    t1=t1,
    dt0=1e-2,              # initial guess for step size
    y0=y0,
    brownian_motion=bm,
    saveat=saveat,
    stepsize_controller=stepsize_controller,
)

# solution.ys will be a list of flattened states at each saved time
# We can unflatten them for analysis or visualization:
import matplotlib.pyplot as plt

for i, (tval, y_flat) in enumerate(zip(solution.ts, solution.ys)):
    Y_t = unflatten_fn(y_flat)
    u_t = Y_t[0]  # shape (Nx, Ny)
    v_t = Y_t[1]  # shape (Nx, Ny)
    # e.g. visualize or store
    if i % 5 == 0:
        plt.figure(figsize=(8,3))
        plt.suptitle(f"t = {tval:.2f}")
        plt.subplot(1,2,1)
        plt.imshow(u_t, origin="lower", cmap="viridis")
        plt.colorbar(label="u (prey)")
        plt.subplot(1,2,2)
        plt.imshow(v_t, origin="lower", cmap="magma")
        plt.colorbar(label="v (predators)")
        plt.show()
