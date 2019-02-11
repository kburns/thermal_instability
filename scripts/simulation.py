"""
Heated RBC
    H = A * sin(π*b/B)

Parameters:
    Lz [L] Domain height
    ν [L**2 / t] Viscosity
    κ [L**2 / t] Diffusivity
    A [L / t**3] Heating amplitude
    B [L / t**2] Heating scale

Scales:
    L = Lz
    t**2 = Lz / B

Nondimensionalized parameters:
    Pr = ν / κ
    Ra = B * Lz**3 / ν / κ
    He = A / B * Lz**2 / κ
    Hs = A * Lz**(1/2) / B**(3/2)

    Lz * 1 / L = 1
    ν * t / L**2 = ν * Lz**(-3/2) * B**(-1/2) = (Ra / Pr)**(-1/2)
    κ * t / L**2 = κ * Lz**(-3/2) * B**(-1/2) = (Ra * Pr)**(-1/2)
    A * t**3 / L = A * Lz**(1/2) * B**(-3/2) = He * (Ra * Pr)**(-1/2) = Hs
    B * t**2 / L = 1

"""

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = (2., 1.)
Prandtl = 1.
Rayleigh = 1e6
Heating = 1e1

# Create bases and domain
x_basis = de.Fourier('x', 256, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', 128, interval=(-Lz/2, Lz/2), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','b','u','w','bz','uz','wz'])
problem.meta[:]['z']['dirichlet'] = True
problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
problem.parameters['π'] = np.pi
problem.parameters['A'] = Heating
problem.substitutions['H'] = "A*sin(π*b)"
problem.add_equation("dx(u) + wz = 0")
problem.add_equation("dt(b) - P*(dx(dx(b)) + dz(bz))             = -(u*dx(b) + w*bz) + H")
problem.add_equation("dt(u) - R*(dx(dx(u)) + dz(uz)) + dx(p)     = -(u*dx(u) + w*uz)")
problem.add_equation("dt(w) - R*(dx(dx(w)) + dz(wz)) + dz(p) - b = -(u*dx(w) + w*wz)")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_bc("left(b) = 1")
problem.add_bc("left(u) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(b) = -1")
problem.add_bc("right(u) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
problem.add_bc("right(p) = 0", condition="(nx == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
b = solver.state['b']
bz = solver.state['bz']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert =  1e-1 * noise * (zt - z) * (z - zb)
b['g'] = -z/(Lz/2) + pert
b.differentiate('z', out=bz)

# Initial timestep
dt = 2**-6

# Integration parameters
solver.stop_sim_time = 10
solver.stop_wall_time = 30 * 60.
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.05, max_writes=10, mode='overwrite')
snapshots.add_system(solver.state)
snapshots.add_task('H')

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.5,
                     max_change=1.1, min_change=0.5, max_dt=dt, threshold=0.05)
CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + w*w) / R", name='Re')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Re = %f' %flow.max('Re'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
