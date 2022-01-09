import numpy as np
import pyswarms as ps

#######################################################################
# Section 1
# Rosenbrock function (Objective)
def rosenbrock(x):
  return ((1.0 - x[:, 0]) ** 2 + 100.0 * (x[:, 1] - x[:, 0] ** 2) ** 2)

x_bounds = (np.array([-1.5, -0.5]), np.array([2, 3]))

## PSO Global Best
#setup PSO parameters, w c1 and c2
pso_parametersg = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=40, bounds=x_bounds, dimensions=2, options=pso_parametersg)
cost, posg = optimizer.optimize(rosenbrock, iters=1000)
print(posg)


#######################################################################
# Section 2
# PSO Local best
# setup PSO parameters : num of neighbour k = 3, L2 distance(p) =2
pso_parametersl = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}
optimizer = ps.single.LocalBestPSO(n_particles=10, dimensions=2, options=pso_parametersl, bounds=x_bounds)
_, posl = optimizer.optimize(rosenbrock, iters=1000)
print(posl)

