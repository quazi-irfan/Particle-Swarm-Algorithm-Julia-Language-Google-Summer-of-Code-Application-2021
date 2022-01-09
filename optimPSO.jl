## Testing Optim.jl library
## Author: Quazi Irfan
# Source of Optim PSO implementation: # https://github.com/JuliaNLSolvers/Optim.jl/blob/master/src/multivariate/solvers/zeroth_order/particle_swarm.jl

using Optim

## Section 1
# Minimize rosenbrock using NelderMead method
rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
result1 = optimize(rosenbrock, [0.0,0.0], NelderMead())
Optim.minimizer(result1) # Result 0.99, 0.99


## Section 2
# Minimize rosenbrock using Aadaptive PSO
const rlow = [-1.5,-0.5]
const rupp = [2.0, 3.0]
# const rlow = [-1.5, -3.0]
# const rupp = [4.0, 4.0]
result2 = optimize(rosenbrock, [0.0,0.0], ParticleSwarm(;lower=rlow, upper=rupp, n_particles=1000))
Optim.minimizer(result2) # Result 1.0, 1.0

# The minimizers obtained via NedlderMead and Adaptive PSO methods are similar






### Testing other functions

# https://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/#minimizing-a-univariate-function-on-a-bounded-interval
result3 = optimize(x -> x^2, -10.0, 10.0) 
Optim.minimizer(result3)

# Source: https://rosettacode.org/wiki/Particle_swarm_optimization#Julia
# McCormick function - bowl-shaped, with a single minimum
# https://www.sfu.ca/~ssurjano/mccorm.html
# http://benchmarkfcns.xyz/benchmarkfcns/mccormickfcn.html
const mcclow = [-1.5, -3.0]
const mccupp = [4.0, 4.0]
mccormick(x) = sin(x[1] + x[2]) + (x[1] - x[2])^2 - 1.5 * x[1] + 2.5 * x[2] + 1
result4 = optimize(mccormick, [0.0, 0.0], ParticleSwarm(;lower=mcclow, upper=mccupp, n_particles=100))
Optim.minimizer(result4) # -0.54, -1.54 (Which matches the expected result)

# Michalewicz function - steep ridges and valleys, with multiple minima
# https://www.sfu.ca/~ssurjano/michal.html
# const miclow = [0.0, 0.0]
# const micupp = Float64.([pi, pi])
# michalewicz(x, m=10) = -sum(i -> sin(x[i]) * (i * sin( x[i]^2/pi))^(2*m), 1:length(x))
# result = optimize(mccormick, [0.0, 0.0], ParticleSwarm(;lower=mcclow, upper=mccupp, n_particles=100))
# Optim.minimizer(result) # -0.54, -1.54 (Which matches the expected result)

