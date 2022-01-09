# Section 1 : PSO with inertia weight parameter
# https://www.rdocumentation.org/packages/metaheuristicOpt/versions/2.0.0
library(metaheuristicOpt)

rosenbrock <- function(x) {
  return ((1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2)
}

Vmax <- 2
ci <- 1.5
cg <- 1.5
w <- 0.7
numVar <- 2
rangeVar <- matrix(c(-1.5,2.0,-0.5,3.0), ncol=2)

resultRosenPSO <- PSO(rosenbrock, optimType="MIN", numVar, numPopulation=20, maxIter = 100, rangeVar, Vmax, ci, cg, w)
resultRosenPSO # returns 0.9996966 0.9993824 and expected value is 1, 1

#####################################################
# Section 2
# Multi objective function Optimization
# Source: https://rdrr.io/cran/mopsocd/man/examples.html

library(mopsocd)

viennet <- function(x){
  f1 <- 0.5*(x[1]^2+x[2]^2)+sin(x[1]^2+x[2]^2)
  f2 <- 0.125*(3*x[1]-2*x[2]+4)^2+(1.0/27.0)*(x[1]-x[2]+1)^2+15
  f3 <- 1.0/(x[1]^2+x[2]^2+1)-1.1*exp(-(x[1]^2+x[2]^2))
  return(c(f1,f2,f3))
}

## Set Arguments
varcount <- 2
fncount <- 3
lbound <- c(-3,-3)
ubound <- c(3,3)
optmin <- 0

ex1 <- mopsocd(viennet,varcnt=varcount,fncnt=fncount,lowerbound=lbound,upperbound=ubound,opt=optmin)
## Access Pareto Object Fields
print(ex1$numsols)
print(ex1$objfnvalues)
print(ex1$paramvalues)

## Plot
library(scatterplot3d)
scatterplot3d(ex1$objfnvalues[,1],ex1$objfnvalues[,2],ex1$objfnvalues[,3])

#########################################################
# Section 3
# PSO with constrain

library(pso)

fitness <- function(x){
  d <- x[1]
  D <- x[2]
  N <- round(x[3])
  
  # define fitness function
  fitness_value <- (N+2)*D*d^2
  
  #define constraint
  g1 <- 1 - D^3*N/(71785*d^4)
  g2 <- (4*D^2-d*D)/(12566*(D * d^3 - d^4)) + 1/(5108 * d^2) - 1
  g3 <- 1 - (140.45*d)/(D^2*N)
  g4 <- (D+d)/1.5 - 1
  
  #penalized constraint violation
  fitness_value <- ifelse( g1 <= 0 & g2 <= 0 & g3 <= 0 & g4 <= 0, fitness_value, fitness_value + 1e3 )
  
  return(fitness_value)
}

set.seed(90)
psoptim(rep(NA,3), fn = fitness, lower = c(0.05, 0.25, 2), upper = c(2, 1.3, 15))

# convergence: An integer code. 0 indicates that the algorithm terminated by reaching the absolute tolerance; otherwise:
#  1: Maximal number of function evaluations reached
#  2: Maximal number of iterations reached.
#  3: Maximal number of restarts reached.
#  4: Maximal number of iterations without improvement reached
