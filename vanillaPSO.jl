## Vanilla Partile Swarm Optimization Algorithm
## Author: Quazi Irfan

using Plots,Random, Statistics, Colors, Distributions
Random.seed!(123)

# setup
# function f(x::AbstractFloat)::AbstractFloat
#     return x^2
#     #return (2-x)^3 + x^2 # https://www.desmos.com/calculator/k5pqd1ydhb
# end

## User function
# f(x)::Float64 = x^2;
f(x)::Float64 = x^4 -8x^2 + 2x;
# f(x)::Float64 = cos(x);
x_min = -4;
x_max = 4;
numOfParticles = 10;
numOfIteration = 25;

## Initilization
w = 1.5;
scale1 = 1;
scale2 = 2;


# Draw the user function
x_interval = (x_max - x_min)/100;
x = [x_min:x_interval:x_max;]; 
plot(x, f.(x),title = "Iteration 0", label="User function") 

# generate particles
particlePosOffset = (x_max - x_min) * .05; # so particle do not appear at the edge, only for visualization
particlePos = collect(range(x_min+particlePosOffset, x_max-particlePosOffset, length=numOfParticles)) #.+ rand(-1:0.01:1, numOfParticles); 
particleVel = zeros(Float64, numOfParticles); 
#particleVel = rand(-1:0.001:1, numOfParticles); #set random velocity


# draw the initial particles
# Source: http://juliagraphics.github.io/Colors.jl/stable/colormapsandcolorscales/#Color-interpolation
# color_names link: https://github.com/JuliaGraphics/Colors.jl/blob/master/src/names_data.jl
tempColor = mc=range(HSL(colorant"red"), stop=HSL(colorant"green"), length=numOfParticles);
scatter!(particlePos, f.(particlePos), mc=tempColor, label="")
plot!([particlePos particlePos.+particleVel]', [f.(particlePos) f.(particlePos)]', color="black", label="")
savefig("C:\\Users\\iamcr\\Desktop\\GSoC\\2021\\Julia\\Plots\\plot0.png")

# if pos < 0 then add c*2pi, if pos > 2pi then subtract c*2pi
function periodicClamp(x, x_min, x_max)
    periodLength = x_max - x_min

    while x < x_min
        x += periodLength
    end
    
    while x > x_max
        x -= periodLength
    end
    
    return x
end;

## Optimization Step
# at the beginning best position of a particle is its current position
particleBestPos = particlePos;
# initialize an empty array to save the cost change 
absAvgCostDiffHistory = zeros(numOfIteration); 

for i = 1:numOfIteration
    swarmBestPos = particlePos[argmin(f.(particlePos))]; #find the particle pos with lowest cost

    m = particleVel;
    stochastic1 =  rand(Uniform(), 1);
    stochastic2 =  rand(Uniform(), 1);
    p = (stochastic1 * scale1) .* (particleBestPos - particlePos);
    s = (stochastic2 * scale2) .* (swarmBestPos .- particlePos);
    particleVel = m .+  p .+ s;
    particleVel = clamp.(particleVel, -.25, .25)
    particlePos = particlePos .+ particleVel.*1; # considering unit time

    particlePos = periodicClamp.(particlePos, x_min, x_max);

    # redraw the new plots with updated particle position
    plot(x, f.(x), title = string("Iteration ", i), label="User function")  
    # plot(x, f.(x), label="User function")  
    scatter!(particlePos, f.(particlePos), color=tempColor, label="")
    plot!([particlePos particlePos.+particleVel]', [f.(particlePos) f.(particlePos)]', color="black", label="")
    savefig("C:\\Users\\iamcr\\Desktop\\GSoC\\2021\\Julia\\Plots\\plot" * string(i) * ".png")
    plot!([particlePos particlePos.+particleVel * 2]', [f.(particlePos) f.(particlePos)]', color="black", label="")
    
    # calculate the improvement before updating candidate solution
    absAvgCostDiffHistory[i] = abs(mean(particleBestPos - particlePos));

    # Before going to a new iteration check if the new position are better than the older ones
    mask = f.(particlePos) .<= f.(particleBestPos);
    particleBestPos = (.!mask .* particleBestPos) .+ (mask .* particlePos);    
end

plot(absAvgCostDiffHistory, title="Cost history", xlabel="Num of Iteration", ylabel = "Average change in position", label="")
savefig("C:\\Users\\iamcr\\Desktop\\GSoC\\2021\\Julia\\Plots\\CostHistory.png")

#vanillaPSO(x -> x^2, -10.0, 10.0)