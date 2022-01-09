using CSV # Source: JuliaCon 2020 | MLJ: A Machine Learning Toolbox for Julia : https://www.youtube.com/watch?v=qSWbCn170HU&t=27s
file = CSV.File("horse.csv");

using DataFrames
horse = DataFrames.DataFrame(file);

using MLJ
coerce!(horse, autotype(horse));
coerce!(horse, Count => Continuous); #convert all Count data to Continuous
# name of column and type
coerce!(horse, :surgery => Multiclass, :age => Multiclass, :mucous_membranes => Multiclass, 
:capillary_refill_time => Multiclass, :outcome => Multiclass, :cp_data => Multiclass);

y, X = unpack(horse,
              ==(:outcome),
              name -> elscitype(Tables.getcolumn(horse, name)) == Continuous);
train, test = partition(eachindex(y), 0.7, shuffle=true);

# put model type into scope, 
# instantiate the model with default parameters
# pkg name is imp when model with same name belong to multiple package
lc = @load LogisticClassifier pkg=MLJLinearModels verbosity=0;  #load model
model = lc(); # initialization
model.lambda = 100;
mach = machine(model, X, y);
fit!(mach, rows=train);
yhat = predict(mach, rows=test);
misclassification_rate(mode.(yhat), y[test])

tuning = RandomSearch(rng=123)
r = range(model, :lambda, lower=1, upper=150, scale=:log);
tuned_model = TunedModel(model=model,
                         ranges=r,
                         resampling=CV(nfolds=6),
                         measures=cross_entropy,
                         tuning=Grid(resolution=10),
                         n=30)
self_tuning_mach = machine(tuned_model, X, y)
fit!(self_tuning_mach) # self tuning - optimizatio happens here - optimizes it using the strategy we've set, and once the best model is found - it will retrain on all available data and store that result
ythat = predict(self_tuning_mach, rows=test);
misclassification_rate(mode.(ythat), y[test])

curve = learning_curve(mach,
                       range=r,
                       resampling=Holdout(fraction_train=0.7), # (default)
                       measure=cross_entropy)

using Plots
plot(curve.parameter_values, curve.measurements)
xlabel!(plt, "epochs")
ylabel!(plt, "cross entropy on holdout set")




using MLJ
X = (x1=rand(100), x2=rand(100), x3=rand(100));
y = 2X.x1 - X.x2 + 0.05*rand(100);
dtr = @load DecisionTreeRegressor pkg=DecisionTree;
tree_model = dtr();

# set the range of hyper parameter
r = range(tree_model, :min_purity_increase, lower=0.001, upper=1.0, scale=:log);

# RandomSearch is a bunch of parameter for the prior distribution 
# these prioor dist will be used for sampling

# TunedModel wrap the model and the construction explain what kind of tuning we want to do
# the new model is kind of self tuning model

tuned_model = TunedModel(model=tree_model,
                                    resampling = CV(nfolds=3),
                                    tuning = Grid(resolution=10),
                                    ranges = r,
                                    measure = rms);
self_tuning_mach = machine(tuned_model, X, y)
fit!(self_tuning_mach) # self tuning - optimizatio happens here - optimizes it using the strategy we've set, and once the best model is found - it will retrain on all available data and store that result
f.best_model # prediction will pick the model with the optimal hyperparamter


# after instantiating the model we construct a machine
# a machine is model and data bounded
# its convenient to give it the complete dataset
mach = machine(model, X, y) # for supervised
fitted_params(mach) # inspect the trained parameter in a mamed tuple
# machine stores the actual learned parameters of the model

fit!(mach, rows=train, verbosity=2) # trains the model using train data and stores the parameters in machine

# if you change a hyperparameter using model.hp, then you retrain the modelj

predict(mach, rows=test) # given the machine(contains the trained model) and new dataset or row indecies
# predict method by default will predict probabilitiesj
# predict_mode returns the highest probability

# measures the performance of the model
cross_entropy(yhat, y[test]) |> mean
missclassification_rate(mode.(yhat), y[test])
measures(matching(y))

model =  @load NeuralNetworkClassifier
model.hyperparameters.epochs = 12


# lets say we have model bounded with data and we want to test it's performance
evaluate!(mach, resampling=Holdout(fraction_train=0.7), measures=[cross_entropy])

evaluate!(mach, resampling=CV(nfolds=6),measures=[cross_entropy])

# range takes the name of model, the name of hyperparameter, and its lower and upper range and a scale 




