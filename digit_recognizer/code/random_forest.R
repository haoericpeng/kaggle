library(doParallel)
library(randomForest)

# Load and process data
source("load_process_data.R")

# Set seed for random number generator
set.seed(888)

# Define values of tuning parameters (subsampling size and number of features
# to try for node splitting) to search over
numTrainCases <- length(trainLabels)
sampleSizes <- floor(numTrainCases * c(0.2, 0.4, 0.6, 0.8, 1))
featureNumbers <- floor(ncol(trainFeatures) ^ (1 / c(3, 2.5, 2, 1.5)))

# Calculate OOB errors for each model
outOfBagErrors <- matrix(nrow = length(sampleSizes),
                         ncol = length(featureNumbers))
numTrees <- 400
for (i in 1:nrow(outOfBagErrors))
{
  sampleSize <- sampleSizes[i]
  withReplacement <- sampleSize == numTrainCases
  for (j in 1:ncol(outOfBagErrors))
  {
    randomForestResults <- randomForest(x = trainFeatures,
                                        y = trainLabels,
                                        ntree = numTrees,
                                        mtry = featureNumbers[j],
                                        replace = withReplacement,
                                        sampsize = sampleSize,
                                        proximity = FALSE)
    outOfBagErrors[i, j] <- randomForestResults$err.rate[numTrees, 1]
  }
}

# Identify parameter values corresponding to lowest OOB error
paramIndices <- which(outOfBagErrors == min(outOfBagErrors), arr.ind = TRUE)

# Train forest with chosen settings
numWorkers <- 4
numTreesPerWorker <- numTrees / numWorkers
cluster <- makeCluster(4)
registerDoParallel(cluster)
sampleSize <- sampleSizes[paramIndices[1]]
randomForestResults <- foreach(nTree = rep(numTreesPerWorker, numWorkers),
                               .combine = combine,
                               .packages = "randomForest") %dopar%
                         randomForest(x = trainFeatures,
                                      y = trainLabels,
                                      ntree = nTree,
                                      mtry = featureNumbers[paramIndices[2]],
                                      replace = sampleSize == numTrainCases,
                                      sampsize = sampleSize,
                                      proximity = FALSE)
stopCluster(cluster)

# Make predictions for test data
predictions <- predict(randomForestResults, testFeatures)
write_csv(data.frame(ImageId = 1:nrow(testFeatures),
                     Label = levels(trainLabels)[predictions]),
          "random_forest_predictions.csv")
