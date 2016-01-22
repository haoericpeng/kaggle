library(readr)

# Read data
trainData <- read_csv("../data/train.csv")
testFeatures <- read_csv("../data/test.csv")
trainLabels <- as.factor(trainData[, 1])
trainFeatures <- trainData[, -1]

# Remove features that take on the same value for all training and test samples
uselessIndices <- which(apply(rbind(trainFeatures, testFeatures),
                              2,
                              function(x) length(unique(x))) == 1)
uselessIndices <- as.vector(uselessIndices)
trainFeatures <- trainFeatures[, -uselessIndices]
testFeatures <- testFeatures[, -uselessIndices]
