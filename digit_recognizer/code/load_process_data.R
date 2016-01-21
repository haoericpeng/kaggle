library(readr)

# Read data
trainData <- read_csv("train.csv")
testFeatures <- read_csv("test.csv")
trainLabels <- as.factor(trainData[, 1])
trainFeatures <- trainData[, -1]
