# VIC in R: Validity Index using supervised Classifiers
# Requires packages 'caret', 'bnlearn', 'forecast', 'ROCR', 'dplyr',
# and additional libraries 'randomForest', 'kernlab', 'naivebayes', 'MASS' for the
# default option of the algorithm
library(caret)
library(bnlearn)
library(forecast)
library(ROCR)
library(dplyr)


# ----- pred (function) -------
# Computes the AUC of a model trained usign k-fold cross validation (default k=10)
# ----- Parameters
# data (data.frame): a data frame containing the data to train and evaluate the model with
# method (character): a string specifying the model to train. From the available models
#        with the 'caret' package (https://topepo.github.io/caret/available-models.html)
# ctrl (list): a 'trainControl' sequence specifying the parameters of the classification 
#        method. If not specified, the default is 10-fold cross-validation
# ----- Value
# auc (numeric): the area under the curve (ROC)
pred <- function(data, method, ctrl = NULL) {
        # setup trainControl
        if (is.null(ctrl)) {
                myctrl <- trainControl(method = "cv",     # Cross-validation
                                       number = 10,      # 10-fold                  
                                       classProbs = TRUE, summaryFunction = multiClassSummary)
        } else {
                myctrl = ctrl
        }
        # train model with specified method
        caret_model <- suppressWarnings(train(class~.,
                                              data = data,
                                              method = method,
                                              metric = "ROC",
                                              trControl = myctrl))
        # find mean 'auc'
        D <- caret_model$results
        mean(D$AUC)
}

# ----- bayesNet (function) -------
# Define a Bayesian Network structure that fits the data, and evaluate the strength
# of the arcs by using k-fold cross-validation
# ----- Parameters
# data (data.frame): a data frame containing the data to train and evaluate the model with
# numCols (integer): a number specifying the maximum number of columns to use as nodes
#         in the Bayesian Network. If not specified, the default is 1/3 of the total
#         number of columns in 'data'
# ----- Value
# auc (numeric): the area under the curve (ROC) 
bayesNet <- function(data, numCols = ncol(data)%/%3, k = 10) { 
        # define the columns to use
        cols <- sample(seq(ncol(data)),numCols)
        # use 10-fold cross-validation to fit the structure and parameters of the BN
        cv.bic <- bn.cv(data[,c(cols, ncol(data))], bn = "hc", 
                        k = k, runs = 1, fit = "mle")
        # find the best fitted structure
        loss.cv <- unlist(lapply(cv.bic, function(x) {x[["loss"]]}))
        index <- match(min(loss.cv), loss.cv) 
        fitted <- cv.bic[[index]][["fitted"]]
        strength <- boot.strength(data[,c(cols, ncol(data))], algorithm = "hc",
                                  R = 20, m = 30)
        # compute the area under the curve (ROC)
        pred <- as.prediction(strength, bn.net(fitted))
        pred <- performance(pred, "auc")
        pred@y.values[[1]]
}


# ----- get.auc (function) -------
# Apply the correct method for computing the 'auc'
# ----- Parameters
# data (data.frame): a data frame containing the data to train and evaluate the model with
# method (character): a string specifying the model to train. Either from the 
#        available models with 'caret', or "bayesNet" for finding the Bayesian Network
# k (integer): a number specifying the number of folds to apply in cross-validation
# ----- Value
# auc (numeric): the area under the curve (ROC) 
## IMPORTANT: requires functions 'bayesNet' and 'pred'
get.auc <- function(data, method, k = 10, ctrl = NULL) {
        ifelse(method == "bayesNet", bayesNet(data, k = k), pred(data, method, ctrl))
}


# ----- VIC (function) -------
# Apply the VIC algorithm to a dataset to find the validity index for a clustering algorithm
# ----- Parameters
# data (data.frame): a labeled data set with a 'class' attribute.
# k (integer): a number specifying the number of folds to apply in cross-validation
# ctrl (list): a 'trainControl' sequence specifying the parameters of the classification 
#        method. If not specified, the default is 10-fold cross-validation
# classifiers (list): a character list specifying the classifiers to test for comparison. 
#             Either from the available models with 'caret', or "bayesNet" for finding 
#             the Bayesian Network
# ----- Value
# results (list): a named list containing the auc results for all the classifiers
#         and the validity index (VIC) 
VIC <- function(data, k = 10, ctrl = NULL,
                classifiers = list("rf", "svmRadial", "naive_bayes", "lda", "bayesNet")) {
        # calculate the AUC for all the 'classifiers' and find the max (VIC)
        vic <- numeric()
        results <- list()
        for (method in classifiers) {
                temp.auc <- get.auc(data, method, k, ctrl)
                vic <- max(vic, temp.auc)
                results[[method]] <- temp.auc
        }
        results[["VIC"]] <- vic
        return(results)
}


# EXAMPLE
data("mtcars")
# find the clusters in the data
cluster <- pam(mtcars, 4)
# append class label to data
class <- as.factor(cluster$clustering)
labels <- levels(class)
labels <- LETTERS[1:length(labels)]
levels(class) <- labels
mtcars[["class"]] <- class

set.seed(1234)
# obtain VIC with default options
defaul.vic <- VIC(mtcars)
defaul.vic
# obtain VIC, applying 5-fold cross-validation, using k-Nearest Neighbors, 
# Multi-Layer Perceptron, Parallel Random Forest Model Averaged Neural Network
myctrl <- trainControl(method = "cv",  
                       number = 5,    # number of folds                   
                       classProbs = TRUE, summaryFunction = multiClassSummary)
my.vic <- VIC(mtcars, k = 5, ctrl = myctrl,
              classifiers = list("knn", "mlp", "parRF", "avNNet", "svmLinear"))
my.vic


# EXAMPLE MULTI-THREAD
# To optimize the time performance for the execution of the algorithm, it is 
# possible to parallelize the processes. Requires package 'doParallel'
# Check https://cran.r-project.org/web/packages/doParallel/vignettes/gettingstartedParallel.pdf
# for further information
library(doParallel)
data("mtcars")

# Two methods are suggestd
# First...
cl <- makeCluster(2) # ... to specify the number of workers to use
# The following lines run in parallel
registerDoParallel(cl)
ptime <- system.time({
        new.vic <- VIC(mtcars) 
})[3]
ptime
stopCluster(cl) # parallel execution ends here


# Second ...
registerDoParallel(cores = 2) # ... to select the number of cores to use
# The following lines run in parallel
stime <- system.time({
        new.vic <- VIC(mtcars) 
})[3]
stime
stopImplicitCluster() # parallel execution ends here


