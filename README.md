# VIC in R
Implementation of the VIC algorithm in R. 

## VIC algorithm
First of all, VIC stands for Validity Index using supervised Classifiers. It is a cluster validity index which evaluates the quality of a partition result of a clustering algorithm. It uses an ensemble of supervised classifiers to alleviate the problem of a lack of a domain expert who helps to evaluate the quality of the partition. The principle that inspires the algorithm lies in the idea that any good partition induces the construction of a good classifier. Hence the performance evaluation of such classifier should reflect the quality of the partition, such that the higher the evaluation metric for the classifier the better the partition found with the clustering algorithm [1].

The method works on a data set which has been labeled by a clustering algorithm, and trains a set of supervised classifers using k-fold cross-validation (suggested k = 5). The evaluation metric used with each of these classifers is AUC (Area under the Curve -ROC-). The maximum AUC among the results from all the classifiers is the VIC.

## Implementation in R
This R implementation of the VIC algorithm uses Random Forest, SVM, Bayesian Network, Naive Bayes and LDA as default classifiers to evaluate the partion in the data, by applying 10-fold cross validation. However, these methods can be updated to meet specific requirements. By default, it requires packages `'caret', 'bnlearn', 'forecast', 'ROCR', 'dplyr'`, and additional libraries `'randomForest', 'kernlab', 'naivebayes', 'MASS'`.

### How to execute VIC
A set of examples are provided with this file. 

The simplest way to test the VIC algorithm is with the default options (10-fold cross-validation, with random forest, SVM, Bayesian network, naive Bayes and LDA) . Tests below are performed with the `mtcars` dataset, though the clusters must be found first. This requires library `'cluster'` to apply a simple clustering algorithm, `pam` (Partitioning Around Medoids):

    library(cluster)
    data("mtcars")
    # find the clusters in the data
    cluster <- pam(mtcars, 4) # 4 clusters
    # append class label to data
    class <- as.factor(cluster$clustering)
    labels <- levels(class)
    labels <- LETTERS[1:length(labels)]
    levels(class) <- labels
    mtcars[["class"]] <- class
    # obtain VIC with default options
    defaul.vic <- VIC(mtcars)
    defaul.vic

However, if we want to personalize the number of folds, we need to specify the `k` and `ctrl` arguments of the function. Additionally, we can specify other `classifiers` (as a list to pass to the function) to test the cluster with:

    myctrl <- trainControl(method = "cv",  
                       number = 5,    # number of folds                   
                       classProbs = TRUE, summaryFunction = multiClassSummary)
    my.vic <- VIC(mtcars, k = 5, ctrl = myctrl,
                  classifiers = list("knn", "mlp", "parRF", "avNNet", "svmLinear"))
    my.vic


### Multi-threaded execution of VIC
To optimize the time performance for the execution of the algorithm, it is possible to parallelize the processes. This process requires package `'doParallel'`.
Check https://cran.r-project.org/web/packages/doParallel/vignettes/gettingstartedParallel.pdf for further information.

There are two suggest methods to parallelize the execution of the code. The first one specifies the number of workers to use via the `makeCluster` function:

    library(doParallel)
    cl <- makeCluster(2) 
    # The following lines run in parallel
    registerDoParallel(cl)
    ptime <- system.time({
            new.vic <- VIC(mtcars) 
    })[3]
    ptime
    stopCluster(cl) # parallel execution ends here


The second one select the number of cores to use, by specifying it in the `registerDoParallel`function:

    registerDoParallel(cores = 2) 
    # The following lines run in parallel
    stime <- system.time({
            new.vic <- VIC(mtcars) 
    })[3]
    stime
    stopImplicitCluster() # parallel execution ends here


## References
[1] J. Rodríguez, M. A. Medina-Pérez, A. E. Gutierrez-Rodríguez, R. Monroy, H. Terashima-Marín, "Cluster validation using an ensemble of supervised classifiers," Knowledge-Based Systems, vol. 145, pp. 134-144, 2018.
