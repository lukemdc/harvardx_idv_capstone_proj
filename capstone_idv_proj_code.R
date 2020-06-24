---
  title: "Comparing Machine Learning Approaches to Predict Wine Quality"
author: "Luke M."
date: "6/23/2020"
output:
  pdf_document: default
html_document:
  df_print: paged
---
  
if(!require(klaR)) install.packages("klar", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")

set.seed(1) # use `set.seed(1, sample.kind = "Rounding")` in R >=3.6 

#load data from github repository 
wine_data<-read.csv(url("https://raw.githubusercontent.com/lukemdc/harvardx_idv_capstone_proj/master/winequality-white.csv"))

#get descriptive statistics for the available variables
kable(summary(wine_data[,1:6]))
kable(summary(wine_data[,7:12]))

#plot the distribution of wine ratings
ggplot(wine_data, aes(x=quality)) +
  geom_histogram(binwidth = 1) +
  theme_light() +
  labs(x="Quality Score", y="Count", title="Distribution of Quality Scores")

#calculate  pearson pair-wise correlations between each possible predictor and ratings
pearsons_coeffs<-data.frame(t(cor(wine_data$quality,select(wine_data,-c(quality)),method = 'pearson')))

#rename the column with the correlation coefficients
names(pearsons_coeffs)[1] <- "correlation_coeff"

#print the correlation coefficients to a kable
kable(arrange(pearsons_coeffs, desc(correlation_coeff)))

# Methods

## Data Partitioning and Setup 

#create index identifying the observations selected for the training dataset
train_index <- createDataPartition(wine_data$quality, times = 1, p=.8, list = F)

#use the index and its inverse to create training and test sets
train <- wine_data[train_index,]
test <- wine_data[-train_index,]

#create one ggplot object with histogram of training data quality ratings
ggpl1<- ggplot(train, aes(x=quality)) + 
  geom_histogram(aes(y = stat(count) / sum(count), binwidth = 1)) +
  scale_y_continuous(labels = scales::percent) +
  labs(x="Trainign Set Quality Ratings", y="Percent of Observations")

#create second ggplot object for test data
ggpl2<- ggplot(test, aes(x=quality)) + 
  geom_histogram(aes(y = stat(count) / sum(count), binwidth = 1)) +
  scale_y_continuous(labels = scales::percent) +
  labs(x="Test Set Quality Ratings", y="Percent of Observations")

#display side-by-side to compare distributions
grid.arrange(ggpl1, ggpl2, ncol = 2)

#create empty tables where rmse results for each model will be stored  
train_rmse_table<-data.frame(model=character(0),rmse=double(0))
test_rmse_table<-data.frame(model=character(0),rmse=double(0))

#create dfs to store the y-hats from each model will be stored  
train_results<-data.frame(rating=train$quality)
test_results<-data.frame(rating=test$quality)

#rename the quality column "rating" for easy reference in subsequent code
names(train)[which(names(train)=="quality")] <- "rating"
names(test)[which(names(train)=="quality")] <- "rating"
names(wine_data)[which(names(wine_data)=="quality")] <- "rating"

#Baseline Models

#create list of possible ratings
poss_ratings<-seq(3,9,1)

#use random sample of possible ratings to generate predicted ratings and RMSEs
train_results[["y_hat_random_guesses"]]<-
  sample(poss_ratings, nrow(train),replace = T)

train_rmse_table[nrow(train_rmse_table)+1,]<-
  c("random_guesses", 
    RMSE(train_results$y_hat_random_guesses, 
         train_results$rating))

test_results[["y_hat_random_guesses"]]<-
  sample(poss_ratings, nrow(test),replace = T)

test_rmse_table[nrow(test_rmse_table)+1,]<-
  c("random_guesses", 
    RMSE(test_results$y_hat_random_guesses, 
         test_results$rating))

##Weighted guesses
#estimate the proportion of observations in the training data that have each rating level
probs<-train %>% group_by(rating) %>% 
  summarize(prob_per_rating=as.double(n()/nrow(train)))

#assign predictions at each rating level according their priors and find RMSEs
train_results[["y_hat_weighted_guesses"]]<-
  sample(poss_ratings, nrow(train), replace = T, prob=c(probs$prob_per_rating))

train_rmse_table[nrow(train_rmse_table)+1,]<-
  c("weighted_guesses", 
    RMSE(train_results$y_hat_weighted_guesses, 
         train_results$rating))

test_results[["y_hat_weighted_guesses"]]<-
  sample(poss_ratings, nrow(test), replace = T, prob=c(probs$prob_per_rating))

test_rmse_table[nrow(test_rmse_table)+1,]<-
  c("weighted_guesses", 
    RMSE(test_results$y_hat_weighted_guesses,
         test_results$rating))

##Global Average
#make predictions solely from overall average rating in the training dataset
train_results[["y_hat_global_mean"]]<-mean(train$rating)
train_rmse_table<-rbind(train_rmse_table,c("global_mean", 
                                           RMSE(train_results$y_hat_global_mean,
                                                train_results$rating)))

test_results[["y_hat_global_mean"]]<-mean(train$rating)
test_rmse_table<-rbind(test_rmse_table,c("global_mean", 
                                         RMSE(test_results$y_hat_global_mean,
                                              test_results$rating)))
rm(probs,poss_ratings, ggpl1, ggpl2, pearsons_coeffs)

## Machine Learning Approaches 

#estimate the best stepwise regression model based on 25 bootstraps
best_stepwise_fit<-train(rating ~ ., 
                         method = "leapSeq", 
                         data = wine_data[train_index,],
                         tuneGrid=data.frame(nvmax=seq(1,ncol(train)-1,1)))
set.seed(1)

#the optimal number of predictors is:
best_stepwise_fit$bestTune

#append the predictions to the results datasets
train_results[["y_hat_stepwise_regression"]]<-
  predict(best_stepwise_fit, wine_data[train_index,])
test_results[["y_hat_stepwise_regression"]]<-
  predict(best_stepwise_fit, wine_data[-train_index,])

#append RMSEs to their respective tables
train_rmse_table<-rbind(train_rmse_table,
                        c("stepwise_regression",
                          min(best_stepwise_fit$results$RMSE)))
test_rmse_table<-rbind(test_rmse_table,
                       c("stepwise_regression",
                         RMSE(test_results[["y_hat_stepwise_regression"]],
                              test_results$rating)))
###K-means Clustering and Stepwise Regression. 

set.seed(1)  

#create list of possible values of k 
k=seq(2,50,1)

#calculate the within-cluster sum of squares for each k
wss<-sapply(k,function(k_){
  kmeans_fit<-kmeans(wine_data[1:ncol(wine_data)-1], 
                     k_, iter.max = 20, nstart=10)
  
  #return the total within-cluster sum of squares
  kmeans_fit$tot.withinss
})

#this plot shows us that the "elbow" occurs somewhere between 10-15, so 
ggplot(NULL, aes(x=k,y=wss)) +
  geom_point() +
  theme_grey() +
  labs(x="Cluster Size k", 
       y="With-in Cluster Sum of Squares", 
       title="Cluster Size K vs. Within-Cluster Sum of Squares")

#identify the best k that minimizes SSD for observations within clusters
best_k<-30

#using the optimal k to cluster the wines 
clusters<-kmeans(wine_data[1:ncol(wine_data)-1], 
                 best_k, iter.max = 20, nstart=10)$cluster

#add the cluster numbers for each observation to their respective rows 
wine_clusters <- cbind(wine_data, clusters) 

#add columns for the wine's average rating for each cluster
wine_clusters <- wine_clusters %>% group_by(clusters) %>%
  mutate(cluster_rating_avg=mean(rating))

#remove the cluster number because we don't need it anymore
wine_clusters <- select(wine_clusters,-c(clusters))

#estimate the best stepwise regression model based on 25 bootstraps
best_stepwise_fit<-train(rating ~ ., 
                         method = "leapSeq", 
                         data = wine_clusters[train_index,],
                         tuneGrid=data.frame(nvmax=seq(1,ncol(train)-1,1)))
#clean up the work space by removing objects we don't need anymore
rm(clusters, best_k, k, wss)

#the optimal number of predictors is:
best_stepwise_fit$bestTune

#append the predictions to the results datasets
train_results[["y_hat_stepwise_regression_clusters"]]<-
  predict(best_stepwise_fit, wine_clusters[train_index,])
test_results[["y_hat_stepwise_regression_clusters"]]<-
  predict(best_stepwise_fit, wine_clusters[-train_index,])

#append RMSEs to their respective tables
train_rmse_table<-rbind(train_rmse_table,
                        c("stepwise_regression_clusters",
                          min(best_stepwise_fit$results$RMSE)))
test_rmse_table<-rbind(test_rmse_table,
                       c("stepwise_regression_clusters",
                         RMSE(test_results[["y_hat_stepwise_regression_clusters"]],
                              test_results$rating)))
##KNN method

#define list of possible ks. 
ks<-seq(5,50,5)

#test all ks and return fit with the optimal k
knn_fit<-train(rating ~ ., method = "knn", data = train,
               tuneGrid=data.frame(k=ks))

#use best knn fit model to make predictions 
train_results[["y_hat_knn"]]<-predict(knn_fit,train)
test_results[["y_hat_knn"]]<-predict(knn_fit,test)

#calculate RMSEs and append them to the RMSE tables
train_rmse_table<-rbind(train_rmse_table,c("knn", 
                                          RMSE(train_results$y_hat_knn,
                                               train_results$rating)))
test_rmse_table<-rbind(test_rmse_table, c("knn",
                                          RMSE(test_results$y_hat_knn, 
                                               test_results$rating)))
trees<-seq(10,150,10)

#calculate fits for each number of trees
fits<-sapply(trees, function(tree){
  fit<-randomForest(as.matrix(train[1:ncol(train)-1]),
                    as.matrix(train$rating), ntree=tree)
  list(tree,RMSE(fit$predicted,train_results$rating),fit,fit$pred)
})

#plot the number of trees vs RMSE to determine the best number of trees
plot(fits[1,],fits[2,])

print(paste("The random forest with the lowest RMSE has",
            unlist(fits[1,which.min(fits[2,])]),"trees with RMSE of",
            fits[2,which.min(fits[2,])]))

#add predicted values to results tables
best_rf_fit<-fits[3,which.min(fits[2,])]
train_results[["y_hat_rf"]]<-unlist(predict(best_rf_fit, train))
test_results[["y_hat_rf"]]<-unlist(predict(best_rf_fit,test))

#update rmse tables
train_rmse_table<-rbind(
  train_rmse_table,c("rf",
                     RMSE(train_results[["y_hat_rf"]],
                          train_results$rating)))
test_rmse_table<-rbind(
  test_rmse_table,c("rf", 
                    RMSE(test_results[["y_hat_rf"]], 
                         test_results$rating)))

##Ensemble method

#rank the modeling methods according to their RMSEs
test_rmse_table[["rank"]]<-as.integer(rank(test_rmse_table$rmse))

#create a list of possible number of models
poss_num_models <- seq(2,nrow(test_rmse_table),1)

#estimate the RMSEs for ensemble models with various numbers of models included
rmses<- sapply(poss_num_models,function(poss_models){
  models_to_keep <- test_rmse_table %>% filter(rank<poss_models)
  results_to_keep <- lapply(list(models_to_keep$model), function(k){
    paste("y_hat",k,sep="_")
  })
  
  #average the models kept and calculate RMSE
  y_hat_meta_model_test <- rowMeans(select(test_results, 
                                           c(unlist(results_to_keep))))
  RMSE(y_hat_meta_model_test, test_results$rating)
})

#show which ensemble minimizes RMSE
which.min(rmses)

#generate list of models to keep for the ensemble model
models_to_keep <- test_rmse_table %>% filter(rank<=which.min(rmses))
results_to_keep <- lapply(list(models_to_keep$model), function(k){
  paste("y_hat",k,sep="_")
})

#print the models kept for the ensemble model
models_to_keep

#append predictions to the results tables
train_results[["y_hat_meta_model"]]<-rowMeans(
  select(train_results, c(unlist(results_to_keep))))
test_results[["y_hat_meta_model"]]<-rowMeans(
  select(test_results,c(unlist(results_to_keep))))

#append RMSEs to RMSE tables
train_rmse_table<-rbind(train_rmse_table,c("meta_model", RMSE(train_results$y_hat_meta_model,train_results$rating)))
test_rmse_table<-rbind(test_rmse_table,c("meta_model", RMSE(test_results$y_hat_meta_model,test_results$rating)))
final_rmse_table<-test_rmse_table

# Final Results  

kable(final_rmse_table)
