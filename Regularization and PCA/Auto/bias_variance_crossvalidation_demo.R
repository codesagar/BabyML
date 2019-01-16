rm(list=ls(all=T))
setwd("~/Desktop/Batch43 - Regularization/")
auto_data <- read.table('auto-data', header = F, dec = '.')
attr_names <- c('mpg','cyl','disp','hp','weight','accl','year','origin','car_name')
colnames(auto_data) <- attr_names

summary(auto_data)
#NAs in mpg and hp
#too few observations for cyl 3 and 5. We will ignore them for now

str(auto_data)
#cyl needs to be discrete 
#year needs to be feature engineered
#origin needs to be discrete
#car_name needs to be dropped

#Since mpg is target attribute, it does not make any sense to impute it. Hence we remove
#all observations with NA values in mpg
auto_data <- auto_data[-which(is.na(auto_data$mpg)),]
auto_data <- auto_data[-which(auto_data$cyl == 3 | auto_data$cyl == 5),]
auto_data$car_name <- NULL
discrete_vars <- c('cyl','origin')
auto_data[,discrete_vars] <- as.data.frame(lapply(auto_data[,discrete_vars], factor))

#We create a new variable 'age' which takes latest year as base year, and calculates
#the age of the car
present <- max(auto_data$year)
auto_data$age <- present - auto_data$year
auto_data$year <- NULL

#Create train and test data partitions
library(caret)
set.seed(2500)
idx <- createDataPartition(auto_data$mpg, p = 0.6)
train <- auto_data[idx$Resample1,]
test <- auto_data[-idx$Resample1,]

#Imputation of missing values in hp using kNN
library(DMwR)
train_imputed <- knnImputation(train[,!names(train) %in% 'mpg'], k = 5, scale = T)
test_imputed <- knnImputation(test[,!names(test) %in% 'mpg'], k = 5, scale = T,
                              distData = train[,!names(train) %in% 'mpg'])

#Attach target to imputed data
train <- cbind(train_imputed, 'mpg' = train$mpg)
test <- cbind(test_imputed, 'mpg' = test$mpg)

#Error Metrics vector
model_name <- vector()
rmse_train <- vector()
rmse_test <- vector()

#Model0 - linear model with only intercept - least complex model
model0 <- lm(mpg ~ 1, data = train)
summary(model0)
mean(train$mpg)
model_name <- append(model_name, 'm0')
rmse_train <- append(rmse_train, RMSE(predict(model0, train), train$mpg))
rmse_test <- append(rmse_test, RMSE(predict(model0, test), test$mpg))

#Complex models
run_model <- function(degree, data_train, data_test){
  for(i in 1:degree){
    x <- as.matrix(data_train[,!names(data_train) %in% c('mpg','cyl','origin')])
    x2 <- poly(x, degree = i)
    final_df <- data.frame(x2, data_train[,c('mpg','cyl','origin')])
    x_test <- as.matrix(data_test[,!names(data_test) %in% c('mpg','cyl','origin')])
    x2_test <- poly(x_test, degree = i)
    final_df_test <- data.frame(x2_test, data_test[,c('mpg','cyl','origin')])
    fit <- lm(mpg ~ ., data = final_df)
    model_name <- append(model_name, paste('m',i, sep = ''))
    rmse_train <- append(rmse_train, RMSE(predict(fit, final_df), final_df$mpg))
    rmse_test <- append(rmse_test, RMSE(predict(fit, final_df_test), final_df_test$mpg))
  }
  return(list(model_name,rmse_train,rmse_test))
}

model_info <- run_model(4, train, test)
model_df <- data.frame('model_name' = model_info[[1]], 
                       'rmse_train' = model_info[[2]], 
                       'rmse_test' = model_info[[3]])
# model_df
model_df$model_name <- factor(model_df$model_name, levels = paste('m',0:4,sep=''))
ggplot(model_df, aes(x = model_name)) + geom_line(aes(y = rmse_train, group = 1)) +  
  geom_line(aes(y = rmse_test, group = 1)) + ylab('RMSE') + xlab('Model Complexity') +
  scale_y_continuous(limits = c(0,25))

library(glmnet)
library(dummies)
dummy_obj <- dummyVars(~ cyl + origin, data = train)
dummy_vars_train <- predict(dummy_obj, newdata = train)
train$cyl <- train$origin <- NULL
train <- cbind(train, dummy_vars_train)
dummy_vars_test <- predict(dummy_obj, newdata = test)
test$cyl <- test$origin <- NULL
test <- cbind(test, dummy_vars_test)

# x <- as.matrix(train[,!names(train) %in% c('mpg','cyl','origin')])
# x2 <- poly(x, degree = 2)
# final_df <- data.frame(x2, train[,c('mpg','cyl.4','cyl.6','cyl.8',
#                                     'origin.1','origin.2','origin.3')])
# x_test <- as.matrix(test[,!names(test) %in% c('mpg','cyl','origin')])
# x2_test <- poly(x_test, degree = 2, raw = T)
# final_df_test <- data.frame(x2_test, test[,c('mpg','cyl.4','cyl.6','cyl.8',
#                                              'origin.1','origin.2','origin.3')])

#Ridge Regression
model_ridge <- glmnet(as.matrix(train), 
                      as.matrix(train[,'mpg']), family = 'gaussian', alpha = 0, 
                      standardize = TRUE)
plot(model_ridge, xvar = 'lambda')
ridge_cv <- cv.glmnet(as.matrix(train), 
                      as.matrix(train[,'mpg']), family = 'gaussian', alpha = 0, 
                      standardize = TRUE)
plot(ridge_cv)
model_ridge_cv <- glmnet(as.matrix(train), 
                         as.matrix(train[,'mpg']), family = 'gaussian', alpha = 0, 
                         standardize = TRUE, lambda = ridge_cv$lambda.1se)
lambda_val <- ridge_cv$lambda.1se
RMSE(predict(model_ridge, as.matrix(train), s = lambda_val), train$mpg)
RMSE(predict(model_ridge, as.matrix(test), s = lambda_val), test$mpg)
