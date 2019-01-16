#------------------------BATCH-48: PCA and REGULARIZATION ASSIGNMENT SOLUTION----------------------------#

# Code for PCA maybe slightly different from the approach discussed in class
# Please feel free to use whichever approach you prefer

# Clear Environment:
rm(list=ls())

# Set working directory
setwd("~/Desktop/Batch 48/PCAandregularization/OnlineNewsPopularity")

# Load the required packages:
library(caret)
library(DMwR)
library(car)
library(glmnet)

# Read the data:
online_news<-read.csv("OnlineNewsPopularity.csv")

# Check structure of the data:
str(online_news) # All the variables are numeric

# removing 'url' variable
online_news$url<-NULL

# checking for NA values: 
sum(is.na(online_news)) #no NA values

# train-test split: 
set.seed(1000)
rows <- createDataPartition(y = online_news$shares,p = 0.7,list = FALSE)
x_train <- online_news[rows,!names(online_news)%in% 'shares']
y_train <- online_news[rows,'shares']
x_validation <- online_news[-rows,!names(online_news)%in% 'shares']
y_validation <- online_news[-rows,'shares'] 

# Lets build a base Linear Regression Model: 
x_train$shares<-y_train
lm_base<- lm(shares~., x_train)

# look at the summary of the model: 
summary(lm_base)
# Model is significant, but few significant attributes
# Extremely low R2 and adjusted R2 -  indicating the presence of large number of independent variables and also that the model is not able to explain the variation in the target variable 
# NA's for few coefficients, this could mean multi collinearity in the data, lets look at VIF values

vif(lm_base)
# Error in vif.default(lm_base) : there are aliased coefficients in the model
# This clearly indicates high multi collinearity in the data
# This occured because of a dummy trap -meaning look at the data set, the "weekday_is_.." variable is present for all the weekdays from Monday to sunday. This leads to collinearity among those variables. 
# Dummy coding should be done for n-1 levels and not all the levels.
# The same issue for "LDA_.." variable too
# The "is_weekend" attribute is again dependent on the "weekday_is_saturday" and "weekday_is_sunday" attributes, therefore there is collinearity and the coefficients are NA
# Therefore, removing  "weekday_is_saturday" and "is_weekend" will suffice the issue, and the other attributes will give the same information
# So lets remove "weekday_is_sunday", "is_weekend", and "LDA_04" variables and run the model again

# Removing "weekday_is_sunday", "is_weekend", and "LDA_04" variables:
x_train$weekday_is_sunday<-NULL
x_train$is_weekend<-NULL
x_train$LDA_04<-NULL

# Build LM model on the new data set: 
lm_base2<- lm(shares~., x_train)
summary(lm_base2)
# The coefficients are fine 
vif(lm_base2)

# Lets look at the variables with high VIF values:
collinear_var<- names(which(vif(lm_base2)>10))
collinear_var

# lets remove these variables and run the model: 
lm_base3<- lm(shares~., x_train[,!names(x_train) %in% collinear_var])
summary(lm_base3)
vif(lm_base3)
# coefficients are fine now, No NA values
# low R2 and adjusted R2

# Lets look at the variables and see if there are any high VIF values in this model:
names(which(vif(lm_base3)>10)) # NuLL

# So the issue of Multicollinearity is resolved.
# However, the R2 and adjusted R2 values are still poor
# Step AIC can also be incorporated here, instead of manually selecting features like we have done, since our focus in this assignment is PCA And Regularization, Step AIC is not touched upon
# Please try Step AIC on this dataset too and see how the multi-collinearity in the data and feature selection is addressed
# We will continue working with lm_base3 model and look at how PCA and regularisation improve the errors against it

# Lets look at the error metrics for lm_base3 model: 
lm_base3_train_preds<- predict(lm_base3, x_train)
lm_base3_val_preds<-predict(lm_base3, x_validation)

# Error metrics evaluation: 
lm_base3_train_error<- regr.eval(y_train, lm_base3_train_preds)
lm_base3_val_error<- regr.eval(y_validation, lm_base3_val_preds)
lm_base3_train_error
lm_base3_val_error

# Extremely large errors and also high variance between the train and validation errors
# PCA - is a dimensionality reduction technique learnt, which can also reduce varaince so lets run PCA and see if any improvement is there:
# Lets run PCA on the train set, 
# The base R function prcomp() is used to perform PCA. By default, it centers the variable to have mean equals to zero. With parameter scale = T, we normalize the variables to have standard deviation equals to 1.
# Its an unsupervised technique so we dont need the target variable, so excluding it
x_train$shares<-NULL

# PCA can handle multi collinearity, so running it on the entire train data set which includes the collinear variables too
prin_comp <- prcomp(x_train, scale = T)

#Lets look at the summary of the prin_comp model:
summary(prin_comp)
# The output has standard deviation, proportion of variance and cumulative proportion
# The individual standard deviation and variance is explained by the 'standard deviation' and 'proportion of variance"
# Cummulative Proportion is including all the variance upto that variable
# PC30 cummulative proportion is 0.90726%, which means 90.723% variation in the data is explained, if we consider the principle components till PC30

# Lets look at the screeplot
# For that we need the proportion of variance for each PC and also the cummulative variance.
# So lets manually calculate them, 

# compute standard deviation of each principal component
std_dev <- prin_comp$sdev

# compute variance
pr_var <- std_dev^2

# check variance of first 30 components
pr_var[1:30]

# proportion of variance explained
prop_var <- pr_var/sum(pr_var)

# Plot the scree plot
plot(prop_var,, xlab = "Principal Component",
       ylab = "Proportion of Variance Explained",
       type = "b")
# Though the elbow is not very clear in this plot, it shows that beyong PC30, the proportion of variance explained is decreasing, so lets consider PC's till 25

# cumulative scree plot
plot(cumsum(prop_var), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

# This plot shows that 25 components results in variance close to ~ 85%. Therefore, in this case, we’ll select number of components as 25 [PC1 to PC25] and proceed to the modeling stage. This completes the steps to implement PCA on train data. For modeling, we’ll use these 25 components as predictor variables and follow the normal procedures.
# We should do exactly the same transformation to the test set as we did to training set, including the center and scaling feature. Let’s do it:

# create a new train data with the transformed training set of principal components and the target variable
train.data <- data.frame(Shares = y_train, prin_comp$x)
str(train.data)

#we are interested in first 25 PCAs
train.data <- train.data[,1:26]
# selecting the columns as 1:26 because we have the target variable in column 1 followed by the 25 PC's
str(train.data)

# Apply PCA on the validation data: 
validation.data <- predict(prin_comp, newdata = x_validation)
validation.data <- as.data.frame(validation.data)

#select the first 25 components
validation.data<- validation.data[,1:25] 
# we select the same 25 components we selected for train data in validation data too
# here columns are 1:25 as the validation data set does not have the target variable

# Lets build Linear regression model on the PCA transformed train data set now: 
lm_PCA<- lm(Shares~., data=train.data)
summary(lm_PCA)
# predict on the train data: 
train_pred<- predict(lm_PCA, train.data)

# error metric evaluation on the train data: 
train_PCA<- regr.eval(trues = train.data$Shares, preds = train_pred)
train_PCA

# predict on the validation data: 
val_pred<- predict(lm_PCA, validation.data)

# error metric evaluation on the validation data: 
val_PCA<- regr.eval(trues = y_validation, preds = val_pred)
val_PCA

# high variance in the data still exists, though we have done PCA
# lets try running regularisation on this model and check if the varaince will reduce
# Intuitively regularisation reduces variance.
# Lets look at all the three regularisation methods learnt
# Since we have already done dimensionality reduction, our aim now is to further reduce variance, so  Ridge model should give the least error

# Regularization 
regularization_analysis_model<- data.frame(matrix(ncol = 3))
names(regularization_analysis_model)= c('Alpha','Train_RMSE','Validation_RMSE')
train.data$Shares<-NULL
for (a in seq(0,1,0.1))
{
   # Rgularisation Model
  model.cv<-cv.glmnet(as.matrix(train.data), as.matrix(y_train), alpha=a,
                      family="gaussian")
  # Train-Validation predictions
  model_train_preds <- predict(model.cv, as.matrix(train.data), s=model.cv$lambda.min)
  model_val_preds <- predict(model.cv, as.matrix(validation.data), s=model.cv$lambda.min)
  # RMSE calculation
  model_train_rmse <- RMSE(pred =  model_train_preds, obs = y_train)
  model_val_rmse <- RMSE(pred = model_val_preds, obs = y_validation)
  # Appending entry to dataframe for later analysis
  regularization_analysis_model <- rbind(regularization_analysis_model, c(a,model_train_rmse,model_val_rmse))
}

print(regularization_analysis_model <- na.omit(regularization_analysis_model))

# alpha = 0 is giving the least error on validation data set, so ridge is more suitable model here as was our intuition
# However the variance is very high, other algorithms that can reduce variance can be used
# But since the scope of this assignment is to understand PCA and regularization techniques we are limiting to only these methods here
# Regularization can be first done followed by PCA. The results would more or less the same
# On the data set, regularization and PCA can be done separately, instead of combining them, and then model building can be done later separately
# The purpose here is to just understand these 2 techniques and code comfortably, please feel free to try any variations.

# Final ridge model: 
ridge.cv<-cv.glmnet(as.matrix(train.data), as.matrix(y_train), alpha=0,
                    family="gaussian")
# Train-Validation predictions
ridge_train_preds <- predict(ridge.cv, as.matrix(train.data), s=model.cv$lambda.min)
ridge_val_preds <- predict(ridge.cv, as.matrix(validation.data), s=model.cv$lambda.min)

# RMSE calculation
train_ridge <- regr.eval(y_train,ridge_train_preds)
val_ridge<- regr.eval(y_validation,ridge_val_preds)

#FINAL COMPILATION OF THE ERRORS:
errors<- rbind(lm_base3_train_error, train_PCA, train_ridge, lm_base3_val_error, val_PCA,val_ridge)
errors
# The base LM model seemed to work best here
# However, the variance in train and validation data is large
# Different variations and advanced machine learning algorithms can be further used for your practise

