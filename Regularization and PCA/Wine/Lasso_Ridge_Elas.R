rm(list=ls(all=TRUE))


CustData = read.csv("CustomerData.csv",header=TRUE)

str(CustData)
summary(CustData)

################################################
numeric_Variables = CustData[,-c(1,12,13,14)]
target_variable = subset(CustData,select="TotalRevenueGenerated")
#Converting categorical attributes into dummy variables 
# removing the intercept column (1 index)
catDummies <- model.matrix(CustData$TotalRevenueGenerated ~ CustData$FavoriteChannelOfTransaction + CustData$FavoriteGame )[,-1]
#####################################################
#Split the data into train and test data sets
rows=seq(1,nrow(CustData),1)
set.seed(123)
trainRows=sample(rows,(70*nrow(CustData))/100)
train1 = data.frame(numeric_Variables, catDummies,TotalRevenueGenerated=target_variable)[trainRows,]
test1 = data.frame(numeric_Variables, catDummies,TotalRevenueGenerated=target_variable)[-trainRows,]


#############################################################################
#Converted the data into matrix form to input into glm model

numeric_Variables <- scale(numeric_Variables)
data2 <- as.matrix(data.frame(numeric_Variables, catDummies))
train = data2[trainRows,] 
test = data2[-trainRows,]

#Target Varaible
y=CustData$TotalRevenueGenerated[trainRows]
ytest = CustData$TotalRevenueGenerated[-trainRows]

library(glmnet)
#####################################################
# fit model


fit2=glmnet(train,y,alpha=0)  #Ridge
plot(fit2,xvar="lambda",label=TRUE)

fit1=glmnet(train,y,alpha=1)  #Lasso
plot(fit1,xvar="lambda",label=TRUE)

#######################################################
#cv.glmnet will help you choose lambda
cv <- cv.glmnet(train,y)  #By default alpha=1


#lambda.min - value of lambda that gives minimum cvm - mean cross-validated error
###################################################
# Lasso Regression  using glmnet - L1 norm
fit1=glmnet(train,y,lambda=cv$lambda.min,alpha=1)
predicted.train <- predict(fit1,train)
library(DMwR)
LASSOtrain = regr.eval(y, predict(fit1,train))
LASSOtest = regr.eval(ytest, predict(fit1,test))
LASSOtrain
LASSOtest


#Model Selection
coef(fit1)
cv.lasso=cv.glmnet(train,y)
plot(cv.lasso)
coef(cv.lasso)

#############################################################################
# Ridge Regression  using glmnet  - L2 norm
library(glmnet)
# fit model
fit2=glmnet(train,y,lambda=cv$lambda.min,alpha=0)
predict(fit2,train)
library(DMwR)
RIDGEtrain = regr.eval(y, predict(fit2,train))
RIDGEtest = regr.eval(ytest, predict(fit2,test))
RIDGEtrain
RIDGEtest
#Model Selection
coef(fit2) 
cv.ridge=cv.glmnet(train,y,alpha=0)
plot(cv.ridge)
coef(cv.ridge)

#cvfit = cv.glmnet(train,y,alpha=0, type.measure = "mae")
################################################################
# Elastic regression
fit3=glmnet(train,y,lambda=cv$lambda.min,alpha=0.5)
# summarize the fit
summary(fit3)
predict(fit3,train)

library(DMwR)
Elastictrain = regr.eval(y, predict(fit3,train))
Elastictest = regr.eval(ytest, predict(fit3,test))
Elastictrain
Elastictest

# make predictions
predictions <- predict(fit3, train, type="link")
# summarize accuracy
rmse <- mean((y - predictions)^2)
print(rmse)

#################################################

LASSOtrain
LASSOtest
RIDGEtrain
RIDGEtest
Elastictrain
Elastictest

finalerros <- data.frame(rbind(
                               LASSOtrain,LASSOtest,
                               RIDGEtrain,RIDGEtest,
                               Elastictrain,Elastictest))
finalerros
