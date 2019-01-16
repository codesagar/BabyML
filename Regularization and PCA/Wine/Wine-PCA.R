
# Chemical analysis of Wine
# These data are the results of a chemical analysis of wines grown 
# in the same region in Italy but derived from three different cultivars. 
# The analysis determined the quantities of 13 constituents found in each 
# of the three types of wines. 

WineData <- read.csv("wine.csv")
str(WineData)

# PCA analysis is done only on the predictors
wine.predictors <- WineData[,-1]

# Since the predictors are of completely different magnitude, 
# we need scale them before the analysis.
scaled.Predictors <- scale(wine.predictors)

# compute PCs
pca.out = princomp(scaled.Predictors)    # princomp(wine.predictors, cor=TRUE) would automatically scale
names(pca.out)
summary(pca.out)


plot(pca.out)

#If we choose 80% explanatory power for variances, we need only first 5 components of PC.

compressed_features = pca.out$scores[,1:5] 

library(nnet)
multout.pca <- multinom(WineData$wine.class ~ compressed_features)
summary(multout.pca)
#Gives us AIC value of 24

multout.full <- multinom(wine.class ~ ., data=WineData)
summary(multout.full)  #Gives us AIC of 56



#Visualizing the spread in the dataset using only the first 2 components.
#

library(ggbiplot)

g <- ggbiplot(pca.out, obs.scale = 1, var.scale = 1, 
              groups = WineData$wine.class, ellipse = TRUE, circle = TRUE) 
  g  + scale_color_discrete(name = '')
  


