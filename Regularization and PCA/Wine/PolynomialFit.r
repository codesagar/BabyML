

x= 0.2*(1:10)
y= 3 + (2*x  )* abs(rnorm(length(x),0.,0.5)) + 4*x - 2*x*x

plot(x,y)

inTrain <- c(1,4,5,7,8,10)

DataSet <- data.frame(x=x,y=y)
TrainSet <- DataSet[inTrain,]
TestSet <- DataSet[-inTrain,]

lm1 <- lm(y~x , data=TrainSet)
y1 <- predict(lm1,newdata=TestSet)
rmse(lm1$fitted.values,TrainSet$y)
rmse(y1,TestSet$y)
plotfit(TrainSet,TestSet,lm1)


lm2 <- lm(y~ x+ I(x^2)+ I(x^3), data=TrainSet)
y2 <- predict(lm2,newdata=TestSet)
rmse(lm2$fitted.values,TrainSet$y)
rmse(y2,TestSet$y)
plotfit(TrainSet,TestSet,lm2)

lm3 <- lm(y~ x+ I(x^2) + I(x^3) + I(x^4) + I(x^5) , data=TrainSet)
y3 <- predict(lm3,newdata=TestSet)
rmse(lm3$fitted.values,TrainSet$y)
rmse(y3,TestSet$y)
plotfit(TrainSet,TestSet,lm3)


plotfit<-function(TrainSet,TestSet,modl) {
  xfit <- 0.02*(0:100) 
  yfit <- predict(modl,newdata=data.frame(x=xfit))
  
  plotout <- ggplot()+geom_point(aes(x=x,y=y),data=TrainSet,col="black",size=3) +geom_point(aes(x=x,y=y),data=TestSet,col="red", size=3) + geom_line(aes(x=xfit,y=yfit),col="blue" )
  plotout <- plotout + coord_cartesian(xlim=c(0,2) ,ylim = c(0,9))
    return(plotout)
   #
}
