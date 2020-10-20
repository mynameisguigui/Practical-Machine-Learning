################################################
###         Practical Machine Learning       ###
################################################

############
#  week 2  #
############

# ================== Covariate creation / feature engineering=============================
library(ISLR)
library(caret)
data(Wage)

inTrain <- createDataPartition( y = Wage$wage,p = 0.7,list = FALSE)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]

table(training$jobclass)

dummies <- dummyVars(wage ~ jobclass,data = training)    # caret package
head(predict(dummies,newdata = training))

# Removing zero covariates
# Another thing that happens is that some of the variables are basically have no variability in them. 
# it has no variability and it's probably not going to be a useful covariate
# so one thing you could use is this near zero variable of function in carrot to identify those variables that 
# have very little variability and will likely not be good predictors.

nsv <- nearZeroVar(training,saveMetrics = TRUE)   
nsv

# Spline basis
library(Splines)
bsBasis <- bs(training$age,df = 3)    # polynomial
bsBasis

# Fitting curves with splines
lm1 <- lm(wage ~ bsBasis, data = training)
plot(training$age,training$wage, pch = 19, cex = 0.5)
points(training$age,predict(lm1,newdata = training),col = 'red', pch = 19,cex = 0.5)
predict(bsBasis,age = testing$age)

# Notes and further reading
# Level 1 raw data to covariates 
#         Google "feature extraction for [data type]"
#         in some application (image,voices) automated feature creation is possible/necessary
# Level 2 feature creation
#         the function preProcessing in caret will handle some preprocessing
# preprocessing with caret: fit spline models use GAM method in the caret package


# ================================== Predicting with regression ======================================

inTrain <- createDataPartition( y = Wage$wage,p = 0.7,list = FALSE)
trainFaith<- Wage[inTrain,]
testFaith <- Wage[-inTrain,]
head(trainFaith)
lm1<-lm(eruption ~ waiting,data = trainFaith)
summary(lm1)

# plot
lines(trainFaith$waiting,lm1$fitted,lwd = 3)

# estimate
coef(lm1)[1] + coef(lm1)[2]*80

# predict
predict(lm1,newdata)

# plot
par(mfrow = c(1,2))

# RMSE
sqrt(sum((lm1$fitted - trainFaith$eruption)^2)) 

# Prediction Intervals

# predicting with regression multiple covariates
featurePlot()

qplot(age,wage,data = training)
qplot(age,wage,color = jobclass,data = training)

modFit <- train(wage ~ age + jobclass + education, method = 'lm', data = training)
finMod <- modFit$finalModel
print(modFit)

library(ISLR)
library(ggplot2)
library(caret)
data(Wage)
Wage <- subset(Wage,select = -c(logwage))
summary(Wage)

inTrain <- createDataPartition(y = Wage$wage, p = 0.7,list = FALSE)
training <- Wage[inTrain,]
testing <-Wage[-inTrain,]
dim(training)
dim(testing)

featurePlot(x = training[,c('age','education','jobclass')],
            y = training$wage,
            plot = 'pairs')

qplot(age,wage,data = training)
qplot(age,wage,colour = jobclass,data = training)
qplot(age,wage,colour = education,data = training)
# Residuals vs Fitted
plot(finMod,1,pch = 19,cex = 0.5,col = '#00000010')
# Color by variables not used in the model
qplot(finMod$fitted,finMod$residuals,colour = race,data = training)
# plot by index
plot(finMod$residuals,pch = 19) # trend --> missing variables

pred <- predict(modFit,testing)
qplot(wage,pred,color = year,data = testing)


# use all the variables
modFitAll <- train(wage~.,data = training,method = 'lm')
pred <- predict(modFitAll,testing)
qplot(wage,pred,data = testing)


# ================================== Test ======================================
library(AppliedPredictiveModeling)
data(AlzheimerDisease)


library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

# variable missing 
library(Hmisc)
library(dplyr)
library(ggplot2)
library(gridExtra)
training <- mutate(training, index=1:nrow(training))
qplot(index, CompressiveStrength, data=training, color=cut2(training$Cement, g=10))
byCement <- qplot(index, CompressiveStrength, data=training, color=cut2(training$Cement, g=breaks))
byBlastFurnaceSlag <- qplot(index, CompressiveStrength, data=training, color=cut2(training$BlastFurnaceSlag, g=breaks))
byFlyAsh <- qplot(index, CompressiveStrength, data=training, color=cut2(training$FlyAsh, g=breaks))
byWater <- qplot(index, CompressiveStrength, data=training, color=cut2(training$Water, g=breaks))
bySuperplasticizer <- qplot(index, CompressiveStrength, data=training, color=cut2(training$Superplasticizer, g=breaks))
byCoarseAggregate <- qplot(index, CompressiveStrength, data=training, color=cut2(training$CoarseAggregate, g=breaks))
byFineAggregate <- qplot(index, CompressiveStrength, data=training, color=cut2(training$FineAggregate, g=breaks))
byAge <- qplot(index, CompressiveStrength, data=training, color=cut2(training$Age, g=breaks))
grid.arrange(byCement, byBlastFurnaceSlag, byFlyAsh, byWater, bySuperplasticizer, byCoarseAggregate, byFineAggregate, byAge)

# histogram
hist(training$Superplasticizer)
hist(log(training$Superplasticizer+1))

# PCA
# match patterns
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
IL_col_idx <- grep("^[Ii][Ll].*", names(training))
preObj <- preProcess(training[, IL_col_idx], method=c("center", "scale", "pca"), thresh=0.8)
preObj

# PCA vs Non PCA

# extract new training and testing sets
IL_col_idx <- grep("^[Ii][Ll].*", names(training))
suppressMessages(library(dplyr))
new_training <- training[, c(names(training)[IL_col_idx], "diagnosis")]
names(new_training)

IL_col_idx <- grep("^[Ii][Ll].*", names(testing))
suppressMessages(library(dplyr))
new_testing <- testing[, c(names(testing)[IL_col_idx], "diagnosis")]
names(new_testing)

# compute the model with non_pca predictors
non_pca_model <- train(diagnosis ~ ., data=new_training, method="glm")
# apply the non pca model on the testing set and check the accuracy
non_pca_result <- confusionMatrix(new_testing[, 13], predict(non_pca_model, new_testing[, -13]))
non_pca_result

# perform PCA extraction on the new training and testing sets
pc_training_obj <- preProcess(new_training[, -13], method=c('center', 'scale', 'pca'), thresh=0.8)
pc_training_preds <- predict(pc_training_obj, new_training[, -13])
pc_testing_preds <- predict(pc_training_obj, new_testing[, -13])
# compute the model with pca predictors
pca_model <- train(new_training$diagnosis ~ ., data=pc_training_preds, method="glm")
# apply the PCA model on the testing set
pca_result <- confusionMatrix(new_testing[, 13], predict(pca_model, pc_testing_preds))
pca_result

############
#  Week 3  #
############
#=============== trees ==============
# Decision Tree

# Measures of Impurity
# Misclassification Error, Gini index, Deviance/information gain ~ 
data(iris)
library(ggplot2)
names(iris)

table(iris$Species)
inTrain <- createDataPartition( y = iris$Species, p = 0.7, list = FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training)
dim(testing)

qplot(Petal.Width,Sepal.Width,colour = Species,data = training)
library(caret)
modFit <- train(Species~.,method = 'rpart',data = training)
print(modFit$finalModel)

plot(modFit$finalModel,uniform = TRUE,main = 'Classification Tree')
text(modFit$finalModel,use.n = TRUE,all = TRUE,cex = .8)

library(rattle)
fancyRpartPlot(modFit$finalModel)

# predicting
predict(modFit,newdata = testing)

#================ bagging =====================
library(ElemStatLearn)
data(ozone,package = 'ElemStatLearn')
ozone <- ozone[order(ozone$ozone),]
head(ozone)

# train --> bagEarth,treebag,bagFDA

# More bagging in caret
library(party)

predictors <- data.frame(ozone = ozone$ozone)
temperature <- ozone$temperature
treebag <- bag(predictors,temperature,B = 10, 
               bagControl = bagControl(fit = ctreeBag$fit,
                                       predict = ctreeBag$pred,
                                       aggregate = ctreeBag$aggregate))


plot(ozone$ozone,temperature,col = 'lightgrey',pch = 9)
points(ozone$ozone,predict(treebag$fits[[1]]$fit,predictors),pch = 19,col = 'red')
points(ozone$ozone,predict(treebag,predictors),pch = 10,col = 'blue')


#================ random forest =====================

data(iris)
library(ggplot2)
inTrain <- createDataPartition(y = iris$Species, p = 0.7,list = FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]

library(caret)
modFit <- train(Species~ ., data = training, method = "rf", prox = TRUE)
modFit

library(randomForest)
getTree(modFit$finalModel, k=2)

irisP <- classCenter(training[,c(3,4)],training$Species,modFit$finalModel$prox)
irisP <- as.data.frame(irisP);irisP$Species <-rownames(irisP)
p <- qplot(Petal.Width,Petal.Length,col = Species,data = training)
p + geom_point(aes(x = Petal.Width,y = Petal.Length,col = Species), size = 5, shape = 4,data = irisP)

# make predictions

pred <- predict(modFit,testing)
testing$predRight <- pred== testing$Species
table(pred,testing$Species)

qplot(Petal.Width,Petal.Length,colour = predRight,data = testing,main = 'newdata predictions')

# ==================== try this quiz ==========================

library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)

inTrain <- createDataPartition(y = segmentationOriginal$Class, p = 0.7,list = FALSE)
training <- segmentationOriginal[inTrain,]
testing <- segmentationOriginal[-inTrain,]

set.seed(125)

library(caret)
modFit <- train(Class~.,method = 'rpart',data = training)
modFit$finalModel
# 1.
testing <- segmentationOriginal[-inTrain,]
testing$TotalIntenCh2 <- 23000
testing$FiberWidthCh1 <- 10
testing$PerimStatusCh1 <- 2
pred <- predict(modFit,testing)

# 2.
testing <- segmentationOriginal[-inTrain,]
testing$TotalIntenCh2 <- 50000
testing$FiberWidthCh1 <- 10
testing$VarIntenCh4 <- 100
pred <- predict(modFit,testing)

# 3.
testing <- segmentationOriginal[-inTrain,]
testing$TotalIntenCh2 <- 57000
testing$FiberWidthCh1 <- 8
testing$VarIntenCh4 <- 100
pred <- predict(modFit,testing)

# 4.
testing <- segmentationOriginal[-inTrain,]
testing$PerimStatusCh1 <- 2
testing$FiberWidthCh1 <- 8
testing$VarIntenCh4 <- 100
pred <- predict(modFit,testing)


suppressMessages(library(rattle))
library(rpart.plot)
fancyRpartPlot(modFit$finalModel)


# Leave-one-out cross validation is K-fold cross validation taken to its logical extreme, with K equal to N,
# the number of data points in the set. 


# Q3.
library(pgmm)
data(olive)
olive = olive[,-1]
modFit <- train(Area ~.,method = 'rpart',data = olive)
modFit
suppressMessages(library(rattle))
library(rpart.plot)
fancyRpartPlot(modFit$finalModel)
newdata = as.data.frame(t(colMeans(olive)))
predict(modFit,newdata)

# Q4.
library(ElemStatLearn)
data(SAheart)
RNGversion("3.5.3")
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]
set.seed(13234)

glmFit <- train(chd ~ age+alcohol+obesity+tobacco+typea+ldl,method = 'glm',family = 'binomial',
                data = trainSA)
missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}

missClass(trainSA$chd,predict(glmFit,trainSA))

missClass(testSA$chd,predict(glmFit,testSA))

# Q5.

library(ElemStatLearn)
data(vowel.train)
data(vowel.test)
vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)

set.seed(33833)
library(randomForest)
modvowel <- randomForest(y ~ ., data = vowel.train)
order(varImp(modvowel),decreasing = T)


############
#  Week 4  #
############
# =================== regularized regression =============
library(ElemStatLearn)
data(prostate)
str(prostate)

# basic idea : fit a linear model and penalize(shrink)large coefficients.
# pros : can help with the bias/variance tradeoff, can help with model selection
# cons : may be computationally demanding on large data sets & does not perform as well as random forests and boosting

# hard thresholding : contrain only n coefficients to be nonzero
# Ridge Regression : if we have a lot of nonzero coefficients, then each coefficient would be pretty small 

# reduce irreducible error + bias^2 + variance overall

#=================== combine predictors ===================

# improve accuracy but reduce interpretability

# combine different classifiers --> stacking, ensembling ~ 

#=================== Forecast ========================

# Data are dependent over time 
# specific pattern types : trends,seasonal patterns,cycles
# subsampling into training/test is more complicated
# spatial data : dependency between nearby observations,location specific effects
# goal is to predict one or more observations into the future

# Beware spurious correlations & extrapolation!

library(quantmod)
from.dat <- as.Date("01/01/08",format = '%m/%d/%y')
to.dat <- as.Date("12/31/13",format = '%m/%d/%y')
getSymbols('GOOG',src='google',from = from.dat,to=to.dat)

head(GOOG)
mGoog <- to.monthly(GOOG)
googOpen <- Op(mGoog)
ts1 <- ts(googOpen,frequency = 12)
plot(ts1,xlab = 'Years+1',ylab='GOOG')

# decompose a time series into parts : trends, seasonal,cyclic

plot(decompose(ts1),xlab="Years+1")

ts1Train <- window(ts1,start = 1,end = 5)
ts1Train <- window(ts1,start = 5,end = (7-0.01))
ts1Train

plot(ts1Train)

# simple moving average 
lines(ma(ts1Train,order = 3),col='red')

# exponential smoothing
ets1 <- ets(ts1Train,model = "MM")
fcast <- forecast(ets1)
plot(fcast)
lines(ts1Test,col='red')

#=================== Unsupervised Predictions ========================

# ======== Q1 =======

library(ElemStatLearn)
library(caret)
data(vowel.train)
data(vowel.test)

vowel.test$y <- as.factor(vowel.test$y)
vowel.train$y <- as.factor(vowel.train$y)
set.seed(33833)

rf_fit <- train(y~.,method = 'rf',data = vowel.train)
gbm_fit <- train(y~.,method = 'gbm',data = vowel.train)

accuracy(vowel.test$y,predict(rf_fit,vowel.test))
accuracy(vowel.test$y,predict(gbm_fit,vowel.test))

pred_rf <- predict(rf_fit,vowel.test)
pred_gbm <- predict(gbm_fit,vowel.test)
predDF <- data.frame(pred_rf, pred_gbm, y = vowel.test$y)
sum(pred_rf[predDF$pred_rf == predDF$pred_gbm] == 
      predDF$y[predDF$pred_rf == predDF$pred_gbm]) / 
  sum(predDF$pred_rf == predDF$pred_gbm)

# ======== Q2 =======
library(caret)

library(gbm)

set.seed(3433)

library(AppliedPredictiveModeling)

data(AlzheimerDisease)

adData = data.frame(diagnosis,predictors)

inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

set.seed(62433)
rf_fit <- train(diagnosis~.,method = 'rf',data = training)
gbm_fit <- train(diagnosis~.,method = 'gbm',data = training)
lda_fit <- train(diagnosis~.,method = 'gbm',data = training)

pred_rf <- predict(rf_fit,testing)
pred_gbm <- predict(gbm_fit,testing)
pred_lda <- predict(lda_fit,testing)

accuracy(testing$diagnosis,pred_rf)
accuracy(testing$diagnosis,pred_gbm)
accuracy(testing$diagnosis,pred_lda)

predDF <- data.frame(pred_rf, pred_gbm, pred_lda, diagnosis = testing$diagnosis)
combModFit <- train(diagnosis ~ ., method = "rf", data = predDF)
combPred <- predict(combModFit, predDF)
accuracy(testing$diagnosis,combPred)

# ======== Q3 =======

set.seed(3523)

library(AppliedPredictiveModeling)

data(concrete)

inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]

training = concrete[ inTrain,]

testing = concrete[-inTrain,]

set.seed(233)
mod_lasso <- train(CompressiveStrength ~ ., data = training, method = "lasso")
library(elasticnet)
plot.enet(mod_lasso$finalModel, xvar = "penalty", use.color = TRUE)

#========= Q4 ========

library(lubridate) # For year() function below
library(readr)

dat<- read_csv("C:/Users/U552KZ/Desktop/gaData.csv")

training = dat[year(dat$date) < 2012,]

testing = dat[(year(dat$date)) > 2011,]

tstrain = ts(training$visitsTumblr)

# Fit a model using the bats() function in the forecast package to the training time series. 
# Then forecast this model for the remaining time points. 
# For how many of the testing points is the true value within the 95% prediction interval bounds?

library(forecast)
mod_ts <- bats(tstrain)
fcast <- forecast(mod_ts, level = 95, h = dim(testing)[1])
sum(fcast$lower < testing$visitsTumblr & testing$visitsTumblr < fcast$upper) / 
  dim(testing)[1]

#========= Q5 =========
set.seed(3523)

library(AppliedPredictiveModeling)

data(concrete)

inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]

training = concrete[ inTrain,]

testing = concrete[-inTrain,]

set.seed(325)
library(e1071)
mod_svm <- svm(CompressiveStrength ~ ., data = training)
pred_svm <- predict(mod_svm, testing)

accuracy(pred_svm, testing$CompressiveStrength)






