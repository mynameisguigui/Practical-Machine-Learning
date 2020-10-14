##################
# FINAL PROJECT  #
##################
# 
# Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large 
# amount of data about personal activity relatively inexpensively. These type of devices are part of
# the quantified self movement - a group of enthusiasts who take measurements about themselves regularly 
# to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing 
# that people regularly do is quantify how much of a particular activity they do, but they rarely quantify 
# how well they do it. In this project, your goal will be to use data from accelerometers on the belt, 
# forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and 
# incorrectly in 5 different ways.

# The goal of your project is to predict the manner in which they did the exercise. 
# This is the "classe" variable in the training set. You may use any of the other variables to predict with.
# You should create a report describing how you built your model, how you used cross validation,
# what you think the expected out of sample error is, and why you made the choices you did.
# You will also use your prediction model to predict 20 different test cases.
# # 
# # Peer Review Portion
# Your submission for the Peer Review portion should consist of a link to a Github repo with your R markdown
# and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words
# and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo 
# with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
# 
# Course Project Prediction Quiz Portion
# Apply your machine learning algorithm to the 20 test cases available in the test data above and submit your 
# predictions in appropriate format to the Course Project Prediction Quiz for automated grading.

# ============ read data & explore ================
library(readr)
library(dplyr)

training <- read_csv("C:/Users/U552KZ/Desktop/pml-training.csv")
testing <- read_csv("C:/Users/U552KZ/Desktop/pml-testing.csv")

colnames(training)
table(training$classe) # A B C D E - pretty balanced dataset
sum(complete.cases(training)) # 324 complete cases

# ============ missing data & useless variables ================
dim(training) # dimension 19622 * 160 

glimpse(training) #lots of variables has high NA rate

# missing rate for each variable
for (i in colnames(training)){
  print(paste0(i,':',sum(is.na(training[[i]]))/dim(training)[1]))
}

# find variables which missing rate > 90%
list <- c()
for (i in colnames(training)){
  if (sum(is.na(training[[i]]))/dim(training)[1]){
    list<-c(list,i)
  }
}
length(list) # we have 100 sparse variables

# find useless variables 
head(training)

for (i in colnames(training)){
  print(i)
  print(summary(training[[i]]))
}

table(training$X1)
list <- c(list,'X1')

training <- training[,!names(training) %in% list] # drop sparse variables, we have 60 variables now
testing <- testing[,!names(testing) %in% list]

# =========== model building with cv ========
library(caret)
library(Metrics)

# split training data to train and validation 
inTrain <- createDataPartition(y = training$classe, p = 0.8, list = FALSE)
train <- training[inTrain,]
validation <- training[-inTrain,]

train.control <- trainControl(method = "cv", number = 5)

rf_mod <- train(classe~.,data = train,method = 'rf',na.action = na.pass,trControl = train.control)
gbm_mod <- train(classe~.,data = train,method = 'gbm',na.action = na.pass,trControl = train.control)
svm_mod <- train(classe~.,data = train,method = 'svmRadial',na.action = na.pass,trControl = train.control)

pred_rf <- predict(rf_mod,validation)
pred_gbm <- predict(gbm_mod,validation)
pred_svm <- predict(svm_mod,validation)

accuracy(validation$classe,pred_rf) # accuracy = 1
accuracy(validation$classe,pred_gbm) # accuracy = 99.54%
accuracy(validation$classe,pred_svm) # accuracy = 93.98%

# I choose random forest model 
# =========== prediction  ========

predict(rf_mod, testing)

