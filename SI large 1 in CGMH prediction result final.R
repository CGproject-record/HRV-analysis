
rm(list=ls(all=TRUE))
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# Part 1 train model
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# --------------------load package
library(caret)
library(kernlab)
library(foreign)
library(mlbench)
library(rpart)
library(rpart.plot)
library(ipred)
library(randomForest)
library(gbm)
library(nnet)
library(dplyr)
library(plyr)
library(hydroGOF)
library(Boruta)
library(pROC)
library(doParallel)
library(SHAPforxgboost)
library(deepnet)
registerDoParallel(10)
getDoParWorkers()


# load data 
mydata=read.csv("Siover1_rf_top10_cgmh_data_with_score.csv",header=TRUE)



availvar<-c("aFdP",
            "fFdP",
            "ARerr",
            "DFA.Alpha.1",
            "Mean.rate",
            "Poincar..SD2",
            "shannEn",
            "LF.HF.ratio.LombScargle","gcount","sgridTAU")



fitControl=trainControl(method="repeatedcv",
                        number=10,
                        repeats=10,
                        classProbs=TRUE,
                        summaryFunction=twoClassSummary,
                        search="random")




mydata_new<-cbind(si1m=mydata$si1m,mydata[,availvar])
mydata_new$si1m<-factor(mydata_new$si1m,levels = c("Y","N"))
##
dim(mydata_new)
set.seed(123)
mydata1=mydata_new[which(mydata_new$si1m=="Y"),]
mydata2=mydata_new[which(mydata_new$si1m=="N"),]
index1=sample(dim(mydata1)[1],nrow(mydata1)*0.25)
index2=sample(dim(mydata2)[1],nrow(mydata2)*0.25)
testing=rbind(mydata1[index1,],mydata2[index2,])
training=rbind(mydata1[-index1,],mydata2[-index2,])



# ----------train model
rp1_train<-train( si1m~.,
                  data=training,
                  method="rpart",
                  metric="ROC",
                  tuneLength=20,
                  trControl=fitControl)


rp2_train<-train(si1m~.,
                 data=training,
                 method="rpart2",
                 metric="ROC",
                 tuneLength=20,
                 trControl=fitControl)






rf_train<-train(si1m~.,
                data=training,
                method="rf",
                metric="ROC",
                tuneLength=20,ntree=40,
                trControl=fitControl)


set.seed(34)
rf_train<-train(si1m~.,
                data=training,
                method="rf",
                metric="ROC",
                tuneLength=20,ntree=40,
                trControl=fitControl)






set.seed(2)
sdwd_train<-train(si1m~.,
                  data=training,
                  method="sdwd",
                  metric="ROC",
                  tuneLength=2,
                  trControl=fitControl)



set.seed(99)
nnet_train<-train(si1m~.,
                  data=training,
                  method="nnet",
                  metric="ROC",
                  tuneLength=10,
                  trControl=fitControl)




set.seed(13)
xgb_train<-train(si1m~.,
                 data=training,
                 method="xgbTree",
                 metric="ROC",
                 tuneLength= 40,
                 trControl=fitControl)


# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# Part 2 show model parameters
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------


cat("Models parameters:\n")
print(sdwd_train)
print(nnet_train)
print(rp1_train)
print(rp2_train)
print(xgb_train)
print(rf_train)


# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# Part 3 show result
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------

predictions_train=predict(xgb_train,newdata=training)
predictions_test=predict(xgb_train,newdata=testing)
xgbtra<-confusionMatrix(predict(xgb_train,training),training$si1m)$overall["Accuracy"]
xgbtea<-confusionMatrix(predict(xgb_train,testing),testing$si1m)$overall["Accuracy"]
train_results=predict(xgb_train,training,type="prob")
test_results=predict(xgb_train,testing,type="prob")
train_results$obs=training$si1m
train_results$pred=predictions_train
test_results$obs=testing$si1m
test_results$pred=predictions_test
xgb_trauc<-roc(training$si1m,train_results[,"Y"],levels=c("Y","N"))$auc
xgb_teauc<-roc(testing$si1m,test_results[,"Y"],levels=c("Y","N"))$auc
xgbresult<-c(xgbtra,xgbtea,xgb_trauc,xgb_teauc)


##2. sdwd
predictions_train=predict(sdwd_train,newdata=training)
predictions_test=predict(sdwd_train,newdata=testing)
sdwdtra<-confusionMatrix(predict(sdwd_train,training),training$si1m)$overall["Accuracy"]
sdwdtea<-confusionMatrix(predict(sdwd_train,testing),testing$si1m)$overall["Accuracy"]
train_results=predict(sdwd_train,training,type="prob")
test_results=predict(sdwd_train,testing,type="prob")
train_results$obs=training$si1m
train_results$pred=predictions_train
test_results$obs=testing$si1m
test_results$pred=predictions_test
sdwd_trauc<-roc(training$si1m,train_results[,"Y"],levels=c("Y","N"))$auc
sdwd_teauc<-roc(testing$si1m,test_results[,"Y"],levels=c("Y","N"))$auc
sdwdresult<-c(sdwdtra,sdwdtea,sdwd_trauc,sdwd_teauc)


##3. nnet
predictions_train=predict(nnet_train,newdata=training)
predictions_test=predict(nnet_train,newdata=testing)
nnettra<-confusionMatrix(predict(nnet_train,training),training$si1m)$overall["Accuracy"]
nnettea<-confusionMatrix(predict(nnet_train,testing),testing$si1m)$overall["Accuracy"]
train_results=predict(nnet_train,training,type="prob")
test_results=predict(nnet_train,testing,type="prob")
train_results$obs=training$si1m
train_results$pred=predictions_train
test_results$obs=testing$si1m
test_results$pred=predictions_test
nnet_trauc<-roc(training$si1m,train_results[,"Y"],levels=c("Y","N"))$auc
nnet_teauc<-roc(testing$si1m,test_results[,"Y"],levels=c("Y","N"))$auc
nnetresult<-c(nnettra,nnettea,nnet_trauc,nnet_teauc)


##4. rpart
predictions_train=predict(rp1_train,newdata=training)
predictions_test=predict(rp1_train,newdata=testing)
rp1tra<-confusionMatrix(predict(rp1_train,training),training$si1m)$overall["Accuracy"]
rp1tea<-confusionMatrix(predict(rp1_train,testing),testing$si1m)$overall["Accuracy"]
train_results=predict(rp1_train,training,type="prob")
test_results=predict(rp1_train,testing,type="prob")
train_results$obs=training$si1m
train_results$pred=predictions_train
test_results$obs=testing$si1m
test_results$pred=predictions_test
rp1_trauc<-roc(training$si1m,train_results[,"Y"],levels=c("Y","N"))$auc
rp1_teauc<-roc(testing$si1m,test_results[,"Y"],levels=c("Y","N"))$auc
rp1result<-c(rp1tra,rp1tea,rp1_trauc,rp1_teauc)



##5. rpart2
predictions_train=predict(rp2_train,newdata=training)
predictions_test=predict(rp2_train,newdata=testing)
rp2tra<-confusionMatrix(predict(rp2_train,training),training$si1m)$overall["Accuracy"]
rp2tea<-confusionMatrix(predict(rp2_train,testing),testing$si1m)$overall["Accuracy"]
train_results=predict(rp2_train,training,type="prob")
test_results=predict(rp2_train,testing,type="prob")
train_results$obs=training$si1m
train_results$pred=predictions_train
test_results$obs=testing$si1m
test_results$pred=predictions_test
rp2_trauc<-roc(training$si1m,train_results[,"Y"],levels=c("Y","N"))$auc
rp2_teauc<-roc(testing$si1m,test_results[,"Y"],levels=c("Y","N"))$auc
rp2result<-c(rp2tra,rp2tea,rp2_trauc,rp2_teauc)


##6. rf
predictions_train=predict(rf_train,newdata=training)
predictions_test=predict(rf_train,newdata=testing)
rftra<-confusionMatrix(predict(rf_train,training),training$si1m)$overall["Accuracy"]
rftea<-confusionMatrix(predict(rf_train,testing),testing$si1m)$overall["Accuracy"]
train_results=predict(rf_train,training,type="prob")
test_results=predict(rf_train,testing,type="prob")
train_results$obs=training$si1m
train_results$pred=predictions_train
test_results$obs=testing$si1m
test_results$pred=predictions_test
rf_trauc<-roc(training$si1m,train_results[,"Y"],levels=c("Y","N"))$auc
rf_teauc<-roc(testing$si1m,test_results[,"Y"],levels=c("Y","N"))$auc
rfresult<-c(rftra,rftea,rf_trauc,rf_teauc)







dnn_train<-read.csv('dnn for si over 1 in cgmh train result ver2 20230503.csv',header = T)
head(dnn_train)

dnn_train$class_pred <-factor(ifelse(dnn_train$class_pred=="0","N","Y"),levels=c("Y","N"))



dnn_test<-read.csv('dnn for si over 1 in cgmh test result ver2 20230503.csv',header = T)
head(dnn_test)

dnn_test$class_pred <-factor(ifelse(dnn_test$class_pred=="0","N","Y"),levels=c("Y","N"))



predictions_train=dnn_train$class_pred 
predictions_test=dnn_test$class_pred
dnntra<-confusionMatrix(predictions_train,training$si1m)$overall["Accuracy"]
dnntea<-confusionMatrix(predictions_test,testing$si1m)$overall["Accuracy"]
dnn_trauc<-roc(training$si1m,dnn_train$prob_pred,levels=c("Y","N"))$auc

dnn_teauc<-roc(testing$si1m,dnn_test$prob_pred,levels=c("Y","N"))$auc
dnnresult<-c(dnntra,dnntea,dnn_trauc,dnn_teauc)

a<-rbind(  sdwdresult ,nnetresult,rp1result,rp2result,rfresult,xgbresult,dnnresult)
colnames(a)<-c("Training Accuracy","Testing Accuracy","Training AUC","Testing AUC")
cat("Models result:\n")
print(a)

# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# Part 4 show models 95 percent CI
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------
transfor_CI<-function( model ,data ) {
  
  tr_con<-confusionMatrix(predict(model,data),data$si1m)$table
  
  library(epiR)
  rval <- epi.tests(tr_con, conf.level = 0.95)
  
  # train
  AUC_result<-roc(data$si1m, predict(model,data,type='prob')[,"Y"],levels=c("Y","N"),CI=TRUE)
  as.numeric(AUC_result$auc)
  ci_auc_result<-ci(AUC_result)
  ci_auc_result<-as.numeric(ci_auc_result)
  ci_auc_result
  Sensitivity<-rval$detail[3,2:4]
  # Sensitivity
  
  Specificity<-rval$detail[4,2:4]
  # Specificity
  
  ci_auc_result<-ci_auc_result[c(2,1,3)]
  all_result<-rbind(ci_auc_result,Sensitivity,Specificity)
  row.names(all_result)<-c("AUC","Sensitivity","Specificity")
  # all_result
  
  
  all_result$CI<-paste(round(all_result$est,4)," [",round(all_result$lower,4),"-",round(all_result$upper,4),"]")
  # all_result
  
  
  final_CI<-as.data.frame(rbind(all_result$CI))
  # final_CI
  names(final_CI)<-c("AUC","Sensitivity","Specificity")
  # final_CI
  return(final_CI)
  
  
}






# ----------------------------train    

train_all_ci<-rbind(transfor_CI(sdwd_train,training),
                    transfor_CI(nnet_train,training),
                    transfor_CI(rp1_train,training),
                    transfor_CI(rp2_train,training),
                    transfor_CI(rf_train,training),
                    transfor_CI(xgb_train,training))


train_all_ci<-as.data.frame(train_all_ci)

rownames(train_all_ci)<-c("SDWD","nnet","raprt1","rpart2","random forest","XGBoost")


# ----------------------------test

test_all_ci<-rbind(transfor_CI(sdwd_train,testing),
                   transfor_CI(nnet_train,testing),
                   transfor_CI(rp1_train,testing),
                   transfor_CI(rp2_train,testing),
                   transfor_CI(rf_train,testing),
                   transfor_CI(xgb_train,testing))


test_all_ci<-as.data.frame(test_all_ci)

rownames(test_all_ci)<-c("SDWD","nnet","raprt1","rpart2","random forest","XGBoost")





# -----------dnn train----------------------


dnn_train<-read.csv('dnn for si over 1 in cgmh train result ver2 20230503.csv',header = T)
head(dnn_train)

dnn_train$class_pred <-factor(ifelse(dnn_train$class_pred=="0","N","Y"),levels=c("Y","N"))
tr_con<-confusionMatrix(dnn_train$class_pred ,training$si1m)$table
library(epiR)
rval <- epi.tests(tr_con, conf.level = 0.95)



# train
AUC_result<-roc(training$si1m, dnn_train$prob_pred,levels=c("Y","N"),CI=TRUE)
as.numeric(AUC_result$auc)
ci_auc_result<-ci(AUC_result)
ci_auc_result<-as.numeric(ci_auc_result)
ci_auc_result
Sensitivity<-rval$detail[3,2:4]
# Sensitivity

Specificity<-rval$detail[4,2:4]
# Specificity

ci_auc_result<-ci_auc_result[c(2,1,3)]
all_result<-rbind(ci_auc_result,Sensitivity,Specificity)
row.names(all_result)<-c("AUC","Sensitivity","Specificity")
# all_result


all_result$CI<-paste(round(all_result$est,4)," [",round(all_result$lower,4),"-",round(all_result$upper,4),"]")
# all_result


final_CI<-as.data.frame(rbind(all_result$CI))
# final_CI
names(final_CI)<-c("AUC","Sensitivity","Specificity")

dnn_train_result<-final_CI




# -----------dnn test----------------------


dnn_test<-read.csv('dnn for si over 1 in cgmh test result ver2 20230503.csv',header = T)
head(dnn_test)

dnn_test$class_pred <-factor(ifelse(dnn_test$class_pred=="0","N","Y"),levels=c("Y","N"))
tr_con<-confusionMatrix(dnn_test$class_pred ,testing$si1m)$table
library(epiR)
rval <- epi.tests(tr_con, conf.level = 0.95)



# train
AUC_result<-roc(testing$si1m, dnn_test$prob_pred,levels=c("Y","N"),CI=TRUE)
as.numeric(AUC_result$auc)
ci_auc_result<-ci(AUC_result)
ci_auc_result<-as.numeric(ci_auc_result)
ci_auc_result
Sensitivity<-rval$detail[3,2:4]
# Sensitivity

Specificity<-rval$detail[4,2:4]
# Specificity

ci_auc_result<-ci_auc_result[c(2,1,3)]
all_result<-rbind(ci_auc_result,Sensitivity,Specificity)
row.names(all_result)<-c("AUC","Sensitivity","Specificity")
# all_result


all_result$CI<-paste(round(all_result$est,4)," [",round(all_result$lower,4),"-",round(all_result$upper,4),"]")
# all_result


final_CI<-as.data.frame(rbind(all_result$CI))
# final_CI
names(final_CI)<-c("AUC","Sensitivity","Specificity")

dnn_test_result<-final_CI



# ---------------combine all 
cat("Training data models 95% CI:\n")
train_all_ci<-rbind(train_all_ci,dnn_train_result)
row.names(train_all_ci)[7]<-"DNN"
train_all_ci


cat("Testing data models 95% CI:\n")
test_all_ci<-rbind(test_all_ci,dnn_test_result)
row.names(test_all_ci)[7]<-"DNN"
test_all_ci





# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# Part 4.1 show external valid result and 95 percent CI
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------


exter_valid<- read.csv('Siover1_rf_top10_mimic_data.csv',header = T)
exter_valid$si1m<-factor(ifelse(exter_valid$si1m=="1","Y","N"),levels=c("Y","N"))
external_ci_xgb<-transfor_CI(xgb_train,exter_valid)

external_ci_xgb<-as.data.frame(external_ci_xgb)
row.names(external_ci_xgb)<-"Xgboost"




model<- xgb_train
#-----------------------------------------------------------
predictions_test_new=predict(model,newdata=exter_valid)
modeltea_new<-confusionMatrix(predict(model,exter_valid),exter_valid$si1m)$overall["Accuracy"]
modeltea_new

test_results_new=predict(model,exter_valid,type="prob")


test_results_new$obs=exter_valid$si1m
test_results_new$pred=exter_valid
model_teauc_new<-roc(exter_valid$si1m,test_results_new[,"Y"],levels=c("Y","N"))$auc
model_teauc_new



xgbresult1<-c(xgbresult,modeltea_new,model_teauc_new)

a1<-rbind(xgbresult1)
# model result
colnames(a1)<-c("Training Accuracy","Testing Accuracy","Training AUC","Testing AUC","new Testing Accuracy","new Testing AUC")
print(a1)





# add CI 

tr_con<-confusionMatrix(predictions_test_new,exter_valid$si1m)$table

library(epiR)
rval <- epi.tests(tr_con, conf.level = 0.95)

# train MEWS
AUC_result<-roc(exter_valid$si1m, test_results_new[,"Y"],levels=c("Y","N"),CI=TRUE)
as.numeric(AUC_result$auc)
ci_auc_result<-ci(AUC_result)
ci_auc_result<-as.numeric(ci_auc_result)
ci_auc_result
Sensitivity<-rval$detail[3,2:4]
# Sensitivity
#Specificity
Specificity<-rval$detail[4,2:4]
# Specificity

ci_auc_result<-ci_auc_result[c(2,1,3)]
all_result<-rbind(ci_auc_result,Sensitivity,Specificity)
row.names(all_result)<-c("AUC","Sensitivity","Specificity")
# all_result


all_result$CI<-paste(round(all_result$est,4)," [",round(all_result$lower,4),"-",round(all_result$upper,4),"]")
# all_result


final_CI<-as.data.frame(rbind(all_result$CI))
# final_CI
names(final_CI)<-c("AUC","Sensitivity","Specificity")
# final_CI
news_test_ci<-final_CI
news_test_ci


# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# Part 5 show score 95 percent CI
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------




mydata=read.csv("Siover1_rf_top10_cgmh_data_with_score.csv",header=TRUE)



availvar<-c("aFdP",
            "fFdP",
            "ARerr",
            "DFA.Alpha.1",
            "Mean.rate",
            "Poincar..SD2",
            "shannEn",
            "LF.HF.ratio.LombScargle","gcount","sgridTAU",'sofam','MEWSm','NEWSm')





mydata_new<-cbind(si1m=mydata$si1m,mydata[,availvar])
mydata_new$si1m<-factor(mydata_new$si1m,levels = c("Y","N"))




##
dim(mydata_new)
set.seed(123)
mydata1=mydata_new[which(mydata_new$si1m=="Y"),]
mydata2=mydata_new[which(mydata_new$si1m=="N"),]
index1=sample(dim(mydata1)[1],nrow(mydata1)*0.25)
index2=sample(dim(mydata2)[1],nrow(mydata2)*0.25)
testing=rbind(mydata1[index1,],mydata2[index2,])
training=rbind(mydata1[-index1,],mydata2[-index2,])

delta_SOFA_train <- training[,'sofam']
delta_SOFA_test<- testing[,'sofam']
MEWS_train <- training[,'MEWSm']
MEWS_test <- testing[,'MEWSm']
NEWS_train <- training[,'NEWSm']
NEWS_test <- testing[,'NEWSm']







cutoff<-1


outcome<-training$si1m
pred<-factor(ifelse(training$sofam>cutoff,"Y","N"),levels = c("Y","N"))

t0<-confusionMatrix(pred,outcome)
t0
acc<-t0$overall[1]



result_delta_sofa_train<-as.data.frame(cbind(cutoff,acc))





cutoff<-4


outcome<-training$si1m
pred<-factor(ifelse(training$MEWSm>cutoff,"Y","N"),levels = c("Y","N"))

t0<-confusionMatrix(pred,outcome)
t0
acc<-t0$overall[1]



result_mews_train<-as.data.frame(cbind(cutoff,acc))
result_mews_train



cutoff<-6

outcome<-training$si1m
pred<-factor(ifelse(training$NEWSm>cutoff,"Y","N"),levels = c("Y","N"))

t0<-confusionMatrix(pred,outcome)
t0
acc<-t0$overall[1]

result_news_train<-as.data.frame(cbind(cutoff,acc))





cutoff<-1

acc<-c()

outcome<-testing$si1m
pred<-factor(ifelse(testing$sofam>cutoff,"Y","N"),levels = c("Y","N"))

t0<-confusionMatrix(pred,outcome)
t0
acc<-t0$overall[1]




result_delta_sofa_test<-as.data.frame(cbind(cutoff,acc))
result_delta_sofa_test


cutoff<-4


outcome<-testing$si1m
pred<-factor(ifelse(testing$MEWSm>cutoff,"Y","N"),levels = c("Y","N"))

t0<-confusionMatrix(pred,outcome)
t0
acc<-t0$overall[1]


result_mews_test<-as.data.frame(cbind(cutoff,acc))
result_mews_test



cutoff<-6

outcome<-testing$si1m
pred<-factor(ifelse(testing$NEWSm>cutoff,"Y","N"),levels = c("Y","N"))

t0<-confusionMatrix(pred,outcome)
t0
acc<-t0$overall[1]




result_news_test<-as.data.frame(cbind(cutoff,acc))





# ----------------------------------------------


cutoff<-1


tr_con<-confusionMatrix(factor(ifelse(delta_SOFA_train>cutoff,"Y","N"),levels=c('Y','N')),training$si1m)$table

library(epiR)
rval <- epi.tests(tr_con, conf.level = 0.95)

# train MEWS
AUC_result<-roc(training$si1m, delta_SOFA_train,levels=c("Y","N"),CI=TRUE)
as.numeric(AUC_result$auc)
ci_auc_result<-ci(AUC_result)
ci_auc_result<-as.numeric(ci_auc_result)
ci_auc_result
Sensitivity<-rval$detail[3,2:4]
# Sensitivity
#Specificity
Specificity<-rval$detail[4,2:4]
# Specificity

ci_auc_result<-ci_auc_result[c(2,1,3)]
all_result<-rbind(ci_auc_result,Sensitivity,Specificity)
row.names(all_result)<-c("AUC","Sensitivity","Specificity")
# all_result


all_result$CI<-paste(round(all_result$est,4)," [",round(all_result$lower,4),"-",round(all_result$upper,4),"]")
# all_result


final_CI<-as.data.frame(rbind(all_result$CI))
# final_CI
names(final_CI)<-c("AUC","Sensitivity","Specificity")
# final_CI
news_test_ci<-final_CI
news_test_ci
ci0 <- rbind(news_test_ci)


ci0$cutoff<-cutoff


delta_sofa_cut_off_result_train<-ci0



delta_sofa_cut_off_result_train$accuracy <-result_delta_sofa_train$acc





cutoff<-4


tr_con<-confusionMatrix(factor(ifelse(MEWS_train>cutoff,"Y","N"),levels=c('Y','N')),training$si1m)$table

library(epiR)
rval <- epi.tests(tr_con, conf.level = 0.95)

# train MEWS
AUC_result<-roc(training$si1m, MEWS_train,levels=c("Y","N"),CI=TRUE)
as.numeric(AUC_result$auc)
ci_auc_result<-ci(AUC_result)
ci_auc_result<-as.numeric(ci_auc_result)
ci_auc_result
Sensitivity<-rval$detail[3,2:4]
# Sensitivity

Specificity<-rval$detail[4,2:4]
# Specificity

ci_auc_result<-ci_auc_result[c(2,1,3)]
all_result<-rbind(ci_auc_result,Sensitivity,Specificity)
row.names(all_result)<-c("AUC","Sensitivity","Specificity")



all_result$CI<-paste(round(all_result$est,4)," [",round(all_result$lower,4),"-",round(all_result$upper,4),"]")



final_CI<-as.data.frame(rbind(all_result$CI))
# final_CI
names(final_CI)<-c("AUC","Sensitivity","Specificity")
# final_CI
news_test_ci<-final_CI
news_test_ci
ci0 <- rbind(news_test_ci)


ci0$cutoff<-cutoff


mews_cut_off_train_result<-ci0
mews_cut_off_train_result


mews_cut_off_train_result$accuracy <- result_mews_train$acc




cutoff<-6


tr_con<-confusionMatrix(factor(ifelse(NEWS_train>cutoff,"Y","N"),levels=c('Y','N')),training$si1m)$table

library(epiR)
rval <- epi.tests(tr_con, conf.level = 0.95)

# train MEWS
AUC_result<-roc(training$si1m, NEWS_train,levels=c("Y","N"),CI=TRUE)
as.numeric(AUC_result$auc)
ci_auc_result<-ci(AUC_result)
ci_auc_result<-as.numeric(ci_auc_result)
ci_auc_result
Sensitivity<-rval$detail[3,2:4]
# Sensitivity

Specificity<-rval$detail[4,2:4]
# Specificity

ci_auc_result<-ci_auc_result[c(2,1,3)]
all_result<-rbind(ci_auc_result,Sensitivity,Specificity)
row.names(all_result)<-c("AUC","Sensitivity","Specificity")
# all_result


all_result$CI<-paste(round(all_result$est,4)," [",round(all_result$lower,4),"-",round(all_result$upper,4),"]")
# all_result


final_CI<-as.data.frame(rbind(all_result$CI))
# final_CI
names(final_CI)<-c("AUC","Sensitivity","Specificity")
# final_CI
news_test_ci<-final_CI
news_test_ci
ci0 <- rbind(news_test_ci)


ci0$cutoff<-cutoff


news_cut_off_train_result<-ci0


news_cut_off_train_result$accuracy <- result_news_train$acc



# ----------------------------------------------------







cutoff<-1


tr_con<-confusionMatrix(factor(ifelse(delta_SOFA_test>cutoff,"Y","N"),levels=c('Y','N')),testing$si1m)$table

library(epiR)
rval <- epi.tests(tr_con, conf.level = 0.95)

# train MEWS
AUC_result<-roc(testing$si1m, delta_SOFA_test,levels=c("Y","N"),CI=TRUE)
as.numeric(AUC_result$auc)
ci_auc_result<-ci(AUC_result)
ci_auc_result<-as.numeric(ci_auc_result)
ci_auc_result
Sensitivity<-rval$detail[3,2:4]
# Sensitivity

Specificity<-rval$detail[4,2:4]
# Specificity

ci_auc_result<-ci_auc_result[c(2,1,3)]
all_result<-rbind(ci_auc_result,Sensitivity,Specificity)
row.names(all_result)<-c("AUC","Sensitivity","Specificity")
# all_result


all_result$CI<-paste(round(all_result$est,4)," [",round(all_result$lower,4),"-",round(all_result$upper,4),"]")
# all_result


final_CI<-as.data.frame(rbind(all_result$CI))
# final_CI
names(final_CI)<-c("AUC","Sensitivity","Specificity")
# final_CI
news_test_ci<-final_CI
news_test_ci
ci0 <- rbind(news_test_ci)


ci0$cutoff<-cutoff


delta_sofa_cut_off__test_result<-ci0
delta_sofa_cut_off__test_result

delta_sofa_cut_off__test_result$accuracy <- result_delta_sofa_test$acc
delta_sofa_cut_off__test_result




cutoff<-4


tr_con<-confusionMatrix(factor(ifelse(MEWS_test>cutoff,"Y","N"),levels=c('Y','N')),testing$si1m)$table

library(epiR)
rval <- epi.tests(tr_con, conf.level = 0.95)

# train MEWS
AUC_result<-roc(testing$si1m, MEWS_test,levels=c("Y","N"),CI=TRUE)
as.numeric(AUC_result$auc)
ci_auc_result<-ci(AUC_result)
ci_auc_result<-as.numeric(ci_auc_result)
ci_auc_result
Sensitivity<-rval$detail[3,2:4]
# Sensitivity

Specificity<-rval$detail[4,2:4]
# Specificity

ci_auc_result<-ci_auc_result[c(2,1,3)]
all_result<-rbind(ci_auc_result,Sensitivity,Specificity)
row.names(all_result)<-c("AUC","Sensitivity","Specificity")
# all_result


all_result$CI<-paste(round(all_result$est,4)," [",round(all_result$lower,4),"-",round(all_result$upper,4),"]")
# all_result


final_CI<-as.data.frame(rbind(all_result$CI))
# final_CI
names(final_CI)<-c("AUC","Sensitivity","Specificity")
# final_CI
news_test_ci<-final_CI
news_test_ci
ci0 <- rbind(news_test_ci)


ci0$cutoff<-cutoff


mews_cut_off_test_result<-ci0

mews_cut_off_test_result$accuracy <- result_mews_test$acc




cutoff<-6

tr_con<-confusionMatrix(factor(ifelse(NEWS_test>cutoff,"Y","N"),levels=c('Y','N')),testing$si1m)$table

library(epiR)
rval <- epi.tests(tr_con, conf.level = 0.95)

# train MEWS
AUC_result<-roc(testing$si1m, NEWS_test,levels=c("Y","N"),CI=TRUE)
as.numeric(AUC_result$auc)
ci_auc_result<-ci(AUC_result)
ci_auc_result<-as.numeric(ci_auc_result)
ci_auc_result
Sensitivity<-rval$detail[3,2:4]
# Sensitivity

Specificity<-rval$detail[4,2:4]
# Specificity

ci_auc_result<-ci_auc_result[c(2,1,3)]
all_result<-rbind(ci_auc_result,Sensitivity,Specificity)
row.names(all_result)<-c("AUC","Sensitivity","Specificity")
# all_result


all_result$CI<-paste(round(all_result$est,4)," [",round(all_result$lower,4),"-",round(all_result$upper,4),"]")
# all_result


final_CI<-as.data.frame(rbind(all_result$CI))
# final_CI
names(final_CI)<-c("AUC","Sensitivity","Specificity")
# final_CI
news_test_ci<-final_CI
news_test_ci
ci0 <- rbind(news_test_ci)


ci0$cutoff<-cutoff


news_cut_off_test_result<-ci0
news_cut_off_test_result$accuracy <- result_news_test$acc



# ----------------------------------------------------------------
cat("Training data score 95% CI:\n")
delta_sofa_cut_off_result_train
mews_cut_off_train_result
news_cut_off_train_result


cat("Testing data score 95% CI:\n")
delta_sofa_cut_off__test_result
mews_cut_off_test_result
news_cut_off_test_result


# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# Part 6 ROC plot
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------



mydata=read.csv("Siover1_rf_top10_cgmh_data_with_score.csv",header=TRUE)



availvar<-c("aFdP",
            "fFdP",
            "ARerr",
            "DFA.Alpha.1",
            "Mean.rate",
            "Poincar..SD2",
            "shannEn",
            "LF.HF.ratio.LombScargle","gcount","sgridTAU","sofam","MEWSm","NEWSm")



fitControl=trainControl(method="repeatedcv",
                        number=10,
                        repeats=10,
                        classProbs=TRUE,
                        summaryFunction=twoClassSummary,
                        search="random")




mydata_new<-cbind(si1m=mydata$si1m,mydata[,availvar])
mydata_new$si1m<-factor(mydata_new$si1m,levels = c("Y","N"))
##
dim(mydata_new)
set.seed(123)
mydata1=mydata_new[which(mydata_new$si1m=="Y"),]
mydata2=mydata_new[which(mydata_new$si1m=="N"),]
index1=sample(dim(mydata1)[1],nrow(mydata1)*0.25)
index2=sample(dim(mydata2)[1],nrow(mydata2)*0.25)
testing=rbind(mydata1[index1,],mydata2[index2,])
training=rbind(mydata1[-index1,],mydata2[-index2,])



training<-training[,c("si1m",availvar[1:10])]



deltaSOFA<-testing$sofam
MEWS<-testing$MEWSm
NEWS<-testing$NEWSm




testing<-testing[,c("si1m",availvar[1:10])]
#----------------------------------------------------------


predictions_train=predict(xgb_train,newdata=training)
predictions_test=predict(xgb_train,newdata=testing)
xgbtra<-confusionMatrix(predict(xgb_train,training),training$si1m)$overall["Accuracy"]
xgbtea<-confusionMatrix(predict(xgb_train,testing),testing$si1m)$overall["Accuracy"]
train_results=predict(xgb_train,training,type="prob")
test_results=predict(xgb_train,testing,type="prob")
train_results$obs=training$si1m
train_results$pred=predictions_train
test_results$obs=testing$si1m
test_results$pred=predictions_test
xgb_trauc<-roc(training$si1m,train_results[,"Y"],levels=c("Y","N"))$auc
xgb_teauc<-roc(testing$si1m,test_results[,"Y"],levels=c("Y","N"))$auc
xgbresult<-c(xgbtra,xgbtea,xgb_trauc,xgb_teauc)

a<-rbind(xgbresult)
colnames(a)<-c("Training Accuracy","Testing Accuracy","Training AUC","Testing AUC")
print(a)








mydata0<-read.csv('Siover1_rf_top10_mimic_data.csv',header = T)


new_test<-mydata0[,names(training)]
new_test$si1m<-factor(ifelse(new_test$si1m=="1","Y","N"),levels = c("Y","N"))
table(new_test$si1m)
predictions_test_new=predict(xgb_train,newdata=new_test)
xgbtea_new<-confusionMatrix(predict(xgb_train,new_test),new_test$si1m)$overall["Accuracy"]
xgbtea_new

test_results_new=predict(xgb_train,new_test,type="prob")


test_results_new$obs=new_test$si1m
test_results_new$pred=predictions_test_new
xgb_teauc_new<-roc(new_test$si1m,test_results_new[,"Y"],levels=c("Y","N"))$auc
xgb_teauc_new



xgbresult1<-c(xgbtra,xgbtea,xgbtea_new,xgb_trauc,xgb_teauc,xgb_teauc_new)

a1<-rbind(xgbresult1)
colnames(a1)<-c("Training Accuracy","Testing Accuracy","new Testing Accuracy","Training AUC","Testing AUC","new Testing AUC")
print(a1)




xgboost_newTest_original<-as.data.frame(cbind(class = testing$si1m,xgboost_prob  =test_results[,"Y"] ))
xgboost_newTest_original$class<-factor(ifelse(xgboost_newTest_original$class=="1","1","0"),levels = c("1",'0'))

rocobj1 <- plot.roc(
  roc(xgboost_newTest_original$class,xgboost_newTest_original$xgboost_prob),
  main="", percent=TRUE, col="red",lty = 1)
rocobj2 <- lines.roc(roc(testing$si1m,NEWS), percent=TRUE, col="green",lty = 2)
rocobj3 <- lines.roc(roc(testing$si1m,testing$Mean.rate), percent=TRUE, col="blue",lty = 3)
rocobj4 <- lines.roc(roc(testing$si1m,testing$sgridTAU),  percent=TRUE, col="darkgoldenrod",lty = 4)
rocobj5 <- lines.roc(roc(testing$si1m,testing$aFdP), percent=TRUE, col="cyan",lty = 5)
rocobj6 <- lines.roc(roc(testing$si1m,MEWS), percent=TRUE, col="firebrick",lty = 6)
rocobj7 <- lines.roc(roc(testing$si1m,deltaSOFA), percent=TRUE, col="darkorchid",lty = 7)


mod_roc<-round( roc(xgboost_newTest_original$class,xgboost_newTest_original$xgboost_prob)$auc,3)
var1_roc<-round(roc(testing$si1m,testing$sgridTAU)$auc,3)
var2_roc<-round(roc(testing$si1m,testing$Mean.rate)$auc,3)
var3_roc<-round(roc(testing$si1m,testing$aFdP)$auc,3)
var4_roc<-round(roc(testing$si1m,deltaSOFA)$auc,3)
var5_roc<-round(roc(testing$si1m,MEWS)$auc,3)
var6_roc<-round(roc(testing$si1m,NEWS)$auc,3)


le0<-c("XGBoost","NEWS","Mean rate","sgridTAU","aFdP","MEWS","delta SOFA")
roc0<-c(mod_roc,var1_roc,var2_roc,var3_roc,var4_roc,var5_roc,var6_roc)


roc_sort<-sort(roc0,decreasing = T)

legend("bottomright", legend=paste(le0," (AUC = ",roc_sort,")" ,sep=''),
       col=c("red","green","blue","darkgoldenrod","cyan","firebrick","darkorchid"), lwd=1.2,cex = 1,lty = 1:7)

