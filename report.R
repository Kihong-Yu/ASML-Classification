install.packages('precrec')
install.packages('skimr')
install.packages('nnet')
install.packages('tidyverse')
install.packages('ggplot2')
install.packages('DataExplorer')
install.packages('maptree')
install.packages('rpart')
install.packages('rsample')
install.packages(c('dplyr',"caret","boot"))
install.packages('mlr3verse')
install.packages('GGally')
library("skimr")
library("nnet")
library("tidyverse")
library("ggplot2")
library("GGally")
library('DataExplorer')
library('maptree')
library("rpart")
library('rsample')
library("dplyr")
library("caret")
library("boot")
library(maptree)
library(rpart)

bank_loan_data <- readr::read_csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")

ggpairs(bank_loan_data %>% select(Income,Family,Education,CD.Account,CreditCard),
        aes(color = ''))

DataExplorer::plot_histogram(bank_loan_data, ncol = 3)
DataExplorer::plot_boxplot(bank_loan_data, by = "Personal.Loan", ncol = 3)

fit3.cart <- rpart(as.factor(Personal.Loan) ~   
                     Income+Family+Education+CD.Account+Online+CreditCard,data=bank_loan_data)

plotcp(fit3.cart)
draw.tree(fit3.cart)


library("mlr3verse")
set.seed(100) # set seed for reproducibility

bank_loan_data$Personal.Loan <-as.factor(bank_loan_data$Personal.Loan)
predit_task <- TaskClassif$new(id = "isLoan",
                               backend = bank_loan_data, 
                               target = "Personal.Loan",
                               positive = "1")
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(predit_task)
train_set = sample(predit_task$row_ids, 0.8 * predit_task$nrow)
test_set = setdiff(predit_task$row_ids, train_set)
lrn_lr <- lrn("classif.log_reg", predict_type = "prob")
lrn_lr$train(predit_task, row_ids = train_set)
res_lr <- resample(predit_task, lrn_lr, cv5, store_models = TRUE)
res_lr$aggregate()

loan_prid <- lrn_lr$predict(predit_task, row_ids = test_set)
loan_prid$confusion

#roc and prc for log_reg
autoplot(res_lr,type="roc")
autoplot(res_lr,type="prc")


#cart model

lrn_cart <- lrn("classif.rpart", predict_type = "prob")
lrn_cart$train(predit_task, row_ids = train_set)
res_cart <- resample(predit_task, lrn_cart, cv5, store_models = TRUE)
res_cart$aggregate()

loan_prid2 <- lrn_cart$predict(predit_task, row_ids = test_set)
loan_prid2$confusion
autoplot(res_cart,type="roc")
autoplot(res_cart,type="prc")

#random forest model
lrn_randForest <- lrn("classif.ranger", importance = "permutation", predict_type = "prob")
lrn_randForest$train(predit_task, row_ids = train_set)
fit4.rf <- lrn_randForest$predict(predit_task, row_ids = test_set)
fit4.rf$confusion
res_rf <- resample(predit_task, lrn_randForest, cv5, store_models = TRUE)
res_rf$aggregate()
autoplot(res_rf,type="roc")
autoplot(res_rf,type="prc")


#benchmark
learners = lrns(c("classif.rpart", "classif.ranger","classif.log_reg"), predict_type = "prob")

bm_design = benchmark_grid(
  tasks = predit_task,
  learners = learners,
  resamplings = rsmp("cv", folds = 10)
)

bmr = benchmark(bm_design)

measures = msrs(c("classif.ce", "classif.acc","classif.auc","classif.tpr","classif.fpr","classif.fnr"))
performances = bmr$aggregate(measures)
performances[, c("learner_id", "classif.ce", "classif.acc","classif.auc","classif.tpr","classif.fpr","classif.fnr")]

#importance
importance = as.data.table(lrn_randForest$importance(), keep.rownames = TRUE)
# edit col name
colnames(importance) = c("Feature", "Importance")

ggplot(data=importance,
       aes(x = reorder(Feature, Importance), y = Importance)) + 
  geom_col() + coord_flip() + xlab("")


#Hyperparameter adjustment default: num.trees =500, mtry=floor(sqrt())
rf_med = lrn("classif.ranger", id = "med", predict_type = "prob")
rf_low = lrn("classif.ranger", id = "low", predict_type = "prob",
             num.trees = 5, mtry = 2)
rf_high = lrn("classif.ranger", id = "high", predict_type = "prob",
              num.trees = 1000, mtry = 10)

learners2 = list(rf_low, rf_med, rf_high)
bm_design2 = benchmark_grid(
  tasks = predit_task,
  learners = learners2,
  resamplings = rsmp("cv", folds = 10)
)
bmr2 = benchmark(bm_design2)

performances2 = bmr2$aggregate(measures)
performances2[, .(learner_id, classif.ce, classif.auc)]

autoplot(bmr2)

