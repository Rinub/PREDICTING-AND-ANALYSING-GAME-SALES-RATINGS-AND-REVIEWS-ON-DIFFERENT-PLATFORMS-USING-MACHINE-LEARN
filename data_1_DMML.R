install.packages('ROSE')
install.packages('dummies')
install.packages('caret')
install.packages('dplyr')
install.packages('caTools')
install.packages('rsample')
install.packages("Amelia")
install.packages("heatmaply")
install.packages('reshape')
install.packages("VIM")  
install.packages("visdat")  
install.packages("ggcorrplot")
install.packages("mice")
install.packages("missForest")
install.packages("psych")
install.packages('randomForest')
install.packages("caTools") 

###############################################
# IMPORTING LIBERARYS
            # Install VIM package
library(VIM)
library(knitr)
library(tidyverse)
library(ggplot2)
library(mice)
library(lattice)
library(reshape2)
library(DataExplorer)
library(ROSE)
library(dummies)
library(caret)
library(rsample)
library(Amelia)
library(ggplot2)
library(ggcorrplot)

################################################

rawdata1 <- read.csv("D:\\PROJECT_DATA\\Video_Games_Sales.csv",header=T,stringsAsFactors = T)
view(rawdata1)

# Checking the first 5 rows of the dataset:
head(rawdata1)

summary(rawdata1)
################################################

#Visual representation of missing data
missmap(rawdata1, main = "MISSING VS OBSERVED")


library(ggplot2)
library(reshape)



sum(is.na(rawdata1)) # Two missings in our vector


aggr(rawdata1)  


library(visdat)
vis_miss(rawdata1)


#############################################################
# check for correlation between the variables





###############################################################
# dropping unwanted rows and loading into new dataframe

df = subset(rawdata1, select = c(EU_Sales, NA_Sales, Global_Sales, Critic_Score) )

################################################################
################################################################
# imputing missing values


library(mice)
#=============MICE package functions
#Other Methods midastouch(weighted predictive mean matching), sample, cart
#rf, mean, norm(Bayesian linear regression), logreg, lda

# imputing critic score
imputed_Data_1 <- mice(df, m=1, maxit = 50, method = 'pmm', seed = 500)
summary(imputed_Data)
# m is 5 imputed data sets; maxit: iterations; 5 datasets are created any one can be selected
#get complete data ( 2nd out of 5)
df_1 <- complete(imputed_Data_1)


vis_miss(df_1)

df_1 = subset(df_1, select = c(Critic_Score) )
head(df_1)

vis_miss(df_1)



###############################################################################
# imputing user score

df = subset(rawdata1, select = c(EU_Sales, NA_Sales, Global_Sales, User_Score) )
imputed_Data_2 <- mice(df, m=1, maxit = 50, method = 'pmm', seed = 500)
df_2 <- complete(imputed_Data_2)
vis_miss(df_2)
df_2 = subset(df_2, select = c(User_Score) )
head(df_2)
vis_miss(df_2)

###############################################################################

imp_1 <- cbind(df_1, df_2) 
head(imp_1)

##############################################################################

df = subset(rawdata1, select = c(EU_Sales, NA_Sales, Global_Sales, Critic_Count, User_Count ) )
imp_1 <- cbind(df, imp_1) 
vis_miss(mis_for_imp)

#########################################################################


#=============Miss Forest functions
#This package usage random forest as the missing value evaluation algorithm
#impute missing values, using all parameters as default values
library(missForest)
imp_2 <- missForest(imp_1)

#check imputation error
#NRMSE is normalized mean squared error (for continous vars)
#PFC is proportion of falsely classified (categorical vars error)


#The treated data is extracted using
imp_2<-imp_2$ximp

vis_miss(imp_2)

head(imp_2)


final_half_1 = subset(imp_2, select = c(Global_Sales, Critic_Score, Critic_Count, User_Count, User_Score ) )

final_half_2 = subset(rawdata1, select = c(Platform, Year_of_Release, Genre, Publisher, Rating ) )


head(rawdata1)

##########################################################################################
# combining dataframes column wise

final_data <- cbind(final_half_1, final_half_2)                       # cbind vector to data frame

vis_miss(final_data)

head(final_data)

############################################################################################

# NA values cant be detected since the some cells are empty, so fill the empy cells with NA

final_data[final_data == ""] <- NA         

imp_rating = subset(final_data, select = c(Rating, Critic_Score , Global_Sales, Critic_Count, User_Count, Platform ) )
vis_miss(imp_rating)

#############################################################################################

library(missForest)
imp_rating <- missForest(imp_rating)

#check imputation error
#NRMSE is normalized mean squared error (for continous vars)
#PFC is proportion of falsely classified (categorical vars error)


#The treated data is extracted using
final_imp_rat<-imp_rating$ximp


final_data = subset(final_data, select = -c(Rating) )

rating = subset(final_imp_rat, select = c(Rating) )

final_data <- cbind(final_data, rating) 
vis_miss(final_data)

final_data_imputed <- na.omit(final_data)
vis_miss(final_data_imputed)

##########################################################################################
##########################################################################################

# checking for outliers in final_data_imputed

###########################################################################################


library(ggplot2)
library(ggcorrplot)
library(psych)
psych::describe(final_data_imputed)

###########################################################################################


outlierTreament<-function(x){
  qnt <- quantile(x, probs=c(.25, .75), na.rm = T)
  caps <- quantile(x, probs=c(.05, .95), na.rm = T)
  H <- 1.5 * IQR(x, na.rm = T)
  x[x < (qnt[1] - H)] <- caps[1]
  x[x > (qnt[2] + H)] <- caps[2]
  return(x)}


numeric_cols<-names(final_data_imputed)[sapply(final_data_imputed, is.numeric)]
numeric_data<-final_data_imputed[,names(final_data_imputed)%in%numeric_cols]


impute_df_IQR<-as.data.frame(sapply(numeric_data,outlierTreament))

###################################################################################

head(impute_df_IQR)

ggplot(impute_df_IQR) +
  aes(x = "", y = Global_Sales) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()

ggplot(impute_df_IQR) +
  aes(x = "", y = Critic_Score) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()

ggplot(impute_df_IQR) +
  aes(x = "", y = Critic_Count) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()

ggplot(impute_df_IQR) +
  aes(x = "", y = User_Count) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()

ggplot(impute_df_IQR) +
  aes(x = "", y = User_Score) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()


head(final_data_imputed)

final_data_pre = subset(final_data_imputed, select = c(Platform, Year_of_Release, Genre, Publisher, Rating) )




final_data_1 <- cbind(final_data_pre, impute_df_IQR)  

##################################################################################################################

# global_sales
# critical_count
# user_count

boxplot.stats(final_data_1$Global_Sales )$out
outliers <-boxplot.stats(final_data_1$Global_Sales )$out
final_data_1 <- final_data_1[-which(final_data_1$Global_Sales %in% outliers),]

boxplot.stats(final_data_1$Global_Sales )$out
outliers <-boxplot.stats(final_data_1$Global_Sales )$out
final_data_1 <- final_data_1[-which(final_data_1$Global_Sales %in% outliers),]

boxplot.stats(final_data_1$Global_Sales )$out
outliers <-boxplot.stats(final_data_1$Global_Sales )$out
final_data_1 <- final_data_1[-which(final_data_1$Global_Sales %in% outliers),]

boxplot.stats(final_data_1$Global_Sales )$out
outliers <-boxplot.stats(final_data_1$Global_Sales )$out
final_data_1 <- final_data_1[-which(final_data_1$Global_Sales %in% outliers),]

boxplot.stats(final_data_1$Global_Sales )$out
outliers <-boxplot.stats(final_data_1$Global_Sales )$out
final_data_1 <- final_data_1[-which(final_data_1$Global_Sales %in% outliers),]

boxplot.stats(final_data_1$Global_Sales )$out
outliers <-boxplot.stats(final_data_1$Global_Sales )$out
final_data_1 <- final_data_1[-which(final_data_1$Global_Sales %in% outliers),]

boxplot.stats(final_data_1$Global_Sales )$out



outliers <-boxplot.stats(final_data$Critic_Score )$out
final_data <- final_data[-which(final_data$Critic_Score %in% outliers),]
boxplot.stats(final_data$Critic_Score)$out

head(final_data_1)



final_data = subset(final_data_1, select = -c(Critic_Count, User_Count) )


###################################################################################################################

ggplot(final_data) +
  aes(x = "", y = Global_Sales) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()

ggplot(final_data) +
  aes(x = "", y = Critic_Score) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()


ggplot(final_data) +
  aes(x = "", y = User_Score) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()

#########################################################################################################

# write.csv(final_data,"C:\\Users\\Asus TUF\\Documents\\R\\DMML project\\FINAL_DATA_1.csv")


vis_miss(final_data)
dim(final_data)
#####################################################################################




#####################################################################################################


sapply(final_data, class)
summary(final_data)



data_fit <- transform(
  final_data,
  Platform=as.numeric(Platform),
  Year_of_Release=as.numeric(Year_of_Release),
  Publisher =as.numeric (Publisher),
  Critic_Score =as.numeric (Critic_Score),
  User_Score =as.numeric (User_Score),
  Global_Sales =as.numeric (Global_Sales),
  Rating =as.factor(Rating)
)

###########################################################################################################

# FACTORIAL ANALYSIS

###########################################################################################################



data_class <- droplevels(data_fit)
numeric_cols<-names(data_class)[sapply(data_class, is.numeric)]
numeric_data<-data_class[,names(data_class)%in%numeric_cols]

library(heatmaply)

heatmaply_cor(
  cor(numeric_data),
  xlab = "Features", 
  ylab = "Features",
  k_col = 2, 
  k_row = 2
)


factor_analysis <- factanal(numeric_data,factor= 3)
factor_analysis



###########################################################################################################

# fitting machine learning algorithm 


###########################################################################################################


# MODEL 1 - RANDOM FOREST REGRESSION MODEL - PREDICTING gLOBAL SALES
#####################################################################

#predicting Global sales using random forest regression

library(randomForest)

library(caTools)

# checcking null values
data_fit[ data_fit == "?"] <- NA
colSums(is.na(data_fit))
head(data_fit)
data_fit_1 = subset(data_fit, select = c(Platform, Year_of_Release,  Genre, Rating, User_Score, Critic_Score, Global_Sales))
head(data_fit_1)
# splitting test and train dataset
sample = sample.split(data_fit_1$Global_Sales,SplitRatio = 0.75)
train = subset(data_fit_1, sample == TRUE)
test  = subset(data_fit_1, sample == FALSE)

dim(train)
dim(test)

model_1 <- randomForest(
  Global_Sales ~ .,
  data=train
)

predict_1 = predict(model_1, newdata=test[-7])

predict_1



#Model performance metrics
########################## 

data.frame(R2 = R2(predict_1, test$Global_Sales), 
           RMSE = RMSE(predict_1, test$Global_Sales), 
           MAE = MAE(predict_1,test$Global_Sales))

#############################################################################################



###############################################################################################

# FACTORIAL ANALYSIS FOR VARIABLE SELECTION TO FIT IN ALGORITHM

################################################################################################

data_class <- droplevels(data_fit)
numeric_cols<-names(data_class)[sapply(data_class, is.numeric)]
numeric_data<-data_class[,names(data_class)%in%numeric_cols]

library(heatmaply)

heatmaply_cor(
  cor(numeric_data),
  xlab = "Features", 
  ylab = "Features",
  k_col = 2, 
  k_row = 2
)


factor_analysis <- factanal(numeric_data,factor= 3)
factor_analysis




###############################################################################################


# MODEL 2 - RANDOM FOREST CLASSIFICATION MODEL - PREDICTING GAME RATING
########################################################################



data_fit[ data_fit == "?"] <- NA
colSums(is.na(data_fit))

# splitting test and train dataset
sample = sample.split(data_class$Rating, SplitRatio = .75)
train_1 = subset(data_class, sample == TRUE)
test_1  = subset(data_class, sample == FALSE)

dim(train_1)
dim(test_1)


model_2 <- randomForest(
  Rating ~ .,
  data= train_1)

predict_2 = predict(model_2, newdata=test_1[-5])


#Model - 2 performance metrics
############################## 

cm = table(test_1[,5], predict_2)
accuracy_1 <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy_1(cm)
cm

confusionMatrix(predict_2, test_1[,5],
                positive = "pos")

#####################################################################################################

# MODEL 3 - Support Vector Machines CLASSIFICATION MODEL - PREDICTING GAME RATING
#################################################################################


library(e1071)

##the normalization function is created
nor <-function(x) { (x -min(x))/(max(x)-min(x))   }

##Run nomalization on first 4 coulumns of dataset because they are the predictors
data_norm_svm <- as.data.frame(lapply(data_class[,c(1,2,4,6,7,8)], nor))
rating_SVM = subset(data_class, select = c(Rating))

SVM_data <- cbind(data_norm_svm, rating_SVM) 

# splitting test and train dataset
sample = sample.split(SVM_data$Rating, SplitRatio = .75)
train_SVM = subset(SVM_data, sample == TRUE)
test_SVM  = subset(SVM_data, sample == FALSE)


classifier = svm(formula = Rating ~ .,
                 data = train_SVM,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
predict_SVM = predict(classifier, newdata = test_SVM[-7])

confusionMatrix(predict_SVM, test_SVM[-7],
                positive = "pos")


#Model - 3 performance metrics
##############################

cm_SVM = table(test_SVM[,7], predict_SVM)
accuracy_SVM <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy_SVM(cm_SVM)
cm_SVM

confusionMatrix(predict_SVM, test_SVM[,7],
                positive = "pos")
######################################################################################################

#IMPORTING REQUIRED PACKAGES

###################################################################################################


#Loading required packages
#install.packages('tidyverse')
library(tidyverse)
#install.packages('ggplot2')
library(ggplot2)
#install.packages('caret')
library(caret)
#install.packages('caretEnsemble')
library(caretEnsemble)
#install.packages('psych')
library(psych)
#install.packages('Amelia')
library(Amelia)
#install.packages('mice')
library(mice)
#install.packages('GGally')
library(GGally)
#install.packages('rpart')
library(rpart)
#install.packages('randomForest')
library(randomForest)


######################################################################################

# VISUAL ANALYSIS OF DATA

######################################################################################

#visualize the missing data
missmap(data_class)


#Data Visualization
#Visual 1
ggplot(data_class, aes(Global_Sales, colour = Rating)) +
  geom_freqpoly(binwidth = 1) + labs(title="Global_Sales Distribution by Rating")


#visual 2
c <- ggplot(data_class, aes(x=Global_Sales, fill=Rating, color=Rating)) +
  geom_histogram(binwidth = 1) + labs(title="Global_Sales Distribution by Rating")
c + theme_bw()


#visual 3
P <- ggplot(data_class, aes(x=Critic_Score, fill=Rating, color=Rating)) +
  geom_histogram(binwidth = 1) + labs(title="Critic_Score Distribution by Rating")
P + theme_bw()


#visual 3
q <- ggplot(data_class, aes(x=User_Score, fill=Rating, color=Rating)) +
  geom_histogram(binwidth = 1) + labs(title="User_Score Distribution by Rating")
q + theme_bw()


#visual 5
ggpairs(data_class)

#visual 4
ggplot(data_class, aes(Platform, colour = Rating)) +
  geom_freqpoly(binwidth = 1) + labs(title="Platform Distribution by Rating")

head(data_class)


################################################################################


# MODEL 4 -  KNN CLASSIFICATION MODEL - PREDICTING GAME RATING
#################################################################################



#Building a model
#split data into training and test data sets

# splitting test and train dataset
data_fit_1 <- droplevels(data_fit)

##the normalization function is created
nor <-function(x) { (x -min(x))/(max(x)-min(x))   }

##Run nomalization on first 4 coulumns of dataset because they are the predictors
data_norm <- as.data.frame(lapply(data_fit_1[,c(1,2,4,6,7,8)], nor))

summary(data_norm)

##Generate a random number that is 90% of the total number of rows in dataset.
ran <- sample(1:nrow(data_fit_1), 0.7 * nrow(data_fit_1)) 
##extract training set
data_train <- data_norm[ran,] 
##extract testing set
data_test <- data_norm[-ran,] 


##extract 5th column of train dataset because it will be used as 'cl' argument in knn function.
target_train <- data_fit_1[ran,5]

##extract testing set
target_test <- data_fit_1[-ran,5] 




##load the package class
library(class)

##run knn function
predict_3 <- knn(data_train,data_test,cl=target_train,k=39)


# Model - 4 performance metrics
############################### 


##create confusion matrix
cm_2 <- table(predict_3,target_test)
cm_2

accuracy_2 <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy_2(cm_2)


confusionMatrix(predict_3, target_test,
                positive = "pos")
##################################################################################################################

