# importing the needed model to perform data mining and machine learning

library(stringr)
library(foreign)
library(caret)
library(pROC)
library(psych)

###############################################################################################################################
# Importing the dataset into r

df_3.df <- read.csv("C:\\Users\\Asus TUF\\Documents\\R\\DMML project\\appstore_strategy_games.csv", header = T, na.strings = c(""), stringsAsFactors = T)
###############################################################################################################################

###############################################################################################################################
# performing cleaning, pre-processing imputation and transformation


summary(df_3.df)
str(df_3.df)

# dropping the columns that is not needed

df_3.df <- df_3.df[ , !names(df_3.df) %in% c("Primary.Genre","URL","ID","Name","Subtitle","Icon.URL","Description","Developer","Genres","Original.Release.Date","Current.Version.Release.Date")]

# checking the null values and adding it column wise 

#visualize the missing data
library(Amelia)
missmap(df_3.df)


sapply(df_3.df, function(x) sum(is.na(x)))

# dropping the Null values from all the columns of the dataframe
df_3.df <- df_3.df[!is.na(df_3.df$Size),]
df_3.df <- df_3.df[!is.na(df_3.df$Price),]
df_3.df <- df_3.df[!is.na(df_3.df$Languages),]

#visualize the missing data
missmap(df_3.df)


########## TRANSFORMATION ##########
summary(df_3.df)
sapply(df_3.df, function(x) sum(is.na(x)))

# Transforming Average User Rating to either be good or bad for logistic regression analysis
df_3.df <- df_3.df[!is.na(df_3.df$Average.User.Rating),]
df_3.df$Average.User.Rating <- as.character(df_3.df$Average.User.Rating)

unique(df_3.df$Average.User.Rating)

df_3.df$Average.User.Rating[which(df_3.df$Average.User.Rating %in% c("0","0.5","1","1.5","2","2.5","3","3.5","4"))] <- "Bad"
df_3.df$Average.User.Rating[which(df_3.df$Average.User.Rating %in% c("4.5","5"))] <- "Good"
df_3.df$Average.User.Rating <- as.factor(df_3.df$Average.User.Rating)

# Converting all NA in In.App.Purchases to 0
df_3.df$In.app.Purchases[is.na(df_3.df$In.app.Purchases)] <- "0"

# converting column size to MB from Bytes for easier understanding
btomb <- function(x){
  varnum <- x/1048576
  varnum <- as.numeric(format(round(varnum, 2), nsmall = 2))
  return (varnum)
}

df_3.df$Size <- sapply(df_3.df$Size, function(x) btomb(x))

# converting languages to have only 3 levels
conlang <- function(x){
  if (str_detect(x, "^.*EN.*$", negate = TRUE)) {
    varlan = "No EN"
  }
  else if (x == "EN"){
    varlan = "Only EN"
  }
  else {
    varlan = "EN +"
  }
  return(varlan)
}

df_3.df$Languages <- as.factor(sapply(df_3.df$Languages, function(x) conlang(x)))

# getting only max in-app purchase amount from vector
maxinapp <- function(x){
  maxval <- unlist(str_split(x, ","))
  maxval <- trimws(maxval)
  nummaxval <- vector()
  for (i in (1:length(maxval))) {
    nummaxval[i] <- as.numeric(maxval[i])
  }
  return (max(nummaxval))
}

df_3.df$In.app.Purchases <- sapply(df_3.df$In.app.Purchases, function(x) maxinapp(x))

# reset the row count
rownames(df_3.df) <- NULL

# checking for multicollinearity
pairs.panels(df_3.df)

# creating dummy variables for multicategorical variables
dmy <- dummyVars(" ~ Age.Rating", data = df_3.df, fullRank = T)
age_rat <- data.frame(predict(dmy, newdata = df_3.df))

dmy <- dummyVars(" ~ Languages", data = df_3.df, fullRank = T)
lang <- data.frame(predict(dmy, newdata = df_3.df))

# removing categorized variables and adding dummy variables
df_3.df <- df_3.df[,c(-5,-6)]
df_3.df <- data.frame(df_3.df, age_rat, lang)

str(df_3.df)
##########################################################################################
# identifying using cooks distance and treating the outliers in df_3.df dataframe

blr_train_model <- glm(Average.User.Rating ~ ., data = df_3.df, family = binomial)

blr_train_model
summary(blr_train_model)
cd_train_model <- cooks.distance(blr_train_model)
cd_train_model[cd_train_model>1]
infl <- as.numeric(names(cd_train_model[cd_train_model>1]))

# (No outliers detected)


#######################################################################################

# factorial analysis for selecting the best model to fit into algorithm
head(df_3.df)




#######################################################################################
# creating testing and training datasets
set.seed(123)
train_sample <- sample(nrow(df_3.df), round(0.7*nrow(df_3.df)))

df_3.df_train <- df_3.df[train_sample, ]
df_3.df_test <- df_3.df[-train_sample, ]

# checking for even split of dataset
prop.table(table(df_3.df_train$Average.User.Rating))
prop.table(table(df_3.df_test$Average.User.Rating))

########## ANALYSIS USING BINOMIAL LOGISTIC REGRESSION ##########
# removing insignificant variables using backward testing from the model
blr_train_model <- glm(Average.User.Rating ~ ., data = df_3.df_train, family = "binomial")

blr_train_model
summary(blr_train_model)

blr_train_model <- glm(Average.User.Rating ~ User.Rating.Count+Price+In.app.Purchases+Size+Age.Rating.17.+Age.Rating.4.+Age.Rating.9., data = df_3.df_train, family = "binomial")

blr_train_model
summary(blr_train_model)

blr_train_model <- glm(Average.User.Rating ~ User.Rating.Count+In.app.Purchases+Size+Age.Rating.17.+Age.Rating.4.+Age.Rating.9., data = df_3.df_train, family = "binomial")

blr_train_model
summary(blr_train_model)

blr_test_predict <- predict(blr_train_model, df_3.df_test, type = "response")
summary(blr_test_predict)

blr_test_predict[blr_test_predict<=0.5] <- "Bad"
blr_test_predict[blr_test_predict != "Bad"] <- "Good"
blr_test_predict <- factor(blr_test_predict, levels=c("Bad","Good"))

# evaluating the model
confusionMatrix(blr_test_predict, df_3.df_test$Average.User.Rating, positive = "Good")

df_3.df_test$Average.User.Rating <- as.ordered(df_3.df_test$Average.User.Rating)
blr_test_predict <- as.ordered(blr_test_predict)
blr_ROC <- roc(df_3.df_test$Average.User.Rating, blr_test_predict)

blr_ROC

par(mfrow=c(1,1))
plot.roc(blr_ROC, print.auc = TRUE, col = "magenta")

# removing unused objects
rm("age_rat","dmy","lang","train_sample")