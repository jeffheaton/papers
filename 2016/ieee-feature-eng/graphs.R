# This R script was used to create graphs for following paper/conference:
#
# Heaton, J. (2016, April). An Empirical Analysis of Feature Engineering for Predictive Modeling.
# In SoutheastCon 2016 (pp. 1-6). IEEE.
#
# http://www.jeffheaton.com
library(ggplot2)
library(Cairo)

#dat = read.csv("/Users/jeff/data/features/results.csv", header = TRUE)
dat = read.csv("/Users/jeff/projects/papers/2016/ieee-feature-eng/results.csv", header = TRUE)

theme_set(theme_grey(base_size = 15)) 

# Support Vector Machine - Regression
CairoPDF("error_svr.pdf")
s <- subset(dat, model == 'GridSearchCV')[,c('experiment','error')]
s$error <- pmin(s$error, 0.05)
ggplot(data=s, aes(x=experiment, y=error)) +
  geom_bar(stat="identity") 
dev.off()

# Random forest
s <- subset(dat, model == 'RandomForestRegressor')[,c('experiment','error')]
CairoPDF("error_rf.pdf")
s$error <- pmin(s$error, 0.05)
ggplot(data=s, aes(x=experiment, y=error)) +
  geom_bar(stat="identity") 
dev.off()

# Deep ANN - Neural Network
CairoPDF("error_dann.pdf")
s <- subset(dat, model == 'NeuralNet')[,c('experiment','error')]
s$error <- pmin(s$error, 0.05)
ggplot(data=s, aes(x=experiment, y=error)) +
  geom_bar(stat="identity") 
dev.off()

# Gradient Boosted Machine (GBM)
CairoPDF("error_gbm.pdf")
s <- subset(dat, model == 'GradientBoostingRegressor')[,c('experiment','error')]
s$error <- pmin(s$error, 0.05)
ggplot(data=s, aes(x=experiment, y=error)) +
  geom_bar(stat="identity") 
dev.off()

0.00001
(2**34*8)/1024/1024/1024
16*8
32/8
