library(ggplot2)

#dat = read.csv("/Users/jeff/data/features/results.csv", header = TRUE)
dat = read.csv("/Users/jeff/projects/papers/2016/ieee-feature-eng/results.csv", header = TRUE)

theme_set(theme_grey(base_size = 20)) 

# Support Vector Machine - Regression
pdf("error_svr.pdf")
s <- subset(dat, model == 'GridSearchCV')[,c('experiment','error')]
s$error <- pmin(s$error, 0.05)
ggplot(data=s, aes(x=experiment, y=error)) +
  geom_bar(stat="identity") 
dev.off()

# Random forest
s <- subset(dat, model == 'RandomForestRegressor')[,c('experiment','error')]
pdf("error_rf.pdf")
s$error <- pmin(s$error, 0.05)
ggplot(data=s, aes(x=experiment, y=error)) +
  geom_bar(stat="identity") 
dev.off()

# Deep ANN - Neural Network
pdf("error_dann.pdf")
s <- subset(dat, model == 'NeuralNet')[,c('experiment','error')]
s$error <- pmin(s$error, 0.05)
ggplot(data=s, aes(x=experiment, y=error)) +
  geom_bar(stat="identity") 
dev.off()

# Gradient Boosted Machine (GBM)
pdf("error_gbm.pdf")
s <- subset(dat, model == 'GradientBoostingRegressor')[,c('experiment','error')]
s$error <- pmin(s$error, 0.05)
ggplot(data=s, aes(x=experiment, y=error)) +
  geom_bar(stat="identity") 
dev.off()

