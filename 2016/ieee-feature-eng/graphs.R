library(ggplot2)

dat = read.csv("/Users/jeff/data/features/results.csv", header = TRUE)
s <- subset(dat, model == 'GridSearchCV')[,c('experiment','error')]
s <- subset(dat, model == 'RandomForestRegressor')[,c('experiment','error')]
s <- subset(dat, model == 'NeuralNet')[,c('experiment','error')]
s <- subset(dat, model == 'GradientBoostingRegressor')[,c('experiment','error')]

#dat[dat$experiment == 'poly']
#dat[dat$model == 'NeuralNetwork', c('experiment','error') ]

s$error <- pmin(s$error, 0.05)
ggplot(data=s, aes(x=experiment, y=error)) +
  geom_bar(stat="identity") 

#
s <- subset(dat, model == 'GridSearchCV')[,c('experiment','error')]
s$error <- pmin(s$error, 0.05)
ggplot(data=s, aes(x=experiment, y=error)) +
  geom_bar(stat="identity") 


#
s <- subset(dat, model == 'RandomForestRegressor')[,c('experiment','error')]
s$error <- pmin(s$error, 0.05)
ggplot(data=s, aes(x=experiment, y=error)) +
  geom_bar(stat="identity") 

#
s <- subset(dat, model == 'NeuralNet')[,c('experiment','error')]
s$error <- pmin(s$error, 0.05)
ggplot(data=s, aes(x=experiment, y=error)) +
  geom_bar(stat="identity") 

#
s <- subset(dat, model == 'GradientBoostingRegressor')[,c('experiment','error')]
s$error <- pmin(s$error, 0.05)
ggplot(data=s, aes(x=experiment, y=error)) +
  geom_bar(stat="identity") 





+  scale_y_log10()

s
1e5

log(59.37163)
0.58961 - 0.14544
0.98726/0.66203

x <- 6.30651

x1 <- 6.30651
x2 <- 2.23126
x3 <- 6.95826
x4 <- 9.88415

(x1-x2)/(x3-x4)

x <- 6.30651
1/(5*x+8*x^2)
