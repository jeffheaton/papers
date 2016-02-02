library(arules)

tr <- read.transactions("C:\\Users\\Jeff\\data\\rules\\def.txt", format = "basket", sep=" ")
itemFrequencyPlot(tr, support = 0.1, cex.names=0.8)
image(tr,ylab='Applicants',xlab='Rx')
#rules <- apriori(tr, parameter = list(support = 0.01, confidence = 0.6))
rules <- apriori(tr, parameter = list(support = 0.01, confidence = 0.6))
rules.sorted <- sort(rules, by="lift")
inspect(rules.sorted)
top_rules <- rules.sorted[0:20]
inspect(top_rules)
summary(rules)