# Generate benchmark data for frequent itemset mining.
__author__ = 'jheaton'
import random

def sizeof_fmt(num):
    for x in ['','k','m','g']:
        if num < 1000.0:
            return "%3.1f%s" % (num, x)
        num /= 1000.0
    return "%3.1f%s" % (num, 't')

# These paramaters can be changed to determine the type of data to generate.
# How many baskets, or transactions
BASKET_COUNT = 10
# Maximum number of items per basket
MAX_ITEMS_PER_BASKET = 7
# The number of unique items
ITEM_COUNT = 100
# The number of frequent itemsets
NUM_FREQUENT_ITEMSETS = 2
# The probability of a basket containing a frequent itemset.
PROB_FREQUENT = 0.5

# Generate the data
pop_frequent = ["F"+str(n) for n in range(0,MAX_ITEMS_PER_BASKET)]
pop_regular = ["I"+str(n) for n in range(MAX_ITEMS_PER_BASKET,ITEM_COUNT)]


freq_itemsets = []

for i in range(NUM_FREQUENT_ITEMSETS):
    cnt = random.randint(1,MAX_ITEMS_PER_BASKET)
    freq_itemsets.append(random.sample(pop_frequent,cnt))

# Create a filename that encodes the MAX_ITEMS_PER_BASKET and BASKET_COUNT into
# the filename.
with open('c:\\freq\\fprob'+str(PROB_FREQUENT)+"_tsz"
        +str(MAX_ITEMS_PER_BASKET)+'_tct'
        +sizeof_fmt(BASKET_COUNT)+'.txt', 'w') as f:
    for i in range(BASKET_COUNT):
        line = []

        cnt = random.randint(1,MAX_ITEMS_PER_BASKET)
        if random.random()<=PROB_FREQUENT:
            idx = random.randint(0,len(freq_itemsets)-1)
            for j in range(len(freq_itemsets[idx])):
                line.append(freq_itemsets[idx][j])

        needed = max(0,cnt - len(line))
        line = line + random.sample(pop_regular,needed)

        f.write(" ".join(line)+"\n")
