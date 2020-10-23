# This Python script was used to collect the data for following paper/conference:
#
# Heaton, J. (2016, April). Comparing Dataset Characteristics that Favor the Apriori, 
# Eclat or FP-Growth Frequent Itemset Mining Algorithms. In SoutheastCon 2015 (pp. 1-6). IEEE.
#
# http://www.jeffheaton.com
#

# Generate benchmark data for frequent itemset mining.
__author__ = 'jheaton'
import random
import csv
from tqdm import tqdm

def sizeof_fmt(num):
    for x in ['','k','m','g']:
        if num < 1000.0:
            return "%3.1f%s" % (num, x)
        num /= 1000.0
    return "%3.1f%s" % (num, 't')

def generate_itemset(row_count, max_per_basket, num_freq_sets, item_count, prob_frequent):
    '''
    Generate a dataset of frequent items. These paramaters can be changed to 
    determine the type of data to generate.

    :param int row_count: The number of rows in the dataset.
    :param int max_per_basket: Maximum number of items per basket.
    :param int num_freq_sets: The number of unique frequent item sets.
    :param int item_count: The number of unique items.
    :param float prob_frequent: The probability of a basket containing a frequent itemset.
    '''
    # Generate the data
    pop_frequent = ["F"+str(n) for n in range(0,max_per_basket)]
    pop_regular = ["I"+str(n) for n in range(max_per_basket,item_count)]
    freq_itemsets = []

    # Create a filename that encodes the max_per_basket and basket_count into
    # the filename.
    filename = str(prob_frequent)+"_tsz" \
        + str(max_per_basket)+'_tct' \
         +sizeof_fmt(row_count)+'.txt'

    for i in tqdm(range(num_freq_sets),desc=f"{filename}:pass 1/2"):
        cnt = random.randint(1,max_per_basket)
        freq_itemsets.append(random.sample(pop_frequent,cnt))

    with open(filename, 'w') as f:
        for i in tqdm(range(row_count),desc=f"{filename}:pass 2/2"):
            line = []

            cnt = random.randint(1,max_per_basket)
            if random.random()<=prob_frequent:
                idx = random.randint(0,len(freq_itemsets)-1)
                for j in range(len(freq_itemsets[idx])):
                    line.append(freq_itemsets[idx][j])

            needed = max(0,cnt - len(line))
            line = line + random.sample(pop_regular,needed)

            f.write(" ".join(line)+"\n")

random.seed(1000)
ROWS = 10000000
#generate_itemset(1000, 10, 100, 50000, 0.5)

for i in range(10,110,10):
    generate_itemset(ROWS, i, 100, 50000, 0.5)

for i in range(1,9):
    generate_itemset(ROWS, 50, 100, 50000, i/10.0)