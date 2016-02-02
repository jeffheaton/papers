__author__ = 'jheaton'

import os
import time
import csv
import numpy
import time
from subprocess import call

PATH = "c:\\freq\\"
TESTS = 20

def process(alog,file,log):
    runs = []

    for i in range(TESTS):
        start_time = time.time()
        command = os.path.join(PATH,alog+".exe")
        datafile = os.path.join(PATH,file)
        call(command + " " + datafile,stdout=log,stderr=log)
        elapsed_time = time.time() - start_time
        print(time.strftime("%c") + ": Result: " + str(elapsed_time) + " seconds.")
        runs.append(elapsed_time)

    return (numpy.mean(runs),numpy.min(runs))


files = [
    #"fprob0.1_tsz100_tct10.0m.txt",

    "fprob0.5_tsz100_tct10.0m.txt",
"fprob0.5_tsz10_tct10.0m.txt",
"fprob0.5_tsz20_tct10.0m.txt",
"fprob0.5_tsz30_tct10.0m.txt",
"fprob0.5_tsz40_tct10.0m.txt",
"fprob0.5_tsz50_tct10.0m.txt",
"fprob0.5_tsz60_tct10.0m.txt",
"fprob0.5_tsz70_tct10.0m.txt",
"fprob0.5_tsz80_tct10.0m.txt",
"fprob0.5_tsz90_tct10.0m.txt"

    #"fprob0.1_tsz50_tct10.0m.txt",
    #"fprob0.2_tsz50_tct10.0m.txt",
    #"fprob0.3_tsz50_tct10.0m.txt",
    #"fprob0.4_tsz50_tct10.0m.txt",
    #"fprob0.5_tsz50_tct10.0m.txt",
    #"fprob0.6_tsz50_tct10.0m.txt",
    #"fprob0.7_tsz50_tct10.0m.txt",
    #"fprob0.8_tsz50_tct10.0m.txt",
]

alogs = [
    "apriori",
    "eclat",
    "fpgrowth"
]

with open(os.path.join(PATH,"log.txt"), 'w') as logFile:
    with open(os.path.join(PATH,"report.csv"), 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='\"', quoting=csv.QUOTE_MINIMAL)

        for file in files:
            for alog in alogs:
                print(time.strftime("%c") + ": Evaluating: " + str([alog,file]))
                t_mean, t_min = process(alog,file,logFile)
                writer.writerow([file,alog,str(t_mean),str(t_min)])
