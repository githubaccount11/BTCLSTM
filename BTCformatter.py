# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:29:28 2020

@author: Leigh Fair-Smiley
"""


import csv

ratio = {"BTC"}
file = "C:/Users/User/.spyder-py3/BTCdata/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv"
csvFile = f'BTCData/BTC.csv'
csvFile2 = f'BTCData/BTC2.csv'
csvFile3 = f'BTCData/BTC3.csv'
csvFile4 = f'BTCData/BTC4.csv'
lastVal = 4.39
index = 0
with open(file, 'r') as f:
    reader = csv.reader(f)
    with open(csvFile, 'w') as csvFile:
        with open(csvFile2, 'w') as csvFile2:
            with open(csvFile3, 'w') as csvFile3:
                with open(csvFile4, 'w') as csvFile4:
                    writer = csv.writer(csvFile)
                    csvwriter = csv.writer(csvFile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csvwriter2 = csv.writer(csvFile2, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csvwriter3 = csv.writer(csvFile3, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csvwriter4 = csv.writer(csvFile4, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    for row in reader:
                        if row[0] == 'Timestamp':
                            csvwriter.writerow(['pct_change'] + ['volume'])
                        else:
                            if row[4] == "NaN":
                                row[4] = lastVal
                                
                            if row[5] == "NaN":
                                row[5] = 0
                            if index < 1000000:
                                csvwriter.writerow(([(float(row[4]) - float(lastVal)) / float(lastVal)]) + [row[5]])
                            elif 1000000 <= index < 2000000:
                                csvwriter2.writerow(([(float(row[4]) - float(lastVal)) / float(lastVal)]) + [row[5]])
                            elif 2000000 <= index < 3000000:
                                csvwriter3.writerow(([(float(row[4]) - float(lastVal)) / float(lastVal)]) + [row[5]])
                            else:
                                csvwriter4.writerow(([(float(row[4]) - float(lastVal)) / float(lastVal)]) + [row[5]])
                            lastVal = row[4]
                            index = index + 1
                            print(index)
                            print(row)
                
print('finished')