import csv
import os

with open ('fer2013.csv') as f:
    csvr = csv.reader(f)
    header = next(csvr)
    rows = [row for row in csvr]

if not os.path.exists(os.path.dirname("data")):
    try:
        os.mkdir("data")
    except OSError as ee:
        raise


trainingData = [row[:-1] for row in rows if row[-1] == 'Training']
csv.writer(open('data/train.csv', 'w+')).writerows([header[:-1]] + trainingData)
print (len(trainingData))

publicTestData = [row[:-1] for row in rows if row[-1] == 'PublicTest']
csv.writer(open('data/testPublic.csv', 'w+')).writerows([header[:-1]] + publicTestData)
print (len(publicTestData))

privateTestData = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
csv.writer(open('data/testPrivate.csv', 'w+')).writerows([header[:-1]] + privateTestData)
print (len(privateTestData))