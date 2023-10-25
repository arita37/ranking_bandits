unique=[]
prev = -1
with open("MQ2007/Fold1/vali.txt") as data:
     for line in data:
             qid = int(line.split(' ')[1].split(':')[1])
             if prev != qid :
                prev =qid 
                if qid in unique:
                    print("failure")
                    break
                else:
                    unique.append(qid)
print(len(unique))

data = []
with open("MQ2007/Fold1/train.txt") as data_all:
     for line in data_all:
          data.append(line)
          

import csv

# Input text data
print(data[0])

# Create a CSV file
with open('data.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write header
    csv_writer.writerow(["qid", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "docid", "inc", "prob"])

    # Parse and write data to the CSV file
    for line in data:

        parts = line.split(' ')
        qid = int(parts[0])
        features = {int(f.split(':')[0]): float(f.split(':')[1]) for f in parts[2:48]}
        print(features)
        docid = parts[-7]
        print(docid)
        inc = parts[-4]
        prob = float(parts[-1].split('=')[-1].replace('\n', ''))
        print(inc, prob)
        row = [qid] + [features[i] for i in range(1, 46)] + [docid, inc, prob]
        csv_writer.writerow(row)

print("CSV file 'data.csv' has been created.")

