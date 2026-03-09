import csv;
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

filename = dir_path + "/books.csv"
fields = []
rows = []

with open(filename, 'r', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)
    
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

    print("Total no. of rows: %d" % csvreader.line_num)  # Row count
print('Field names are: ' + ', '.join(fields))

print('\nFirst 5 rows are:\n')
for row in rows[:5]:
    for col in row:
        print("%10s" % col, end=" ")
    print('\n')