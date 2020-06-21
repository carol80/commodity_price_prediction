import csv
with open('Oil2.csv','rt')as f:
    data = csv.reader(f)

    for row in data :
        print(row)
    for index, col in enumerate(row[1:], 1):  # Skip the header row
        row[index][1] = (long)(col[1])

    for row in data :
        col=row
        c = 0
        for col in data:
            if row[0] != col[0] :
                row[0] = row[0] / c
                with open('newdata.csv', mode='w') as file:
                    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow( row[0], row[2])
                    row[0] = col[0]

            elif row[0] == col[0] :
                c = c + 1
                row[2] = ( row[2] + col[2] ) / 2