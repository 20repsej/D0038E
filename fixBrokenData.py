# D0038E AIPR
#
#
# Imports
import csv

# File
file = "train-final.csv"

fields = []
rows = []


afternoon = []
baby = []
big = []
born = []
bye = []
calendar = []
child = []
cloud = []
come = []
daily = []
dance = []
dark = []
day = []
enjoy = []
go = []
hello = []
home = []
love = []
my = []
name = []
no = []
rain = []
sorry = []
strong = []
study = []
thankyou = []
welcome = []
wind = []
yes = []
you = []

with open(file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)

    fields = next(csvreader)

    for row in csvreader:
        rows.append(row)

    print("Total no of rows: %d"%(csvreader.line_num))

def printRow(row):
    print("Row to print: ")
    for col in row:
        print("%10s"%col,end=" "),

def removeIncompleteRows(input_file, output_file):

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        for row in reader:
            if all(row):
                writer.writerow(row)

def divideIntoClass(input_file):
    with open(input_file) as f_in:
        reader = csv.reader(f_in)

        for row in reader:
            if (row[-1] == '1'):
                afternoon.append(row)
            elif(row[-1] == '2'):
                baby.append(row)
            elif(row[-1] == '3'):
                big.append(row)
            elif(row[-1] == '4'):
                born.append(row)
            elif(row[-1] == '5'):
                bye.append(row)
            elif(row[-1] == '6'):
                calendar.append(row)
            elif(row[-1] == '7'):
                child.append(row)
            elif(row[-1] == '8'):
                cloud.append(row)
            elif(row[-1] == '9'):
                come.append(row)
            elif(row[-1] == '10'):
                daily.append(row)
            elif(row[-1] == '11'):
                dance.append(row)
            elif(row[-1] == '12'):
                dark.append(row)
            elif(row[-1] == '13'):
                day.append(row)
            elif(row[-1] == '14'):
                enjoy.append(row)
            elif(row[-1] == '15'):
                go.append(row)
            elif(row[-1] == '16'):
                hello.append(row)
            elif(row[-1] == '17'):
                home.append(row)
            elif(row[-1] == '18'):
                love.append(row)
            elif(row[-1] == '19'):
                my.append(row)
            elif(row[-1] == '20'):
                name.append(row)
            elif(row[-1] == '21'):
                no.append(row)
            elif(row[-1] == '22'):
                rain.append(row)
            elif(row[-1] == '23'):
                sorry.append(row)
            elif(row[-1] == '24'):
                strong.append(row)
            elif(row[-1] == '25'):
                study.append(row)
            elif(row[-1] == '26'):
                thankyou.append(row)
            elif(row[-1] == '27'):
                welcome.append(row)
            elif(row[-1] == '28'):
                wind.append(row)
            elif(row[-1] == '29'):
                yes.append(row)
            elif(row[-1] == '30'):
                you.append(row)
            else:
              print("BAD")

        printRow(afternoon)


def averageValue(rows):



#removeIncompleteRows(file, "removed.csv")
divideIntoClass(file)


# printing the field names
#print('Field names are:' + ', '.join(field for field in fields))
 
# printing first 5 rows
#print('\nFirst 5 rows are:\n')
#for row in rows[8:15]:
#    # parsing each column of a row
#    for col in row:
#        print("%10s"%col,end=" "),
#    print('\n')