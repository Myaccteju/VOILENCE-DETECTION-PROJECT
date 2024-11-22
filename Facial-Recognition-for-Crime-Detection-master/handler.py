import csv
def insertData(data):
    field=['Name', "Father's Name", "Gender", "DOB","Crimes"]
    x=[data['Name'], data["Father's Name"], data['Gender'], data['DOB(yyyy-mm-dd)'], data['Crimes Done']]
    filen = "Criminal.csv"
    with open(filen, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(field)
        csvwriter.writerow(x)
