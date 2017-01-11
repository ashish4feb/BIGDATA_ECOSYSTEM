import csv
import numpy

# Open up the csv file in to a Python object
csv_file_object = csv.reader(open('train.csv', 'rU')) 
header = next(csv_file_object)  # The next() command just skips the 
                                 # first line which is a header
data=[]                          # Create a variable called 'data'.
for row in csv_file_object:      # Run through each row in the csv file,
    data.append(row)             # adding each row to the data variable
data = numpy.array(data) 	         # Then convert from a list to an array
			         # Be aware that each item is currently
                                 # a string in this format

print(data[:,0]);
