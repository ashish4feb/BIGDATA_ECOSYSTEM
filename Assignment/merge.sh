head -1 train.csv > final.csv

for filename in $(ls foo*.csv); 
do 
sed 1d $filename >> final.csv;
done;
