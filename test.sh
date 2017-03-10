#!/bin/bash
#./test.sh

path=/home/lca80/Desktop/data/
array=(${path}gutenberg/5000-8.txt ${path}100mb/pagecounts-20141201-070000 ${path}100mb ${path}300mb ${path}499mb [6]=${path}a1-pagecounts-2)

echo "" > logfile.txt

#------------nltk-------------------------------------------------
echo ".....nltk....." >> logfile.txt
echo

for item in ${array[*]}
do
	echo "time python py/runtok.py $item 3 >> logfile.txt"
	time python gpu_py/runtok.py $item 3 >> logfile.txt
done
echo
exit
#------------cpp-------------------------------------------------
echo ".....Tok.cpp(gpu)....." >> logfile.txt
echo

for item in ${array[*]}
do
	echo "time ./cpp/tok $item 1 >> logfile.txt"
	time ./gpu_cpp/tok $item 1 >> logfile.txt
done
echo
#------------GpuTokenizer.py-------------------------------------------------
echo ".....GpuTokenizer.py....." >> logfile.txt
echo

for item in ${array[*]}
do
	echo "time python py/runtok.py $item 1 >> logfile.txt"
	time python gpu_py/runtok.py $item 1 >> logfile.txt
done
echo
#------------cpp-------------------------------------------------
echo ".....Tok.cpp(cpu)....." >> logfile.txt
echo

for item in ${array[*]}
do
	echo "time ./cpp/tok $item 2 >> logfile.txt"
	time ./gpu_cpp/tok $item 2 >> logfile.txt
done
echo
#------------OneScan.py-------------------------------------------------
echo ".....OneScan.py....." >> logfile.txt
echo

for item in ${array[*]}
do
	echo "time python py/runtok.py $item 2 >> logfile.txt"
	time python gpu_py/runtok.py $item 2 >> logfile.txt
done
echo
#-----------------------------------------------------------------------

python Myplot.py


