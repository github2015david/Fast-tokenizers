#!/bin/bash
#./test.sh

path=/home/lca80/Desktop/data/
array=(${path}gutenberg/5000-8.txt ${path}100mb/pagecounts-20141201-070000 ${path}100mb ${path}300mb ${path}499mb ${path}a1-pagecounts-2)

NUM=6

NUM=`expr $NUM - 1`
echo "" > logfile.txt
#------------nltk-------------------------------------------------
echo ".....nltk....." >> logfile.txt
echo

for i in $(seq 0 $NUM)
do
	echo "time python src_py/runtok.py ${array[$i]} 3 >> logfile.txt"
	time python src_py/runtok.py ${array[$i]} 3 >> logfile.txt
done
echo
#------------cpp-------------------------------------------------
echo ".....Tok.cpp(gpu)....." >> logfile.txt
echo

for i in $(seq 0 $NUM)
do
	echo "time ./src_cpp/tok ${array[$i]} 1 >> logfile.txt"
	time ./src_cpp/tok ${array[$i]} 1 >> logfile.txt
done
echo
#------------GpuTokenizer.py-------------------------------------------------
echo ".....GpuTokenizer.py....." >> logfile.txt
echo

for i in $(seq 0 $NUM)
do
	echo "time python src_py/runtok.py ${array[$i]} 1 >> logfile.txt"
	time python src_py/runtok.py ${array[$i]} 1 >> logfile.txt
done
echo
#------------cpp-------------------------------------------------
echo ".....Tok.cpp(cpu)....." >> logfile.txt
echo

for i in $(seq 0 $NUM)
do
	echo "time ./src_cpp/tok ${array[$i]} 2 >> logfile.txt"
	time ./src_cpp/tok ${array[$i]} 2 >> logfile.txt
done
echo
#------------OneScan.py-------------------------------------------------
echo ".....OneScan.py....." >> logfile.txt
echo

for i in $(seq 0 $NUM)
do
	echo "time python src_py/runtok.py ${array[$i]} 2 >> logfile.txt"
	time python src_py/runtok.py ${array[$i]} 2 >> logfile.txt
done
echo
#-----------------------------------------------------------------------

python Myplot.py

