#!/usr/bin/python
# python Compare2PTB.py <data_path> <optional: 1/2/3>
#	1: compare TreebankWordTokenizer to src_py/GpuTokenizer.py
#	2: compare TreebankWordTokenizer to src_py/OneScan.py
#	3: compare TreebankWordTokenizer to src_cpp/GpuTokenize.cu
#
# sample:
#	python Compare2PTB.py /home/lca80/Desktop/data/300mb 3
#
import sys, os
import time
from nltk.tokenize import TreebankWordTokenizer 
sys.path.append("src_py")
import GpuTokenizer
import OneScan

from os import listdir
from os.path import isfile, join

t0 = time.time()
t1 = time.time()
if len(sys.argv) < 2:
	print "\nUsage: python <app.py> <path_to_data> <optional: path_to_output>\n"
	sys.exit()

path = sys.argv[1]
if isfile(path):
	fs_len = 1;
	fs_lst = [path]
else:	
	fs_lst = [ path + "/" + f for f in listdir(path) if isfile(join(path, f))]
	fs_len = len(fs_lst)

for i, f in enumerate(fs_lst):
	print "%d/%d %s"%(i, fs_len, f)
print "files: ",fs_len


def getCpp(fname):
	print "\nGpuTokenizer.cpp.............."
	t1 = time.time()
	os.system("./src_cpp/tok "+fname + " -1")
	buf = ""
	with open("output_cpp/part-00000", 'r') as f:
		buf += f.read()
	f.closed
	lst2 = buf.replace("...", " ... ").replace("--", " -- ").split()
	t = time.time()-t1
	print("time: %.3f secs to tokenize (tokens:%d)."%(t, len(lst2))) 
	return lst2


def getOneScan(buf):
	print "\nOnescan.oneScanTokenizer()..............."
	t1 = time.time()
	lst2 = OneScan.oneScanTokenizer(buf)
	t = time.time()-t1
	print("time: %.3f secs to tokenize %db, %dkb/s (tokens:%d)."%(t, len(buf), len(buf)/1024/t, len(lst2)))
	return lst2


def getGpu(buf):
	print "\nGpuTokenizer.gpuTokenize(buf)..............."
	t1 = time.time()
	lst2 = GpuTokenizer.gpuTokenize(buf)
	t = time.time()-t1
	print("time: %.3f secs to tokenize %db, %dkb/s (tokens:%d)."%(t, len(buf), len(buf)/1024/t, len(lst2)))
	return lst2


def compare(fname):
	print "\n\n-------------------",fname

	#1) get result from nltk
	print "\nnltk.TreebankWordTokenizer().............."
	t1 = time.time()
	buf = ""
	with open(fname, 'r') as f:
		buf += f.read()
	f.closed
	t1 = time.time()
	n_bytes = len(buf)
	print n_bytes
	lst1 = TreebankWordTokenizer().tokenize(buf)
	len1=len(lst1)
	print("time: %.3f secs to tokenize %db, (tokens:%d)."%(time.time()-t1, n_bytes, len1))

	#2)get result from py or cpp
	cmd = 1
	if len(sys.argv) > 2:
		cmd = int(sys.argv[2])

	if cmd == 3:
		lst2 = getCpp(fname)
	elif cmd == 2:
		lst2 = getOneScan(buf)
	else:
		lst2 = getGpu(buf)

	len2 = len(lst2)
	print " (tokens:%d)."%(len2); t1 = time.time()

	#3) compare
	if len2 < len1:
		len1 = len2

	i = 0;
	j = 0;
	count = 0;

	while (i < len1-1) and (j <len1-1):
		x1 = lst1[i]
		x2 = lst2[j]
		if x1 <> x2:
			if lst1[i+1] == lst2[j+1]:
				print "\nlen1:%3d, len2:%3d not matched skip 1 -----%d\n"%(len(lst1[i]),len(lst2[j]), i)
				print "<<",i,">>",' '.join(lst1[i-2:i+5])
				print "<<",j,">>",' '.join(lst2[j-2:j+5])
			else:
				if lst1[i+1] == lst2[j+2]:
					print "\nnot matched lst2 jump 1 -----", i ,"\n"
					print "<<",i,">>",' '.join(lst1[i-2:i+3])
					print "<<",j,">>",' '.join(lst2[j-2:j+3])
	#				print lst1[i+1]; 
	#				print lst2[j+2]; 
					i += 1
					j += 2
				elif lst1[i+2] == lst2[j+1]: 
					print "\nnot matched lst1 jump 1 -----", i,"\n"
					print "<<",i,">>",' '.join(lst1[i-2:i+3])
					print "<<",j,">>",' '.join(lst2[j-2:j+3])
	#				print lst1[i+2]; 
	#				print lst2[j+1]; 
					i += 2
					j += 1
				elif lst1[i+3] == lst2[j+1]: 
					print "\nnot matched lst1 jump 2 -----", i,"\n"
					print "<<",i,">>",' '.join(lst1[i-2:i+4])
					print "<<",j,">>",' '.join(lst2[j-2:j+4])
	#				print lst1[i+2]; 
	#				print lst2[j+1]; 
					i += 3
					j += 1
					count +=1
				elif lst1[i+4] == lst2[j+1]: 
					print "\nnot matched lst1 jump 3 -----", i,"\n"
					print "<<",i,">>",' '.join(lst1[i-2:i+5])
					print "<<",j,">>",' '.join(lst2[j-2:j+5])
	#				print lst1[i+2]; 
	#				print lst2[j+1]; 
					i += 4
					j += 1
					count +=2
				elif lst1[i+6] == lst2[j+1]: 
					print "\nnot matched lst1 jump 5 -----", i,"\n"
					print "<<",i,">>",' '.join(lst1[i-2:i+7])
					print "<<",j,">>",' '.join(lst2[j-2:j+7])
	#				print lst1[i+2]; 
	#				print lst2[j+1]; 
					i += 6
					j += 1
					count +=4
				else:
					print "\nmore than 2 not matched  -----", i,"\n"
					print "<<",i,">>",' '.join(lst1[i-2:i+10])
					print "<<",j,">>",' '.join(lst2[j-2:j+10])
					print "Try to adjust the WORD_MAX_LEN. stop!"
					break
			count +=1
		i += 1
		j += 1	
			
	print "\n%d/%d words not matched. Done!\n"%(count,len1)
	return [count, i]

tokens = 0
total_count = 0
for fname in fs_lst:
#	try:
		[count, token] = compare(fname)
		total_count += count
		tokens += token
		"""
	except:
		print "error in ",fname
		cmd = raw_input('Enter cmd(q: quit)\n>>')
		if cmd == 'q':
			break
		else:
			continue
		"""
print "nltk.TreebankWordTokenizer vs",
cmd = 1
if len(sys.argv) > 2:
	cmd = int(sys.argv[2])

if cmd == 3:
	print "GpuTokenizer.cpp"
if cmd == 2:
	print "OneScan.py"
else:
	print "GpuTokenizer.py"


print ("mismatched : %d/%d"%(total_count, tokens))
