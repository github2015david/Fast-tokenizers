#!/usr/bin/python
# python runtok.py <data_path> <optional: 1/2/3/-1/-2/-3>
#	1: GpuTokenizer.gpuTokenizer
#	2: OneScan.oneScanTokenizer
#	3: TreebankWordTokenizer().tokenize
#	-1/-2/-3 : save output to files "output_py" folder
#
# sample:
#	python runtok.py /home/lca80/Desktop/data/a1-pagecounts-2/pagecounts-20141201-000000
#
import sys, os
import time
import math
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
import GpuTokenizer
import OneScan

verbose = 1		# 0: no wording, 1: wording

MEM_MAX = 520000000	# max gpu size 500mb (can be up to 700mb practically, 2000/4 = 500mb theoretically)
OUTPUT_TOKENS = 11000000 # output max size

#get files form path
def getFiles(in_path):
	fs_lst = []
        if os.path.isfile(in_path):
		fs_lst += [(in_path, os.path.getsize(in_path))]		
	else:
		for f in os.listdir(in_path):
			fname = in_path + '/' + f
			if os.path.isfile(fname):
				fs_lst += [(fname, os.path.getsize(fname))]
	return fs_lst


#select different tokenizer
def tokenize(buf):
	n_bytes = len(buf)

	cmd = 1	#default gpu
	if len(sys.argv) > 2:
		cmd = abs(int(sys.argv[2]))
	if cmd == 3:
		tokens = TreebankWordTokenizer().tokenize(buf)
	elif cmd == 2:
		tokens = OneScan.oneScanTokenizer(buf)
	elif cmd == 4:
		tokens = Cpu.gpuTokenize(buf)
	else:
		tokens = GpuTokenizer.gpuTokenize(buf)

	return tokens

#save and display output
def saveAndDisplay(tokens, i_file):
	len1=len(tokens)

	out=""
	if verbose:
		out += "\ntokens:\n"
		for t in tokens[:50]:
			out += "%s,"%(t)
		out += "........................................"
		for t in tokens[-10:]:
			out += "%s,"%(t)
		out += "(%d tokens)\n"%(len1)

	cmd = 1	#default gpu
	if len(sys.argv) > 2:
		cmd = int(sys.argv[2])

	if cmd < 0:
		if verbose:
			out += "output saved in output_py/part-0000%d\n"%(i_file)

		os.system("rm -r output_py; mkdir output_py")
		with open("output_py/part-0000%d"%(i_file), 'w') as f:
			f.write("\n".join(tokens[:OUTPUT_TOKENS]))
		f.closed
	
	return out

#Main
if __name__ == "__main__":
	t0 = time.time()
	t1 = time.time()

	if len(sys.argv) < 2:
		print "\nUsage: python <app.py> <path_to_data> <optional: path_to_output>\n"
		sys.exit()

	fs_lst = getFiles(sys.argv[1])
	fs_len = len(fs_lst)
	if fs_len == 0:
		print ("Error: read empty directory!")

	if verbose:
		for i, f in enumerate(fs_lst):
			print "%d/%d %s"%(i, fs_len, f)
		print "files: ",fs_len

	t_read = time.time() - t1 ; t1 = time.time()
	t_tokenize = 0
	t_save = 0

	#limit data size for each chunk of data 500mb approx.
	n_tokens = 0
	all_bytes = 0
	out = ""

	buf = ""
	n_bytes = 0
	i_outfile = 0
	for (fname, n) in fs_lst:
		n_bytes += n

		#limit MEM_MAX going to gpu
		if n_bytes < MEM_MAX:
			with open(fname, 'r') as f:
				buf += f.read()
			f.closed
		else :		
			t_read += time.time() - t1 ; t1 = time.time()

			tokens = tokenize(buf)

			t_tokenize += time.time() - t1 ; t1 = time.time()

			out += saveAndDisplay(tokens,i_outfile)

			t_save += time.time() - t1 ; t1 = time.time()

			n_tokens += len(tokens)
			all_bytes += len(buf)
			i_outfile += 1

			n_bytes = n
			buf = ""
			with open(fname, 'r') as f:
				buf += f.read()
			f.closed
		

	if len(buf) >0:
		t_read += time.time() - t1 ; t1 = time.time()

		tokens = tokenize(buf)

		t_tokenize += time.time() - t1 ; t1 = time.time()

		out += saveAndDisplay(tokens, i_outfile)

		t_save += time.time() - t1 ; t1 = time.time()

		n_tokens += len(tokens)
		all_bytes += len(buf)


	if verbose:
		print out
		print "read data time    : %f"%(t_read)
		print "tokenize time     : %f"%(t_tokenize)
		print "write out time    : %f"%(t_save)

		t = time.time()-t0
		print("time: %.3f secs to tokenize %db, %dkb/s (tokens:%d)."%(t, all_bytes, n_bytes/1024/t, n_tokens))
