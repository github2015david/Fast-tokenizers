#!/usr/bin/env python
import sys
import matplotlib.pyplot as plt

with open("logfile.txt", 'r') as f:
	data = f.read()
f.closed
lst1 = data.split('\n')
lst = []
for line in lst1:
	lst2 = line.split()
	if len(lst2) < 5:
		continue
	if lst2[0] == "time:":
		lst += [float(lst2[1])]
print lst

len1 = len(lst)/5
ys_group = []
start = 0

for i in xrange(5):
	ys_group += [lst[start:start+len1]]
	start += len1

print ys_group

ys_g = []
for i in xrange(1,5):
	ys_g += [map(lambda (x,y): x/y, zip(ys_group[0],ys_group[i]))]

print ys_g
xs = [1,10,100,300,500,1000]

my_labels=["gpu_cpp", "gpu_py", "cpu_cpp", "onescan_py"]

#sys.exit(0)
xs = xs[:len(ys_g[0])]

for i, ys in enumerate(ys_g):
	plt.semilogx(xs, ys, label = my_labels[i])

plt.ylabel('Speedup')
plt.title('Compare to TreebankWordTokenizer.tokenize()')
plt.xlabel('Mb')
plt.ylim(0.,30.)
plt.legend( loc='upper left', numpoints = 1 )
plt.figure(1)

plt.show()
