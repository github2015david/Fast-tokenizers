#!/usr/bin/env python
import sys
import matplotlib.pyplot as plt

with open("logfile1.txt", 'r') as f:
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

"""
ys_g1 = [[1,2,3,4,5,6,7,8,9],[2,3,4,5,6,7,8,9,10],[3,4,5,6,7,8,9,10,11]]
for ys in ys_g1:
	print len(ys),ys

# Plot error over iterations
plt.figure()

for i, ys in enumerate(ys_g1):
	plt.plot(ys, label = my_labels[i])

plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch')
plt.ylim(0.,20.)
plt.legend( loc='upper right', numpoints = 1 )
plt.figure(1)


ys_g = [[1,2,3,4],[2,3,4,5],[3,4,5,6]]
xs = [1,10,100,1000]
for ys in ys_g:
	print len(ys),ys	

# Plot error over iterations
plt.figure()


for i, ys in enumerate(ys_g):
	plt.semilogx(xs, ys, label = my_labels[i])

plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch')
plt.ylim(0.,20.)
plt.legend( loc='upper right', numpoints = 1 )
plt.figure(2)

plt.show()

"""
