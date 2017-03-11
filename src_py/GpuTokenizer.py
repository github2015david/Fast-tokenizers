#!/usr/bin/python
#
# to use:
#import GpuTokenizer
#...
#tokens = GpuTokenizer.gpuTokenize(text)
#...

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy as np
import time
import math
import sys

verbose = 2

PATCHING = '\1'		#define in device
ENDING = '\0'
DOUBLE1 = '\2'		#define in device, for open '"'
DOUBLE2 = '\3'		#define in device, for close '"'

HALF_WINDOW = 7		#define in device
FULL_WINDOW =  14	#define in device

#patching-kernel
BLOCK_DIM_1D = 128
GRID_DIM_1D = 4096
GLOBAL_REPEAT = 30

#gpu kernel wrapped cuda code
def patchingModule():
	mod = SourceModule("""
	#include <stdio.h>

	__device__ char PATCHING = '\\1';

	__device__ char DOUBLE1 = '\\2';

	__device__ char DOUBLE2 = '\\3';

	__device__ char QM = '\\'';

	__device__ int HALF_WINDOW = 7;

	__device__ const int FULL_WINDOW = 14;

	//lst are sorted for binary search
	__device__ const char *wspace_lst="\\t\\n\\v\\f\\r ";	//6

	__device__ const char *b_lst = "\\t\\n\\f\\r !\\"#$%&'()*+,-./:;<=>?@[\\\]^`{|}~"; //36

	__device__ const char *changed_lst = " !#$%&(),:;<>?@[]{}"; //19

	__device__ const char *not_changed_lst = "*+-./=\\\^`|~";  //11

	__device__ char tolower(char c) {
		if 	(c >= 'A' && c <= 'Z')
			c += 32;
		return c;
	}

	//binary_search(): return index +1, to indicate 0 is not found. to get the index = 'return' -1
	__device__ int inAt(const char *arr, int len2, char val) {
		int start = 0;
		int end = len2-1;
		if( val < arr[start] || val > arr[end] ){
			return -1 +1;
		}

		while(start <= end){
			int pivot = (start+end) >> 1; 
			if(arr[pivot] == val){
				if (pivot == 0)
					return 0+1;
				while (arr[pivot-1] == val)
					pivot -= 1;	
				return pivot+1;
			}else if(arr[pivot] < val){
				start = pivot+1;
			}else if(arr[pivot] > val){
				end = pivot-1;
			} 
		}
		return -1+1;
	}

	__device__ int isBackChanged(char* window, int p)  {
		if (inAt(changed_lst, 19, window[p]) 
					|| (window[p] == '.' && window[p+1] == '.' && window[p+2] == '.') 
					|| (window[p] == '-' && window[p+1] == '-') 				
					|| window[p] == '"')
			return 1;
		else
			return 0;
	}

	__device__ int isFrontChanged(char* window, int p)  {
		if (inAt(changed_lst, 19, window[p]) 
					|| (window[p] == '.' && window[p-1] == '.' && window[p-2] == '.') 
					|| (window[p] == '-' && window[p-1] == '-') 				
					|| window[p] == '"')
			return 1;
		else
			return 0;
	}

	__device__ void rules_kernel(char *c, char *l, char *r, char *window)
	{
		int i = HALF_WINDOW;
		char c_lower = tolower(*c);

		if (c_lower == '`') {
			if (window[i-1] != '`' && window[i+1] == '`') {
				*l = ' ';
				*r = PATCHING;
			}
			else if (window[i-1] == '`' && window[i+1] != '`') {
				*l = PATCHING;
				*r = ' ';
			}
		}
		else if (c_lower == '"') {
			if (inAt(" (<[`{", 6, window[i-1])) {
				*l = ' ';
				*c = '`';
				*r = '\\2';	//double in replace
			}
			else {
				*l = ' ';
				*c = QM;
				*r = '\\3';	//double in replace
				return ;
			}
		}
		else if ((c_lower == ':' || c_lower == ',')) {
			if ( !inAt("0123456789", 10, window[i+1]) 
				&& ( window[i -1] != *c 
				|| (window[i -1] == *c and window[i-2] == *c))) {
				*l = ' ';
				*r = ' ';
			}
			else if (window[i-2] == *c && window[i-1] == *c && window[i+1] == '_') {
				*l = ' ';
				*r = ' ';
			} 			
		}
		else if ( inAt("!#$%&();<>?@[]{}", 16, *c)) { //delimiter
			*l = ' ';
			*r = ' ';
		}
		else if ( inAt(not_changed_lst, 11, *c)) {  //boundary and not delimiter		 
			if (*c == '.' && window[i+1] == '.' && window[i+2] == '.' && window[i-1] != '.') {// every "..."
				*l = ' ';
				*r = PATCHING;
			}
			else if (*c == '.' && window[i-1] == '.' && window[i-2] == '.' && window[i-3] != '.') {
				*l = PATCHING;
				*r = ' ';	
			}
			else if (*c == '.' && window[i-1] == '.' && window[i-2] == '.' && window[i-3] == '.'
				&& window[i-4] == '.' && window[i-5] == '.' && window[i-6] != '.') {
				*l = PATCHING;
				*r = ' ';	
			}
			else if (*c == '.' && window[i+1] == '\\0') {
				*l = ' ';
				*r = PATCHING;
			}
			else if ((window[i+1] == 'c' && window[i+2] == 'a' && window[i+3] == 'n' && window[i+4] == 'n' && window[i+5] == 'o' && window[i+6] == 't')
				|| (window[i+1] == 'd' && window[i+2] == QM && window[i+3] == 'y' && window[i+4] == 'e' && inAt(b_lst, 11, window[i+5]))
				|| (window[i+1] == 'g' && window[i+2] == 'i' && window[i+3] == 'm' && window[i+4] == 'm' && window[i+5] == 'e' && inAt(b_lst, 11, window[i+6]))
				|| (window[i+1] == 'g' && window[i+2] == 'o' && window[i+3] == 'n' && window[i+4] == 'n' && window[i+5] == 'a' && inAt(b_lst, 11, window[i+6]))
				|| (window[i+1] == 'g' && window[i+2] == 'o' && window[i+3] == 't' && window[i+4] == 't' && window[i+5] == 'a' && inAt(b_lst, 11, window[i+6]))
				|| (window[i+1] == 'l' && window[i+2] == 'e' && window[i+3] == 'm' && window[i+4] == 'm' && window[i+5] == 'e' && inAt(b_lst, 11, window[i+6]))
				|| (window[i+1] == 'm' && window[i+2] == 'o' && window[i+3] == 'r' && window[i+4] == QM && window[i+5] == 'n' && inAt(b_lst, 11, window[i+6]))
				|| (window[i+1] == 'w' && window[i+2] == 'a' && window[i+3] == 'n' && window[i+4] == 'n' && window[i+5] == 'a' && inAt(b_lst, 11, window[i+6])))  {
				*l = PATCHING;
				*r = ' ';
			}
			else if ((window[i-6] == 'c' && window[i-5] == 'a' && window[i-4] == 'n' && window[i-3] == 'n' && window[i-2] == 'o' && window[i-1] == 't')
				|| (inAt(b_lst, 11, window[i-5]) && window[i-4] == 'd' && window[i-3] == QM && window[i-2] == 'y' && window[i-1] == 'e')  
				|| (inAt(b_lst, 11, window[i-6]) && window[i-5] == 'g' && window[i-4] == 'i' && window[i-3] == 'm' && window[i-2] == 'm' && window[i-1] == 'e')
				|| (inAt(b_lst, 11, window[i-6]) && window[i-5] == 'g' && window[i-4] == 'o' && window[i-3] == 'n' && window[i-2] == 'n' && window[i-1] == 'a')
				|| (inAt(b_lst, 11, window[i-6]) && window[i-5] == 'g' && window[i-4] == 'o' && window[i-3] == 't' && window[i-2] == 't' && window[i-1] == 'a')
				|| (inAt(b_lst, 11, window[i-6]) && window[i-5] == 'l' && window[i-4] == 'e' && window[i-3] == 'm' && window[i-2] == 'm' && window[i-1] == 'e')
				|| (inAt(b_lst, 11, window[i-6]) && window[i-5] == 'w' && window[i-4] == 'a' && window[i-3] == 'n' && window[i-2] == 'n' && window[i-1] == 'a')
				|| (inAt(b_lst, 11, window[i-6]) && window[i-5] == 'm' && window[i-4] == 'o' && window[i-3] == 'r' && window[i-2] == QM && window[i-1] == 'n')
				|| (inAt(b_lst, 11, window[i-5]) && window[i-4] == QM && window[i-3] == 't' && window[i-2] == 'i' && window[i-1] == 's')
				|| (inAt(b_lst, 11, window[i-6]) && window[i-5] == QM && window[i-4] == 't' && window[i-3] == 'w' && window[i-2] == 'a' && window[i-1] == 's')) {
				*l = ' ';
				*r = PATCHING;
			}

			else if (*c == '-') { //(*c != '*')  every "--"
				if (window[i-1] != '-' && window[i+1] == '-') {
					*l = ' ';
					*r = PATCHING;
				}
				else if (window[i-2] != '-' && window[i-1] == '-') {
					*l = PATCHING;
					*r = ' ';
				}
				else if (window[i-4] != '-' && window[i-3] == '-'
					&& window[i-2] == '-' && window[i-1] == '-') {
					*l = PATCHING;
					*r = ' ';
				}
				else if (window[i-6] != '-' && window[i-5] == '-'
					&& window[i-4] == '-' && window[i-3] == '-'
					&& window[i-2] == '-' && window[i-1] == '-') {
					*l = PATCHING;
					*r = ' ';
				}
			}

			else {
				*l = PATCHING;
				*r = PATCHING;
			}
		}
		else if (*c == QM) {
			if (window[i-1]!= QM && window[i+1] == QM) {
				*l = ' ';
				*r = PATCHING;
			}
			else if (window[i-1] == QM) {
				*l = PATCHING;
				*r = ' ';
			}
			else if ( window[i+1] == '"' || (window[i+1] == '-' && window[i+2] == '-') || (window[i+1] == '.' && window[i+2] == '.' && window[i+3] == '.')) {
				*l = ' ';
				*r = ' ';
			}
			else if ( window[i-1] != QM) {
				if ( inAt("DMSdms", 6, window[i+1])) {
					if (isBackChanged(window, i+2)) {
						*l = ' ';
						*r = PATCHING;
					}
				}
				else if ( window[i+1] == 'l' && window[i+2] == 'l') {
					if (isBackChanged(window, i+3))  {
						*l = ' ';
						*r = PATCHING;
					}
				}
				else if ( window[i+1] == 'L' && window[i+2] == 'L') {
					if (isBackChanged(window, i+3))  {
						*l = ' ';
						*r = PATCHING;
					}
				}
				else if ( window[i+1] == 'r' && window[i+2] == 'e') {
					if (isBackChanged(window, i+3))  {
						*l = ' ';
						*r = PATCHING;
					}
				}
				else if ( window[i+1] == 'R' && window[i+2] == 'E' && isBackChanged(window, i+3) )  {
					*l = ' ';
					*r = PATCHING;
				}
				else if ( window[i+1] == 'v' && window[i+2] == 'e' && isBackChanged(window, i+3) )  {
					*l = ' ';
					*r = PATCHING;
				}
				else if ( window[i+1] == 'V' && window[i+2] == 'E') {
					if (isBackChanged(window, i+3))  {
						*l = ' ';
						*r = PATCHING;
					}
				}
				else if ( inAt(b_lst, 36, window[i-2]) && (window[i-1] == 'd' || window[i-1] == 'D')
					&& window[i+1] == 'y' && window[i+2] == 'e' && inAt(b_lst, 36, window[i+3]))  {
					*l = ' ';
					*r = PATCHING;
				}
				else if ( inAt(b_lst, 36, window[i-4]) && window[i-3] == 'm' 
					&& window[i-2] == 'o' && window[i-1] == 'r' && window[i+1] == 'n') { 
					if (inAt(b_lst, 36, window[i+2]))  {
						*l = ' ';
						*r = PATCHING;
					}
				}
				else if ( window[i-1] != QM) {
					if (inAt(" !#$%&(),:;<>?@[]{}", 19, window[i+1])) {
						*l = ' ';
						*r = ' ';
					}
				}
			}
		}
		else if ((c_lower == 'm' || c_lower == 'n' || c_lower == 't')) { 
			if (isFrontChanged(window, i-2) && window[i-1] == QM && c_lower == 't')  {  //constrations3
				if (( window[i+1] == 'i' && window[i+2] == 's' && inAt(b_lst, 36, window[i+3])) 
					|| ( window[i+1] == 'w' && window[i+2] == 'a' && window[i+3] == 's' && inAt(b_lst, 36, window[i+4])) )  {
					*l = PATCHING;
					*r = ' ';
				}
			}
			else if ( window[i-1] != QM && c_lower == 'n' && window[i+1] == QM && window[i+2] == 't') { // || window[i+2] == 'T')) {
				if (isBackChanged(window, i+3))  {
					*l = ' ';
					*r = PATCHING;
				}
			}
			else if ( window[i-2] == 'n' && c_lower == 't') {  //|| window[i-2] == 'N') && window[i-1] == QM && c_lower == 't') { 
				if (isBackChanged(window, i+1))  {
					*l = PATCHING;
					*r = ' ';
				}
			}
			else if (inAt(b_lst, 36, window[i-4])) { //break in middle of contrations2
				if ( window[i-3] == 'c' && window[i-2] == 'a' && window[i-1] == 'n'
					&& c_lower == 'n' && window[i+1] == 'o' && window[i+2] == 't') { 
					if (inAt(b_lst, 36, window[i+3]))  {
						*l = ' ';
						*r = PATCHING;
					}
				} 
				if ( window[i-3] == 'g' && window[i-2] == 'i' && window[i-1] == 'm'
					&& c_lower == 'm' && window[i+1] == 'e') {
					if (inAt(b_lst, 36, window[i+2]))  {
						*l = ' ';
						*r = PATCHING;
					}
				}
				else if ( window[i-3] == 'g' && window[i-2] == 'o' && window[i-1] == 'n'
					&& c_lower == 'n' && window[i+1] == 'a') {
					if (inAt(b_lst, 36, window[i+2]))  {
						*l = ' ';
						*r = PATCHING;
					}
				}
				else if ( window[i-3] == 'g' && window[i-2] == 'o' && window[i-1] == 't'
					&& c_lower == 't' && window[i+1] == 'a') {
					if (inAt(b_lst, 36, window[i+2]))  {
						*l = ' ';
						*r = PATCHING;
					}
				}
				else if ( window[i-3] == 'l' && window[i-2] == 'e' && window[i-1] == 'm'
					&& c_lower == 'm' && window[i+1] == 'e') {
					if (inAt(b_lst, 36, window[i+2]))  {
						*l = ' ';
						*r = PATCHING;
					}
				}
				else if ( window[i-3] == 'l' && window[i-2] == 'e' && window[i-1] == 'm'
					&& c_lower == 'm' && window[i+1] == 'e') {
					if (inAt(b_lst, 36, window[i+2]))  {
						*l = ' ';
						*r = PATCHING;
					}
				}
				else if ( window[i-3] == 'w' && window[i-2] == 'a' && window[i-1] == 'n'
					&& c_lower == 'n' && window[i+1] == 'a') {
					if (inAt(changed_lst, 19, window[i+2]))  {
						*l = ' ';
						*r = PATCHING;
					}
				}
			}
		}
		return;
	}

	__global__ void patch_kernel( char *buffer, char *patched, int *para)
	{
		int repeat = para[1];
		int n = para[0];
		int tid = (blockDim.x*blockIdx.x + threadIdx.x); 		
		int x_start = tid*repeat;

		int j;
		char window[FULL_WINDOW];

		char l, c, r;
		for (int i=x_start; i < x_start+repeat; i++) {
			if (i < n) {
				j = i*3;	

				c = buffer[i];
				patched[j] = PATCHING;
				patched[j+1] = c;
				patched[j+2] = PATCHING;
				memcpy(window, buffer + i - HALF_WINDOW, 2*HALF_WINDOW);
				if (!inAt(wspace_lst, 6, c)) 
				{
					l = PATCHING;
					r = PATCHING;
					for (int i = 0; i < FULL_WINDOW; i++)
						window[i] = tolower(window[i]);

					rules_kernel(&c, &l, &r, window);
					patched[j] = l;
					patched[j+1] = c;
					patched[j+2] = r;			
				}
				else {
					patched[j+1] = ' ';
				}				
			}
		}//end for

	}
	""")

	return mod.get_function("patch_kernel")


#gpu tokenizer
def gpuTokenize(buff):
	t1 = time.time()

	#loading data
	a_npa = np.array([PATCHING*HALF_WINDOW + buff + PATCHING*HALF_WINDOW], dtype=np.str)
	buffer_npa = a_npa.view('S1').reshape(-1, 1)

	buffer_npa[HALF_WINDOW-1] = ' '
	buffer_npa[-HALF_WINDOW] = ENDING
	n_bytes = buffer_npa.size

	#patching-kernel
	block_x = BLOCK_DIM_1D
	grid_x = GRID_DIM_1D

	repeat = int(math.ceil(float(n_bytes) /grid_x/block_x))
	while (repeat > GLOBAL_REPEAT): 
		grid_x *= 2;
		block_x *= 2;
		repeat = int(math.ceil(float(n_bytes) /grid_x/block_x))

	if verbose > 1:
		print("patching-kernel   - grid_x:%d, block_x:%d, repeat:%d, n:%d"%\
			(grid_x, block_x, repeat, n_bytes))

	para_npa = np.array([n_bytes,repeat], dtype=np.int32)
	out_npa = np.empty([buffer_npa.size*3,1], dtype=np.str)

	if verbose:
		print "1load data time   - %f"%(time.time()-t1) ; t1 = time.time()

	module = patchingModule()
	module(drv.In(buffer_npa), drv.Out(out_npa), drv.In(para_npa), grid=(grid_x,1), block=(block_x,1,1) )

	if verbose:
		print "2patching time    - %f"%(time.time()-t1) ; t1 = time.time()

	patched = out_npa.reshape(-1,out_npa.size)
	patched = patched.view('S'+str(patched.size))[0][0]

	if verbose:
		print "3reshap time      - %f"%(time.time()-t1) ; t1 = time.time()

#	patched = patched.replace(PATCHING,'').replace(DOUBLE1,'` ').replace(DOUBLE2,"' ").replace(ENDING,'')
	patched = patched.translate(None, PATCHING + ENDING)
	patched = patched.replace(DOUBLE1,'` ').replace(DOUBLE2,"' ").replace("..."," ... ").replace("--"," -- ")

	if verbose:
		print "4replace time     - %f"%(time.time()-t1) ; t1 = time.time()

	tokens = patched.split()

	if verbose:
		print "5splitting time   - %f"%(time.time()-t1) ; t1 = time.time()

	return tokens


#for testing
#from Onescan import oneScanTokenizer
from nltk.tokenize import TreebankWordTokenizer

if __name__ == "__main__":
	s = ", _in_-4*\".--\"" #what I cannot acannotb acannot cannot/ caneeenot. gotta\t #'tis *cannot cannot9 cannot."
	print ' '.join(TreebankWordTokenizer().tokenize(s))
#	print ' '.join(oneScanTokenizer(s))
	print ' '.join(gpuTokenize(s))
	#what I can not acannotb acannot can not / caneeenot. got ta # 't is * can not cannot9 can not .
	sys.exit(0)
	s = '"<> denotes" I\'ve an in-equa...ti--on\'s ... aa("not equal#$ to AT@T"). '
	print ' '.join(TreebankWordTokenizer().tokenize(s))
#	print ' '.join(oneScanTokenizer(s))
	print ' '.join(gpuTokenize(s))
	#`` < > denotes '' I 've an in-equa ... ti -- on 's ... aa ( `` not equal # $ to AT @ T '' ) .

	s = '''Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.'''
	print TreebankWordTokenizer().tokenize(s)
#	print oneScanTokenizer(s)
	print gpuTokenize(s)
	#['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks', '.']

	s = "They'll save and invest more."
	print TreebankWordTokenizer().tokenize(s)
#	print oneScanTokenizer(s)
	print gpuTokenize(s)
	#['They', "'ll", 'save', 'and', 'invest', 'more', '.']

	s = "hi, my name can't%thello,"
	print TreebankWordTokenizer().tokenize(s)
#	print oneScanTokenizer(s)
	print gpuTokenize(s)
	#['hi', ',', 'my', 'name', 'ca', "n't", 'hello', ',']

