//GpuTokenize.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "GpuTokenize.h"
#include "Tok.h"	/* verbose */
#include <math.h>       /* ceil */

#define PADDING '\1'
#define DOUBLE1 '\2'
#define DOUBLE2 '\3'
#define HALF_WINDOW 7
#define FULL_WINDOW 14

//padding-kernel
#define BLOCK_DIM_1D 128
#define GRID_DIM_1D 4096
#define GLOBAL_REPEAT 50

__device__ char padding='\1';

__device__ char d1='\2';

__device__ char d2='\3';

__device__ const int half_window= 7;

__device__ const int full_window= 14;

__device__ char QM= '\'';

//lst are sorted for binary search
__device__ const char *wspace_lst="\t\n\v\f\r ";	//6

__device__ const char *b_lst = "\t\n\f\r !\"#$%&'()*+,-./:;<=>?@[\\]^`{|}~"; //36

__device__ const char *changed_lst = " !#$%&(),:;<>?@[]{}"; //19

__device__ const char *not_changed_lst = "*+-./=\\^`|~";  //11

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
	int i = half_window;
	char c_lower = tolower(*c);

	if (c_lower == '`') {
		if (window[i-1] != '`' && window[i+1] == '`') {
			*l = ' ';
			*r = padding;
		}
		else if (window[i-1] == '`' && window[i+1] != '`') {
			*l = padding;
			*r = ' ';
		}
	}
	else if (c_lower == '"') {
		if (inAt(" (<[`{", 6, window[i-1])) {
			*l = ' ';
			*c = '`';
			*r = d1;	//double in replace
		}
		else {
			*l = ' ';
			*c = QM;
			*r = d2;	//double in replace
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
			*r = padding;
		}
		else if (*c == '.' && window[i-1] == '.' && window[i-2] == '.' && window[i-3] != '.') {
			*l = padding;
			*r = ' ';	
		}
		else if (*c == '.' && window[i-1] == '.' && window[i-2] == '.' && window[i-3] == '.'
			&& window[i-4] == '.' && window[i-5] == '.' && window[i-6] != '.') {
			*l = padding;
			*r = ' ';	
		}
		else if (*c == '.' && window[i+1] == '\0') {
			*l = ' ';
			*r = padding;
		}
		else if ((window[i+1] == 'c' && window[i+2] == 'a' && window[i+3] == 'n' && window[i+4] == 'n' && window[i+5] == 'o' && window[i+6] == 't')
			|| (window[i+1] == 'd' && window[i+2] == QM && window[i+3] == 'y' && window[i+4] == 'e' && inAt(b_lst, 11, window[i+5]))
			|| (window[i+1] == 'g' && window[i+2] == 'i' && window[i+3] == 'm' && window[i+4] == 'm' && window[i+5] == 'e' && inAt(b_lst, 11, window[i+6]))
			|| (window[i+1] == 'g' && window[i+2] == 'o' && window[i+3] == 'n' && window[i+4] == 'n' && window[i+5] == 'a' && inAt(b_lst, 11, window[i+6]))
			|| (window[i+1] == 'g' && window[i+2] == 'o' && window[i+3] == 't' && window[i+4] == 't' && window[i+5] == 'a' && inAt(b_lst, 11, window[i+6]))
			|| (window[i+1] == 'l' && window[i+2] == 'e' && window[i+3] == 'm' && window[i+4] == 'm' && window[i+5] == 'e' && inAt(b_lst, 11, window[i+6]))
			|| (window[i+1] == 'm' && window[i+2] == 'o' && window[i+3] == 'r' && window[i+4] == QM && window[i+5] == 'n' && inAt(b_lst, 11, window[i+6]))
			|| (window[i+1] == 'w' && window[i+2] == 'a' && window[i+3] == 'n' && window[i+4] == 'n' && window[i+5] == 'a' && inAt(b_lst, 11, window[i+6])))  {
			*l = padding;
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
			*r = padding;
		}

		else if (*c == '-') { //(*c != '*')  every "--"
			if (window[i-1] != '-' && window[i+1] == '-') {
				*l = ' ';
				*r = padding;
			}
			else if (window[i-2] != '-' && window[i-1] == '-') {
				*l = padding;
				*r = ' ';
			}
			else if (window[i-4] != '-' && window[i-3] == '-'
				&& window[i-2] == '-' && window[i-1] == '-') {
				*l = padding;
				*r = ' ';
			}
			else if (window[i-6] != '-' && window[i-5] == '-'
				&& window[i-4] == '-' && window[i-3] == '-'
				&& window[i-2] == '-' && window[i-1] == '-') {
				*l = padding;
				*r = ' ';
			}
		}

		else {
			*l = padding;
			*r = padding;
		}
	}
	else if (*c == QM) {
		if (window[i-1]!= QM && window[i+1] == QM) {
			*l = ' ';
			*r = padding;
		}
		else if (window[i-1] == QM) {
			*l = padding;
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
					*r = padding;
				}
			}
			else if ( window[i+1] == 'l' && window[i+2] == 'l') {
				if (isBackChanged(window, i+3))  {
					*l = ' ';
					*r = padding;
				}
			}
			else if ( window[i+1] == 'L' && window[i+2] == 'L') {
				if (isBackChanged(window, i+3))  {
					*l = ' ';
					*r = padding;
				}
			}
			else if ( window[i+1] == 'r' && window[i+2] == 'e') {
				if (isBackChanged(window, i+3))  {
					*l = ' ';
					*r = padding;
				}
			}
			else if ( window[i+1] == 'R' && window[i+2] == 'E' && isBackChanged(window, i+3) )  {
				*l = ' ';
				*r = padding;
			}
			else if ( window[i+1] == 'v' && window[i+2] == 'e' && isBackChanged(window, i+3) )  {
				*l = ' ';
				*r = padding;
			}
			else if ( window[i+1] == 'V' && window[i+2] == 'E') {
				if (isBackChanged(window, i+3))  {
					*l = ' ';
					*r = padding;
				}
			}
			else if ( inAt(b_lst, 36, window[i-2]) && (window[i-1] == 'd' || window[i-1] == 'D')
				&& window[i+1] == 'y' && window[i+2] == 'e' && inAt(b_lst, 36, window[i+3]))  {
				*l = ' ';
				*r = padding;
			}
			else if ( inAt(b_lst, 36, window[i-4]) && window[i-3] == 'm' 
				&& window[i-2] == 'o' && window[i-1] == 'r' && window[i+1] == 'n') { 
				if (inAt(b_lst, 36, window[i+2]))  {
					*l = ' ';
					*r = padding;
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
				*l = padding;
				*r = ' ';
			}
		}
		else if ( window[i-1] != QM && c_lower == 'n' && window[i+1] == QM && window[i+2] == 't') { // || window[i+2] == 'T')) {
			if (isBackChanged(window, i+3))  {
				*l = ' ';
				*r = padding;
			}
		}
		else if ( window[i-2] == 'n' && c_lower == 't') {  //|| window[i-2] == 'N') && window[i-1] == QM && c_lower == 't') { 
			if (isBackChanged(window, i+1))  {
				*l = padding;
				*r = ' ';
			}
		}
		else if (inAt(b_lst, 36, window[i-4])) { //break in middle of contrations2
			if ( window[i-3] == 'c' && window[i-2] == 'a' && window[i-1] == 'n'
				&& c_lower == 'n' && window[i+1] == 'o' && window[i+2] == 't') { 
				if (inAt(b_lst, 36, window[i+3]))  {
					*l = ' ';
					*r = padding;
				}
			} 
			if ( window[i-3] == 'g' && window[i-2] == 'i' && window[i-1] == 'm'
				&& c_lower == 'm' && window[i+1] == 'e') {
				if (inAt(b_lst, 36, window[i+2]))  {
					*l = ' ';
					*r = padding;
				}
			}
			else if ( window[i-3] == 'g' && window[i-2] == 'o' && window[i-1] == 'n'
				&& c_lower == 'n' && window[i+1] == 'a') {
				if (inAt(b_lst, 36, window[i+2]))  {
					*l = ' ';
					*r = padding;
				}
			}
			else if ( window[i-3] == 'g' && window[i-2] == 'o' && window[i-1] == 't'
				&& c_lower == 't' && window[i+1] == 'a') {
				if (inAt(b_lst, 36, window[i+2]))  {
					*l = ' ';
					*r = padding;
				}
			}
			else if ( window[i-3] == 'l' && window[i-2] == 'e' && window[i-1] == 'm'
				&& c_lower == 'm' && window[i+1] == 'e') {
				if (inAt(b_lst, 36, window[i+2]))  {
					*l = ' ';
					*r = padding;
				}
			}
			else if ( window[i-3] == 'l' && window[i-2] == 'e' && window[i-1] == 'm'
				&& c_lower == 'm' && window[i+1] == 'e') {
				if (inAt(b_lst, 36, window[i+2]))  {
					*l = ' ';
					*r = padding;
				}
			}
			else if ( window[i-3] == 'w' && window[i-2] == 'a' && window[i-1] == 'n'
				&& c_lower == 'n' && window[i+1] == 'a') {
				if (inAt(changed_lst, 19, window[i+2]))  {
					*l = ' ';
					*r = padding;
				}
			}
		}
	}
	return;
}


__global__ void mykernel( char *buffer, char *padded, int *para)
{
	int repeat = para[0];
	int n = para[1];
	int tid = (blockDim.x*blockIdx.x + threadIdx.x); 		
	int x_start = tid*repeat;

	int j;
	char window[full_window];
	char l, c, r;
	for (int i=x_start; i < x_start+repeat; i++) {
		if (i < n) {
			j = i*3;	

			c = buffer[i];
			padded[j] = padding;
			padded[j+1] = c;
			padded[j+2] = padding;
			memcpy(window, buffer + i - half_window, 2*half_window);
			if (!inAt(wspace_lst, 6, c)) 
			{
				l = padding;
				r = padding;
				for (int i = 0; i < full_window; i++)
					window[i] = tolower(window[i]);

				rules_kernel(&c, &l, &r, window);
				padded[j] = l;
				padded[j+1] = c;
				padded[j+2] = r;			
			}
			else {
				padded[j+1] = ' ';
			}				
		}
	}//end for

}

char* padding_kernel(char *buffer, int *n_bytes)
{
	int n = *n_bytes;
	char *cpu_a = (char*) malloc(n + FULL_WINDOW);
	memset(cpu_a, PADDING, HALF_WINDOW);		//done in cuda kernel
	memset(cpu_a + n+ HALF_WINDOW, PADDING, HALF_WINDOW);
	memcpy(cpu_a + HALF_WINDOW, buffer, n);	
	cpu_a[HALF_WINDOW - 1] = ' ';
	cpu_a[n+HALF_WINDOW] = '\0';
	n += FULL_WINDOW;

	char *cpu_b = (char*) malloc(3 * n);
	memset(cpu_b, PADDING, 3 * n);

	char *gpu_a;	//with size n
	char *gpu_b;	//with size 3*n

	cudaMalloc( (void**)&gpu_a, n * sizeof(char) );
	cudaMalloc( (void**)&gpu_b, 3 * n * sizeof(char) );
	cudaMemcpy( gpu_a, cpu_a, sizeof(char) * n, cudaMemcpyHostToDevice);

	//set grid, block size and get repeats(# of strides), set para
	int block_x = BLOCK_DIM_1D;
	int grid_x = GRID_DIM_1D;

	int repeat = (int)ceil((float)n /(grid_x*block_x));
	while (repeat > GLOBAL_REPEAT) {
		grid_x *= 2;
		block_x *= 2;
		repeat = (int)ceil((float)n /(grid_x*block_x));
	}

	if (verbose > 1)
		printf("padding-kernel    - grid_x:%d, block_x:%d, repeat:%d, n:%d\n", grid_x, block_x, repeat, n);

	int cpu_para[3]={repeat, n, 0};
	int *gpu_para;

	cudaMalloc( (void**)&gpu_para, 3*sizeof(int));
	cudaMemcpy( gpu_para, cpu_para, 3*sizeof(int), cudaMemcpyHostToDevice);

	//call gpu
	dim3 gridDim(grid_x,1,1);
	dim3 blockDim(block_x,1,1);
	mykernel<<<gridDim, blockDim>>>(gpu_a, gpu_b, gpu_para);

	//get result
	cudaMemcpy( cpu_b, gpu_b, sizeof(char) * 3 * n, cudaMemcpyDeviceToHost);

	//free
	cudaFree(gpu_para);
	cudaFree(gpu_a);
	cudaFree(gpu_b);
	free (cpu_a);

	*n_bytes = n;

	return cpu_b;
}


//remove, duplicate and split
int splitAndSaveToFile(char *str, int n_bytes, char **tokens, int n_toks, char *output_file) {
	int i, j, k, start, w_len;
	char buff[1000];
	char *s, *wk;

	wk = s = strdup(str);

	FILE *fw;
	fw= fopen(output_file, "wb");

	int len= (n_bytes - HALF_WINDOW)*3;
	i = HALF_WINDOW*3;	//old
	j = 0;			//new
	k = 0;			//tokens

	start = j;
	while (i < len) {
		if (s[i] == PADDING){
			i += 1;
		} else if (s[i] == ' ')  {
			w_len = j - start;
			if (w_len > 0) {
				memcpy(buff, str + start, w_len);
				buff[w_len] = '\0';
				fwrite(buff, strlen(buff), 1, fw);
				fwrite("\n", 1, 1, fw);
//				printf ("%s\n",buff);
				if (k < n_toks)
					sprintf(tokens[k],"%s",buff);
				k += 1;
				start += w_len + 1; 
			}
			else  {
				start += 1;
			}
			str[j] = s[i];
			i += 1;
			j += 1;
		} else if (s[i] == DOUBLE1 || s[i] == DOUBLE2 ){
			str[j] = s[i-1];
			str[j+1] = ' ';

			memcpy(buff, str + start, 2);
			buff[2] = '\0';
			fwrite(buff, strlen(buff), 1, fw);
			fwrite("\n", 1, 1, fw);
//			printf ("%s\n",buff);
			if (k < n_toks)
				sprintf(tokens[k],"%s",buff);

			k += 1;
			start += 2 + 1; 

			i += 1;
			j += 2;
		} else {
			str[j] = s[i];
			i += 1;
			j += 1;
		}
	}
	str[j] = '\0';

	fclose(fw);
	free(wk);

	return k;
}//end splitAndSaveToFile(()


//remove, duplicate and split
int split(char *str, int n_bytes, char **tokens, int n_toks) {
	int i, j, k, start, w_len;
	char buff[1000];
	char *s, *wk;

	wk = s = strdup(str);

	int len= (n_bytes - HALF_WINDOW)*3;
	i = HALF_WINDOW*3;	//old
	j = 0;			//new
	k = 0;			//tokens

	start = j;
	while (i < len) {
		if (s[i] == PADDING){
			i += 1;
		} else if (s[i] == ' ')  {
			w_len = j - start;
			if (w_len > 0) {
				memcpy(buff, str + start, w_len);
				buff[w_len] = '\0';

				if (k < n_toks)
					sprintf(tokens[k],"%s",buff);
				k += 1;
				start += w_len + 1; 
			}
			else  {
				start += 1;
			}
			str[j] = s[i];
			i += 1;
			j += 1;
		} else if (s[i] == DOUBLE1 || s[i] == DOUBLE2 ){
			str[j] = s[i-1];
			str[j+1] = ' ';

			memcpy(buff, str + start, 2);
			buff[2] = '\0';

			if (k < n_toks)
				sprintf(tokens[k],"%s",buff);
			k += 1;
			start += 2 + 1; 

			i += 1;
			j += 2;
		} else {
			str[j] = s[i];
			i += 1;
			j += 1;
		}
	}
	str[j] = '\0';

	free(wk);

	return k;
}//end split(()


//gpu tokenizing
int gpuTokenizer(char *buffer, int n_bytes, char **tokens, int n_toks, char *outfile)
{
	double start_time = now();

	//1) padded (3*n_bytes) for l, c and r for each char
	//return 'padded' with ' 's according to PTB's rules
	char *padded = padding_kernel(buffer, &n_bytes);

	if (verbose > 1)
		printf("padding(gpu) time - %f\n",now() - start_time);  start_time = now();

	//3) remove padding and split tokens
	int n_tokens;
	if (strlen(outfile) > 0) {
		n_tokens = splitAndSaveToFile(padded, n_bytes, tokens, n_toks, outfile);
		if (verbose > 1) {
			printf("split time         - %f\n",now() - start_time);  start_time = now();
			printf ("\nsave to - %s\n",outfile);
		}
	}
	else {
		n_tokens = split(padded, n_bytes, tokens, n_toks);
		if (verbose > 1)
			printf("split time         - %f\n",now() - start_time);  start_time = now();
	}


	free (padded);

	return n_tokens;
}//end of gpu

