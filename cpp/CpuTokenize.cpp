//tokenizer.h
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <dirent.h>
#include <math.h>

#include "CpuTokenize.h"
#include "Tok.h"

char patching='\1';

char d1='\2';

char d2='\3';

int half_window= 7;

int full_window= 14;

char QM= '\'';

//lst are sorted for binary search
char *wspace_lst="\t\n\v\f\r ";	//6

char *b_lst = "\t\n\f\r !\"#$%&'()*+,-./:;<=>?@[\\]^`{|}~"; //36

char *changed_lst = " !#$%&(),:;<>?@[]{}"; //19

char *not_changed_lst = "*+-./=\\^`|~";  //11

char tolower_cpu(char c) {
	if 	(c >= 'A' && c <= 'Z')
		c += 32;
	return c;
}

//binary_search(): return index +1, to indicate 0 is not found. to get the index = 'return' -1
int inAt(char *arr, int len2, char val) {
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
}//end of inAt()


int isBackChanged_cpu(char* window, int p)  {
	if (inAt(changed_lst, 19, window[p]) 
		|| (window[p] == '.' && window[p+1] == '.' && window[p+2] == '.') 
		|| (window[p] == '-' && window[p+1] == '-') 				
		|| window[p] == '"')
		return 1;
	else
		return 0;
}

int isFrontChanged_cpu(char* window, int p)  {
	if (inAt(changed_lst, 19, window[p]) 
		|| (window[p] == '.' && window[p-1] == '.' && window[p-2] == '.') 
		|| (window[p] == '-' && window[p-1] == '-') 				
		|| window[p] == '"')
		return 1;
	else
		return 0;
}

//apply ptb rules
void rules(char *c, char *l, char *r, char *window)
{
	int i = half_window;
	char c_lower = tolower_cpu(*c);

	if (c_lower == '`') {
		if (window[i-1] != '`' && window[i+1] == '`') {
			*l = ' ';
			*r = patching;
		}
		else if (window[i-1] == '`' && window[i+1] != '`') {
			*l = patching;
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
			*r = patching;
		}
		else if (*c == '.' && window[i-1] == '.' && window[i-2] == '.' && window[i-3] != '.') {
			*l = patching;
			*r = ' ';	
		}
		else if (*c == '.' && window[i-1] == '.' && window[i-2] == '.' && window[i-3] == '.'
			&& window[i-4] == '.' && window[i-5] == '.' && window[i-6] != '.') {
			*l = patching;
			*r = ' ';	
		}
		else if (*c == '.' && window[i+1] == '\0') {
			*l = ' ';
			*r = patching;
		}
		else if ((window[i+1] == 'c' && window[i+2] == 'a' && window[i+3] == 'n' && window[i+4] == 'n' && window[i+5] == 'o' && window[i+6] == 't')
			|| (window[i+1] == 'd' && window[i+2] == QM && window[i+3] == 'y' && window[i+4] == 'e' && inAt(b_lst, 11, window[i+5]))
			|| (window[i+1] == 'g' && window[i+2] == 'i' && window[i+3] == 'm' && window[i+4] == 'm' && window[i+5] == 'e' && inAt(b_lst, 11, window[i+6]))
			|| (window[i+1] == 'g' && window[i+2] == 'o' && window[i+3] == 'n' && window[i+4] == 'n' && window[i+5] == 'a' && inAt(b_lst, 11, window[i+6]))
			|| (window[i+1] == 'g' && window[i+2] == 'o' && window[i+3] == 't' && window[i+4] == 't' && window[i+5] == 'a' && inAt(b_lst, 11, window[i+6]))
			|| (window[i+1] == 'l' && window[i+2] == 'e' && window[i+3] == 'm' && window[i+4] == 'm' && window[i+5] == 'e' && inAt(b_lst, 11, window[i+6]))
			|| (window[i+1] == 'm' && window[i+2] == 'o' && window[i+3] == 'r' && window[i+4] == QM && window[i+5] == 'n' && inAt(b_lst, 11, window[i+6]))
			|| (window[i+1] == 'w' && window[i+2] == 'a' && window[i+3] == 'n' && window[i+4] == 'n' && window[i+5] == 'a' && inAt(b_lst, 11, window[i+6])))  {
			*l = patching;
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
			*r = patching;
		}

		else if (*c == '-') { //(*c != '*')  every "--"
			if (window[i-1] != '-' && window[i+1] == '-') {
				*l = ' ';
				*r = patching;
			}
			else if (window[i-2] != '-' && window[i-1] == '-') {
				*l = patching;
				*r = ' ';
			}
			else if (window[i-4] != '-' && window[i-3] == '-'
				&& window[i-2] == '-' && window[i-1] == '-') {
				*l = patching;
				*r = ' ';
			}
			else if (window[i-6] != '-' && window[i-5] == '-'
				&& window[i-4] == '-' && window[i-3] == '-'
				&& window[i-2] == '-' && window[i-1] == '-') {
				*l = patching;
				*r = ' ';
			}
		}

		else {
			*l = patching;
			*r = patching;
		}
	}
	else if (*c == QM) {
		if (window[i-1]!= QM && window[i+1] == QM) {
			*l = ' ';
			*r = patching;
		}
		else if (window[i-1] == QM) {
			*l = patching;
			*r = ' ';
		}
		else if ( window[i+1] == '"' || (window[i+1] == '-' && window[i+2] == '-') || (window[i+1] == '.' && window[i+2] == '.' && window[i+3] == '.')) {
			*l = ' ';
			*r = ' ';
		}
		else if ( window[i-1] != QM) {
			if ( inAt("DMSdms", 6, window[i+1])) {
				if (isBackChanged_cpu(window, i+2)) {
					*l = ' ';
					*r = patching;
				}
			}
			else if ( window[i+1] == 'l' && window[i+2] == 'l') {
				if (isBackChanged_cpu(window, i+3))  {
					*l = ' ';
					*r = patching;
				}
			}
			else if ( window[i+1] == 'L' && window[i+2] == 'L') {
				if (isBackChanged_cpu(window, i+3))  {
					*l = ' ';
					*r = patching;
				}
			}
			else if ( window[i+1] == 'r' && window[i+2] == 'e') {
				if (isBackChanged_cpu(window, i+3))  {
					*l = ' ';
					*r = patching;
				}
			}
			else if ( window[i+1] == 'R' && window[i+2] == 'E' && isBackChanged_cpu(window, i+3) )  {
				*l = ' ';
				*r = patching;
			}
			else if ( window[i+1] == 'v' && window[i+2] == 'e' && isBackChanged_cpu(window, i+3) )  {
				*l = ' ';
				*r = patching;
			}
			else if ( window[i+1] == 'V' && window[i+2] == 'E') {
				if (isBackChanged_cpu(window, i+3))  {
					*l = ' ';
					*r = patching;
				}
			}
			else if ( inAt(b_lst, 36, window[i-2]) && (window[i-1] == 'd' || window[i-1] == 'D')
				&& window[i+1] == 'y' && window[i+2] == 'e' && inAt(b_lst, 36, window[i+3]))  {
				*l = ' ';
				*r = patching;
			}
			else if ( inAt(b_lst, 36, window[i-4]) && window[i-3] == 'm' 
				&& window[i-2] == 'o' && window[i-1] == 'r' && window[i+1] == 'n') { 
				if (inAt(b_lst, 36, window[i+2]))  {
					*l = ' ';
					*r = patching;
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
		if (isFrontChanged_cpu(window, i-2) && window[i-1] == QM && c_lower == 't')  {  //constrations3
			if (( window[i+1] == 'i' && window[i+2] == 's' && inAt(b_lst, 36, window[i+3])) 
				|| ( window[i+1] == 'w' && window[i+2] == 'a' && window[i+3] == 's' && inAt(b_lst, 36, window[i+4])) )  {
				*l = patching;
				*r = ' ';
			}
		}
		else if ( window[i-1] != QM && c_lower == 'n' && window[i+1] == QM && window[i+2] == 't') { // || window[i+2] == 'T')) {
			if (isBackChanged_cpu(window, i+3))  {
				*l = ' ';
				*r = patching;
			}
		}
		else if ( window[i-2] == 'n' && c_lower == 't') {  //|| window[i-2] == 'N') && window[i-1] == QM && c_lower == 't') { 
			if (isBackChanged_cpu(window, i+1))  {
				*l = patching;
				*r = ' ';
			}
		}
		else if (inAt(b_lst, 36, window[i-4])) { //break in middle of contrations2
			if ( window[i-3] == 'c' && window[i-2] == 'a' && window[i-1] == 'n'
				&& c_lower == 'n' && window[i+1] == 'o' && window[i+2] == 't') { 
				if (inAt(b_lst, 36, window[i+3]))  {
					*l = ' ';
					*r = patching;
				}
			} 
			if ( window[i-3] == 'g' && window[i-2] == 'i' && window[i-1] == 'm'
				&& c_lower == 'm' && window[i+1] == 'e') {
				if (inAt(b_lst, 36, window[i+2]))  {
					*l = ' ';
					*r = patching;
				}
			}
			else if ( window[i-3] == 'g' && window[i-2] == 'o' && window[i-1] == 'n'
				&& c_lower == 'n' && window[i+1] == 'a') {
				if (inAt(b_lst, 36, window[i+2]))  {
					*l = ' ';
					*r = patching;
				}
			}
			else if ( window[i-3] == 'g' && window[i-2] == 'o' && window[i-1] == 't'
				&& c_lower == 't' && window[i+1] == 'a') {
				if (inAt(b_lst, 36, window[i+2]))  {
					*l = ' ';
					*r = patching;
				}
			}
			else if ( window[i-3] == 'l' && window[i-2] == 'e' && window[i-1] == 'm'
				&& c_lower == 'm' && window[i+1] == 'e') {
				if (inAt(b_lst, 36, window[i+2]))  {
					*l = ' ';
					*r = patching;
				}
			}
			else if ( window[i-3] == 'l' && window[i-2] == 'e' && window[i-1] == 'm'
				&& c_lower == 'm' && window[i+1] == 'e') {
				if (inAt(b_lst, 36, window[i+2]))  {
					*l = ' ';
					*r = patching;
				}
			}
			else if ( window[i-3] == 'w' && window[i-2] == 'a' && window[i-1] == 'n'
				&& c_lower == 'n' && window[i+1] == 'a') {
				if (inAt(changed_lst, 19, window[i+2]))  {
					*l = ' ';
					*r = patching;
				}
			}
		}
	}
	return;
}//end of rules()


//patching l and r for each char
char* patching_cpu(char *cpu_a, int *n_bytes)
{
	int n = *n_bytes;
	char *buffer = (char*) malloc(n + full_window);
	memset(buffer, patching, half_window);		//done in cuda kernel
	memset(buffer + n+ half_window, patching, half_window);
	memcpy(buffer + half_window, cpu_a, n);	
	buffer[half_window - 1] = ' ';
	buffer[n+half_window] = '\0';
	n += full_window;

	char *patched = (char*) malloc(3 * n);
	memset(patched, patching, 3 * n);

	int i, j;
	char window[full_window];
	char l, c, r;

	for (int i = half_window; i < n - 3; i ++)
	{	
		j = i*3;	

		c = buffer[i];
		patched[j] = patching;
		patched[j+1] = c;
		patched[j+2] = patching;
		memcpy(window, buffer+i-half_window, full_window);

		if (!inAt(wspace_lst, strlen(wspace_lst), c)) 
		{
			l = patching;
			r = patching;
			for (int i = 0; i < full_window; i++)
				window[i] = tolower_cpu(window[i]);

			rules(&c, &l, &r, window);
			patched[j] = l;
			patched[j+1] = c;
			patched[j+2] = r;			
		}
		else {
			patched[j+1] = ' ';
		}
	}
	
	free (buffer);
	*n_bytes = n;

	return patched;
}//patching()


//remove, duplicate and split
int splitAndSaveToFile_cpu(char *str, int n_bytes, char **tokens, int n_toks, char *output_file) {
	int i, j, k, start, w_len;
	char buff[1000];
	char *s, *wk;

	wk = s = strdup(str);

	FILE *fw;
	fw= fopen(output_file, "wb");

	int len= (n_bytes - half_window)*3;
	i = half_window*3;	//old
	j = 0;			//new
	k = 0;			//tokens

	start = j;
	while (i < len) {
		if (s[i] == patching){
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
		} else if (s[i] == d1 || s[i] == d2 ){
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
int split_cpu(char *str, int n_bytes, char **tokens, int n_toks) {
	int i, j, k, start, w_len;
	char buff[1000];
	char *s, *wk;

	wk = s = strdup(str);

	int len= (n_bytes - half_window)*3;
	i = half_window*3;	//old
	j = 0;			//new
	k = 0;			//tokens

	start = j;
	while (i < len) {
		if (s[i] == patching){
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
		} else if (s[i] == d1 || s[i] == d2 ){
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
int cpuTokenizer(char *buffer, int n_bytes, char **tokens, int n_toks, char *outfile)
{
	double start_time = now();

	//1) patched (3*n_bytes) for l, c and r for each char
	//return 'patched' with ' 's according to PTB's rules
	char *patched = patching_cpu(buffer, &n_bytes);

	if (verbose > 1)
		printf("patching(cpu) time - %f\n",now() - start_time);  start_time = now();

	//2) remove patching and split tokens
	int n_tokens;
	if (strlen(outfile) > 0) {
		n_tokens = splitAndSaveToFile_cpu(patched, n_bytes, tokens, n_toks, outfile);
		if (verbose > 1) {
			printf("split time         - %f\n",now() - start_time);  start_time = now();
			printf ("\nsave to: %s\n",outfile);
		}
	}
	else {
		n_tokens = split_cpu(patched, n_bytes, tokens, n_toks);
		if (verbose > 1)
			printf("split time         - %f\n",now() - start_time);  start_time = now();
	}


	free (patched);

	return n_tokens;
}//end of gpu




