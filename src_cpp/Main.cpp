//Main.cpp
// ./tok /home/lca80/Desktop/data/300mb, outfile.txt
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h> 
#include <cstring> 
#include <string>
#include <dirent.h>
#include <math.h>
#include "CpuTokenize.h"
#include "GpuTokenize.h"
#include "Tok.h"

using namespace std;

int verbose = 1;	//default 1: print 50 tokens and time, 2: all, 0: nothing
int cmd = 1;		//default 1: gpu, 2: cpu, -1/-2: save output to "output_cpp" folder

double tokenize(char *buffer, int n_bytes, char *outfile, int *n_tokens_ptr) {
	double total_time = now();
	double start_time = now();

	//2) get tokens
	char **tokens;
//	int n_toks = n_bytes/3;	//for store tokens, take time to allocate
	int n_toks = 1000;	//only keep 1000 for display. tokens are save to file
	tokens = (char **)malloc(n_toks * sizeof(char *));
	for (int i=0; i<n_toks; i++)
		tokens[i] = (char *)malloc(WORD_MAX_LEN * sizeof(char));

	if (verbose)
		printf("\ntokens malloc time - %f\n",now() - start_time);  start_time = now();
	
	int n_tokens;
	if (cmd == 2)
		n_tokens = cpuTokenizer(buffer, n_bytes, tokens, n_toks, outfile);
	else
		n_tokens = gpuTokenizer(buffer, n_bytes, tokens, n_toks, outfile);

	//3) print tokens
	if (verbose)  {
		printf ("\ntokens(first 50):\n");
		int n_print = 50;
		if (n_print > n_tokens)
			n_print = n_tokens;
		for (int i = 0; i < n_print - 1; i ++) {
			printf ("%s, ",tokens[i]);
		}
		printf ("%s .......... \n\n",tokens[n_print - 1]);

		total_time = now() - total_time;

		printf("sub-total time: %.3f secs to tokenize %db, %.0f kb/s (tokens:%d).\n", 
			total_time, n_bytes, n_bytes/1024/total_time, n_tokens);
	}

	*n_tokens_ptr = n_tokens;

	for (int i = 0; i < n_toks; i++ )
        	free (tokens[i] );
	free (tokens);

	return total_time;
}

//THE MAIN	
int main (int argc, char *argv[])
{
	double total_time = now();
	double start_time = now();
	double t_read = 0.0;
	double t_tokenize = 0.0;

	char outfile[100];
	outfile[0] = '\0';
	if (argc > 2) {
		cmd = atoi(argv[2]);
		if (cmd < 0) {
			strcpy(outfile, "output_cpp/part-0000");
			system("rm -r output_cpp; mkdir output_cpp");
		}
		cmd = abs(cmd);
	}
		
	//1) get data
	int n_size[500];	//assume # of files < 1000
	char **fname = (char **)malloc(500 * sizeof(char *));
	for (int i=0; i < 500; i++)
		fname[i] = (char *)malloc(FNAME_MAX_LEN * sizeof(char));

	//get all datafile names 
	int n_fname = getFileName(argc, argv, fname);
	int total_bytes = getSize(fname, n_fname, n_size);
	if (verbose)  {
		for(int i = 0; i < n_fname; i++)  {
			printf("%d/%d %s(%d)\n",i+1, n_fname, fname[i], n_size[i]);
		} 
		printf("n_files:%d-----\n",n_fname);
	}

	//get size of files
	FILE *fp;
	char *buffer = (char*) malloc(MEM_MAX);
	int total_tokens = 0;
	int n = 0;

	int n_bytes = 0;
	int n_next = 0;
	int n_tokens = 0;
	int i_outfile = 0;

	//2) tokenize
	for (int i = 0; i < n_fname; i ++) { 
		n_next += n_size[i];
		//limit MEM_MAX going to gpu
		if (n_next < MEM_MAX) {
			fp = fopen(fname[i], "rb");
			fread(buffer + n_bytes, n_size[i], 1, fp);
			n_bytes += n_size[i];
			fclose(fp);
		}
		else {
			t_read += now() - start_time;
			if (strlen(outfile) > 0) {
				sprintf(outfile, "%s%d", outfile, i_outfile);
				i_outfile += 1;
			}
			t_tokenize += tokenize(buffer, n_bytes, outfile, &n_tokens);
			start_time = now();

			total_tokens += n_tokens;
			n += n_bytes;
			n_bytes = 0;
			n_next = 0;
			fp = fopen(fname[i], "rb");
			fread(buffer + n_bytes, n_size[i], 1, fp);
			n_bytes += n_size[i];
			n_next = n_bytes;
			fclose(fp);
		}
	}

	if (n < total_bytes) {
		t_read += now() - start_time;
		if (strlen(outfile) > 0) {
			sprintf(outfile, "%s%d", outfile, i_outfile);
			i_outfile += 1;
		}
		t_tokenize += tokenize(buffer, n_bytes, outfile, &n_tokens);
		total_tokens += n_tokens;
		n += n_bytes;
	}

	//3) print time
	if (verbose)  {
		printf("bytes:%dKb\n", total_bytes);   ///1024);
		printf("getdata time       : %f\n",t_read);
		printf("tokenize time      : %f\n",t_tokenize);
		total_time = now() - total_time;
		printf("time: %.3f secs to tokenize %db, %.0f kb/s (tokens:%d).\n", 
			total_time, n_bytes, n_bytes/1024/total_time, n_tokens);
	}

	free (buffer);
	free (fname);

	return 0;
}//end of main()


