//Tok.cpp
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

#include "Tok.h"

//timer
double now()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec / 1000000.;
}//end of timer()


//get file names from a folder
int getFileName(int argc, char* argv[], char **fname)
{
	int i, n;
	char *name = (char *)malloc(FNAME_MAX_LEN * sizeof(char));
	if ( argc < 2 ) /* argc should be 2 for correct execution */
	{
		printf( "Usage: provide a .txt file or path to data files\n");
		exit(EXIT_SUCCESS);
	}
	else 
	{
		struct stat statbuf;

		stat(argv[1], &statbuf);

		if(S_ISDIR(statbuf.st_mode))
		{// is directory			
			DIR *dp;
			struct dirent *ep;     
			dp = opendir (argv[1]);

			if (dp != NULL)
			{
				i = 0;
				while (ep = readdir (dp))
				{
					name = ep->d_name;
					if (name[0] != '.')
					{
						n = sprintf (fname[i],"%s/%s",argv[1], name);
						i += 1;
					}
				}
			}
			(void) closedir (dp);		
		}
		else
		{// is a file
			i = 1;
			fname[0] = argv[1];
		}			
	}

	return i;
}//end of getfilename()


//get data from files
int getSize(char **fname, int n_fname, int *n_size)
{
	struct stat file_stat;

	int n_bytes = 0;
	for (int i = 0; i < n_fname; i++)
	{
		stat(fname[i], &file_stat);
		n_size[i] = file_stat.st_size;
		n_bytes += file_stat.st_size;
	}

	return n_bytes;
}//end of getData()




