//Tok.h
#ifndef __TOK_H__
#define __TOK_H__

#define MEM_MAX 320000000  //520000000 //
#define FNAME_MAX_LEN 1000
#define WORD_MAX_LEN 200

extern int verbose;

double now();
int getFileName(int argc, char* argv[], char **fname);
int getSize(char **fname, int n_fname, int *n_size);

#endif
