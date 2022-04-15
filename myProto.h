#ifndef __MYPROTO_H
#define __MYPROTO_H


// includes

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <math.h>

             
// constants

#define INPUT_FILE_NAME 	"input.txt"
#define OUTPUT_FILE_NAME 	"output.txt"
#define MAXIMUM_STR 		"maximum"
#define MINIMUM_STR 		"minimum"

const char HYPHEN = '-';

const float ERROR_VAL = 0.001;

const int FOUND = -20;
const int ERROR = -10;
const int MASTER = 0;
const int SLAVE = 1; 
const int FAILED = 0;
const int SUCCEEDED = 1;
const int MINIMUM = 10;
const int MAXIMUM = 20;
const int NUM_OF_WEIGHTS = 4;
const int CONS_GROUP_SIZE = 9;
const int SEMI_CONS_GROUP_SIZE = 11;
const int ABC_SIZE = 26;
const int MAX_LEN_SEQ1 = 10000;
const int MAX_LEN_SEQ2 = 5000;
const int BUFFER_SIZE = 16384;

typedef enum {STAR, COLON, POINT, SPACE, NUM_OF_SIGNS} sign;


// functions

int computeOnGPU
    (const char* seq1, const char* seq2, char* best_seq, int len1, int len2, int from, int to, int* cons_mat, int* semi_cons_mat, double* W, double* best_score, int* best_offset, int goal);

double  calcAlignmentScore(const char* seq1, const char* seq2, int length, int* cons_mat, int* semi_cons_mat, double* W);
void 	convertGroupToMatrix(int matrix[ABC_SIZE * ABC_SIZE], const char** group, int size);
double  getWeight(char c1, char c2, int* cons_mat, int* semi_cons_mat, double* W);
int 	isWordConatinsLetter(char c, const char* word, int length);
void 	printDetails(double W[NUM_OF_WEIGHTS], const char* seq1, const char* seq2, int goal);
int 	readDataFromTextFile(double* W, char* seq1, char* seq2, int* goal, const char* file_name);
int     test(char* best_seq, const char* seq1, int length, int* cons_mat, int* semi_cons_mat, double* W, double best_score, int best_offset);
void    testSequential(const char* seq1, const char* seq2, int len1, int len2, int* cons_mat, int* semi_cons_mat, double* W, int goal, char* test_best_seq, int* test_best_offset, double* test_best_score);
int 	writeResultsToTextFile(char* seq, int best_offset, double best_score, const char* file_name);


#endif // __MYPROTO_H
