#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"


// Error handling macro

#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        fprintf(stderr, "CUDA error calling \""#call"\", code is %d\n", err); \
        exit(EXIT_FAILURE); }
              


// Device helping unctions 


/*
CUDA_AreLettersExchangeable gets 2 characters (c and sub), and the conservative matrix
the function checks if the original character (c) is allowed to be substituted by another character (sub)
the substitute is allowed if there is no conservative group that contains both letters
*/
__device__ int CUDA_AreLettersExchangeable(char c, char sub, int* cons_mat)
{
	if(sub == HYPHEN)
		return 1;

	int pos1 = ABC_SIZE * (sub - 'A') + (c - 'A');
	int pos2 = ABC_SIZE * (c - 'A') + (sub - 'A');

	if(cons_mat[pos1] != 1 && cons_mat[pos2] != 1) // sub and c are not in the same conservative group
		return 1;
	else
		return 0;
}


/*
CUDA_IsBetterForGoal gets 2 numbers (d and best_d) and the goal of the program (MAX/MIN)
the function returns 1 if the number d is better for the program than the best_d
else - function returns 0
*/
__device__ int CUDA_IsBetterForGoal(double d, double best_d, int goal)
{
	if((goal == MAXIMUM && d > best_d) || (goal == MINIMUM && d < best_d))
		return 1;
	
	return 0;
}


/*
CUDA_GetLettersWeight gets 2 characters (c1 and c2), the conservative and semi conservative matrices and weights array
the function checks the relation between the 2 characters according to the matrices
and returns the matching weight for this pair of characters
*/
__device__ double CUDA_GetLettersWeight(char c1, char c2, int* cons_mat, int* semi_cons_mat, double* W)
{
	if(c1 == HYPHEN || c2 == HYPHEN)
	{
		if(c1 == c2)
			return W[STAR];
		else
			return -W[SPACE];
	}

	int pos1 = ABC_SIZE * (c1 - 'A') + (c2 - 'A');
	int pos2 = ABC_SIZE * (c2 - 'A') + (c1 - 'A');

	if(c1 == c2) // same letter
	{
		return W[STAR];
	}
	else
	{
		if(cons_mat[pos1] == 1 || cons_mat[pos2] == 1) // same conservative group
		{
			return -W[COLON];
		}
		else if(semi_cons_mat[pos1] == 1 || semi_cons_mat[pos2] == 1) // same semi-conservative group
		{
			return -W[POINT];
		}
		else
		{
			return -W[SPACE];
		}
	}
}


// Global functions

/*
this global function finds the best mutant for a specific offset between sequence #1 & sequnce #2
*/
__global__ void
	findBestMutant(const char* seq1, const char* seq2, char* best_seq, int length, int from, int best_offset, int* cons_mat, int* semi_cons_mat, double* W, int goal)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	
	// Check if current thread has a job to do
	if(index < length)
	{
		// Each thread take care of a different letter of sequence 2

		// current letters of seq1 & seq2 and best char for this pair
		char current_letter_seq1 = seq1[index + best_offset + from];
		char current_letter_seq2 = seq2[index];
		char best_char;

		// current & best weight for this letter in seq2
		double hyphen_weight = CUDA_GetLettersWeight(current_letter_seq1, HYPHEN, cons_mat, semi_cons_mat, W);
		double current_weight = CUDA_GetLettersWeight(current_letter_seq1, current_letter_seq2, cons_mat, semi_cons_mat, W);
		double best_weight;
		if(CUDA_IsBetterForGoal(current_weight, hyphen_weight, goal))
		{
			best_weight = current_weight;
			best_char = current_letter_seq2;
		}
		else
		{
			best_weight = hyphen_weight;
			best_char = HYPHEN;
		}
		
		for(char c = 'A'; c <= 'Z'; c++) // for each word in the ABC
		{
			if(c == current_letter_seq2)
				continue;

			if(CUDA_AreLettersExchangeable(current_letter_seq2, c, cons_mat))
			{
				// get current wight of letters between current_letter_seq1 & c
				current_weight = CUDA_GetLettersWeight(current_letter_seq1, c, cons_mat, semi_cons_mat, W);

				if(current_weight == best_weight)
					continue;

				if (CUDA_IsBetterForGoal(current_weight, best_weight, goal))
				{
					best_weight = current_weight;
					best_char = c;
				}
			}
		}

		best_seq[index] = best_char;
	}
}


/*
this global function fills the best-scores array
for each cell in the array - the function saves the best score that matches the goal of the program (MAX/MIN)
each cell represent a different offset between sequence #1 & sequence #2
*/
__global__  void
	fillScoreArr(const char* seq1, const char* seq2, int length, int from, int to, int* cons_mat, int* semi_cons_mat, double* W, double* best_score_arr, int goal)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
	int limit = to - from;

	// Check if current thread has a job to do
	if(index < limit)
	{	
		
		// Each thread take care of a different offset
		int my_offset = index + from;
		best_score_arr[index] = 0;

		for(int j = 0; j < length; j++) // for each letter in sequence #2
		{
			// current character of seq1 & seq2
			char current_letter_seq1 = seq1[my_offset + j];
			char current_letter_seq2 = seq2[j];

			// current & best weight for this round
			double hyphen_weight = CUDA_GetLettersWeight(current_letter_seq1, HYPHEN, cons_mat, semi_cons_mat, W);
			double current_weight = CUDA_GetLettersWeight(current_letter_seq1, current_letter_seq2, cons_mat, semi_cons_mat, W);
			double best_weight;
			if(CUDA_IsBetterForGoal(current_weight, hyphen_weight, goal))
				best_weight = current_weight;
			else
				best_weight = hyphen_weight;

			for(char c = 'A'; c <= 'Z'; c++) // for each word in the ABC
			{
				if(c == current_letter_seq2)
					continue;

				if(CUDA_AreLettersExchangeable(current_letter_seq2, c, cons_mat))
				{
					current_weight = CUDA_GetLettersWeight(current_letter_seq1, c, cons_mat, semi_cons_mat, W);
						
					if(current_weight == best_weight)
						continue;

					if (CUDA_IsBetterForGoal(current_weight, best_weight, goal))
					{
						best_weight = current_weight;
					}
				}
			}

			best_score_arr[index] += best_weight;
		}
	}
}


int computeOnGPU(const char* seq1, const char* seq2, char* best_seq, int len1, int len2, int from, int to, int* cons_mat, int* semi_cons_mat, double* W, double* best_score, int* best_offset, int goal) 
{
	int size = to - from; /* number of offsets to take care */

	
    // Allocate data on GPU memory
    char *d_seq1 = NULL, *d_seq2 = NULL, *d_best_seq = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_seq1, len1 * sizeof(char)));
    CUDA_CHECK(cudaMalloc((void **)&d_seq2, len2 * sizeof(char)));
	CUDA_CHECK(cudaMalloc((void **)&d_best_seq, len2 * sizeof(char)));
    
    int *d_cons_mat = NULL, *d_semi_cons_mat = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_cons_mat, ABC_SIZE * ABC_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_semi_cons_mat, ABC_SIZE * ABC_SIZE * sizeof(int)));
    
    double *d_W = NULL, *d_best_score_arr = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_W, NUM_OF_WEIGHTS * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_best_score_arr, size * sizeof(double)));
    
    
	// Copy to GPU memory
    CUDA_CHECK(cudaMemcpy(d_seq1, seq1, len1 * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq2, seq2, len2 * sizeof(char), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_best_seq, seq2, len2 * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cons_mat, cons_mat,  ABC_SIZE * ABC_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_semi_cons_mat, semi_cons_mat,  ABC_SIZE * ABC_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, W, NUM_OF_WEIGHTS * sizeof(double), cudaMemcpyHostToDevice));
	
	
	// Fill best scores array
	// Run Kernel
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    fillScoreArr<<<blocks_per_grid, threads_per_block>>>(d_seq1, d_seq2, len2, from, to, d_cons_mat, d_semi_cons_mat, d_W, d_best_score_arr, goal);
    CUDA_CHECK(cudaGetLastError());

	    
	// Copy data back to CPU memory
	double* best_score_arr = (double*)malloc(size * sizeof(double));
    CUDA_CHECK(cudaMemcpy(best_score_arr, d_best_score_arr, size * sizeof(double), cudaMemcpyDeviceToHost));
    

	// Find best Offset in array using OpenMp
	omp_set_num_threads(4);
	*best_offset = 0;
	int i;
	*best_score = best_score_arr[0];
#pragma omp parallel for
	for(i = 1; i < size; i++)
	{
		if ((goal == MAXIMUM && *best_score < best_score_arr[i]) || (goal == MINIMUM && *best_score > best_score_arr[i]))
		{
#pragma omp critical
		{
			*best_offset = i;
			*best_score = best_score_arr[i];	
		}
		}
	}  
	

	// Find best mutant of best offset
	// Run Kernel
    threads_per_block = 256;
    blocks_per_grid = (len2 + threads_per_block - 1) / threads_per_block;
    findBestMutant<<<blocks_per_grid, threads_per_block>>>(d_seq1, d_seq2, d_best_seq, len2, from, *best_offset, d_cons_mat, d_semi_cons_mat, d_W, goal);
    CUDA_CHECK(cudaGetLastError());


    // Copy data back to CPU memory
    CUDA_CHECK(cudaMemcpy(best_seq, d_best_seq, len2 * sizeof(char), cudaMemcpyDeviceToHost));


    // Free GPU memory
    CUDA_CHECK(cudaFree(d_seq1));
    CUDA_CHECK(cudaFree(d_seq2));
	CUDA_CHECK(cudaFree(d_best_seq));
    CUDA_CHECK(cudaFree(d_cons_mat));
    CUDA_CHECK(cudaFree(d_semi_cons_mat));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_best_score_arr));

	free(best_score_arr);
    

    return 0;
}

