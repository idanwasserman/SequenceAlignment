/*
 ============================================================================
 Name        : FinalProject.c
 Author      : Idan Wasserman
 Version     :
 Copyright   : 
 Description : Final Project in parallel programming
 ============================================================================
 */
 
 
#include <mpi.h>
#include "myProto.h"

// conservative and semi conservative groups
const char* consGroup[CONS_GROUP_SIZE] =
{
	"NDEQ", "NEQK", "STA", "MILV", "QHRK", "NHQK", "FYW", "HY", "MILF"
};
const char* semi_consGroup[SEMI_CONS_GROUP_SIZE] =
{
	"SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM"
};


// Error handling macro
#define MPI_CHECK(call) \
    if((call) != MPI_SUCCESS) { \
        fprintf(stderr, "MPI error calling \""#call"\"\n"); \
        my_abort(-1); }
        
// Shut down MPI cleanly if something goes wrong
void my_abort(int err)
{
    printf("Program FAILED\n");
    MPI_Abort(MPI_COMM_WORLD, err);
}


int main(int argc, char* argv[])
{

	// defining variables

    int num_of_proc; 				/* number of processes */
    int my_rank; 				/* rank of process */
    int dest;					/* rank of destination process */
    int goal; 					/* goal of search (MAXIMUM/MINIMUM) */
    int position; 				/* position for packaging data into buffer */
    int len1, len2; 			/* length of sequence #1 / #2 */
	int offset_size; 			/* number of total offsets between the two sequences */
	int from, to; 				/* from where to begin and till where */
	int best_offset; 			/* the offset where the best sequence has been produces */
	int tag = 0; 				/* for MPI communication */
	
	int cons_mat[ABC_SIZE * ABC_SIZE];
	int semi_cons_mat[ABC_SIZE * ABC_SIZE];
	
	double best_score; 			/* best alignment score */
	double t1, t2; 				/* for time measuring */
    double W[NUM_OF_WEIGHTS]; 	/* input weight coefficients */
    
	char seq1[MAX_LEN_SEQ1]; 	/* input sequence #1 */
	char seq2[MAX_LEN_SEQ2]; 	/* input sequence #2 */
	char temp_seq2[MAX_LEN_SEQ2];
	char best_seq[MAX_LEN_SEQ2]; /* the sequence that produced MAX/MIN score alignment */
	char buffer[BUFFER_SIZE]; 	/* buffer for packaging data */
	
	MPI_Status status; 			/* return status for receive MPI */


	// Initialize MPI state
    MPI_CHECK(MPI_Init(&argc, &argv));

    // Get MPI node number and node count
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &num_of_proc));
    if (num_of_proc != 2)
    {
       printf("Run the project with two processes only!\n");
       MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));


    // Read data and send it to the other process
    if (my_rank == MASTER)
    {
    	dest = SLAVE;
    	
    	// Read data from input text file
		if(!readDataFromTextFile(W, seq1, seq2, &goal, INPUT_FILE_NAME))
		{
			printf("Reading File ERROR!");
			MPI_Abort(MPI_COMM_WORLD, __LINE__);
		}

		// Start packing data from the beginning of the buffer
		position = 0;
		MPI_Pack(W, NUM_OF_WEIGHTS, MPI_DOUBLE, buffer, BUFFER_SIZE, &position, MPI_COMM_WORLD);
		MPI_Pack(seq1, MAX_LEN_SEQ1, MPI_CHAR, buffer, BUFFER_SIZE, &position, MPI_COMM_WORLD);
		MPI_Pack(seq2, MAX_LEN_SEQ2, MPI_CHAR, buffer, BUFFER_SIZE, &position, MPI_COMM_WORLD);
		MPI_Pack(&goal, 1, MPI_INT, buffer, BUFFER_SIZE, &position, MPI_COMM_WORLD);

		// Send buffer to the other process
		MPI_Send(buffer, position, MPI_PACKED, dest, tag, MPI_COMM_WORLD);
		
    }
    else
    {
    	dest = MASTER;
    	
    	// Receive package
    	MPI_Recv(buffer, BUFFER_SIZE, MPI_PACKED, dest, tag, MPI_COMM_WORLD, &status);

    	// Start unpacking package from the beginning of the buffer
    	position = 0;
    	MPI_Unpack(buffer, BUFFER_SIZE, &position, W, NUM_OF_WEIGHTS, MPI_DOUBLE, MPI_COMM_WORLD);
    	MPI_Unpack(buffer, BUFFER_SIZE, &position, seq1, MAX_LEN_SEQ1, MPI_CHAR, MPI_COMM_WORLD);
    	MPI_Unpack(buffer, BUFFER_SIZE, &position, seq2, MAX_LEN_SEQ2, MPI_CHAR, MPI_COMM_WORLD);
    	MPI_Unpack(buffer, BUFFER_SIZE, &position, &goal, 1, MPI_INT, MPI_COMM_WORLD);
    	
    	// Print details just to check
    	//printDetails(W, seq1, seq2, goal);
    	
    }


    // Prepare data before starting calculations
    
    // Convert groups to matrix/array for efficiency and convenience
    convertGroupToMatrix(cons_mat, consGroup, CONS_GROUP_SIZE);
    convertGroupToMatrix(semi_cons_mat, semi_consGroup, SEMI_CONS_GROUP_SIZE);
        	
    // The same data for both processes
    len1 = strlen(seq1);
    len2 = strlen(seq2);
    offset_size = len1 - len2 + 1;
    strcpy(temp_seq2, seq2);   

    // Different data
    if(my_rank == MASTER)
    {
    	from = 0;
    	to = offset_size / 2 + offset_size % 2;
    }
    else
    {
    	from = (offset_size / 2) + (offset_size % 2);
		to = offset_size;
    }


    // Start calculations together
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();


	// Perform heavy task with CUDA & OMP
    if(computeOnGPU(seq1, seq2, best_seq, len1, len2, from, to, cons_mat, semi_cons_mat, W, &best_score, &best_offset, goal) != 0)
		my_abort(__LINE__);


	// Master collect results
	if(my_rank == MASTER)
	{
		double slave_score;
		int slave_offset;
		char slave_seq[MAX_LEN_SEQ2] = { 0 };
		MPI_Recv(&slave_score, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &status); 
		MPI_Recv(&slave_offset, 1, MPI_INT, dest, tag, MPI_COMM_WORLD, &status); 
		MPI_Recv(&slave_seq, len2, MPI_CHAR, dest, tag, MPI_COMM_WORLD, &status); 
		
		if((goal == MAXIMUM && slave_score > best_score) || (goal == MINIMUM && slave_score < best_score))
		{
			best_score = slave_score;
			best_offset = slave_offset;
			strcpy(best_seq, slave_seq);
		}
	}
	else
	{
		MPI_Send(&best_score, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
		MPI_Send(&best_offset, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
		MPI_Send(&best_seq, len2, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
	}
		
	// End measuring time
	t2 = MPI_Wtime();


	//#define WITH_SEQUENTIAL_SOLUTION
	#ifdef WITH_SEQUENTIAL_SOLUTION

	if(my_rank == MASTER)
    {
		double paralled_time = t2 - t1, sequential_time;
		
		char test_best_seq[MAX_LEN_SEQ2];
		int test_best_offset;
		double test_best_score;

		t1 = MPI_Wtime();
		testSequential(seq1, seq2, len1, len2, cons_mat, semi_cons_mat, W, goal, test_best_seq, &test_best_offset, &test_best_score);
		t2 = MPI_Wtime();
		sequential_time = t2 - t1;

		if(test_best_offset != best_offset || test_best_score != best_score)
		{
			printf("\nTest FAILED!\n");
			printf("\nbest offset: expected: %d, actual: %d\n", best_offset, test_best_offset);
			printf("\nbest score: expected: %.2lf, actual: %.2lf\n", best_score, test_best_score);
			printf("\nbest seq: expected: %s, actual: %s\n", best_seq, test_best_seq);
		}
		else
		{
			printf("\nTest SUCCEEDED!\n");
			writeResultsToTextFile(best_seq, best_offset, best_score, OUTPUT_FILE_NAME);
		}

		printf("\nParalleled program measured time: %lf\n", paralled_time);
		printf("\nSequential program measured time: %lf\n", sequential_time);
		printf("\nParalleled solution is %.2lf times faster than the sequential solution\n", sequential_time / paralled_time);
	}

	#else
	
    // Master write results to output text file and test results
    if(my_rank == MASTER)
    {
		if(test(best_seq, seq1, len2, cons_mat, semi_cons_mat, W, best_score, best_offset))
		{
			printf("\nTest SUCCEEDED!\n");
			writeResultsToTextFile(best_seq, best_offset, best_score, OUTPUT_FILE_NAME);
		}
		else
		{
			printf("\nTest FAILED!\n");
		}

		printf("\nProgram measured time: %lf\n", t2 - t1);
	}

	#endif // WITH_SEQUENTIAL_SOLUTION


	/*
	for more details printed to the terminal - please #define PRINT_DETAILS
	*/
	//#deine PRINT_DETAILS
	#ifdef PRINT_DETAILS

	if(my_rank == MASTER)
	{

		int same_letter_counter = 0, changed_letter_counter = 0, hyphen_counter = 0;
		for(int i = 0; i < len2; i++)
		{
			if(best_seq[i] == seq1[best_offset + i])
				same_letter_counter++;

			if(best_seq[i] != temp_seq2[i])
				changed_letter_counter++;

			if(best_seq[i] == HYPHEN)
				hyphen_counter++;
		}

		printf("\nTotal changes between seq2 and best_seq: %d\n", changed_letter_counter);
		printf("\nTotal same letters between best_seq and seq1: %d\n", same_letter_counter);
		printf("\nTotal hyphens in best seq: %d\n", hyphen_counter);

		printf("\nSeq1 length: %d \t Seq2 length: %d \n", len1, len2);
		printf("\nBest Offset: %d \t Best Score: %.2lf \n", best_offset, best_score);
		printf("Best Sequence: %s\n", best_seq);
		
		
	}

	#endif // PRINT_DETAILS
		
	
	// Shut down MPI
    MPI_Finalize();
	
	return 0;
}

