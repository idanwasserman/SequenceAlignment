#include "myProto.h"


double calcAlignmentScore(const char* seq1, const char* seq2, int length, int* cons_mat, int* semi_cons_mat, double* W)
{
	omp_set_num_threads(4);
	int num_of_stars = 0, num_of_colons = 0, num_of_points = 0, num_of_spaces = 0;
	//double score = 0;

#pragma omp parallel for reduction(+: num_of_stars, num_of_colons, num_of_points, num_of_spaces)
	for (int i = 0; i < length; i++)
	{

		if(seq1[i] == HYPHEN || seq2[i] == HYPHEN)
		{
			if(seq1[i] == seq2[i])
			{
				//score += W[STAR];
				num_of_stars++;
			}
			else
			{
				//score -= W[SPACE];
				num_of_spaces++;
			}
			continue;
		}

		if (seq1[i] == seq2[i]) // same letter
		{
			num_of_stars++;
			//score += W[STAR];
		}
		else
		{
			int pos1 = (seq1[i] - 'A') * ABC_SIZE + (seq2[i] - 'A');
			int pos2 = (seq2[i] - 'A') * ABC_SIZE + (seq1[i] - 'A');

			if (cons_mat[pos1] == 1 || cons_mat[pos2] == 1) // same conservative groups
			{
				num_of_colons++;
				//score -= W[COLON];
			}
			else if (semi_cons_mat[pos1] == 1 || semi_cons_mat[pos2] == 1) // same semi-conservative groups
			{
				num_of_points++;
				//score -= W[POINT];
			}
			else
			{
				num_of_spaces++;
				//score -= W[SPACE];
			}
		}
	}

	return (num_of_stars * W[STAR] - num_of_colons * W[COLON] - num_of_points * W[POINT] - num_of_spaces * W[SPACE]);
	//return score;
}

void convertGroupToMatrix(int matrix[ABC_SIZE * ABC_SIZE], const char** group, int size)
{
	for (char c = 'A'; c <= 'Z'; c++) // for each letter in the ABC
	{
		for (int i = 0; i < size; i++) // for each word in the group
		{
			const char* word = group[i];
			int length = strlen(group[i]);

			// if letter in word - add all word's letters to this letter
			if (isWordConatinsLetter(c, word, length))
			{
				for (int j = 0; j < length; j++) // for each letter in group[i]
				{
					int pos1 = ABC_SIZE * (c - 'A') + (word[j] - 'A');
					int pos2 = ABC_SIZE * (word[j] - 'A') + (c - 'A');
					matrix[pos1] = 1;
					matrix[pos2] = 1;
				}
			}
		}
	}
}

double getWeight(char c1, char c2, int* cons_mat, int* semi_cons_mat, double* W)
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

int isWordConatinsLetter(char c, const char* word, int length)
{
	if (!word)
		return 0;
		
	for (int i = 0; i < length; i++)
	{
		if (c == word[i])
			return 1;
	}

	return 0;
}

void printDetails(double W[NUM_OF_WEIGHTS], const char* seq1, const char* seq2, int goal)
{
	printf("\n");
	printf("W1: %.2lf, W2: %.2lf, W3: %.2lf, W4: %.2lf\n", W[0], W[1], W[2], W[3]);

	int len1 = strlen(seq1), len2 = strlen(seq2);
	printf("seq1 length - %d , seq2 length - %d\n", len1, len2);

	if(goal == MAXIMUM)
		printf("Looking for maximum\n");
	else
		printf("Looking for minimum\n");
		
	printf("Sequence1:\t %s\n", seq1);
	printf("Sequence2:\t %s\n", seq2);

	printf("\n\n");
}

int readDataFromTextFile(double* W, char* seq1, char* seq2, int* goal, const char* file_name)
{
	if(!file_name)
		return ERROR;

	FILE* fp = fopen(file_name, "r");
	if(!fp)
		return FAILED;

	char temp[16];

	fscanf(fp, "%lf %lf %lf %lf\n", &W[0], &W[1], &W[2], &W[3]);
	fscanf(fp, "%s\n", seq1);
	fscanf(fp, "%s\n", seq2);
	fscanf(fp, "%s", temp);

	if(strcmp(temp, MAXIMUM_STR) == 0)
		*goal = MAXIMUM;
	else
		*goal = MINIMUM;

	fclose(fp);
	return SUCCEEDED;
}

int test(char* best_seq, const char* seq1, int length, int* cons_mat, int* semi_cons_mat, double* W, double best_score, int best_offset)
{
	double test_score = calcAlignmentScore(seq1 + best_offset, best_seq, length, cons_mat, semi_cons_mat, W);
	float error = fabs(best_score / test_score - 1);

	if(error < 0)
		error *= -1;

	if(error > ERROR_VAL)
	{
		printf("Expected: %.2lf\n", best_score);
		printf("Actual: %.2lf\n", test_score);
		return FAILED;	
	}

	return SUCCEEDED;
}

void testSequential(const char* seq1, const char* seq2, int len1, int len2, int* cons_mat, int* semi_cons_mat, double* W, int goal, char* test_best_seq, int* test_best_offset, double* test_best_score)
{

	int offset_size = len1 - len2 + 1;
	*test_best_score = 0;
	*test_best_offset = 0;
	strcpy(test_best_seq, "");

	for(int i = 0; i < offset_size; i++) // for each offset between seq1 and seq2
	{

		double current_score = 0;
		char current_seq[MAX_LEN_SEQ2] = { 0 };

		for(int j = 0; j < len2; j++) // for each letter in seq2
		{

			char c1 = seq1[j + i];
			char c2 = seq2[j];
			char best_c = HYPHEN;
			double hyphen_weight = getWeight(c1, HYPHEN, cons_mat, semi_cons_mat, W);
			double current_weight = getWeight(c1, c2, cons_mat, semi_cons_mat, W);
			double best_weight;
			if((goal == MAXIMUM && current_weight > hyphen_weight) || (goal == MINIMUM && current_weight < hyphen_weight))
				best_weight = current_weight;
			else
				best_weight = hyphen_weight;

			for(char c = 'A'; c <= 'Z'; c++) // for each letter in the ABC
			{

				// if c can be substituted with seq2[j]
				int pos1 = ABC_SIZE * (c1 - 'A') + (c2 - 'A');
				int pos2 = ABC_SIZE * (c2 - 'A') + (c1 - 'A');
				
				if(c == c2)
					continue;

				if(cons_mat[pos1] != 1 && cons_mat[pos2] != 1) // sub and c are not in the same conservative group
				{
					current_weight = getWeight(c1, c, cons_mat, semi_cons_mat, W);

					if((goal == MAXIMUM && current_weight > best_weight) || (goal == MINIMUM && current_weight < best_weight))
					{
						best_weight = current_weight;
						best_c = c;
					}
				}

			}

			current_score += best_weight;
			current_seq[j] = best_c;

		}

		if(i == 0 || (goal == MAXIMUM && current_score > *test_best_score) || (goal == MINIMUM && current_score < *test_best_score))
		{
			*test_best_offset = i;
			*test_best_score = current_score;
			strcpy(test_best_seq, current_seq);
		}
	}
}

int writeResultsToTextFile(char* seq, int best_offset, double best_score, const char* file_name)
{
	if(!file_name)
		return ERROR;

	FILE* fp = fopen(file_name, "w");
	if(!fp)
		return FAILED;

	fprintf(fp, "%s\n", seq);
	fprintf(fp, "%d %f", best_offset, best_score);

	fclose(fp);
	return SUCCEEDED;
}
