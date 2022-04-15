build:
	mpicxx -fopenmp -c main.c -o main.o
	mpicxx -fopenmp -c cFunctions.c -o cFunctions.o
	nvcc -I./inc -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -o finalProject  main.o cFunctions.o cudaFunctions.o  /usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./finalProject

run:
	mpiexec -np 2 ./finalProject

runOn2:
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./finalProject


