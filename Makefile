all: normal cuda test alt_test

normal:
	g++ -fopenmp main.cpp graph.cpp -o graph.exe

cuda:
	nvcc graph.cu -o graphcu.exe

test:
	g++ -fopenmp test.cpp graph.cpp -o test.exe

alt_test:
	g++ -fopenmp alt_test.cpp graph.cpp -o alt_test.exe