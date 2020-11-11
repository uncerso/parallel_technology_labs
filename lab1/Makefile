buil_all:
	g++ -std=c++20 -O3 seq.cpp -o seq
	g++ -std=c++20 -fopenmp -O3 par.cpp -o par
	g++ -std=c++20 -fopenmp -O3 seq2.cpp -o seq2
	g++ -std=c++20 -fopenmp -O3 par2.cpp -o par2

run_all:
	nice ./seq
	nice ./par 1
	nice ./par 2
	nice ./seq2
	nice ./par2 
