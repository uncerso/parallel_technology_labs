EXTRA_FLAGS:=-Werror -pedantic-errors -Wall -Wextra -Wpedantic -Wcast-align -Wcast-qual -Wconversion -Wctor-dtor-privacy -Wduplicated-branches -Wduplicated-cond -Wextra-semi -Wfloat-equal -Wlogical-op -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wsign-conversion -Wsign-promo
LIBS:=-fopenmp
STD:=-std=c++14
OUT_NAME:=run
FILES:=source.cpp matrix.cpp
.PHONY: run build build_debug

build:
	g++ -O3 $(STD) $(EXTRA_FLAGS) $(LIBS) $(FILES) -o $(OUT_NAME)

build_debug:
	g++ -g -no-pie $(STD) $(LIBS) $(FILES) -o $(OUT_NAME)

run:
	./run 8 <<< "1000 100"
