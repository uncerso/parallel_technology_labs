EXTRA_FLAGS:=#-Werror -Wall -Wextra -pedantic-errors -Wpedantic -Wcast-align -Wcast-qual -Wconversion -Wctor-dtor-privacy -Wduplicated-branches -Wduplicated-cond -Wextra-semi -Wfloat-equal -Wlogical-op -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wsign-conversion -Wsign-promo
LIBS:=#-finstrument-functions
STD:=-std=c++17
OUT_NAME:=run
FILES:=source.cpp matrixes.cpp
CXX:=mpigxx
.PHONY: run build build_debug

build:
	$(CXX) -O3 $(STD) $(EXTRA_FLAGS) $(LIBS) $(FILES) -o $(OUT_NAME)

build_debug:
	$(CXX) -g -no-pie $(STD) $(LIBS) $(FILES) -o $(OUT_NAME)

run:
	./run 8 <<< "1000 100"
