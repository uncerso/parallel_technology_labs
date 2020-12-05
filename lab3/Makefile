EXTRA_FLAGS:=-Werror -Wall -Wextra -Wcast-align -Wcast-qual -Wconversion -Wctor-dtor-privacy -Wduplicated-branches -Wduplicated-cond -Wextra-semi -Wfloat-equal -Wlogical-op -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wsign-conversion -Wsign-promo
EXTRA_FLAGS_WITH_PEDANTIC:=-pedantic-errors -Wpedantic $(EXTRA_FLAGS)
LIBS:=
STD:=-std=c++17
OUT_NAME:=run
FILES:=source.cpp matrixes.cpp
CXX:=mpicxx
.PHONY: run build build_debug

build_no_pedantic:
	$(CXX) -O3 $(STD) $(EXTRA_FLAGS) $(LIBS) $(FILES) -o $(OUT_NAME)

build:
	$(CXX) -O3 $(STD) $(EXTRA_FLAGS_WITH_PEDANTIC) $(LIBS) $(FILES) -o $(OUT_NAME)

build_debug:
	$(CXX) -g -no-pie $(STD) $(LIBS) $(FILES) -o $(OUT_NAME)

run:
	./run 8 <<< "1000 100"
