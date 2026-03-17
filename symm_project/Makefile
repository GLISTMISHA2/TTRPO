CXX = g++
CXXFLAGS = -O3 -march=native -pthread -I./include
OPENBLAS_LIBS = -lopenblas -lpthread

all: test_symm_openblas_float test_symm_openblas_double \
     my_symm_float my_symm_double

test_symm_openblas_float: src/test_symm_openblas_float.cpp
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@ $(OPENBLAS_LIBS)

test_symm_openblas_double: src/test_symm_openblas_double.cpp
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@ $(OPENBLAS_LIBS)

my_symm_float: src/my_symm_float.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lpthread

my_symm_double: src/my_symm_double.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -lpthread

clean:
	rm -f test_symm_openblas_float test_symm_openblas_double \
	      my_symm_float my_symm_double

.PHONY: all clean
