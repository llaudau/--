CXX=g++
ISPC=ispc
CXXFLAGS=-std=c++11 -O3
ISPCFLAGS=-O3 --target=avx2-i32x8 --arch=x86-64

TARGET=sum
CXXSRC=main.cpp
ISPCSRC=sum.ispc
HEADER=sum_ispc.h

ISPCOBJ=$(ISPCSRC:.ispc=.o)
LIB=-lpthread


sum : main.o
	$(CXX) main.o sum_ispc.o -o $(TARGET) $(LIB)
main.o :
	$(CXX) $(CXXFLAGS) -c $(CXXSRC) -o $@

main.o :sum_ispc.h main.cpp

%_ispc.h %_ispc.o : %.ispc
	$(ISPC) $(ISPCFLAGS) $< -o $*_ispc.o -h $*_ispc.h

clean:
	rm -f sum_ispc.h sum_ispc.o sum main.o