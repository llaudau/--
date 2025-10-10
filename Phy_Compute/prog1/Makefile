CXX=g++

CXXFLAGS=-std=c++11 -O3
ISPCFLAGS=-O3 --target=avx2-i32x8 --arch=x86-64

TARGET=sum
CXXSRC=main.cpp

ISPCOBJ=$(ISPCSRC:.ispc=.o)
LIB=-lpthread


sum : main.o
	$(CXX) main.o  -o $(TARGET) $(LIB)
main.o :
	$(CXX) $(CXXFLAGS) -c $(CXXSRC) -o $@

main.o : main.cpp

clean:
	rm -f sum main.o