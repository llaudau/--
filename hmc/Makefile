# Top-level Makefile for Tianhe HMC build
# Usage: make        (build)
#        make clean   (clean build)
#        make rebuild (clean + build)

MPICXX   := /usr/local/ompi/bin/mpicxx
BUILDDIR := build
JOBS     := 16

.PHONY: all clean rebuild

all:
	@mkdir -p $(BUILDDIR)
	@cd $(BUILDDIR) && cmake .. -DCMAKE_CXX_COMPILER=$(MPICXX) -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" 2>&1 | tail -1
	@cd $(BUILDDIR) && make -j$(JOBS)

clean:
	rm -rf $(BUILDDIR)

rebuild: clean all
