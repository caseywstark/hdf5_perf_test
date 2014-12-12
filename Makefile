
CXX := CC
CPPFLAGS :=
CXXFLAGS := -pedantic -Wall -Wextra -Wno-long-long -Wfatal-errors -O3
LDFLAGS :=
LIBS := -lhdf5

TARGETS := io_test_serial.ex io_test_mpi.ex

all: $(TARGETS)

%.ex: %.o
	@echo [MAKE] linking
	$(CXX) $(LDFLAGS) $^ $(LIBS) -o $@

.PHONY: clean
clean:
	rm -f $(TARGETS)
