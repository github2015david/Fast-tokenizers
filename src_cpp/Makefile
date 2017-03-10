SRC_CU1 = GpuTokenize.cu
SRC_CPP1 = Main.cpp 
SRC_CPP2 = Tok.cpp
SRC_CPP3 = CpuTokenize.cpp
OBJ = tok


# make and compile
$(OBJ):cpp1.o cpp2.o cpp3.o cu1.o 
	nvcc -arch=sm_20 cu1.o cpp1.o cpp2.o cpp3.o -o $(OBJ)

cu1.o: $(SRC_CU1)
	nvcc -c -arch=sm_20 $(SRC_CU1) -o cu1.o

cpp1.o: $(SRC_CPP1)
	g++ -c -Wno-write-strings $(SRC_CPP1) -o cpp1.o -lm

cpp2.o: $(SRC_CPP2)
	g++ -c -Wno-write-strings $(SRC_CPP2) -o cpp2.o -lm

cpp3.o: $(SRC_CPP3)
	g++ -c -Wno-write-strings $(SRC_CPP3) -o cpp3.o -lm

.PHONY: clean

clean:
	rm -f $(OBJ) *.o
  
