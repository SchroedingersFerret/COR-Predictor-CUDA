OBJECTS = COR-predictor.o COR-optimization.o COR-genetic.o COR-anneal.o COR-main.o

all : $(OBJECTS)
	/usr/local/cuda-10.1/bin/nvcc -arch=sm_50 $(OBJECTS) -o COR-predictor
 
%.o : %.cpp 
	/usr/local/cuda-10.1/bin/nvcc -x cu -arch=sm_50 -std=c++11 -dc $< -o $@

%.o : %.cu
	/usr/local/cuda-10.1/bin/nvcc -arch=sm_50 -std=c++11 -dc $< -o $@

clean:
	rm -f *.o COR-predictor
