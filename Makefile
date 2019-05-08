OBJECTS := COR-main.o COR-predictor.o COR-genetic.o COR-anneal.o COR-optimization.o 

all : $(OBJECTS)
	/usr/local/cuda-10.1/bin/nvcc -arch=sm_50 $(OBJECTS) -o COR-Predictor
 
%.o : %.cpp 
	/usr/local/cuda-10.1/bin/nvcc -x cu -arch=sm_50 -std=c++11 -I. -dc $< -o $@

%.o : %.cu
	/usr/local/cuda-10.1/bin/nvcc -arch=sm_50 -std=c++11 -I. -dc $< -o $@

clean:
	rm -f *.o COR-predictor
