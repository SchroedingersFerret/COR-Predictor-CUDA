/*
 * COR-genetic.cpp
 *
 *  Copyright 2019
 *      J. Ball (SchroedingersFerret)
 */
 
//This file is part of COR-Predictor-CUDA.
//
//   COR-Predictor-CUDA is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   COR-Predictor-CUDA is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//   along with COR-Predictor-CUDA.  If not, see <https://www.gnu.org/licenses/>.

#include <curand.h>
#include <curand_kernel.h>
#include <COR-optimization>
#include <COR-anneal.hpp>
#include <k-hArray.hpp>

//encodes the parameters into an offset binary array
__device__ __host__ void genetic::encode(KernelArray<bool> bin, KernelArray<float> param, int i)
{
	for (int j=0; j<param.size_j; ++j)
	{
		for (int k=0; k<param.size_k; ++k)
		{
			float sum = param.array[k + param.size_k*(j+param.size_j*i)];
			bin.array[k*32 + bin.size_k*(j + param.size_j*i)] = true;
			if (param.array[k + param.size_k*(j + param.size_j*i)] < 0)
			{
				bin.array[k*32 + bin.size_k*(j + bin.size_j*i)] = false;
				sum *= -1;
			}
			bin.array[k*32 + 1 + bin.size_k*(j + bin.size_j*i)] = false;
			if ((int)(0.5 + sum)==1)
				bin.array[k*32 + 1 + bin.size_k*(j + bin.size_j*i)] = true;
			float d = 2.f;
			for (int l=2; l<32; ++l)
			{
				if (bin.array[k*32 + l - 1 + bin.size_k*(j + bin.size_j*i)])
					sum -= 1.f/d;
				bin.array[k*32 + l + bin.size_k*(j + bin.size_j*i)] = false;
				if ((int)(0.5+d*sum) == 1)
					bin.array[k*32 + l + bin.size_k*(j + bin.size_j*i)] = true;
				d *= 2;
			}
		}
	}
}

//recovers the parameters from a binary chromosome
__device__ __host__ void genetic::decode(KernelArray<float> param, KernelArray<bool> bin, int i)
{
	for (int j=0; j<param.size_j; ++j)
	{
		
		for (int k=0; k<param.size_k; ++k)
		{
			float d = 2.f;
			float sum = 0;
			for (int l=1; l<32; ++l)
			{
				if (bin.array[k*32 + l + bin.size_k*(j + bin.size_j*i)])
					sum += 1.f/d;
				d *= 2;
			}
			param.array[k + param.size_k*(j + param.size_j*i)] = sum + 1.f/d;
			if (!bin.array[k*32 + bin.size_k*(j + bin.size_j*i)])
				param.array[k + param.size_k*(j + param.size_j*i)] *= -1;
		}
	}
}

//fills a parameter array with random floats
__device__ void genetic::Get_random_parameters(KernelArray<float> param, int i)
{
	unsigned int seed = (unsigned long long)clock() + threadIdx.x + (blockIdx.x*blockDim.x);
	curandState_t state;
	curand_init(seed,0,0,&state);
	for (int j=0; j<param.size_j; ++j)
	{
		for(int k=0; k<param.size_k; ++k)
		{
			float r = curand(&state) % RAND_MAX + (RAND_MAX/2 - 1);
			r /= RAND_MAX;
			param.array[k + param.size_k*(j+param.size_j*i)] = r;
		}
	}
}

//fills elite population arrays with parameters read from file
void genetic::Get_global_parameters(thrust::device_vector<float> &d_param)
{
	size_t ni = n_initial;
	size_t nj = parameters_global.size();
	size_t nk = parameters_global[0].size();
	thrust::host_vector<float> h_param(ni*nj*nk);
	for (int i=0; i<n_elite; ++i)
		for (int j=0; j<nj; ++j)
			for (int k=0; k<nk; ++k)
				h_param[k+nk*(j+nj*i)] = parameters_global[j][k];
	
	d_param = h_param;
}

//Kernel to initiate the population, MSE, and index arrays
__global__ void initiate_kernel(KernelArray<bool> bin, KernelArray<float> param, KernelArray<float> x, KernelArray<float> y, KernelArray<float> cost, KernelArray<int> index, bool random, int start)
{
	int i = start + threadIdx.x;
	
	//fills parameter array randomly if specified
	if (random)
		genetic::Get_random_parameters(param,i);
	
	//calculate MSE for set of parameters 
	cost.array[i] = optimization::Mean_square_error(x,y,param,i);
	
	//encode parameter array in 32-bit offset binary
	genetic::encode(bin,param,i);
	
	//thread id is assigned as index
	index.array[i] = i;
}

//partition function for quicksort
int genetic::partition(thrust::host_vector<float> &cost, thrust::host_vector<int> &index, int low, int high)
{
	float pivot = cost[(int)(low + high)/2];
	int i = low - 1;
	int j = high + 1;
	for(;;)
	{
		do
			i++;
		while (cost[i] < pivot);
		
		do 
			j--;
		while (cost[j] > pivot);
		
		if (i >= j)
			return j;
		
		std::swap(cost[i],cost[j]);
		std::swap(index[i],index[j]);
	}
}

//quicksorts indices by cost
void genetic::quicksort(thrust::host_vector<float> &cost, thrust::host_vector<int> &index, int low, int high)
{
	if (low < high)
	{
		int pi = partition(cost, index, low, high);
		quicksort(cost, index, low, pi);
		quicksort(cost, index, pi+1, high);
	}
}

//Kernel for arranging the population and MSE's in the order of the quicksorted indices
__global__ void rearrange_kernel(KernelArray<bool> population, KernelArray<bool> bin, KernelArray<float> mean_squared, KernelArray<float> cost, KernelArray<int> index)
{
	int i = threadIdx.x;
	for (int j=0; j<population.size_j; ++j)
		for (int k=0; k<population.size_k; ++k)
			population.array[k + population.size_k*(j + population.size_j*i)] = bin.array[k + population.size_k*(j + population.size_j*index.array[i])];
		
	mean_squared.array[i] = cost.array[i];
	__syncthreads();
}

//initiates genetic algorithm
void genetic::Initiate(HostArray<bool> &population, HostArray<float> &mean_squared, KernelArray<float> &x, KernelArray<float> &y)
{
	//allocate device memory
	thrust::device_vector<bool> bin(n_initial*population.size_j*population.size_k);
	thrust::device_vector<float> param(n_initial*parameters_global.size()*parameters_global[0].size());
	thrust::device_vector<float> cost(n_initial);
	thrust::device_vector<int> index(n_initial);
	
	Get_global_parameters(param);
	
	//fills the elite population with the parameters read from file unless user specifies an entirely random population
	initiate_kernel<<<1,n_elite>>>(convertToKernel(bin,n_initial,population.size_j,population.size_k), 
									convertToKernel(param,n_initial,parameters_global.size(),parameters_global[0].size()),
									x,y,cost,index,random_parameters,0);
	
	
	random_parameters = true;
	
	//The remaining population is initialized randomly
	initiate_kernel<<<1,n_initial-n_elite >>>(convertToKernel(bin,n_initial,population.size_j,population.size_k), 
									convertToKernel(param,n_initial,parameters_global.size(),parameters_global[0].size()),
									x,y,cost,index,random_parameters,n_elite);
	
	//sort population by cost
	thrust::host_vector<float> h_cost = cost;
	thrust::host_vector<int> h_index = index;
	
	quicksort(h_cost,h_index, 0, n_initial-1);
	
	cost = h_cost;
	index = h_index;
		
	thrust::device_vector<bool> d_population(population.size_i*population.size_j*population.size_k);
	thrust::device_vector<float> d_mean_squared(n_gpool);
	
	rearrange_kernel<<<1, n_gpool>>>(convertToKernel(d_population, n_gpool, population.size_j, population.size_k),
						convertToKernel(bin, n_initial, population.size_j, population.size_k),
						d_mean_squared, cost, index);
	
	//copy to host
	population.array = d_population;
	mean_squared.array = d_mean_squared;
}

//shuffles the indices into a random configuration
void genetic::shuffle(thrust::host_vector<int> &index)
{
	for (int i = 0; i<n_gpool; ++i)
	{
		int j = rand() % n_gpool;
		if (j!= i)
			std::swap(index[i],index[j]);
	}
}

//Kernel to run tournaments
__global__ void tourney_kernel(KernelArray<bool> population, KernelArray<bool> bin, KernelArray<float> mean_squared, KernelArray<float> cost, KernelArray<int> index)
{
	int i = threadIdx.x;
	int l = 2*i;
	
	//overwrite the array with the greater cost
	if (cost.array[index.array[l]] < cost.array[index.array[l+1]])
	{
		for (int j=0; j<bin.size_j; ++j)
			for(int k=0; k<bin.size_k; ++k)
				bin.array[k + bin.size_k*(j + bin.size_j*index.array[l+1])] = bin.array[k + bin.size_k*(j + bin.size_j*index.array[l])];
		cost.array[index.array[l+1]] = cost.array[index.array[l]];	
	}
	else
	{
		for (int j=0; j<bin.size_j; ++j)
			for(int k=0; k<bin.size_k; ++k)
				bin.array[k + bin.size_k*(j + bin.size_j*index.array[l])] = bin.array[k + bin.size_k*(j + bin.size_j*index.array[l+1])];
		cost.array[index.array[l]] = cost.array[index.array[l+1]];
	}
	for (int j=0; j<bin.size_j; ++j)
		for(int k=0; k<bin.size_k; ++k)
			population.array[k + population.size_k*(j + population.size_j*i)] = bin.array[k + bin.size_k*(j + bin.size_j*index.array[l])];
	mean_squared.array[i] = cost.array[index.array[l]];
}

//performs tournament selection on the chromosome population
void genetic::tournament(HostArray<bool>  &population, HostArray<float> &mean_squared)
{
	//allocate device memory and initialize
	thrust::device_vector<bool> d_population = population.array;
	thrust::device_vector<bool> bin = population.array;
	thrust::device_vector<float> d_mean_squared = mean_squared.array;
	thrust::device_vector<float> cost = mean_squared.array;

	//initialize index
	thrust::host_vector<int> h_index(n_gpool);
	for (int i=0; i<n_gpool; ++i)
		h_index[i] = i;
	
	//shuffle index
	shuffle(h_index);	
	
	//copy shuffled index to device
	thrust::device_vector<int> index = h_index;
	
	//run tournaments
	tourney_kernel<<<1,n_repro>>>(convertToKernel(bin, population.size_i, population.size_j, population.size_k),
									convertToKernel(d_population, population.size_i, population.size_j, population.size_k),
									d_mean_squared,cost,index);
										
	//copy to host	
	population.array = d_population;
	mean_squared.array = d_mean_squared;
}

//Kernel to produce children from two parent arrays
__global__ void repro_kernel(KernelArray<bool> bin, int n_repro, int n_repro2)
{
	int i = n_repro + threadIdx.x;
	int l = 2*threadIdx.x;
	
	//seed RNG
	unsigned int seed = (unsigned long long)clock() + threadIdx.x + (blockIdx.x*blockDim.x);
	curandState_t state;
	curand_init(seed,0,0,&state);
	
	for (int j=0; j<bin.size_j; ++j)
	{
		for (int k=0; k<bin.size_k; ++k)
		{
			int rnd = curand(&state) % (RAND_MAX-1);
			rnd /= (RAND_MAX-1)/2;	
			bool parent = (bool) rnd;
			
			//select bit from each parent at random
			if (parent)
			{
				bin.array[k + bin.size_k*(j + bin.size_j*i)] = bin.array[k + bin.size_k*(j + bin.size_j*l)];
				bin.array[k + bin.size_k*(j + bin.size_j*(i + n_repro2))] = bin.array[k + bin.size_k*(j + bin.size_j*(l + 1))];
			}
			if (!parent)
			{
				bin.array[k + bin.size_k*(j + bin.size_j*i)] = bin.array[k + bin.size_k*(j + bin.size_j*(l + 1))];
				bin.array[k + bin.size_k*(j + bin.size_j*(i + n_repro2))] = bin.array[k + bin.size_k*(j + bin.size_j*l)];
			}
		}
	}
}

//performs uniform crossover reproduction on the chromosomes
void genetic::reproduction(HostArray<bool> &population, HostArray<float> &mean_squared)
{
	int n_repro2 = n_repro/2;
	
	//allocate memory and initialize
	thrust::device_vector<bool> bin = population.array;
	
	//perform reproduction ( ͡° ͜ʖ ͡°)
	repro_kernel<<<1,n_repro2>>>(convertToKernel(bin,population.size_i,population.size_j, population.size_k),n_repro,n_repro2);
	
	//copy to host
	population.array = bin;
}

//kernel to evaluate the cost of each member of the population
__global__ void eval_kernel(KernelArray<bool> bin, KernelArray<float> param, KernelArray<float> x, KernelArray<float> y, KernelArray<float> cost, KernelArray<int> index)
{
	int i = threadIdx.x;
	genetic::decode(param, bin, i);
	cost.array[i] = optimization::Mean_square_error(x, y, param, i);
	index.array[i] = i;
}

//ranks the chromosomes by cost
void genetic::rankChromosomes(HostArray<bool> &population, HostArray<float> &mean_squared, KernelArray<float> &x, KernelArray<float> &y)
{
	//allocate device memory
	thrust::device_vector<bool> d_population(population.size_i*population.size_j*population.size_k);
	thrust::device_vector<bool> bin = population.array;
	thrust::device_vector<float> d_mean_squared(mean_squared.size_i*mean_squared.size_j*mean_squared.size_k);
	thrust::device_vector<float> cost = mean_squared.array;
	thrust::device_vector<float> param(n_gpool*parameters_global.size()*parameters_global[0].size());
	thrust::device_vector<int> index(n_gpool);
		
	//evaluate cost of each array
	eval_kernel<<<1,n_gpool>>>(convertToKernel(bin,population.size_i, population.size_j, population.size_k),
								convertToKernel(param,n_gpool,parameters_global.size(),parameters_global[0].size()),
								x,y,cost,index);
	//sort population by cost					
	thrust::host_vector<float> h_cost = cost;
	thrust::host_vector<int> h_index = index;
	
	cost = h_cost;
	index = h_index;
	
	rearrange<<<1, n_gpool>>>(convertToKernel(d_population, n_gpool, population.size_j, population.size_k),
						convertToKernel(bin, n_gpool, population.size_j, population.size_k),
						d_mean_squared, cost, index);
	
	//copy to host
	population.array = d_population;
	mean_squared.array = d_mean_squared;

}

//kernel to mutate bits of the elite population
__global__ void elite_kernel(KernelArray<bool> bin, KernelArray<float> param, KernelArray<float> x, KernelArray<float> y, KernelArray<float> cost, int n_elite, int iterations)
{
	//seed RNG
	unsigned int seed = (unsigned long long)clock() + threadIdx.x + (blockIdx.x*blockDim.x);
	
	curandState_t state;
	curand_init(seed,0,0,&state);
	
	for (int m=0; m<iterations; ++m)
	{
		//choose random bit
		int i = threadIdx.x;
		int j = curand(&state) % bin.size_j;
		int k = curand(&state) % bin.size_k;
	
		//flip bit
		bin.array[k + bin.size_k*(j + bin.size_j*i)] = !bin.array[k + bin.size_k*(j + bin.size_j*i)];
	
		//evaluate array
		genetic::decode(param, bin, i);
		float cost2 = optimization::Mean_square_error(x, y, param, i);
	
		//if new array has greater cost, flip bit back
		if (cost2 > cost.array[i])
		{
			bin.array[k + bin.size_k*(j + bin.size_j*i)] = !bin.array[k + bin.size_k*(j + bin.size_j*i)];
		}
		else
			cost.array[i] = cost2;
	}
}

//kernel to mutate bits of the remaining population
__global__ void normal_kernel(KernelArray<bool> bin, int n_elite, int n_normal)
{
	//seed RNG
	unsigned int seed = (unsigned long long)clock() + threadIdx.x + (blockIdx.x*blockDim.x);
	curandState_t state;
	curand_init(seed,0,0,&state);
	
	//choose random bit
	int i = curand(&state) % n_normal + n_elite;
	int j = curand(&state) % bin.size_j;
	int k = curand(&state) % bin.size_k;
	
	//flip bit
	bin.array[k + bin.size_k*(j + bin.size_j*i)] = !bin.array[k + bin.size_k*(j + bin.size_j*i)];
}

//Mutates bits to introduce variation to the population
void genetic::mutate(HostArray<bool> &population, HostArray<float> &mean_squared, KernelArray<float> &x, KernelArray<float> &y)
{
	
	//allocate memory
	thrust::device_vector<bool> bin = population.array;
	thrust::device_vector<float> cost = mean_squared.array;
	thrust::device_vector<float> param(n_gpool*parameters_global.size()*parameters_global[0].size());
	
	int iterations = (int) (population.size_j*population.size_k*pm+1);
	
	//mutate elite population
	elite_kernel<<<1, n_elite>>>(convertToKernel(bin, population.size_i, population.size_j, population.size_k),
								convertToKernel(param, n_gpool, parameters_global.size(), parameters_global[0].size()),
								x, y, cost, n_elite, iterations);
								
	int num_blocks = (int) (n_normal*population.size_j*population.size_k*pm+1)/100;
	
	//mutate normal population
	normal_kernel<<<num_blocks, 100 >>>(convertToKernel(bin, population.size_i, population.size_j, population.size_k), n_elite, n_normal);
	
	//copy to host
	population.array = bin;
	mean_squared.array = cost;
}

//returns the percentage of differing bits between two chromosomes
float genetic::percentDifference(HostArray<bool> &population, int i)
{
	float pd = 0.f;
	for (int j=0; j<population.size_j; ++j)
	{
		for (int k=0; k<population.size_k; ++k)
		{
			if (population.array[k + population.size_k*j] != population.array[k + population.size_k*(j + population.size_j*i)])
				pd++;
		}
	}
	
	pd /= (population.size_j*population.size_k);
	return pd;
}

//returns the diversity of the population
float genetic::getDiversity(HostArray<bool> &population)
{
	float diversity = 0.f;
	for (int i=1; i<n_gpool; ++i)
	{
		diversity += percentDifference(population,i);
	}
	diversity /= n_gpool-1;
	return diversity;
}

//aborts the program if the population diverges
void genetic::DivergenceError()
{
	std::cout << "\nError: Population divergence.\n";
	abort();
}

//aborts the program if the population bottlenecks
void genetic::BottleneckError()
{
	std::cout << "\nError: Population bottleneck.\n";
	abort();
}

//checks the diversity of the population and aborts if it is too large or small
void genetic::CheckDiversity(HostArray<bool> &population)
{
	float diversity = getDiversity(population);
	//std::cout << diversity << "\n";
	if (diversity > 0.60)
		DivergenceError();
	if (diversity < 0.00001)
		BottleneckError();
}

//displays least squares regression
void genetic::show_mean_squared(float S)
{
	std::cout << std::scientific << "\rmean square error = " << S << std::flush;
}

//execute genetic algorithm
void genetic::run()
{
	//population stores each binary genome
	HostArray<bool> binary_population;
	binary_population.array.resize(n_gpool*parameters_global.size()*parameters_global[0].size()*32);
	binary_population.size_i = n_gpool;
	binary_population.size_j = parameters_global.size();
	binary_population.size_k = parameters_global[0].size()*32;
	
	//sum of the square of each residual
	HostArray<float> mean_squared_error;
	mean_squared_error.array.resize(n_gpool);
	mean_squared_error.size_i = n_gpool;
	mean_squared_error.size_j = 1;
	mean_squared_error.size_k = 1;
	
	thrust::device_vector<float> x1(n_data*nx);
	flatten2dToDevice(x1, independent);
	KernelArray<float> d_x = convertToKernel(x1, n_data, nx, 1);
	
	thrust::host_vector<float> x2(n_data*nx);
	flatten2dToHost(x2, independent);
	KernelArray<float> h_x = convertToKernel(x2, n_data, nx, 1);
	
	thrust::device_vector<float> y1(n_data);
	thrust::copy(dependent.begin(), dependent.end(), y1.begin());
	KernelArray<float> d_y = convertToKernel(y1, n_data, 1, 1);
	
	thrust::host_vector<float> y2(n_data);
	thrust::copy(dependent.begin(), dependent.end(), y2.begin());
	KernelArray<float> h_y = convertToKernel(y2, n_data, 1, 1);
	
	Initiate(binary_population, mean_squared_error, d_x, d_y);
	
	std::cout << "Running genetic algorithm...\nPress 'Enter' to stop.\n\n";
	int iterations = 0;
	float old_S = 0;
	
	//stops the loop when 'Enter' is pressed
	bool stop = false;
	std::thread stop_loop([&stop]()
	{
		std::cin.get();
		stop = true;
	});
	
	while(mean_squared_error.array[0] > error && !stop)
	{
		rankChromosomes(binary_population, mean_squared_error, d_x, d_y);
		
		tournament(binary_population, mean_squared_error);
		
		reproduction(binary_population, mean_squared_error);
		
		rankChromosomes(binary_population, mean_squared_error, d_x, d_y);

		if (iterations >= 50)
		{
			
			CheckDiversity(binary_population);
			
			thrust::host_vector<float> param(parameters_global.size()*parameters_global[0].size());
			
			decode(convertToKernel(param,1,parameters_global.size(), parameters_global[0].size()), 
									convertToKernel(binary_population.array, binary_population.size_i, binary_population.size_j, binary_population.size_k), 0);
			
			anneal::run(param, h_x, h_y);
			float cost = Mean_square_error(h_x, h_y,convertToKernel(param,parameters_global.size(), parameters_global[0].size(), 1), 0);
			
			encode(convertToKernel(binary_population.array, binary_population.size_i, binary_population.size_j, binary_population.size_k), 
									convertToKernel(param, parameters_global.size(), parameters_global[0].size(), 1), n_elite-1);
									
			mean_squared_error.array[n_elite-1] = cost;
			
			iterations = 0;
		}
		
		mutate(binary_population,mean_squared_error,d_x,d_y);
		
		rankChromosomes(binary_population,mean_squared_error,d_x,d_y);

		//displays new values in terminal
		float new_S = mean_squared_error.array[0];
		if (new_S != old_S)
		{
			show_mean_squared(mean_squared_error.array[0]);
			old_S = new_S;
		}
		
		iterations++;
	};
	stop_loop.join();
	thrust::host_vector<float> param(parameters_global.size()*parameters_global[0].size());
	decode(convertToKernel(param,1,parameters_global.size(),parameters_global[0].size()), 
			convertToKernel(binary_population.array,binary_population.size_i,binary_population.size_j,binary_population.size_k),0);
	parameters_global = recover2d(param,parameters_global.size(),parameters_global[0].size());
}
