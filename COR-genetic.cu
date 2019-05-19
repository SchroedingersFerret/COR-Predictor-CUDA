/*
 * COR-genetic.cu
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


#include <iostream>
#include <vector>
#include <float.h>
#include <stdlib.h>
#include <stddef.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thread>
#include "k-hArray.h"
#include "COR-genetic.h"
#include "COR-anneal.h"

//encodes the parameters into an offset binary array
__device__ __host__ void genetic::encode(KernelArray<bool> &bin, KernelArray<float> &param, int i_bin, int i_param)
{
	for (int j=0; j<param.size_j; ++j)
	{
		for (int k=0; k<param.size_k; ++k)
		{
			float sum = param.array[k + param.size_k*(j+param.size_j*i_param)];
			bin.array[k*32 + bin.size_k*(j + param.size_j*i_param)] = true;
			if (param.array[k + param.size_k*(j + param.size_j*i_param)] < 0)
			{
				bin.array[k*32 + bin.size_k*(j + bin.size_j*i_bin)] = false;
				sum *= -1;
			}
			bin.array[k*32 + 1 + bin.size_k*(j + bin.size_j*i_bin)] = false;
			if ((int)(0.5 + sum)==1)
				bin.array[k*32 + 1 + bin.size_k*(j + bin.size_j*i_bin)] = true;
			float d = 2.f;
			for (int l=2; l<32; ++l)
			{
				if (bin.array[k*32 + l - 1 + bin.size_k*(j + bin.size_j*i_bin)])
					sum -= 1.f/d;
				bin.array[k*32 + l + bin.size_k*(j + bin.size_j*i_bin)] = false;
				if ((int)(0.5+d*sum) == 1)
					bin.array[k*32 + l + bin.size_k*(j + bin.size_j*i_bin)] = true;
				d *= 2;
			}
		}
	}
}

//recovers the parameters from a binary chromosome
__device__ __host__ void genetic::decode(KernelArray<float> &param, KernelArray<bool> &bin, int i_param, int i_bin)
{
	for (int j=0; j<param.size_j; ++j)
	{

		for (int k=0; k<param.size_k; ++k)
		{
			float d = 2.f;
			float sum = 0;
			for (int l=1; l<32; ++l)
			{
				if (bin.array[k*32 + l + bin.size_k*(j + bin.size_j*i_bin)])
					sum += 1.f/d;
				d *= 2;
			}
			param.array[k + param.size_k*(j + param.size_j*i_param)] = sum + 1.f/d;
			if (!bin.array[k*32 + bin.size_k*(j + bin.size_j*i_param)])
				param.array[k + param.size_k*(j + param.size_j*i_param)] *= -1;
		}
	}
}

//fills a parameter array with random floats
__device__ void genetic::Get_random_parameters(KernelArray<float> &param, int i)
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
void genetic::Get_global_parameters(DeviceArray<float> &d_param)
{
	size_t nj = parameters_global.size();
	size_t nk = parameters_global[0].size();
	HostArray<float> h_param = convertToHost(d_param);
	for (int i=0; i<n_elite; ++i)
		for (int j=0; j<h_param.size_j; ++j)
			for (int k=0; k<h_param.size_k; ++k)
				h_param.array[k+h_param.size_k*(j+h_param.size_j*i)] = parameters_global[j][k];

	d_param = convertToDevice(h_param);
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
	genetic::encode(bin,param,i,i);

	//thread id is assigned as index
	index.array[i] = i;
}

//partition function for parallel quicksort
__global__ void partition(KernelArray<float> cost, KernelArray<int> index, KernelArray<int> low, KernelArray<int> high)
{
	int id = threadIdx.x;
	if (low.array[id] < high.array[id])
	{
		//use median as pivot value
		float pivot = cost.array[(int)(low.array[id] + high.array[id])/2];
		int i = low.array[id] - 1;
		int j = high.array[id] + 1;
		int p = 0;
		//partition array
		for(;;)
		{
			do
				i++;
			while (cost.array[i] < pivot);

			do
				j--;
			while (cost.array[j] > pivot);

			if (i >= j)
			{
				p = j;
				break;
			}
			float temp_f = cost.array[i];
			cost.array[i] = cost.array[j];
			cost.array[j] = temp_f;

			int temp_i = index.array[i];
			index.array[i] = index.array[j];
			index.array[j] = temp_i;
		}
	
		int low_temp = low.array[id];
		int high_temp = high.array[id];
	
		__syncthreads();
		
		//assign new high and low value
		low.array[2*id] = low_temp;
		low.array[2*id+1] = p+1;
		high.array[2*id] = p;
		high.array[2*id+1] = high_temp;
	}
}

//insertion sort for small partitioned array
__global__ void insertion(KernelArray<float> cost, KernelArray<int> index, KernelArray<int> low, KernelArray<int> high)
{
	int id = threadIdx.x;
	for (int i=low.array[id]; i<=high.array[id]; ++i)
	{
		for (int j=i+1; j<=high.array[id]; ++j)
		{
			if (cost.array[j] < cost.array[i])
			{
				float temp_f = cost.array[i];
				cost.array[i] = cost.array[j];
				cost.array[j] = temp_f;

				int temp_i = index.array[i];
				index.array[i] = index.array[j];
				index.array[j] = temp_i;
			}
		}
	}
}

//parallel quicksort function
void genetic::quicksort(DeviceArray<float> &cost, DeviceArray<int> &index)
{
	//preallocate memory on device
	KernelArray<float> k_cost(cost);
	KernelArray<int> k_index(index);
	int size = index.array.size();
	thrust::host_vector<int> h_low(size);
	h_low[0] = 0;
	thrust::device_vector<int> d_low = h_low;
	thrust::host_vector<int> h_high(size);
	h_high[0] = size-1;
	thrust::device_vector<int> d_high = h_high;
	
	int n_arrays = 1;
	
	//partition into smaller arrays
	for (int i=0; n_arrays<size/2; ++i)
	{
		partition<<<1,n_arrays>>>(k_cost, k_index, d_low, d_high);
		cudaDeviceSynchronize();	
		n_arrays *= 2;
	}
	//insertion sort when arrays are small
	insertion<<<1,n_arrays/2>>>(k_cost, k_index, d_low, d_high);
	cudaDeviceSynchronize();
}

//Kernel for arranging the population and MSE's in the order of the quicksorted indices
__global__ void rearrange_kernel(KernelArray<bool> population, KernelArray<bool> bin, KernelArray<float> mean_squared, KernelArray<float> cost, KernelArray<int> index)
{
	int i = threadIdx.x;
	for (int j=0; j<population.size_j; ++j)
		for (int k=0; k<population.size_k; ++k)
			population.array[k + population.size_k*(j + population.size_j*i)] = bin.array[k + population.size_k*(j + population.size_j*index.array[i])];

	mean_squared.array[i] = cost.array[i];
}

//initiates genetic algorithm
void genetic::Initiate(DeviceArray<bool> &population, DeviceArray<float> &mean_squared, KernelArray<float> &x, KernelArray<float> &y)
{
	//allocate device memory
	DeviceArray<bool> d_bin(n_initial*population.size_j*population.size_k, n_initial, population.size_j,population.size_k);
	KernelArray<bool> k_bin(d_bin);
	DeviceArray<float> param(n_initial*parameters_global.size()*parameters_global[0].size(),n_initial,parameters_global.size(), parameters_global[0].size());

	DeviceArray<float> d_cost(n_initial, n_initial, 1, 1);
	KernelArray<float> k_cost(d_cost);
	DeviceArray<int> d_index(n_initial, n_initial, 1, 1);
	KernelArray<int> k_index(d_index);

	Get_global_parameters(param);
	KernelArray<float> k_param(param);

	//fills the elite population with the parameters read from file unless user specifies an entirely random population
	initiate_kernel<<<1,n_elite>>>(k_bin, k_param, x, y, k_cost, k_index, random_parameters, 0);

	random_parameters = true;

	//The remaining population is initialized randomly
	initiate_kernel<<<1,n_initial-n_elite >>>(k_bin, k_param,	x, y, k_cost,k_index, random_parameters, n_elite);
	cudaDeviceSynchronize();

	//sorts population by cost
	quicksort(d_cost,d_index);

	KernelArray<bool> k_population(population);
	KernelArray<float> k_mean_squared(mean_squared);

	rearrange_kernel<<<1, n_gpool>>>(k_population, k_bin, k_mean_squared, k_cost, k_index);
	cudaDeviceSynchronize();
}

//shuffles the indices into a random configuration
void genetic::shuffle(HostArray<int> &index)
{
	for (int i = 0; i<index.size_i; ++i)
	{
		int j = rand() % index.size_i;
		if (j!= i)
			std::swap(index.array[i],index.array[j]);
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
void genetic::tournament(DeviceArray<bool>  &population, DeviceArray<float> &mean_squared)
{
	//allocate device memory and initialize
	KernelArray<bool> k_population(population);
	DeviceArray<bool> bin = population;
	KernelArray<bool> k_bin(bin);
	KernelArray<float> k_mean_squared(mean_squared);
	DeviceArray<float> cost = mean_squared;
	KernelArray<float> k_cost = cost;

	//initialize index
	HostArray<int> h_index(n_gpool,n_gpool,1,1);
	for (int i=0; i<n_gpool; ++i)
		h_index.array[i] = i;

	//shuffle index
	shuffle(h_index);

	//copy shuffled index to device
	DeviceArray<int> index = convertToDevice(h_index);
	KernelArray<int> k_index(index);

	//run tournaments
	tourney_kernel<<<1,n_repro>>>(k_population, k_bin, k_mean_squared, k_cost, k_index);
	cudaDeviceSynchronize();
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
void genetic::reproduction(DeviceArray<bool> &population)
{
	int n_repro2 = n_repro/2;

	//allocate memory and initialize
	KernelArray<bool> k_population(population);

	//perform reproduction ( ͡° ͜ʖ ͡°)
	repro_kernel<<<1,n_repro2>>>(k_population, n_repro, n_repro2);
	cudaDeviceSynchronize();
}

//kernel to evaluate the cost of each member of the population
__global__ void eval_kernel(KernelArray<bool> bin, KernelArray<float> param, KernelArray<float> x, KernelArray<float> y, KernelArray<float> cost, KernelArray<int> index)
{
	int i = threadIdx.x;
	genetic::decode(param, bin, i, i);
	cost.array[i] = optimization::Mean_square_error(x, y, param, i);
	index.array[i] = i;
}

//ranks the chromosomes by cost
void genetic::rankChromosomes(DeviceArray<bool> &population, DeviceArray<float> &mean_squared, KernelArray<float> &x, KernelArray<float> &y)
{
	//allocate device memory
	KernelArray<bool> k_population(population);
	DeviceArray<bool> bin = population;
	KernelArray<bool> k_bin(bin);
	KernelArray<float> k_mean_squared(mean_squared);
	DeviceArray<float> d_cost = mean_squared;
	KernelArray<float> k_cost(d_cost);
	DeviceArray<float> param(n_gpool*parameters_global.size()*parameters_global[0].size(),n_gpool,parameters_global.size(), parameters_global[0].size());
	KernelArray<float> k_param(param);
	DeviceArray<int> d_index(n_gpool, n_gpool, 1, 1);
	KernelArray<int> k_index(d_index);


	//evaluate cost of each array
	eval_kernel<<<1,n_gpool>>>(k_bin, k_param, x, y, k_cost, k_index);
	cudaDeviceSynchronize();

	//sorts population by cost
	quicksort(d_cost,d_index);

	rearrange_kernel<<<1, n_gpool>>>(k_population, k_bin, k_mean_squared, k_cost, k_index);
	cudaDeviceSynchronize();
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
		genetic::decode(param, bin, i, i);
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
void genetic::mutate(DeviceArray<bool> &population, DeviceArray<float> &mean_squared, KernelArray<float> &x, KernelArray<float> &y)
{

	//allocate memory
	KernelArray<bool> k_population(population);
	KernelArray<float> k_mean_squared(mean_squared);
	DeviceArray<float> param(n_gpool*parameters_global.size()*parameters_global[0].size(),n_gpool,parameters_global.size(), parameters_global[0].size());
	KernelArray<float> k_param(param);
	int iterations = (int) (population.size_j*population.size_k*pm+1);

	//mutate elite population
	elite_kernel<<<1, n_elite>>>(k_population, k_param, x, y, k_mean_squared, n_elite, iterations);

	int num_blocks = (int) (n_normal*population.size_j*population.size_k*pm+1)/100;

	//mutate normal population
	normal_kernel<<<num_blocks, 100 >>>(k_population, n_elite, n_normal);
	cudaDeviceSynchronize();
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
	DeviceArray<bool> binary_population(n_gpool*parameters_global.size()*parameters_global[0].size()*32,
										n_gpool,parameters_global.size(),parameters_global[0].size()*32);

	//sum of the square of each residual
	DeviceArray<float> mean_squared_error(n_gpool, n_gpool, 1, 1);

	thrust::device_vector<float> x1(n_data*nx);
	flatten2dToDevice(x1, independent);
	KernelArray<float> d_x = convertToKernel(x1, n_data, nx, 1);

	thrust::host_vector<float> x2(n_data*nx);
	flatten2dToHost(x2, independent);
	KernelArray<float> h_x = convertToKernel(x2, n_data, nx, 1);

	thrust::host_vector<float> y2(n_data);
	thrust::copy(dependent.begin(), dependent.end(), y2.begin());
	KernelArray<float> h_y = convertToKernel(y2, n_data, 1, 1);

	thrust::device_vector<float> y1 = y2;
	KernelArray<float> d_y = convertToKernel(y1, n_data, 1, 1);

	Initiate(binary_population, mean_squared_error, d_x, d_y);

	std::cout << "Running genetic algorithm...\nPress 'Enter' to stop.\n\n";
	int iterations = 0;
	float old_S = 0;

	//stops the loop when 'Enter' is pressed
	bool stop = false;
	std::thread stop_loop([&stop]() // @suppress("Type cannot be resolved")
	{
		std::cin.get();
		stop = true;
	});

	while(mean_squared_error.array[0] > error && !stop)
	{
		tournament(binary_population, mean_squared_error);

		reproduction(binary_population);

		rankChromosomes(binary_population, mean_squared_error, d_x, d_y);

		if (iterations >= 50)
		{
			HostArray<bool> h_population = convertToHost(binary_population);
			CheckDiversity(h_population);

			HostArray<float> param(1*parameters_global.size()*parameters_global[0].size(),1,parameters_global.size(), parameters_global[0].size());
			KernelArray<float> k_param(param);
			KernelArray<bool> k_population(h_population);
			decode(k_param, k_population, 0, 0);

			anneal::run(param, h_x, h_y);
			float cost = Mean_square_error(h_x, h_y,k_param, 0);

			encode(k_population, k_param, n_elite-1, 0);

			mean_squared_error.array[n_elite-1] = cost;

			iterations = 0;
		}

		mutate(binary_population,mean_squared_error,d_x,d_y);

		rankChromosomes(binary_population,mean_squared_error,d_x,d_y);

		//displays new values in terminal
		float new_S = mean_squared_error.array[0];
		if (new_S != old_S)
		{
			show_mean_squared(new_S);
			old_S = new_S;
		}

		iterations++;
	};
	stop_loop.join();
	thrust::host_vector<float> param(parameters_global.size()*parameters_global[0].size());
	KernelArray<float> k_param = convertToKernel(param,1,parameters_global.size(), parameters_global[0].size());
	HostArray<bool> h_population = convertToHost(binary_population);
	KernelArray<bool> k_population(h_population);
	decode(k_param, k_population, 0, 0);
	parameters_global = recover2d(param,parameters_global.size(),parameters_global[0].size());
}
