/*
 * COR-genetic.h
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

#ifndef COR_GENETIC_H_
#define COR_GENETIC_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "k-hArray.h"
#include "COR-optimization.h"

class genetic : public optimization
{
	private:
		static void Get_global_parameters(thrust::device_vector<float> &d_param);
		//static void quicksort(thrust::host_vector<float> &cost, thrust::host_vector<int> &index, int low, int high);
		static void Initiate(HostArray<bool> &population, HostArray<float> &mean_squared, KernelArray<float> &x, KernelArray<float> &y);
		static void shuffle(thrust::host_vector<int> &index);
		static void tournament(HostArray<bool> &population, HostArray<float> &mean_squared);
		static void reproduction(HostArray<bool> &population, HostArray<float> &mean_squared);
		static void rankChromosomes(HostArray<bool> &population, HostArray<float> &mean_squared, KernelArray<float> &x, KernelArray<float> &y);
		static void mutate(HostArray<bool> &population, HostArray<float> &mean_squared, KernelArray<float> &x, KernelArray<float> &y);
		static float percentDifference(HostArray<bool> &population, int i);
		static float getDiversity(HostArray<bool> &population);
		static void DivergenceError();
		static void BottleneckError();
		static void CheckDiversity(HostArray<bool> &population);
		static void show_mean_squared(float mean_squared);
	public:
		__device__ static int partition(KernelArray<float> cost, KernelArray<int> index, int low, int high);
		__device__ static void Get_random_parameters(KernelArray<float> param, int i);
		__device__ __host__ static void encode(KernelArray<bool> bin, KernelArray<float> param, int i);
		__device__ __host__ static void decode(KernelArray<float>param, KernelArray<bool> bin, int i);
		static void run();
};

#endif /* COR_GENETIC_H_ */
