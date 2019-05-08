/*
 * COR-anneal.cpp
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

#include <cmath>
#include <float.h>
#include <stdlib.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "k-hArray.h"
#include "COR-anneal.h"

//returns a random number from an gaussian distribution
float anneal::Gaussian_move(float mean, float error, int accepted)
{
	float u,v,x,xx;
	do
	{
		u = RandInit();
		v = RandInit();
		x = 1.71552776992141*(v-0.5)/u;
		xx = x*x;
	}while(xx >= 5.f-5.13610166675097*u &&
		(xx <= 1.036961042583566/u+1.4 || xx <= -4*log(u)));
	return mean + error*v/u*1.f/(n_data*(1+accepted));
}

//returns neighboring state
thrust::host_vector<float> anneal::neighbor(thrust::host_vector<float> &old_state, float error, int accepted)
{
	thrust::host_vector<float> neighbor(parameters_global.size()*parameters_global[0].size());
	size_t ni = old_state.size();
	for (int i=0; i<ni; ++i)
	{
		neighbor[i] = Gaussian_move(old_state[i],error,accepted);
	}
	return neighbor;
}

//returns temperature given a change in energy and entropy
float anneal::Temperature(float initial_temperature, int accepted)
{
	return initial_temperature*exp(-sqrt(accepted));
}

//generates a random float between 0 and 1.0
float anneal::rand_float()
{
	return ((float) rand() / (RAND_MAX));
}

//runs simulated annealing to make aid in optimization
void anneal::run(thrust::host_vector<float> &old_state, KernelArray<float> &x, KernelArray<float> &y)
{
	float old_energy = Mean_square_error(x,y,
			convertToKernel(old_state,1,parameters_global.size(),parameters_global[0].size()),0);
	float initial_temperature = FLT_MAX*old_energy;
	float old_temperature = initial_temperature;
	int accepted = 0;
	int iterations = 0;

	while (old_temperature > 0 && iterations < 10000)
	{

		thrust::host_vector<float> new_state = neighbor(old_state,error,accepted);

		float new_energy = Mean_square_error(x,y,
				convertToKernel(new_state,1,parameters_global.size(),parameters_global[0].size()),0);
		float delta_energy = new_energy-old_energy;
		float new_temperature = Temperature(old_temperature,accepted);

		float P = rand_float();
		float probability;
		if (delta_energy < 0)
			probability = 1.f;
		else
			probability = exp(-delta_energy/new_temperature);

		if (P < probability)
		{
			old_state = new_state;
			old_energy = new_energy;
			old_temperature = new_temperature;
			accepted++;
		}
		iterations++;
	}
}
