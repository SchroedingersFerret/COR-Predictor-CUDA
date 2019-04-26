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

#ifndef COR_ANNEAL_HPP_
#define COR_ANNEAL_HPP_

class anneal : private optimization
{
	private:
		static float Gaussian_move(float mean, float std_dev,int accepted);
		static thrust::host_vector<float> neighbor(thrust::host_vector<float> &old_state,float error,int accepted);
		static float Temperature(float new_energy, int accepted);
		static float rand_float();
	public:
		static void run(thrust::host_vector<float> &old_state, KernelArray<float> &x, KernelArray<float> &y);
};

#endif /* COR_ANNEAL_HPP_ */
