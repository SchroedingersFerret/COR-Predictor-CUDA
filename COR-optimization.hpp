/*
 * COR-optimization.h
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

#ifndef COR_OPTIMIZATION_H_
#define COR_OPTIMIZATION_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include "COR-predictor.h"
#include "k-hArray.h"

class optimization : public COR_predictor
{
	public:
		static float RandInit();
		__device__ __host__ static float Chebyshev(float x, KernelArray<float> param, int i, int j, float a, float b);
		__device__ __host__ static float f(KernelArray<float> x, int xi, KernelArray<float> param, int i);
		__device__ __host__ static float Mean_square_error(KernelArray<float> x, KernelArray<float> y, KernelArray<float> param, int i);
};

#endif /* COR_OPTIMIZATION_H_ */
