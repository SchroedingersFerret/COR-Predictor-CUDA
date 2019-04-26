/*
 * COR-optimization.cpp
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

//generates a random float between -1.0 and 1.0
float optimization::RandInit()
{
	float r = rand() % RAND_MAX + (RAND_MAX/2 - 1);
	r /= RAND_MAX;
	return r;
};


//Evaluates Chebyshev approximation at x with coefficients from param[]
__device__ __host__ float optimization::Chebyshev(float x, KernelArray<float> param, int i, int j, float a, float b)
{
	float b1 = 0.f, b2 = 0.f;
	float y = (2.f*x-a-b)/(b-a);
	float y2 = 2.f*y;
	for (int k=param.size_k-1; k>0; --k)
	{
		float temp = b1;
		b1 = y2*b1-b2+param.array[k + param.size_k*(j + param.size_j*i)];
		b2 = temp;
	}
	return y*b1-b2+0.5*param.array[param.size_k*(j + param.size_j*i)];
}

//returns the approximate COR with independent variables x[] and coefficients parameters[][]
__device__ __host__ float optimization::f(KernelArray<float> x, int xi, KernelArray<float> param, int i)
{
	float y1 = pow(Chebyshev(x.array[0 + x.size_j*xi], param, i, 0, 0.00018, 1.03),0.5);
	y1 /= pow(Chebyshev(x.array[2 + x.size_j*xi], param, i, 1, 0.002701, 370.f),0.5);
	y1 /= pow(Chebyshev(x.array[4 + x.size_j*xi], param, i, 2, 1.9, 8.553),0.5);
	float y2 = pow(Chebyshev(x.array[1 + x.size_j*xi], param, i, 0, 0.00018, 1.03),0.5);
	y2 /= pow(Chebyshev(x.array[3 + x.size_j*xi], param, i, 1, 0.002701, 370.f),0.5);
	y2 /= pow(Chebyshev(x.array[5 + x.size_j*xi], param,i, 2, 1.9, 8.553),0.5);
	float E = pow(1.f/(0.5/(y1*y1) + 0.5/(y2*y2)), 0.5);
	float v = pow(Chebyshev(x.array[6 + x.size_j*xi], param, i, 3, 0.f, 6.f),0.5);
	float e = E*v*v;
	if (e > 1.f)
		return 1.f;
	return e;
}

//returns the mean of the square of each residual
__device__ __host__ float optimization::Mean_square_error(KernelArray<float> x, KernelArray<float> y, KernelArray<float> param, int i)
{
	float sum = 0;
	for (int xi=0; xi<y.size_i; ++xi)
	{
		float yi = f(x,xi,param,i);
		float residual = y.array[xi]-yi;
		if (isnan(residual))
			residual = 1.f;
		sum += residual*residual;
	}
	return sum/y.size_i;
}
