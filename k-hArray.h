/*
 * k-hArray.h
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

#ifndef KH_ARRAY_H_
#define KH_ARRAY_H_

#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

/*Thanks to Ashwin on Code Yarns for this method: https://codeyarns.com/2011/04/09/how-to-pass-thrust-device-vector-to-kernel/
The Kernel Array structure is a way to convert higher dimensional arrays into 1-d arrays that the kernel can understand
The three size variables keep track of the dimensions of the original array*/

template<typename T>
struct KernelArray
{
	T* array;
	size_t size_i;
	size_t size_j;
	size_t size_k;

	KernelArray(thrust::device_vector<T>& dVec)
	{
		array = thrust::raw_pointer_cast( &dVec[0] );
		size_i = dVec.size();
		size_j = 1;
		size_k = 1;
	}

	KernelArray(thrust::host_vector<T>& dVec)
	{
		array = thrust::raw_pointer_cast( &dVec[0] );
		size_i = dVec.size();
		size_j = 1;
		size_k = 1;
	}
	KernelArray(){};
};

template<typename T>
struct HostArray
{
	thrust::host_vector<T> array;
	size_t size_i;
	size_t size_j;
	size_t size_k;
};

template <typename T>
KernelArray<T> convertToKernel(thrust::device_vector<T> &dVec, int i, int j, int k);

template <typename T>
KernelArray<T> convertToKernel(thrust::host_vector<T> &hVec, int i, int j, int k);

template <typename T>
HostArray<T> convertToHost(thrust::host_vector<T> &hVec, int i, int j, int k);

template <typename T>
void flatten2dToDevice(thrust::device_vector<T> &device, std::vector<std::vector<T> > &v);

template <typename T>
void flatten2dToHost(thrust::host_vector<T> &host, std::vector<std::vector<T> > &v);

template <typename T>
std::vector<std::vector<T> > recover2d(thrust::host_vector<T> &host, int ni, int nj);

#endif /* KH_ARRAY_H_ */
