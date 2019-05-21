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
struct DeviceArray
{
	thrust::device_vector<T> array;
	size_t size_i;
	size_t size_j;
	size_t size_k;
	DeviceArray(size_t array_size, size_t i, size_t j, size_t k)
	{
		array.resize(array_size);
		size_i = i;
		size_j = j;
		size_k = k;
	}
	DeviceArray(){}
};

template<typename T>
struct HostArray
{
	thrust::host_vector<T> array;
	size_t size_i;
	size_t size_j;
	size_t size_k;
	HostArray(size_t array_size, size_t i, size_t j, size_t k)
	{
		array.resize(array_size);
		size_i = i;
		size_j = j;
		size_k = k;
	}
	HostArray(){}
};

template<typename T>
DeviceArray<T> convertToDevice(HostArray<T> &hArr)
{
	DeviceArray<T> dArr;
	dArr.array = hArr.array;
	dArr.size_i = hArr.size_i;
	dArr.size_j = hArr.size_j;
	dArr.size_k = hArr.size_k;
	return dArr;
}

template<typename T>
HostArray<T> convertToHost(DeviceArray<T> &dArr)
{
	HostArray<T> hArr;
	hArr.array = dArr.array;
	hArr.size_i = dArr.size_i;
	hArr.size_j = dArr.size_j;
	hArr.size_k = dArr.size_k;
	return hArr;
}

template<typename T>
struct KernelArray
{
	T* array;
	size_t size_i;
	size_t size_j;
	size_t size_k;

	KernelArray(DeviceArray<T>& dArr)
	{
		array = thrust::raw_pointer_cast(&dArr.array[0]);
		size_i = dArr.size_i;
		size_j = dArr.size_j;
		size_k = dArr.size_k;
	}

	KernelArray(HostArray<T>& hArr)
	{
		array = thrust::raw_pointer_cast(&hArr.array[0]);
		size_i = hArr.size_i;
		size_j = hArr.size_j;
		size_k = hArr.size_k;
	}

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

template <typename T>
KernelArray<T> convertToKernel(thrust::device_vector<T> &dVec, int i, int j, int k)
{
	KernelArray<T> kArray;
	kArray.array = thrust::raw_pointer_cast(&dVec[0]);
	kArray.size_i = i;
	kArray.size_j = j;
	kArray.size_k = k;
	return kArray;
}

template <typename T>
KernelArray<T> convertToKernel(thrust::host_vector<T> &hVec, int i, int j, int k)
{
	KernelArray<T> kArray;
	kArray.array = thrust::raw_pointer_cast(&hVec[0]);
	kArray.size_i = i;
	kArray.size_j = j;
	kArray.size_k = k;
	return kArray;
}

template <typename T>
HostArray<T> convertToHost(thrust::host_vector<T> &hVec, int i, int j, int k)
{
	HostArray<T> hArray;
	hArray.array = hVec;
	hArray.size_i = i;
	hArray.size_j = j;
	hArray.size_k = k;
	return hArray;
}

template <typename T>
void flatten2dToDevice(DeviceArray<T> &device, std::vector<std::vector<T> > &v)
{
	int ni = v.size();
	int nj = v[0].size();
	for (int i=0; i<ni; ++i)
		for (int j=0; j<nj; ++j)
			device.array[j+nj*i] = v[i][j];
}

template <typename T>
void flatten2dToHost(HostArray<T> &host, std::vector<std::vector<T> > &v)
{
	int ni = v.size();
	int nj = v[0].size();
	for (int i=0; i<ni; ++i)
		for (int j=0; j<nj; ++j)
			host.array[j+nj*i] = v[i][j];
}

template <typename T>
std::vector<std::vector<T> > recover2d(HostArray<T> &host, int ni, int nj)
{
	std::vector<std::vector<T> > result(ni,std::vector<T>(nj));
	for (int i=0; i<ni; ++i)
		for (int j=0; j<nj; ++j)
			result[i][j] = host.array[j+nj*i];
	return result;
}

#endif /* KH_ARRAY_H_ */

