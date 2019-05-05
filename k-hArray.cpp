/*
 * k-hArray.cpp
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

#include "k-hArray.h"
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

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
void flatten2dToDevice(thrust::device_vector<T> &device, std::vector<std::vector<T> > &v)
{
	int ni = v.size();
	int nj = v[0].size();
	for (int i=0; i<ni; ++i)
		for (int j=0; j<nj; ++j)
			device[j+nj*i] = v[i][j];
}

template <typename T>
void flatten2dToHost(thrust::host_vector<T> &host, std::vector<std::vector<T> > &v)
{
	int ni = v.size();
	int nj = v[0].size();
	for (int i=0; i<ni; ++i)
		for (int j=0; j<nj; ++j)
			host[j+nj*i] = v[i][j];
}

template <typename T>
std::vector<std::vector<T> > recover2d(thrust::host_vector<T> &host, int ni, int nj)
{
	std::vector<std::vector<T> > result(ni,std::vector<T>(nj));
	for (int i=0; i<ni; ++i)
		for (int j=0; j<nj; ++j)
			result[i][j] = host[j+nj*i];
	return result;
}
