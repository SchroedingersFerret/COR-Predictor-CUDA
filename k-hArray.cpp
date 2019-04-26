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
