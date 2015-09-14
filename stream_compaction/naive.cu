#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"


namespace StreamCompaction {
namespace Naive {

__global__ void gpu_scan(int n, int d, int *odata, int *idata) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);

		if( index < n ) {
			if( index >= (1 << (d-1)) ) {
				odata[index] = idata[index] + idata[index-(1 << (d-1))];
			} else {
				odata[index] = idata[index];
			}
		}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	int *dev_idata;
	int *dev_odata;
	int blockSize=1024;

//	int idata_new[n];
//	idata_new[0] = 0;
//	for(int k=0; k<n-1; k++) {
//		idata_new[k+1] = idata[k];
//	}
//
    int n_new = 1 << ilog2ceil(n);
	int idata_new[n_new];
	idata_new[0] = 0;
    for(int i=0; i<n_new; i++) {
    	if(i<n) {
    		idata_new[i+1] = idata[i];
    	} else {
    		idata_new[i] = 0;
    	}
    }

	dim3 fullBlocksPerGrid((n_new + blockSize - 1) / blockSize);

    cudaMalloc((void**)&dev_idata, n_new * sizeof(int));
    cudaMalloc((void**)&dev_odata, n_new * sizeof(int));

    cudaMemcpy(dev_idata, idata_new, n_new * sizeof(int), cudaMemcpyHostToDevice);

	float time = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	for(int i = 1; i<=ilog2ceil(n_new); i++) {
		gpu_scan<<<fullBlocksPerGrid, blockSize>>>(n_new, i, dev_odata, dev_idata);
		cudaMemcpy(dev_idata, dev_odata, n_new * sizeof(int), cudaMemcpyDeviceToDevice);
		//dev_idata = dev_odata;
	}
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("GPU naive scan time is %.4f ms \n", time);

	cudaMemcpy(odata, dev_odata, n_new * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev_idata);
	cudaFree(dev_odata);

}

}
}
