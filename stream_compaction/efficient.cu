#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

__global__ void gpu_scan_up(int n, int d, int *odata, int *idata) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if( index<n ) {
		if( index>0 && (index+1)%(1<<d)==0 ) {
			odata[index] = idata[index] + idata[index - (1<<(d-1)) ];
		} else {
			odata[index] = idata[index];
		}
//		if( index%( 1<<(d+1) ) ==0 ) {
//			odata[index + (1<<(d+1)) - 1] = idata[index + (1<<(d+1)) - 1] + idata[ index + (1<<d) - 1 ];
//		} else {
//			odata[index] = idata[index];
//		}
	}
}

__global__ void gpu_scan_down(int n, int d, int *odata, int *idata) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if( index<n ) {
		if( index%( 1<<(d+1) ) ==0 ) {
			int t = idata[index + (1<<d) - 1];
			odata[index + (1<<d) - 1] = idata[index + (1<<(d+1)) - 1];
			odata[index + (1<<(d+1)) - 1] = idata[index + (1<<(d+1)) - 1] + t;
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
	int blockSize = 1024;

    int n_new = 1 << ilog2ceil(n);
    //printf("n_new is %d \n", n_new);
	int idata_new[n_new];
    for(int i=0; i<n_new; i++) {
    	if(i<n) {
    		idata_new[i] = idata[i];
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
    for(int j=1; j<=ilog2ceil(n); j++) {
    	gpu_scan_up<<<fullBlocksPerGrid, blockSize>>>(n_new, j, dev_odata, dev_idata);
    	//cudaMemcpy(dev_idata, dev_odata, n_new * sizeof(int), cudaMemcpyDeviceToDevice);
    	dev_idata = dev_odata;
    }
    cudaMemcpy(idata_new, dev_odata, n_new * sizeof(int), cudaMemcpyDeviceToHost);

    idata_new[n_new-1] = 0;
    cudaMemcpy(dev_idata, idata_new, n_new * sizeof(int), cudaMemcpyHostToDevice);
    for(int k=ilog2ceil(n)-1; k>=0; k--) {
    	gpu_scan_down<<<fullBlocksPerGrid, blockSize>>>(n_new, k, dev_odata, dev_idata);
    	//cudaMemcpy(dev_idata, dev_odata, n_new * sizeof(int), cudaMemcpyDeviceToDevice);
    	dev_idata = dev_odata;
    }
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("GPU work-efficient scan time is %.4f ms \n", time);

    cudaMemcpy(odata, dev_odata, n_new * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_idata);
    cudaFree(dev_odata);
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {

	int *dev_idata;
	int *dev_odata;
	int *dev_bools;
	int *dev_indices;

	int hst_bools[n];
	int hst_indices[n];

	int blockSize = 1024;

	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

    cudaMalloc((void**)&dev_idata, n * sizeof(int));
    cudaMalloc((void**)&dev_odata, n * sizeof(int));
    cudaMalloc((void**)&dev_bools, n * sizeof(int));
    cudaMalloc((void**)&dev_indices, n * sizeof(int));

    cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
    Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(n, dev_bools, dev_idata);
    cudaMemcpy(hst_bools, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);

	scan(n, hst_indices, hst_bools);
	//printf("n is %d \n", n);

	cudaMemcpy(dev_indices, hst_indices, n * sizeof(int), cudaMemcpyHostToDevice);
	Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
	cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_idata);
	cudaFree(dev_odata);
	cudaFree(dev_bools);
	cudaFree(dev_indices);

	if(idata[n-1] == 0) {
		return hst_indices[n-1];
	} else {
		return hst_indices[n-1] + 1;
	}

}

}
}
