#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {

	thrust::host_vector<int> thrustHst_idata(idata, idata+n);
	thrust::device_vector<int> thrustDev_idata(thrustHst_idata);
	thrust::device_vector<int> thrustDev_odata(n);

	float time = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	thrust::exclusive_scan(thrustDev_idata.begin(), thrustDev_idata.end(), thrustDev_odata.begin());
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Thrust scan time is %.4f ms \n", time);

	thrust::copy(thrustDev_odata.begin(), thrustDev_odata.end(), odata);
}

}
}
