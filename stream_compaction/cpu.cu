#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {

	float time = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	odata[0] = 0;
	for(int i = 1; i<n; i++) {
		odata[i] = odata[i-1] + idata[i-1];
	}
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("CPU scan time is %.4f ms \n", time);
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
    // TODO
	int count = 0;
	for( int i = 0; i<n; i++) {
		if ( idata[i] != 0 ) {
			odata[count] = idata[i];
			count++;
		}
	}
    return count;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
    // TODO
	int temp[n];
	int idx[n];
	for(int i = 0; i<n; i++) {
		if(idata[i] != 0) {
			temp[i] = 1;
		} else {
			temp[i] = 0;
		}
	}
	scan(n, idx, temp);
	for(int j = 0; j<n; j++) {
		if(temp[j] == 1) {
			odata[idx[j]] = idata[j];
		}
	}
    return idx[n-1];
}

}
}
