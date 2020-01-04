#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include "vector_functions.hpp"
#include "vector_types.h"
#include "helper_math.h"
#include "device_functions.h" 
#include "commonDefines.h"
#include <device_atomic_functions.h>

__device__ float euclideianDistance(float x1, float y1, float x2, float y2){
    float sum = float(pow(x1 - x2, 2) + pow(y1 - y2, 2));
    float distance = float(sqrt(sum));
    return distance;
}

__global__ void cuda_kMeans_clearAll(float* inputSums_x, float* inputSums_y, int* inputCounts) {
	int tx = threadIdx.x;
	inputSums_x[tx] = 0;
	inputSums_y[tx] = 0;
	inputCounts[tx] = 0;
}

__global__ void cuda_kMeans_CalculateDistances(float* points, float* inputClusters_x, float* inputClusters_y, int* clustersCount, float* outputSums_x, float* outputSums_y, int inputDimension, int vectorDimension, int clusterDimension) {
	int tx = threadIdx.x;
	int row = blockIdx.x * blockDim.x + tx;
	int it = row * vectorDimension;
	float distance;
	float minDistance;
	int clusterIndex;

	float2 point;
	float2 cluster;

	if (row < inputDimension) {
		point = make_float2(points[it], points[it + 1]); //load input point
		minDistance = 10000;
		clusterIndex = 0;

		for (int j = 0; j < clusterDimension; j++) {
			cluster = make_float2(inputClusters_x[j], inputClusters_y[j]); //from central gpu memory
			distance = euclideianDistance(point.x, point.y, cluster.x, cluster.y);

			if (distance < minDistance) {
				minDistance = distance;
				clusterIndex = j;
			}

		}
	
		atomicAdd(&outputSums_x[clusterIndex], point.x);
		atomicAdd(&outputSums_y[clusterIndex], point.y);
		atomicAdd(&clustersCount[clusterIndex], 1);
		
	}

}

__global__ void cuda_kMeans_updateCentroids(float* inputClusters_x, float* inputClusters_y, float* inputSums_x, float* inputSums_y, int* inputCounts) {
	int cluster = threadIdx.x;
	int count = max(1, inputCounts[cluster]);
	inputClusters_x[cluster] = inputSums_x[cluster] / count;
	inputClusters_y[cluster] = inputSums_y[cluster] / count;
}


extern "C"
void cuda_kMeans_CalculateDistances_wrapper(float* points, float* inputClusters_x, float* inputClusters_y, int* clustersCount, float* outputSums_x, float* outputSums_y, int inputDimension, int vectorDimension, int clusterDimension, dim3 gridDim, dim3 blockDim) {
	cuda_kMeans_CalculateDistances << <gridDim, blockDim >> > (points, inputClusters_x, inputClusters_y, clustersCount, outputSums_x, outputSums_y, inputDimension, vectorDimension, clusterDimension);
}

extern "C"
void cuda_kMeans_UpdateClusters_wrapper(float* inputClusters_x, float* inputClusters_y, float* inputSums_x, float* inputSums_y, int* inputCounts, dim3 gridDim, dim3 blockDim) {
	cuda_kMeans_updateCentroids << <gridDim, blockDim >> > (inputClusters_x, inputClusters_y, inputSums_x, inputSums_y, inputCounts);
}

extern "C"
void cuda_kMeans_ClearAll_wrapper(float* inputSums_x, float* inputSums_y, int* inputCounts, dim3 gridDim, dim3 blockDim) {
	cuda_kMeans_clearAll << <gridDim, blockDim >> > (inputSums_x, inputSums_y, inputCounts);
}



