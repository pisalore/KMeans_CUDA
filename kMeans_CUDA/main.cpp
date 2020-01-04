#include <iostream>
#include <omp.h>
#include <iostream>
#include "csvio.h"
#include "commonDefines.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

extern "C"
void cuda_kMeans_CalculateDistances_wrapper(float* inputPoints, float* inputClusters_x, float* inputClusters_y, int* clustersCount, float* outputSums_x, float* outputSums_y, int inputDimension, int vectorDimension, int clusterNumber, dim3 gridDim, dim3 blockDim);
extern "C"
void cuda_kMeans_UpdateClusters_wrapper(float* inputClusters_x, float* inputClusters_y, float* inputSums_x, float* inputSums_y, int* inputCounts, dim3 gridDim, dim3 blockDim);
extern "C"
void cuda_kMeans_ClearAll_wrapper(float* inputSums_x, float* inputSums_y, int* inputCounts, dim3 gridDim, dim3 blockDim);



int main(int argc, char* argv[]) {
    //load file and data structures initialization using thrust library
    std::string filename = "datasets/dataset1000.csv";
    std::vector<float> inputPoints;
    std::vector<float> inputClusters_x; //clusters x coordinates
    std::vector<float> inputClusters_y; //clusters y coordinates
    std::vector<float> clusterSums_x(CLUSTERS_NUMBER , 0); // x accumulator
    std::vector<float> clusterSums_y(CLUSTERS_NUMBER, 0); // y accumulator
    std::vector<int> clustersCounter(CLUSTERS_NUMBER, 0); //clusters points counter
    std::string delimiter = ";";

    read2VecFrom(filename, delimiter, inputPoints); //points initialization
    initializeClusters(CLUSTERS_NUMBER, inputClusters_x, inputClusters_y, inputPoints);
    int datasetDim = inputPoints.size() / 2;

    thrust::device_vector<float> inputPoints_device = inputPoints;
    thrust::device_vector<float> inputClusters_x_device = inputClusters_x;
    thrust::device_vector<float> inputClusters_y_device = inputClusters_y;
    thrust::device_vector<float> outputSums_x_device = clusterSums_x;
    thrust::device_vector<float> outputSums_y_device = clusterSums_y;
    thrust::device_vector<int> outputClustersCount_device = clustersCounter;

    //pointer to data for CPU-GPU comunications
    float* inputPoints_ptr_device = thrust::raw_pointer_cast(&inputPoints_device[0]);
    float* inputClusters_x_ptr_device = thrust::raw_pointer_cast(&inputClusters_x_device[0]);
    float* inputClusters_y_ptr_device = thrust::raw_pointer_cast(&inputClusters_y_device[0]);
    float* outputSums_x_ptr_device = thrust::raw_pointer_cast(&outputSums_x_device[0]);
    float* outputSums_y_ptr_device = thrust::raw_pointer_cast(&outputSums_y_device[0]);
    int* outputClustersCount_ptr_device = thrust::raw_pointer_cast(&outputClustersCount_device[0]);

    dim3 gridDim = dim3(ceil((float)(datasetDim) / BLOCK_DIM));
    dim3 blockDim = dim3(BLOCK_DIM);
    
    //computation start
    std::cout << "K Means started. " << std::endl;
    double t1 = omp_get_wtime();

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        cuda_kMeans_ClearAll_wrapper(outputSums_x_ptr_device, outputSums_y_ptr_device, outputClustersCount_ptr_device, 1, CLUSTERS_NUMBER);
        cuda_kMeans_CalculateDistances_wrapper(inputPoints_ptr_device, inputClusters_x_ptr_device, inputClusters_y_ptr_device, outputClustersCount_ptr_device, outputSums_x_ptr_device, outputSums_y_ptr_device, datasetDim, VECTOR_DIM, CLUSTERS_NUMBER, gridDim, blockDim);
        cudaDeviceSynchronize();
        cuda_kMeans_UpdateClusters_wrapper(inputClusters_x_ptr_device, inputClusters_y_ptr_device, outputSums_x_ptr_device, outputSums_y_ptr_device, outputClustersCount_ptr_device, 1, CLUSTERS_NUMBER);
    }

    double t2 = omp_get_wtime() - t1;
    std::cout << "K Means completed in: " << t2 << std::endl;

    //Store and save results
    std::vector<float> outputSums_host_x_stl;
    std::vector<float> outputSums_host_y_stl;
    std::vector<float> centroids;

    thrust::host_vector<float> outputClusters_x_host = inputClusters_x_device;
    thrust::host_vector<float> outputClusters_y_host = inputClusters_y_device;
    std::vector<float> outputClusters_host_x_st1;
    std::vector<float> outputClusters_host_y_st1;
    outputClusters_host_x_st1.resize(outputClusters_x_host.size());
    thrust::copy(outputClusters_x_host.begin(), outputClusters_x_host.end(), outputClusters_host_x_st1.begin());
    outputClusters_host_y_st1.resize(outputClusters_y_host.size());
    thrust::copy(outputClusters_y_host.begin(), outputClusters_y_host.end(), outputClusters_host_y_st1.begin());

    for (int i = 0; i < outputClusters_host_x_st1.size(); i++) {
        std::cout << "Cluster " << i << ": " << "(" << outputClusters_host_x_st1[i] << ", " << outputClusters_host_y_st1[i]<< ")" << std::endl;
        centroids.push_back(outputClusters_host_x_st1[i]);
        centroids.push_back(outputClusters_host_y_st1[i]);
    }

    write2VecTo(std::string("centroids.csv"), delimiter, centroids);

}