#include <cudnn.h>
#include <curand.h>
#include <iostream>
#include <chrono>

using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

int main()
{
	const size_t batchSize = 2048;
	const size_t inputFeatures = 1024;
	const size_t outputFeatures = 2048;

	//float* input = new float[batchSize * inputFeatures];		// Input data
	//float* weights = new float[outputFeatures * inputFeatures];	// Weights, but stored transposed
	//float* output = new float[batchSize * outputFeatures];		// Output data

	cudnnHandle_t handle;
	cudnnCreate(&handle);

	///
	
	cudaEvent_t start, stop;
	float elapsedTime;
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	cudnnTensorDescriptor_t inputDescriptor;
	cudnnFilterDescriptor_t weightDescriptor;
	cudnnTensorDescriptor_t outputDescriptor;
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "cudnnTensorDescriptor_t: " << elapsedTime << " ms" << endl;
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	cudnnCreateTensorDescriptor(&inputDescriptor);
	cudnnCreateFilterDescriptor(&weightDescriptor);
	cudnnCreateTensorDescriptor(&outputDescriptor);
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "cudnnCreateTensorDescriptor: " << elapsedTime << " ms" << endl;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	cudnnSetTensor4dDescriptor(inputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, inputFeatures, 1, 1);
	cudnnSetFilter4dDescriptor(weightDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outputFeatures, inputFeatures, 1, 1);
	cudnnSetTensor4dDescriptor(outputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, outputFeatures, 1, 1);
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "cudnnSetTensor4dDescriptor: " << elapsedTime << " ms" << endl;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	cudnnConvolutionDescriptor_t convolutionDescriptor;
	cudnnCreateConvolutionDescriptor(&convolutionDescriptor);
	cudnnSetConvolution2dDescriptor(convolutionDescriptor, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "cudnnSetConvolution2dDescriptor: " << elapsedTime << " ms" << endl;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	int maxAlgorithms;
	cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &maxAlgorithms);
	cudnnConvolutionFwdAlgoPerf_t* forwardPropagationAlgorithms = new cudnnConvolutionFwdAlgoPerf_t[maxAlgorithms];
	cudnnFindConvolutionForwardAlgorithm(handle, inputDescriptor, weightDescriptor, convolutionDescriptor, outputDescriptor, maxAlgorithms, &maxAlgorithms, forwardPropagationAlgorithms);
	cudnnConvolutionFwdAlgo_t bestForwardAlgorithm = forwardPropagationAlgorithms[0].algo;	// automatically sorted by fastest
	delete[] forwardPropagationAlgorithms;
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "cudnnFindConvolutionForwardAlgorithm: " << elapsedTime << " ms" << endl;
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	size_t workspaceBytes;
	cudnnGetConvolutionForwardWorkspaceSize(handle, inputDescriptor, weightDescriptor, convolutionDescriptor, outputDescriptor, bestForwardAlgorithm, &workspaceBytes);
	void* workspace;
	cudaMalloc(&workspace, workspaceBytes);
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "workspace cudaMalloc: " << elapsedTime << " ms" << endl;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	float* gpuInput;
	float* gpuWeights;
	float* gpuOutput;
	cudaMalloc(&gpuInput, batchSize * inputFeatures * sizeof(float));
	cudaMalloc(&gpuWeights, outputFeatures * inputFeatures * sizeof(float));
	cudaMalloc(&gpuOutput, batchSize * outputFeatures * sizeof(float));
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "gpu memory cudaMalloc: " << elapsedTime << " ms" << endl;
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	curandGenerator_t randomGenerator;
	curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(randomGenerator, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());
	curandGenerateNormal(randomGenerator, gpuInput, batchSize * inputFeatures + (batchSize * inputFeatures & 2), 0, 1);
	curandGenerateNormal(randomGenerator, gpuWeights, outputFeatures * inputFeatures + (outputFeatures * inputFeatures & 2), 0, 1);
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "curandGenerateNormal: " << elapsedTime << " ms" << endl;
	
	/*cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	cudaMemcpy(input, gpuInput, batchSize * inputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(weights, gpuWeights, outputFeatures * inputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "gpu memory cudaMemcpy: " << elapsedTime << " ms" << endl;*/
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	float alpha = 1.0f;
	float beta = 0.0f;
	cudnnConvolutionForward(handle, &alpha, inputDescriptor, gpuInput, weightDescriptor, gpuWeights, convolutionDescriptor, bestForwardAlgorithm, workspace, workspaceBytes, &beta, outputDescriptor, gpuOutput);
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "cudnnConvolutionForward: " << elapsedTime << " ms" << endl;

	/*cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	cudaMemcpy(output, gpuOutput, batchSize * outputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "gpu memory cudaMemcpy: " << elapsedTime << " ms" << endl;*/

	// backpropagation
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	cudnnConvolutionBwdDataAlgoPerf_t* backwardPropogationAlgorithms = new cudnnConvolutionBwdDataAlgoPerf_t[maxAlgorithms];
	cudnnFindConvolutionBackwardDataAlgorithm(handle, weightDescriptor, outputDescriptor, convolutionDescriptor, inputDescriptor, maxAlgorithms, &maxAlgorithms, backwardPropogationAlgorithms);
	cudnnConvolutionBwdDataAlgo_t bestBackwardAlgorithm = backwardPropogationAlgorithms[0].algo;	// automatically sorted by fastest
	delete[] backwardPropogationAlgorithms;
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "cudnnFindConvolutionBackwardDataAlgorithm: " << elapsedTime << " ms" << endl;
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	cudnnGetConvolutionBackwardDataWorkspaceSize(handle, weightDescriptor, outputDescriptor, convolutionDescriptor, inputDescriptor, bestBackwardAlgorithm, &workspaceBytes);
	cudaFree(workspace);
	cudaMalloc(&workspace, workspaceBytes);
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "workspace cudaMalloc: " << elapsedTime << " ms" << endl;
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	float* gpuBackpropogationOutput;
	cudaMalloc(&gpuBackpropogationOutput, batchSize* inputFeatures * sizeof(float));
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "gpu memory cudaMalloc: " << elapsedTime << " ms" << endl;
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	curandGenerateNormal(randomGenerator, gpuBackpropogationOutput, batchSize* inputFeatures + (batchSize * inputFeatures & 2), 0, 1);
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "curandGenerateNormal: " << elapsedTime << " ms" << endl;
	
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	cudnnConvolutionBackwardData(handle, &alpha, weightDescriptor, gpuWeights, outputDescriptor, gpuOutput, convolutionDescriptor, bestBackwardAlgorithm, workspace, workspaceBytes, &beta, inputDescriptor, gpuBackpropogationOutput);
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "cudnnConvolutionBackwardData: " << elapsedTime << " ms" << endl;

	// cleanup
	cudaFree(gpuInput);
	cudaFree(gpuWeights);
	cudaFree(gpuOutput);
	cudaFree(workspace);
	
	cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
	cudnnDestroyTensorDescriptor(inputDescriptor);
	cudnnDestroyFilterDescriptor(weightDescriptor);
	cudnnDestroyTensorDescriptor(outputDescriptor);

	///
	
	cudnnDestroy(handle);

	/*
	make handle, workspace, initial input and output on the main modeler
	*/
	
	return 0;
}