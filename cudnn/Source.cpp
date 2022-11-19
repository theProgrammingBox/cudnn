#include <cudnn.h>
#include <curand.h>
#include <cublas_v2.h>
#include <iostream>
#include <algorithm>
#include <chrono>

using std::cout;
using std::endl;
using std::max;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

int main()
{
	const size_t batchSize = 1;
	const size_t inputFeatures = 4;
	const size_t outputFeatures = 4;

	const size_t inputSize = batchSize * inputFeatures;
	const size_t weightSize = outputFeatures * inputFeatures;
	const size_t biasSize = outputFeatures;
	const size_t outputSize = batchSize * outputFeatures;
	
	const size_t inputBytes = inputSize * sizeof(float);
	const size_t weightBytes = weightSize * sizeof(float);
	const size_t biasBytes = biasSize * sizeof(float);
	const size_t outputBytes = outputSize * sizeof(float);

	const float alpha = 1.0f;
	const float beta = 0.0f;

	curandGenerator_t randomGenerator;
	cudnnHandle_t cudnnHandle;
	cublasHandle_t cublasHandle;

	float* gpuInput;
	cudnnTensorDescriptor_t inputDescriptor;
	
	float* gpuWeight;
	cudnnFilterDescriptor_t weightDescriptor;
	
	float* gpuBias;
	cudnnTensorDescriptor_t biasDescriptor;
	
	float* gpuOutput;
	cudnnTensorDescriptor_t outputDescriptor;
	
	cudnnConvolutionDescriptor_t propagationDescriptor;

	int maxForwardPropagationAlgorithms;
	cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle, &maxForwardPropagationAlgorithms);
	cudnnConvolutionFwdAlgo_t bestForwardPropagationAlgorithm;

	int maxWeightBackwardPropagationAlgorithms;
	cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle, &maxWeightBackwardPropagationAlgorithms);
	cudnnConvolutionBwdFilterAlgo_t bestWeightBackwardPropagationAlgorithm;
	
	int maxInputBackwardPropagationAlgorithms;
	cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle, &maxInputBackwardPropagationAlgorithms);
	cudnnConvolutionBwdDataAlgo_t bestInputBackwardPropagationAlgorithm;

	size_t workspaceBytes = 0;
	void* gpuWorkspace;

	

	curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(randomGenerator, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());
	cudnnCreate(&cudnnHandle);
	cublasCreate(&cublasHandle);

	cudaMalloc(&gpuInput, inputBytes);
	cudnnCreateTensorDescriptor(&inputDescriptor);
	cudnnSetTensor4dDescriptor(
		inputDescriptor,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		batchSize,
		inputFeatures,
		1,
		1);

	cudaMalloc(&gpuWeight, weightBytes);
	curandGenerateNormal(randomGenerator, gpuWeight, weightSize + (weightSize & 1), 0, 1);
	cudnnCreateFilterDescriptor(&weightDescriptor);
	cudnnSetFilter4dDescriptor(
		weightDescriptor,
		CUDNN_DATA_FLOAT,
		CUDNN_TENSOR_NCHW,
		outputFeatures,
		inputFeatures,
		1,
		1);

	cudaMalloc(&gpuBias, biasBytes);
	curandGenerateNormal(randomGenerator, gpuBias, biasSize + (biasSize & 1), 0, 1);
	cudnnCreateTensorDescriptor(&biasDescriptor);
	cudnnSetTensor4dDescriptor(
		biasDescriptor,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		1,
		outputFeatures,
		1,
		1);

	cudaMalloc(&gpuOutput, outputBytes);
	cudnnCreateTensorDescriptor(&outputDescriptor);
	cudnnSetTensor4dDescriptor(
		outputDescriptor,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		batchSize,
		outputFeatures,
		1,
		1);

	cudnnCreateConvolutionDescriptor(&propagationDescriptor);
	cudnnSetConvolution2dDescriptor(
		propagationDescriptor,
		0,
		0,
		1,
		1,
		1,
		1,
		CUDNN_CROSS_CORRELATION,
		CUDNN_DATA_FLOAT);

	cudnnConvolutionFwdAlgoPerf_t* forwardPropagationAlgorithms = new cudnnConvolutionFwdAlgoPerf_t[maxForwardPropagationAlgorithms];
	cudnnFindConvolutionForwardAlgorithm(
		cudnnHandle,
		inputDescriptor,
		weightDescriptor,
		propagationDescriptor,
		outputDescriptor,
		maxForwardPropagationAlgorithms,
		&maxForwardPropagationAlgorithms,
		forwardPropagationAlgorithms);
	bestForwardPropagationAlgorithm = forwardPropagationAlgorithms[0].algo;
	delete[] forwardPropagationAlgorithms;

	cudnnConvolutionBwdDataAlgoPerf_t* inputBackwardPropagationAlgorithms = new cudnnConvolutionBwdDataAlgoPerf_t[maxInputBackwardPropagationAlgorithms];
	cudnnFindConvolutionBackwardDataAlgorithm(
		cudnnHandle,
		weightDescriptor,
		outputDescriptor,
		propagationDescriptor,
		inputDescriptor,
		maxInputBackwardPropagationAlgorithms,
		&maxInputBackwardPropagationAlgorithms,
		inputBackwardPropagationAlgorithms);
	bestInputBackwardPropagationAlgorithm = inputBackwardPropagationAlgorithms[0].algo;
	delete[] inputBackwardPropagationAlgorithms;

	cudnnConvolutionBwdFilterAlgoPerf_t* weightBackwardPropagationAlgorithms = new cudnnConvolutionBwdFilterAlgoPerf_t[maxWeightBackwardPropagationAlgorithms];
	cudnnFindConvolutionBackwardFilterAlgorithm(
		cudnnHandle,
		inputDescriptor,
		outputDescriptor,
		propagationDescriptor,
		weightDescriptor,
		maxWeightBackwardPropagationAlgorithms,
		&maxWeightBackwardPropagationAlgorithms,
		weightBackwardPropagationAlgorithms);
	bestWeightBackwardPropagationAlgorithm = weightBackwardPropagationAlgorithms[0].algo;
	delete[] weightBackwardPropagationAlgorithms;

	size_t tempBytes;
	cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
		inputDescriptor,
		weightDescriptor,
		propagationDescriptor,
		outputDescriptor,
		bestForwardPropagationAlgorithm,
		&tempBytes);
	workspaceBytes = max(workspaceBytes, tempBytes);

	cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
		weightDescriptor,
		outputDescriptor,
		propagationDescriptor,
		inputDescriptor,
		bestInputBackwardPropagationAlgorithm,
		&tempBytes);
	workspaceBytes = max(workspaceBytes, tempBytes);
	
	cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
		inputDescriptor,
		outputDescriptor,
		propagationDescriptor,
		weightDescriptor,
		bestWeightBackwardPropagationAlgorithm,
		&tempBytes);
	workspaceBytes = max(workspaceBytes, tempBytes);
	
	cudaMalloc(&gpuWorkspace, workspaceBytes);
	
	
	
	curandGenerateNormal(randomGenerator, gpuInput, inputSize + (inputSize & 1), 0, 1);
	cudnnConvolutionForward(
		cudnnHandle,
		&alpha,
		inputDescriptor,
		gpuInput,
		weightDescriptor,
		gpuWeight,
		propagationDescriptor,
		bestForwardPropagationAlgorithm,
		gpuWorkspace,
		workspaceBytes,
		&beta,
		outputDescriptor,
		gpuOutput);
	
	cudnnAddTensor(
		cudnnHandle,
		&alpha,
		biasDescriptor,
		gpuBias,
		&alpha,
		outputDescriptor,
		gpuOutput);

	float* cpuOutput = new float[outputSize];
	cudaMemcpy(cpuOutput, gpuOutput, outputBytes, cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < batchSize; i++)
	{
		for (size_t j = 0; j < outputFeatures; j++)
		{
			cout << cpuOutput[i * outputFeatures + j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	float* cpuInput = new float[inputSize];
	cudaMemcpy(cpuInput, gpuInput, inputBytes, cudaMemcpyDeviceToHost);
	
	float* cpuWeight = new float[weightSize];
	cudaMemcpy(cpuWeight, gpuWeight, weightBytes, cudaMemcpyDeviceToHost);
	
	float* cpuBias = new float[outputFeatures];
	cudaMemcpy(cpuBias, gpuBias, biasBytes, cudaMemcpyDeviceToHost);
	
	float err = 0.0f;
	for (size_t i = 0; i < batchSize; i++)
	{
		for (size_t j = 0; j < outputFeatures; j++)
		{
			float sum = cpuBias[j];
			for (size_t m = 0; m < inputFeatures; m++)
			{
				sum += cpuWeight[j * inputFeatures + m] * cpuInput[i * inputFeatures + m];
			}
			cout << sum << " ";
			err += abs(sum - cpuOutput[i * outputFeatures + j]);
		}
		cout << endl;
	}
	cout << endl;
	cout << "error: " << err / (batchSize * outputFeatures) << endl << endl;

	return 0;
}