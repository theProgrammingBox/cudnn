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
	const size_t inputFeatures = 3;
	const size_t outputFeatures = 3;

	const size_t inputSize = batchSize * inputFeatures;
	const size_t weightSize = outputFeatures * inputFeatures;
	const size_t biasSize = outputFeatures;
	const size_t outputSize = batchSize * outputFeatures;
	
	const size_t inputBytes = inputSize * sizeof(float);
	const size_t weightBytes = weightSize * sizeof(float);
	const size_t biasBytes = biasSize * sizeof(float);
	const size_t outputBytes = outputSize * sizeof(float);

	const float one = 1.0f;
	const float zero = 0.0f;
	const float minusOne = -1.0f;

	curandGenerator_t randomGenerator;
	cudnnHandle_t cudnnHandle;
	cublasHandle_t cublasHandle;

	float* gpuInput;
	float* gpuInputGradient;
	cudnnTensorDescriptor_t inputDescriptor;
	
	float* gpuWeight;
	float* gpuWeightGradient;
	cudnnFilterDescriptor_t weightDescriptor;
	
	float* gpuBias;
	float* gpuBiasGradient;
	cudnnTensorDescriptor_t biasDescriptor;
	
	float* gpuOutput;
	float* gpuOutputGradient;
	cudnnTensorDescriptor_t outputDescriptor;
	
	cudnnConvolutionDescriptor_t propagationDescriptor;

	int maxForwardPropagationAlgorithms;
	cudnnGetConvolutionForwardAlgorithmMaxCount(
		cudnnHandle, 
		&maxForwardPropagationAlgorithms);
	cudnnConvolutionFwdAlgo_t bestForwardPropagationAlgorithm;

	int maxWeightBackwardPropagationAlgorithms;
	cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
		cudnnHandle, 
		&maxWeightBackwardPropagationAlgorithms);
	cudnnConvolutionBwdFilterAlgo_t bestWeightBackwardPropagationAlgorithm;
	
	int maxInputBackwardPropagationAlgorithms;
	cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
		cudnnHandle, 
		&maxInputBackwardPropagationAlgorithms);
	cudnnConvolutionBwdDataAlgo_t bestInputBackwardPropagationAlgorithm;

	size_t workspaceBytes = 0;
	void* gpuWorkspace;

	

	curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(
		randomGenerator, 
		duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());
	cudnnCreate(&cudnnHandle);
	cublasCreate(&cublasHandle);

	cudaMalloc(&gpuInput, inputBytes);
	cudaMalloc(&gpuInputGradient, inputBytes);
	cudnnCreateTensorDescriptor(&inputDescriptor);
	cudnnSetTensor4dDescriptor(
		inputDescriptor,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		batchSize,
		inputFeatures,
		1,
		1);

	// weight is stored transposed
	cudaMalloc(&gpuWeight, weightBytes);
	cudaMalloc(&gpuWeightGradient, weightBytes);
	curandGenerateNormal(
		randomGenerator, 
		gpuWeight, 
		weightSize + (weightSize & 1), 
		0, 
		1);
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
	cudaMalloc(&gpuBiasGradient, biasBytes);
	curandGenerateNormal(
		randomGenerator, 
		gpuBias, 
		biasSize + (biasSize & 1), 
		0, 
		1);
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
	cudaMalloc(&gpuOutputGradient, outputBytes);
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
	
	
	
	curandGenerateNormal(
		randomGenerator, 
		gpuInput, 
		inputSize + (inputSize & 1), 
		0, 
		1);
	
	cudnnConvolutionForward(
		cudnnHandle,
		&one,
		inputDescriptor,
		gpuInput,
		weightDescriptor,
		gpuWeight,
		propagationDescriptor,
		bestForwardPropagationAlgorithm,
		gpuWorkspace,
		workspaceBytes,
		&zero,
		outputDescriptor,
		gpuOutput);
	
	cudnnAddTensor(
		cudnnHandle,
		&one,
		biasDescriptor,
		gpuBias,
		&one,
		outputDescriptor,
		gpuOutput);


	
	float* cpuInput = new float[inputSize];
	cudaMemcpy(cpuInput, gpuInput, inputBytes, cudaMemcpyDeviceToHost);
	
	float* cpuWeight = new float[weightSize];
	cudaMemcpy(cpuWeight, gpuWeight, weightBytes, cudaMemcpyDeviceToHost);
	
	float* cpuBias = new float[outputFeatures];
	cudaMemcpy(cpuBias, gpuBias, biasBytes, cudaMemcpyDeviceToHost);

	float* cpuOutput = new float[outputSize];
	cudaMemcpy(cpuOutput, gpuOutput, outputBytes, cudaMemcpyDeviceToHost);
	
	cout << "Input:" << endl;
	for (size_t i = 0; i < batchSize; i++)
	{
		for (size_t j = 0; j < inputFeatures; j++)
		{
			cout << cpuInput[i * inputFeatures + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	
	cout << "Weight:" << endl;
	for (size_t i = 0; i < inputFeatures; i++)
	{
		for (size_t j = 0; j < outputFeatures; j++)
		{
			cout << cpuWeight[i * outputFeatures + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	
	cout << "Bias:" << endl;
	for (size_t i = 0; i < outputFeatures; i++)
	{
		cout << cpuBias[i] << " ";
	}
	cout << endl << endl;

	cout << "Output:" << endl;
	for (size_t i = 0; i < batchSize; i++)
	{
		for (size_t j = 0; j < outputFeatures; j++)
		{
			cout << cpuOutput[i * outputFeatures + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	
	cout << "Manual output:" << endl;
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



	cudaMemcpy(gpuOutputGradient, gpuInput, inputBytes, cudaMemcpyDeviceToDevice);

	cublasSaxpy(cublasHandle, outputSize, &minusOne, gpuOutput, 1, gpuOutputGradient, 1);
	
	cudnnConvolutionBackwardData(
		cudnnHandle,
		&one,
		weightDescriptor,
		gpuWeight,
		outputDescriptor,
		gpuOutputGradient,
		propagationDescriptor,
		bestInputBackwardPropagationAlgorithm,
		gpuWorkspace,
		workspaceBytes,
		&zero,
		inputDescriptor,
		gpuInputGradient);

	cudnnConvolutionBackwardFilter(
		cudnnHandle,
		&one,
		inputDescriptor,
		gpuInput,
		outputDescriptor,
		gpuOutputGradient,
		propagationDescriptor,
		bestWeightBackwardPropagationAlgorithm,
		gpuWorkspace,
		workspaceBytes,
		&zero,
		weightDescriptor,
		gpuWeightGradient);

	cudnnConvolutionBackwardBias(
		cudnnHandle,
		&one,
		outputDescriptor,
		gpuOutputGradient,
		&zero,
		biasDescriptor,
		gpuBiasGradient);

	

	float* cpuInputGradient = new float[inputSize];
	cudaMemcpy(cpuInputGradient, gpuInputGradient, inputBytes, cudaMemcpyDeviceToHost);

	float* cpuWeightGradient = new float[weightSize];
	cudaMemcpy(cpuWeightGradient, gpuWeightGradient, weightBytes, cudaMemcpyDeviceToHost);

	float* cpuBiasGradient = new float[outputFeatures];
	cudaMemcpy(cpuBiasGradient, gpuBiasGradient, biasBytes, cudaMemcpyDeviceToHost);

	float* cpuOutputGradient = new float[outputSize];
	cudaMemcpy(cpuOutputGradient, gpuOutputGradient, outputBytes, cudaMemcpyDeviceToHost);

	cout << "Output gradient:" << endl;
	for (size_t i = 0; i < batchSize; i++)
	{
		for (size_t j = 0; j < outputFeatures; j++)
		{
			cout << cpuOutputGradient[i * outputFeatures + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	
	cout << "Manual output gradient:" << endl;
	err = 0.0f;
	for (size_t i = 0; i < batchSize; i++)
	{
		for (size_t j = 0; j < outputFeatures; j++)
		{
			float sum = cpuInput[i * outputFeatures + j] - cpuOutput[i * outputFeatures + j];
			cout << sum << " ";
			err += abs(sum - cpuOutputGradient[i * outputFeatures + j]);
		}
		cout << endl;
	}
	cout << endl;
	cout << "error: " << err / (batchSize * outputFeatures) << endl << endl;

	cout << "Bias gradient:" << endl;
	for (size_t i = 0; i < outputFeatures; i++)
	{
		cout << cpuBiasGradient[i] << " ";
	}
	cout << endl << endl;
	
	cout << "Manual bias gradient:" << endl;
	err = 0.0f;
	for (size_t i = 0; i < outputFeatures; i++)
	{
		float sum = 0.0f;
		for (size_t j = 0; j < batchSize; j++)
		{
			sum += cpuOutputGradient[j * outputFeatures + i];
		}
		cout << sum << " ";
		err += abs(sum - cpuBiasGradient[i]);
	}
	cout << endl << endl;
	cout << "error: " << err / outputFeatures << endl << endl;

	cout << "Weight gradient:" << endl;
	for (size_t i = 0; i < inputFeatures; i++)
	{
		for (size_t j = 0; j < outputFeatures; j++)
		{
			cout << cpuWeightGradient[i * outputFeatures + j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	cout << "Manual weight gradient:" << endl;
	err = 0.0f;
	for (size_t i = 0; i < outputFeatures; i++)
	{
		for (size_t j = 0; j < inputFeatures; j++)
		{
			float sum = 0.0f;
			for (size_t m = 0; m < batchSize; m++)
			{
				sum += cpuInput[m * inputFeatures + j] * cpuOutputGradient[m * outputFeatures + i];
			}
			cout << sum << " ";
			err += abs(sum - cpuWeightGradient[i * inputFeatures + j]);
		}
		cout << endl;
	}
	cout << endl;
	cout << "error: " << err / (inputFeatures * outputFeatures) << endl << endl;
	
	cout << "Input gradient:" << endl;
	for (size_t i = 0; i < batchSize; i++)
	{
		for (size_t j = 0; j < inputFeatures; j++)
		{
			cout << cpuInputGradient[i * inputFeatures + j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	cout << "Manual input gradient:" << endl;
	err = 0.0f;
	for (size_t i = 0; i < batchSize; i++)
	{
		for (size_t j = 0; j < inputFeatures; j++)
		{
			float sum = 0.0f;
			for (size_t m = 0; m < outputFeatures; m++)
			{
				sum += cpuWeight[m * inputFeatures + j] * cpuOutputGradient[i * outputFeatures + m];
			}
			cout << sum << " ";
			err += abs(sum - cpuInputGradient[i * inputFeatures + j]);
		}
		cout << endl;
	}
	cout << endl;
	cout << "error: " << err / (batchSize * inputFeatures) << endl << endl;

	return 0;
}