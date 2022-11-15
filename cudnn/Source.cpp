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
	const size_t batchSize = 2;
	const size_t inputFeatures = 3;
	const size_t outputFeatures = 2;

	float* input = new float[batchSize * inputFeatures];		// Input data
	float* weights = new float[outputFeatures * inputFeatures];	// Weights, but stored transposed
	float* output = new float[batchSize * outputFeatures];		// Output data

	cudnnHandle_t handle;
	cudnnCreate(&handle);

	///

	cudnnTensorDescriptor_t inputDescriptor;
	cudnnFilterDescriptor_t weightDescriptor;
	cudnnTensorDescriptor_t outputDescriptor;

	cudnnCreateTensorDescriptor(&inputDescriptor);
	cudnnCreateFilterDescriptor(&weightDescriptor);
	cudnnCreateTensorDescriptor(&outputDescriptor);

	cudnnSetTensor4dDescriptor(inputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, inputFeatures, 1, 1);
	cudnnSetFilter4dDescriptor(weightDescriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outputFeatures, inputFeatures, 1, 1);
	cudnnSetTensor4dDescriptor(outputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, outputFeatures, 1, 1);

	cudnnConvolutionDescriptor_t convolutionDescriptor;
	cudnnCreateConvolutionDescriptor(&convolutionDescriptor);
	cudnnSetConvolution2dDescriptor(convolutionDescriptor, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

	int maxAlgorithms;
	cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &maxAlgorithms);
	cudnnConvolutionFwdAlgoPerf_t* algorithms = new cudnnConvolutionFwdAlgoPerf_t[maxAlgorithms];
	cudnnFindConvolutionForwardAlgorithm(handle, inputDescriptor, weightDescriptor, convolutionDescriptor, outputDescriptor, maxAlgorithms, &maxAlgorithms, algorithms);
	cudnnConvolutionFwdAlgo_t bestAlgorithm = algorithms[0].algo;
	delete[] algorithms;
	
	size_t workspaceBytes;
	cudnnGetConvolutionForwardWorkspaceSize(handle, inputDescriptor, weightDescriptor, convolutionDescriptor, outputDescriptor, bestAlgorithm, &workspaceBytes);
	void* workspace;
	cudaMalloc(&workspace, workspaceBytes);

	float* gpuInput;
	float* gpuWeights;
	float* gpuOutput;
	cudaMalloc(&gpuInput, batchSize * inputFeatures * sizeof(float));
	cudaMalloc(&gpuWeights, outputFeatures * inputFeatures * sizeof(float));
	cudaMalloc(&gpuOutput, batchSize * outputFeatures * sizeof(float));
	cudaMemcpy(gpuInput, input, batchSize * inputFeatures * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuWeights, weights, outputFeatures * inputFeatures * sizeof(float), cudaMemcpyHostToDevice);
	
	curandGenerator_t randomGenerator;
	curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(randomGenerator, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());
	curandGenerateNormal(randomGenerator, gpuInput, batchSize * inputFeatures, 0, 1);
	curandGenerateNormal(randomGenerator, gpuWeights, outputFeatures * inputFeatures, 0, 1);
	cudaMemcpy(input, gpuInput, batchSize * inputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(weights, gpuWeights, outputFeatures * inputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	
	float alpha = 1.0f;
	float beta = 0.0f;
	cudnnConvolutionForward(handle, &alpha, inputDescriptor, gpuInput, weightDescriptor, gpuWeights, convolutionDescriptor, bestAlgorithm, workspace, workspaceBytes, &beta, outputDescriptor, gpuOutput);
	cudaMemcpy(output, gpuOutput, batchSize * outputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
	
	for (size_t i = 0; i < batchSize; i++)
	{
		for (size_t j = 0; j < outputFeatures; j++)
		{
			cout << output[i * outputFeatures + j] << " ";
		}
		cout << endl;
	}
	
	float error = 0.0f;
	for (size_t i = 0; i < batchSize; i++)
	{
		for (size_t j = 0; j < outputFeatures; j++)
		{
			float sum = 0.0f;
			for (size_t k = 0; k < inputFeatures; k++)
			{
				sum += input[i * inputFeatures + k] * weights[j * inputFeatures + k];
			}
			cout << sum << " ";
			error += abs(sum - output[i * outputFeatures + j]);
		}
		cout << endl;
	}
	cout << "Error: " << error / (batchSize * outputFeatures) << endl;

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
	
	return 0;
}