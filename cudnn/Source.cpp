#include <cudnn.h>
#include <curand.h>
#include <iostream>
#include <chrono>
#include <cmath>

using std::cout;
using std::endl;
using std::max;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::fabs;

/*
write a program that uses cublas to multiply two matrices
A x B = C
a x b * b x c = a x c
*/

int main()
{
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());

	// initialize cudnn
	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);

	// initialize parameters
	int a = 700;
	int b = 500;
	int c = 300;
	
	// initialize data
	float* A, * B, * C;
	cudaMalloc(&A, a * b * sizeof(float));
	cudaMalloc(&B, b * c * sizeof(float));
	cudaMalloc(&C, a * c * sizeof(float));
	
	curandGenerateUniform(gen, A, a * b);
	curandGenerateUniform(gen, B, b * c);
	
	// initialize descriptors
	cudnnTensorDescriptor_t A_desc, C_desc;
	cudnnFilterDescriptor_t B_desc;
	cudnnCreateTensorDescriptor(&A_desc);
	cudnnCreateTensorDescriptor(&C_desc);
	cudnnCreateFilterDescriptor(&B_desc);
	
	cudnnSetTensor4dDescriptor(A_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, a, b, 1, 1);
	cudnnSetTensor4dDescriptor(C_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, a, c, 1, 1);
	cudnnSetFilter4dDescriptor(B_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, c, b, 1, 1);

	cudnnConvolutionDescriptor_t propagationDescriptor;
	cudnnCreateConvolutionDescriptor(&propagationDescriptor);
	cudnnSetConvolution2dDescriptor(propagationDescriptor, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
	
	// initialize algorithm using cudnnFindConvolutionForwardAlgorithm
	cudnnConvolutionFwdAlgoPerf_t algorithmPerf;
	int maxForwardPropagationAlgorithms = 1;
	cudnnFindConvolutionForwardAlgorithm(cudnn, A_desc, B_desc, propagationDescriptor, C_desc, maxForwardPropagationAlgorithms, &maxForwardPropagationAlgorithms, &algorithmPerf);
	
	// initialize workspace
	size_t workspace_size;
	cudnnGetConvolutionForwardWorkspaceSize(cudnn, A_desc, B_desc, propagationDescriptor, C_desc, algorithmPerf.algo, &workspace_size);
	void* workspace;
	cudaMalloc(&workspace, workspace_size);
	
	// initialize alpha and beta
	float alpha = 1.0f;
	float beta = 0.0f;
	
	// initialize time
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	// perform matrix multiplication
	int iterations = 10;
	while (iterations--) {
		cudnnConvolutionForward(cudnn, &alpha, A_desc, A, B_desc, B, propagationDescriptor, algorithmPerf.algo, workspace, workspace_size, &beta, C_desc, C);
	}
	
	// stop timer
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Elapsed time: " << elapsedTime / 10 << " ms" << endl;
	
	// print results
	float* C_host = new float[a * c];
	cudaMemcpy(C_host, C, a * c * sizeof(float), cudaMemcpyDeviceToHost);
	/*cout << "C: " << endl;
	for (int i = 0; i < a; i++)
	{
		for (int j = 0; j < c; j++)
		{
			cout << C_host[i * c + j] << " ";
		}
		cout << endl;
	}
	cout << endl;*/
	
	// calculate result on host
	float* C_host2 = new float[a * c];
	float* A_host = new float[a * b];
	float* B_host = new float[b * c];
	cudaMemcpy(A_host, A, a * b * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(B_host, B, b * c * sizeof(float), cudaMemcpyDeviceToHost);

	//the weight is transposed
	for (int i = 0; i < a; i++)
	{
		for (int j = 0; j < c; j++)
		{
			C_host2[i * c + j] = 0;
			for (int k = 0; k < b; k++)
			{
				C_host2[i * c + j] += A_host[i * b + k] * B_host[j * b + k];
			}
		}
	}
	/*cout << "C_host2: " << endl;
	for (int i = 0; i < a; i++)
	{
		for (int j = 0; j < c; j++)
		{
			cout << C_host2[i * c + j] << " ";
		}
		cout << endl;
	}
	cout << endl;*/

	/*cout << "A: " << endl;
	for (int i = 0; i < a; i++)
	{
		for (int j = 0; j < b; j++)
		{
			cout << A_host[i * b + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	
	cout << "B: " << endl;
	for (int i = 0; i < b; i++)
	{
		for (int j = 0; j < c; j++)
		{
			cout << B_host[i * c + j] << " ";
		}
		cout << endl;
	}
	cout << endl;*/
	
	// print average error
	float error = 0;
	for (int i = 0; i < a * c; i++)
	{
		error += fabs(C_host[i] - C_host2[i]);
	}
	error /= a * c;
	cout << "Average error: " << error << endl;
	
	// free memory
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	cudaFree(workspace);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudnnDestroyTensorDescriptor(A_desc);
	cudnnDestroyTensorDescriptor(C_desc);
	cudnnDestroyFilterDescriptor(B_desc);
	cudnnDestroyConvolutionDescriptor(propagationDescriptor);
	cudnnDestroy(cudnn);
	
	return 0;
}