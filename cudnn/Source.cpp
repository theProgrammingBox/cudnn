#include <cudnn.h>
#include <iostream>
#include <assert.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>

using std::cout;
using std::endl;
using std::max;

int main(int argc, char const* argv[]) {
	srand(time(NULL));

	if (false)
	{
		// creating handle, it combines every detail of a convolution we are about to specify
		// into a single object describing the convolution
		cudnnHandle_t cudnn;
		cudnnCreate(&cudnn);

		// creating the parameter descriptor, basically tell me the details of your input
		// output, and filter tensors or images

		const uint64_t batchSize = 1;			// unique images running in parallel
		const uint64_t inputFeatures = 3;		// 3 image "features" for rgb
		const uint64_t outputFeatures = 3;		// 3 output "features", they are the result of the convolution
		const uint64_t inputImageRows = 3;		// height of the input images
		const uint64_t inputImageCols = 3;		// width of the input images
		const uint64_t outputImageRows = 5;	// height of the output images
		const uint64_t outputImageCols = 5;	// width of the output images
		const uint64_t filterRows = 3;			// assuming square filter for simplicity
		const uint64_t filterCols = 3;			// assuming square filter for simplicity

		//NHWC or NCHW
		// n is batch dimensions
		// c is channel dimensions, rgb for example
		// think of it as the input features in a fully connected layer
		cudnnTensorDescriptor_t input_descriptor;
		cudnnCreateTensorDescriptor(&input_descriptor);
		cudnnSetTensor4dDescriptor(input_descriptor,
			/*format=*/CUDNN_TENSOR_NCHW,		// NHWC or NCHW
			/*dataType=*/CUDNN_DATA_FLOAT,		// data type
			/*batch_size=*/batchSize,			// unique images running in parallel
			/*channels=*/inputFeatures,			// number of "features" in each image like rgb in this case
			/*image_height=*/inputImageRows,	// height of each image
			/*image_width=*/inputImageCols);	// width of each image

		// output channels is the number of filters being used on the input
		// think of it like output features in a fully connected layer

		cudnnTensorDescriptor_t output_descriptor;
		cudnnCreateTensorDescriptor(&output_descriptor);
		cudnnSetTensor4dDescriptor(output_descriptor,
			/*format=*/CUDNN_TENSOR_NCHW,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*batch_size=*/batchSize,			// same as input, expect a unique result for every unique input
			/*channels=*/outputFeatures,		// number of filters being used on the input
			/*image_height=*/outputImageRows,	// in this case the output is the same size as the input
			/*image_width=*/outputImageCols);	// in this case the output is the same size as the input

		// kernels are filters, give it inputFeatures and outputFeatures
		// also give it the filter size, which is the same for both
		// height and width for this example and most other cases

		cudnnFilterDescriptor_t kernel_descriptor;
		cudnnCreateFilterDescriptor(&kernel_descriptor);
		cudnnSetFilter4dDescriptor(kernel_descriptor,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*format=*/CUDNN_TENSOR_NCHW,
			/*out_channels=*/outputFeatures,	// number of filters being used on the input
			/*in_channels=*/inputFeatures,		// number of "features" in each image like rgb in this case
			/*kernel_height=*/filterRows,		// height of each filter
			/*kernel_width=*/filterCols);		// width of each filter

		// convolution descriptor, this is where we specify the details of the convolution
		// like the padding, stride, and dilation

		const uint64_t verticalStride = 1;		// how many pixels to move the filter down, 1 is normal
		const uint64_t horizontalStride = 1;	// how many pixels to move the filter right
		// use an equation to calculate the padding using the input, output, and filter sizes
		assert(((outputImageRows - inputImageRows + filterRows - 1) & 1) == 0);
		const uint64_t verticalPadding = max(uint64_t((outputImageRows + filterRows - 1 - inputImageRows) * 0.5), uint64_t(0));
		assert(((outputImageCols + filterCols - 1 - inputImageCols) & 1) == 0);
		const uint64_t horizontalPadding = max(uint64_t((outputImageCols + filterCols - 1 - inputImageCols) * 0.5), uint64_t(0));
		const uint64_t verticalDilation = 1;	// how many pixels to skip when convolving, use 1 for no skipping
		const uint64_t horizontalDilation = 1;	// how many pixels to skip when convolving, use 1 for no skipping

		cudnnConvolutionDescriptor_t convolution_descriptor;
		cudnnCreateConvolutionDescriptor(&convolution_descriptor);
		cudnnSetConvolution2dDescriptor(convolution_descriptor,
			/*pad_height=*/verticalPadding,
			/*pad_width=*/horizontalPadding,
			/*vertical_stride=*/verticalStride,
			/*horizontal_stride=*/horizontalStride,
			/*dilation_height=*/verticalDilation,
			/*dilation_width=*/horizontalDilation,
			/*mode=*/CUDNN_CROSS_CORRELATION,	// convolution mode
			/*computeType=*/CUDNN_DATA_FLOAT);	// data type
		// there are a few convolution modes, cross correlation is the most common one
		// convolution is the other one, it is the same as cross correlation but flips the filter

		// now we need a more detailed description of the convolution as well as the memory limitations

		int requestedAlgoCount = 0;
		cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &requestedAlgoCount);
		cudnnConvolutionFwdAlgoPerf_t* perfResults = new cudnnConvolutionFwdAlgoPerf_t[requestedAlgoCount];
		cudnnFindConvolutionForwardAlgorithm(cudnn,
			input_descriptor,
			kernel_descriptor,
			convolution_descriptor,
			output_descriptor,
			requestedAlgoCount,
			&requestedAlgoCount,
			perfResults);

		// now we can choose the best algorithm by perfResults[0].algo
		// next we need to allocate memory for the output
		uint64_t workspaceBytes = 0;
		void* workspace = nullptr;
		cudnnGetConvolutionForwardWorkspaceSize(cudnn,
			input_descriptor,
			kernel_descriptor,
			convolution_descriptor,
			output_descriptor,
			perfResults[0].algo,
			&workspaceBytes);
		cudaMalloc(&workspace, workspaceBytes);

		// now we can run the convolution
		float alpha = 1.0f;
		float beta = 0.0f;

		// creating the random input, output, and filter data
		float* input = new float[batchSize * inputFeatures * inputImageRows * inputImageCols];
		float* output = new float[batchSize * outputFeatures * outputImageRows * outputImageCols];
		float* filter = new float[outputFeatures * inputFeatures * filterRows * filterCols];
		for (uint64_t i = 0; i < batchSize * inputFeatures * inputImageRows * inputImageCols; i++)
			input[i] = rand() / float(RAND_MAX);
		for (uint64_t i = 0; i < outputFeatures * inputFeatures * filterRows * filterCols; i++)
			filter[i] = rand() / float(RAND_MAX);

		// allocating memory on the gpu
		float* gpuInput;
		float* gpuOutput;
		float* gpuFilter;
		cudaMalloc(&gpuInput, batchSize * inputFeatures * inputImageRows * inputImageCols * sizeof(float));
		cudaMalloc(&gpuOutput, batchSize * outputFeatures * outputImageRows * outputImageCols * sizeof(float));
		cudaMalloc(&gpuFilter, outputFeatures * inputFeatures * filterRows * filterCols * sizeof(float));
		cudaMemcpy(gpuInput, input, batchSize * inputFeatures * inputImageRows * inputImageCols * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gpuFilter, filter, outputFeatures * inputFeatures * filterRows * filterCols * sizeof(float), cudaMemcpyHostToDevice);

		// running the convolution
		cudnnConvolutionForward(cudnn,
			&alpha,
			input_descriptor,
			gpuInput,
			kernel_descriptor,
			gpuFilter,
			convolution_descriptor,
			perfResults[0].algo,
			workspace,
			workspaceBytes,
			&beta,
			output_descriptor,
			gpuOutput);

		// copying the output back to the cpu
		cudaMemcpy(output, gpuOutput, batchSize * outputFeatures * outputImageRows * outputImageCols * sizeof(float), cudaMemcpyDeviceToHost);

		// printing the output
		for (uint64_t i = 0; i < batchSize; i++)
		{
			for (uint64_t j = 0; j < outputFeatures; j++)
			{
				for (uint64_t k = 0; k < outputImageRows; k++)
				{
					for (uint64_t l = 0; l < outputImageCols; l++)
					{
						cout << output[i * outputFeatures * outputImageRows * outputImageCols + j * outputImageRows * outputImageCols + k * outputImageCols + l] << " ";
					}
					cout << endl;
				}
				cout << endl;
			}
			cout << endl;
		}

		//cout padding
		cout << "verticalPadding: " << verticalPadding << endl;
		cout << "horizontalPadding: " << horizontalPadding << endl;
		cout << "verticalStride: " << verticalStride << endl;
		cout << "horizontalStride: " << horizontalStride << endl;

		// cpu implementation
		float err = 0;
		for (uint64_t i = 0; i < batchSize; i++)
		{
			for (uint64_t j = 0; j < outputFeatures; j++)
			{
				for (uint64_t k = -verticalPadding; k < outputImageRows * verticalStride - verticalPadding; k += verticalStride)
				{
					for (uint64_t l = -horizontalPadding; l < outputImageCols * horizontalStride - horizontalPadding; l += horizontalStride)
					{
						float sum = 0.0f;
						for (uint64_t m = 0; m < inputFeatures; m++)
						{
							for (uint64_t n = 0; n < filterRows; n++)
							{
								for (uint64_t o = 0; o < filterCols; o++)
								{
									if (k + n >= 0 && k + n < inputImageRows && l + o >= 0 && l + o < inputImageCols)
									{
										sum += input[i * inputFeatures * inputImageRows * inputImageCols + m * inputImageRows * inputImageCols + (k + n) * inputImageCols + (l + o)] * filter[j * inputFeatures * filterRows * filterCols + m * filterRows * filterCols + n * filterCols + o];
									}
								}
							}
						}
						err += abs(sum - output[i * outputFeatures * outputImageRows * outputImageCols + j * outputImageRows * outputImageCols + (k + verticalPadding) / verticalStride * outputImageCols + (l + horizontalPadding) / horizontalStride]);
					}
				}
			}
		}
		cout << "error: " << err / (batchSize * outputFeatures * outputImageRows * outputImageCols) << endl;
	}
	else
	{
		// now we are going to do matrix multiplication on cuDNN
		const uint64_t batchSize = 1;
		const uint64_t inputFeatures = 4;
		const uint64_t outputFeatures = 4;

		// make cudnnHandle
		cudnnHandle_t cudnnHandle;
		cudnnCreate(&cudnnHandle);
		
		// creating the input and output descriptors
		cudnnTensorDescriptor_t input_descriptor;
		cudnnTensorDescriptor_t output_descriptor;
		cudnnCreateTensorDescriptor(&input_descriptor);
		cudnnCreateTensorDescriptor(&output_descriptor);
		cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, inputFeatures, 1, 1);
		cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, outputFeatures, 1, 1);
		
		// creating the filter descriptor
		cudnnFilterDescriptor_t kernel_descriptor;
		cudnnCreateFilterDescriptor(&kernel_descriptor);
		cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outputFeatures, inputFeatures, 1, 1);
		
		// creating the convolution descriptor
		cudnnConvolutionDescriptor_t convolution_descriptor;
		cudnnCreateConvolutionDescriptor(&convolution_descriptor);
		cudnnSetConvolution2dDescriptor(convolution_descriptor, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
		
		// getting the workspace size
		uint64_t workspaceBytes = 0;
		void* workspace = nullptr;
		cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
			input_descriptor,
			kernel_descriptor,
			convolution_descriptor,
			output_descriptor,
			perfResults[0].algo,
			&workspaceBytes);
		cudaMalloc(&workspace, workspaceBytes);

		int outputImageRows = 1;
		int outputImageCols = 1;
		
		// allocating the output
		float* output = nullptr;
		cudaMalloc(&output, batchSize* outputFeatures* outputImageRows* outputImageCols * sizeof(float));
		
		// doing the convolution
		const float alpha = 1.0f;
		const float beta = 0.0f;

		// creating the random input, output, and filter data
		float* input = new float[batchSize * inputFeatures];
		float* filter = new float[outputFeatures * inputFeatures];
		for (uint64_t i = 0; i < batchSize * inputFeatures; i++)
		{
			input[i] = rand() / (float)RAND_MAX;
		}
		for (uint64_t i = 0; i < outputFeatures * inputFeatures; i++)
		{
			filter[i] = rand() / (float)RAND_MAX;
		}
		
		// copying the input and filter data to the GPU
		float* input_gpu = nullptr;
		float* filter_gpu = nullptr;
		cudaMalloc(&input_gpu, batchSize* inputFeatures * sizeof(float));
		cudaMalloc(&filter_gpu, outputFeatures* inputFeatures * sizeof(float));
		cudaMemcpy(input_gpu, input, batchSize* inputFeatures * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(filter_gpu, filter, outputFeatures* inputFeatures * sizeof(float), cudaMemcpyHostToDevice);
		
		// doing the convolution
		cudnnConvolutionForward(cudnnHandle, &alpha, input_descriptor, input_gpu, kernel_descriptor, filter_gpu, convolution_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, workspace, workspaceBytes, &beta, output_descriptor, output);
		
		// copying the output data back to the CPU
		float* output_cpu = new float[batchSize * outputFeatures];
		cudaMemcpy(output_cpu, output, batchSize* outputFeatures * sizeof(float), cudaMemcpyDeviceToHost);
		
		// printing the output
		for (uint64_t i = 0; i < batchSize; i++)
		{
			for (uint64_t j = 0; j < outputFeatures; j++)
			{
				cout << output_cpu[i * outputFeatures + j] << " ";
			}
			cout << endl;
		}
		
		// calculating the error
		float err = 0.0f;
		for (uint64_t i = 0; i < batchSize; i++)
		{
			for (uint64_t j = 0; j < outputFeatures; j++)
			{
				float sum = 0.0f;
				for (uint64_t m = 0; m < inputFeatures; m++)
				{
					sum += input[i * inputFeatures + m] * filter[j * inputFeatures + m];
				}
				err += abs(sum - output_cpu[i * outputFeatures + j]);
			}
		}
		cout << "error: " << err / (batchSize * outputFeatures) << endl;
	}
}