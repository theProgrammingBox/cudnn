#include <cudnn.h>
#include <iostream>
#include <assert.h>
#include <algorithm>

using std::cout;
using std::endl;
using std::max;

int main(int argc, char const* argv[]) {
	// creating handle, it combines every detail of a convolution we are about to specify
	// into a single object describing the convolution
	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);

	// creating the parameter descriptor, basically tell me the details of your input
	// output, and filter tensors or images

	const uint64_t batchSize = 1;			// unique images running in parallel
	const uint64_t inputFeatures = 3;		// 3 image "features" for rgb
	const uint64_t outputFeatures = 3;		// 3 output "features", they are the result of the convolution
	const uint64_t inputImageRows = 64;		// height of the input images
	const uint64_t inputImageCols = 64;		// width of the input images
	const uint64_t outputImageRows = 64;	// height of the output images
	const uint64_t outputImageCols = 64;	// width of the output images
	const uint64_t filterRows = 3;			// assuming square filter for simplicity
	const uint64_t filterCols = 3;			// assuming square filter for simplicity
	
	//NHWC or NCHW
	// n is batch dimensions
	// c is channel dimensions, rgb for example
	// think of it as the input features in a fully connected layer
	cudnnTensorDescriptor_t input_descriptor;
	cudnnCreateTensorDescriptor(&input_descriptor);
	cudnnSetTensor4dDescriptor(input_descriptor,
		/*format=*/CUDNN_TENSOR_NHWC,		// NHWC or NCHW
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
		/*format=*/CUDNN_TENSOR_NHWC,
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

	cudnnConvolutionFwdAlgo_t convolution_algorithm;
	cudnnGetConvolutionForwardAlgorithm(cudnn,
		input_descriptor,						// think of each image in the channels as the input nodes
		kernel_descriptor,						// think of each filter as the weights from each input node to each output node
		convolution_descriptor,					// convolution details
		output_descriptor,						// think of each image in the channels as the output nodes
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,	// algorithm preference, i want the fastest one
		/*memoryLimitInBytes=*/0,				// memory limit, 0 for no limit
		&convolution_algorithm);				// outputs the algorithm
	// it can return CUDNN_CONVOLUTION_FWD_ALGO_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_FFT,
	// CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD, etc

	// now we need to ask how much memory the algorithm needs
	uint64_t workspace_bytes = 0;
	cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		input_descriptor,
		kernel_descriptor,
		convolution_descriptor,
		output_descriptor,
		convolution_algorithm,
		&workspace_bytes);
	cout << "Workspace size: " << workspace_bytes << " bytes" << endl;

	cout << "end" << endl;
}