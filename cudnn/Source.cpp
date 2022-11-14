#include <cudnn.h>
#include <iostream>

int main(int argc, char const* argv[]) {
	// creating handle, it combines every detail of a convolution we are about to specify
	// into a single object describing the convolution
	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);

	// creating the parameter descriptor, basically tell me the details of your input
	// output, and filter tensors or images

	const uint64_t imageRows = 64;		// using 64 as a placeholder
	const uint64_t imageCols = 64;		// assuming square images for input
	const uint64_t batchSize = 1;
	const uint64_t inputFeatures = 3;	// 3 images for rgb
	const uint64_t outputFeatures = 3;	// 3 output "images", they are the result of the convolution
	const uint64_t filterSize = 3;		// assuming square filter for simplicity
	
	//NHWC or NCHW
	// n is batch dimensions
	// c is channel dimensions, rgb for example
	// think of it as the input features in a fully connected layer
	cudnnTensorDescriptor_t input_descriptor;
	cudnnCreateTensorDescriptor(&input_descriptor);
	cudnnSetTensor4dDescriptor(input_descriptor,
		/*format=*/CUDNN_TENSOR_NHWC,	// NHWC or NCHW
		/*dataType=*/CUDNN_DATA_FLOAT,	// data type
		/*batch_size=*/batchSize,		// unique images running in parallel
		/*channels=*/inputFeatures,		// number of "features" in each image like rgb in this case
		/*image_height=*/imageRows,		// height of each image
		/*image_width=*/imageCols);		// width of each image

	// output channels is the number of filters being used on the input
	// think of it like output features in a fully connected layer

	cudnnTensorDescriptor_t output_descriptor;
	cudnnCreateTensorDescriptor(&output_descriptor);
	cudnnSetTensor4dDescriptor(output_descriptor,
		/*format=*/CUDNN_TENSOR_NHWC,
		/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/batchSize,		// same as input, expect a unique result for every unique input
		/*channels=*/outputFeatures,	// number of filters being used on the input
		/*image_height=*/imageRows,		// in this case the output is the same size as the input
		/*image_width=*/imageCols);		// in this case the output is the same size as the input

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
		/*kernel_height=*/filterSize,		// height of each filter, assuming square filter for simplicity
		/*kernel_width=*/filterSize);		// width of each filter, assuming square filter for simplicity
	
	// convolution descriptor, this is where we specify the details of the convolution
	// like the padding, stride, and dilation

	const uint64_t verticalPadding = 1;
	const uint64_t horizontalPadding = 1;
	const uint64_t verticalStride = 1;
	const uint64_t horizontalStride = 1;
	const uint64_t verticalDilation = 1;
	const uint64_t horizontalDilation = 1;

	cudnnConvolutionDescriptor_t convolution_descriptor;
	cudnnCreateConvolutionDescriptor(&convolution_descriptor);
	cudnnSetConvolution2dDescriptor(convolution_descriptor,
		/*pad_height=*/1,		// padding height
		/*pad_width=*/1,		// padding width
		/*vertical_stride=*/1,	// vertical stride
		/*horizontal_stride=*/1,	// horizontal stride
		/*dilation_height=*/1,	// dilation height
		/*dilation_width=*/1,	// dilation width
		/*mode=*/CUDNN_CROSS_CORRELATION,	// convolution mode
		/*computeType=*/CUDNN_DATA_FLOAT);	// data type

	std::cout << "end\n";
}