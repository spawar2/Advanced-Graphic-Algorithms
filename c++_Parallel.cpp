void processUsingCpu(std::string input_file, std::string output_file) {
	// pointers to images in CPU's memory (h_) and GPU's memory (d_)
	uchar4        *h_rgbaImage, *d_rgbaImage;
	unsigned char *h_greyImage, *d_greyImage;
 
	//load the image and give us our input and output pointers
	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);
 
	GpuTimer timer;
	timer.Start();
	rgbaToGreyscaleCpu(h_rgbaImage, h_greyImage, numRows(), numCols());
	timer.Stop();
 
	int err = printf("Implemented CPU serial code ran in: %f msecs.\n", timer.Elapsed());
 
	if (err < 0) {
		//Couldn't print!
		std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
		exit(1);
	}
 
	//check results and output the grey image
	postProcess(output_file, h_greyImage);
}




// Parallel implementation for running on CPU using a single thread.
void rgbaToGreyscaleCpu(const uchar4* const rgbaImage, unsigned char *const greyImage,
		const size_t numRows, const size_t numCols)
{
	for (size_t r = 0; r < numRows; ++r) {
		for (size_t c = 0; c < numCols; ++c) {
			const uchar4 rgba = rgbaImage[r * numCols + c];
			const float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
			greyImage[r * numCols + c] = channelSum;
		}
	}
}


