const sampler_t sampler =   CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE |
                            CLK_FILTER_NEAREST;

__kernel void grayscale(__read_only  image2d_t src,
                        __write_only image2d_t dst)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    uint4 color;

    color = read_imageui(src, sampler, (int2)(x, y));
    uint gray = (color.x + color.y + color.z) / 3;
    write_imageui(dst, (int2)(x,y), (uint4)(gray, gray, gray, 0));
}


cl_program  prog;
        cl_kernel   kernel;
        String      kernelSrc = null;
        int[] err = new int[1];
        BufferedImage grayimg = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        // --------------------

        // Get the Sourcecodes
        try {
            kernelSrc  = CLFileLoader.GetSource(KERNEL_GRAYSCALE_PATH);
        }
        catch(IOException exc) {
            System.err.println("Error while loading Kernelfile: " + exc);
        }

        prog = clCreateProgramWithSource(cl.getContext(), 1, new String[] {kernelSrc}, null, err);
        clBuildProgram(prog, 0, null, null, null, err);
        kernel = clCreateKernel(prog, "grayscale", err);

        if(err[0] != CL_SUCCESS) {
            System.err.println("Couldnt create Graykernel");
            System.exit(1);
        }        
        byte dataBuffer[] = ((DataBufferByte)img.getRaster().getDataBuffer()).getData();

        cl_image_format imageFormat = new cl_image_format();
        imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;
        imageFormat.image_channel_order = CL_RGBA;

        cl_image_format formats[] = new cl_image_format[1];
        formats[0] = imageFormat;

        cl_mem input = clCreateImage2D(cl.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, formats, img.getWidth(),
                img.getHeight(), 0, Pointer.to(dataBuffer), err);
        cl_mem output = clCreateImage2D(cl.getContext(), CL_MEM_WRITE_ONLY, formats, img.getWidth(),
                img.getHeight(), 0, null, err);

        if(err[0] != CL_SUCCESS) {
            System.err.println("Couldnt create objects");
            System.exit(1);
        }

        // Set Args
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(input));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(output));

        // Transfer Image to the OpenCL-Device
        if(clEnqueueWriteImage(cl.getCommandQueue(), input, CL_TRUE, 
                new long[] {0,0,0}, new long [] {img.getWidth(), img.getHeight(), 1}, 
                img.getWidth(), 0, Pointer.to(dataBuffer), 0, null, null) != CL_SUCCESS) {
            System.err.println("Cant write Image");
            System.exit(1);
        }

        long workSize[] = new long[] {img.getWidth(), img.getHeight()};
        long workSizelocal[] = new long[] {2,2};

        if(clEnqueueNDRangeKernel(cl.getCommandQueue(), kernel, 2, null, 
                workSize, workSizelocal, 0, null, null) != CL_SUCCESS) {
            System.err.println("Cant apply Blackwhite-kernel");
            System.exit(1);
        }

        byte grayimg_byte[] = ((DataBufferByte)grayimg.getRaster().getDataBuffer()).getData();

        if(clEnqueueReadImage(cl.getCommandQueue(), output, CL_TRUE, new long[] {0,0,0}, 
                new long[] {img.getWidth(), img.getHeight(), 1}, img.getWidth(), 0, Pointer.to(grayimg_byte), 0, null, null) != CL_SUCCESS) {
            System.err.println("Cant read values from Blackwhite-kernel");
            System.exit(1);
        }

        clReleaseMemObject(input);
        clReleaseMemObject(output);

        return grayimg;