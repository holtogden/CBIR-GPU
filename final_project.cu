/* Final Project - Content Based Image Retrieval (CBIR)
   Author: Holt Ogden, Surbhi Gupta, Rob Foskin
   Course: CSS 535
   Date: 3/14/24
   Instructions to Run program:
      Run final_project.exe with the arguments <image_path>
      <image_path> should point to a folder containing .jpg images that will be processed into histograms
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <chrono>
#include <bitset>


const int INTENSITY_BINS = 25;
const int COLOR_CODE_BINS = 64;
const double RED_INTENSITY_WEIGHT = 0.299;
const double GREEN_INTENSITY_WEIGHT = 0.587;
const double BLUE_INTENSITY_WEIGHT = 0.114;

/* CUDA kernel to calculate Intensity of a given image */
__global__ void intensityCUDA_kernel(const unsigned char* imageData, size_t width, size_t height, int* intensityHistogram, int numBins, float binSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float red = static_cast<float>(imageData[4 * idx]);       // R
    float green = static_cast<float>(imageData[4 * idx + 1]); // G
    float blue = static_cast<float>(imageData[4 * idx + 2]);  // B

    // Convert RGB to intensity and determine which histogram bin it goes in
    float intensity = 0.299 * red + 0.587 * green + 0.114 * blue;
    int binIndex = ((int)intensity / 10);

    // Increment bin value atomically to avoid race conditions with other threads
    //atomicAdd(&intensityHistogram[binIndex], 1); // DISABLED FOR TESTING --- I am having trouble using this function
}

/* Calculate and return Intensity histogram sequentially */
int* calculateIntensitySeq(const unsigned char* imageData, size_t width, size_t height, std::chrono::duration<double>* runtimes_sec) {
    //Start clock
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    // Calculate Intensity for each pixel and sort into histogram bins
    double red, green, blue, intensity;
    int binIndex;
    int* intensityHistogram = new int[INTENSITY_BINS];
    memset(intensityHistogram, 0, INTENSITY_BINS * sizeof(int)); // Set initial bin values to 0
    for (int i = 0; i < width * height * 4; i += 4) {
        red = static_cast<double>(imageData[i]);       // R
        green = static_cast<double>(imageData[i + 1]); // G
        blue = static_cast<double>(imageData[i + 2]);  // B
        intensity = RED_INTENSITY_WEIGHT * red + GREEN_INTENSITY_WEIGHT * green + BLUE_INTENSITY_WEIGHT * blue;
        binIndex = ((int)intensity / 10);
        intensityHistogram[binIndex]++;
    }

    // Stop clock and store runtime
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    runtimes_sec[0] += std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    return intensityHistogram;
}

/* Calculate and return Color Code histogram sequentially */
int* calculateColorCodeSeq(const unsigned char* imageData, size_t width, size_t height, std::chrono::duration<double>* runtimes_sec) {
    //Start clock
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    // Calculate Color Code for each pixel and sort into histogram bins
    double red, green, blue;
    int binIndex;
    int* colorCodeHistogram = new int[COLOR_CODE_BINS];
    memset(colorCodeHistogram, 0, COLOR_CODE_BINS * sizeof(int)); // Set initial bin values to 0
    std::string current;
    std::string colorCodeString;
    for (int i = 0; i < width * height * 4; i += 4) {
        colorCodeString = "";
        red = static_cast<double>(imageData[i]); // Get RGB value as integer
        current = std::bitset<8>(red).to_string(); // Convert integer to 8 bit binary string
        colorCodeString += current.substr(0, 2); // Add leftmost two digits of binary string to Color Code string

        green = static_cast<double>(imageData[i + 1]); 
        current = std::bitset<8>(green).to_string();
        colorCodeString += current.substr(0, 2); 

        blue = static_cast<double>(imageData[i + 2]);
        current = std::bitset<8>(blue).to_string();
        colorCodeString += current.substr(0, 2); // After adding leftmost two digits of the three RGB strings, Color Code string will now represent a 6 bit binary number

        binIndex = std::stoi(colorCodeString, nullptr, 2); // Convert 6 digit binary string to integer. Value will be between 0 and 63 because 2^6 = 64
        colorCodeHistogram[binIndex]++;
    }

    // Stop clock and store runtime
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    runtimes_sec[1] += std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    return colorCodeHistogram;
}

/* Calculate and return Intensity histogram using CUDA kernel */
int* calculateIntensityCUDA(const unsigned char* h_imageData, size_t width, size_t height, std::chrono::duration<double>* runtimes_sec) {
    //Start clock
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    // Allocate device memory for image data and copy to device
    size_t imgSize = width * height * 4 * sizeof(unsigned char);
    unsigned char* d_imageData;
    cudaMalloc(&d_imageData, imgSize);  
    cudaMemcpy(d_imageData, h_imageData, imgSize, cudaMemcpyHostToDevice);

    // Allocate memory for Intensity histogram
    int* d_intensityHistogram;
    size_t histSize = INTENSITY_BINS * sizeof(int);
    cudaMalloc(&d_intensityHistogram, histSize);
    cudaMemset(d_intensityHistogram, 0, histSize); // Initialize histogram array to 0

    // Determine block and grid size
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Create CUDA kernel to calculate Intensity
    //intensityCUDA_kernel<<<gridSize, blockSize>>>(d_imageData, imgWidth, imgHeight, d_intensityHistogram, numBins, binSize); // DISABLED FOR TESTING

    // Copy results back to host and free memory
    int* h_intensityHistogram;
    cudaMemcpy(h_intensityHistogram, d_intensityHistogram, histSize, cudaMemcpyDeviceToHost);
    cudaFree(d_imageData);
    cudaFree(d_intensityHistogram);

    // Stop clock and store runtime
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    runtimes_sec[2] += std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    return h_intensityHistogram;
}

/* Calculate and return Color Code histogram using CUDA kernel */
int* calculateColorCodeCUDA(const unsigned char* imageData, size_t width, size_t height, std::chrono::duration<double>* runtimes_sec) {
    //Start clock
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    // Stop clock and store runtime
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    runtimes_sec[3] += std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    return nullptr; //TEST
}

/* Output Intensity and Color Code histograms for Sequential and CUDA to console for testing */
void outputHistograms(std::vector<int*> intensityHistograms_Seq, std::vector<int*> colorCodeHistograms_Seq, std::vector<int*> intensityHistograms_CUDA, std::vector<int*> colorCodeHistograms_CUDA) {
    // Output Sequential Intensity Histograms
    std::cout << "Sequential Intensity Histograms: " << std::endl;
    for (int i = 0; i < intensityHistograms_Seq.size(); i++) {
        std::cout << "Image " << i + 1 << ": ";
        for (int j = 0; j < INTENSITY_BINS; j++) {
            std::cout << (intensityHistograms_Seq[i])[j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;

    // Output Sequential Color Code Histograms
    std::cout << "Sequential Color Code Histograms: " << std::endl;
    for (int i = 0; i < colorCodeHistograms_Seq.size(); i++) {
        std::cout << "Image " << i + 1 << ": ";
        for (int j = 0; j < COLOR_CODE_BINS; j++) {
            std::cout << (colorCodeHistograms_Seq[i])[j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;

    // Output CUDA Intensity Histograms
    std::cout << "CUDA Intensity Histograms: " << std::endl;
    for (int i = 0; i < intensityHistograms_CUDA.size(); i++) {
        std::cout << "Image " << i + 1 << ": ";
        for (int j = 0; j < INTENSITY_BINS; j++) {
            std::cout << (intensityHistograms_CUDA[i])[j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;

    // Output CUDA Color Code Histograms
    std::cout << "CUDA Color Code Histograms: " << std::endl;
    for (int i = 0; i < colorCodeHistograms_CUDA.size(); i++) {
        std::cout << "Image " << i + 1 << ": ";
        for (int j = 0; j < COLOR_CODE_BINS; j++) {
            std::cout << (colorCodeHistograms_CUDA[i])[j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
}

void outputResults(std::chrono::duration<double>* runtimes_sec) {
    // Output runtimes
    std::cout << "Intensity Sequential Runtime: " << runtimes_sec[0].count() << std::endl;
    std::cout << "Color Code Sequential Runtime: " << runtimes_sec[1].count() << std::endl;
    std::cout << "Intensity CUDA Runtime: " << runtimes_sec[2].count() << std::endl;
    std::cout << "Color Code CUDA Runtime: " << runtimes_sec[3].count() << std::endl;
}


int main(int argc, char** argv) {
    // Read in arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <Image_Path>" << std::endl;
        return -1;
    }
    std::string imagePath = argv[1];

    // Create vectors to store histograms for Intensity and ColorCode
    std::vector<int*> intensityHistograms_Seq;
    std::vector<int*> colorCodeHistograms_Seq;
    std::vector<int*> intensityHistograms_CUDA;
    std::vector<int*> colorCodeHistograms_CUDA;

    // Create array to store timing data for each Sequential and CUDA histogram creation
    std::chrono::duration<double>* runtimes_sec = new std::chrono::duration<double>[4]; // Contains runtimes in seconds
    memset(runtimes_sec, 0, 4 * sizeof(std::chrono::duration<double>)); // Set all initial runtime values to 0

    // Read in each file, convert it to an array of pixels, then create histograms
    std::vector<std::string> filenames;
    cv::glob(imagePath + "/*.jpg", filenames, false); // Saves the filenames of every file in the given folder into a vector
    for (int i = 0; i < filenames.size(); i++) {
        // Read in image data
        cv::Mat image = cv::imread(filenames[i], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error: Image " << i << " cannot be loaded." << std::endl;
            return -1;
        }

        // Convert image data to RGBA data. Data will be stored as a 1D unsigned char array containing an RGBA value for each pixel
        cv::Mat imageRGBA;
        cv::cvtColor(image, imageRGBA, cv::COLOR_BGR2RGBA);

        // Calculate Intensity and Color code for image using Sequential and CUDA methods
        const int imgWidth = image.cols;
        const int imgHeight = image.rows;
        intensityHistograms_Seq.push_back(calculateIntensitySeq(imageRGBA.data, imgWidth, imgHeight, runtimes_sec));
        colorCodeHistograms_Seq.push_back(calculateColorCodeSeq(imageRGBA.data, imgWidth, imgHeight, runtimes_sec));
        //intensityHistograms_CUDA.push_back(calculateIntensityCUDA(imageRGBA.data, imgWidth, imgHeight, runtimes_sec)); // DISABLED FOR TESTING
        //colorCodeHistograms_CUDA.push_back(calculateColorCodeCUDA(imageRGBA.data, imgWidth, imgHeight, runtimes_sec)); // DISABLED FOR TESTING
    }

    // Output results to console
    outputHistograms(intensityHistograms_Seq, colorCodeHistograms_Seq, intensityHistograms_CUDA, colorCodeHistograms_CUDA);
    outputResults(runtimes_sec);

    /* Delete histogram arrays stored in vector ----- I am having trouble getting this to work, I think it's a problem with deleting a C array stored in a vector
    while (!intensityHistograms_Seq.empty()) {
        //std::cout << "Deleted histogram for image " << i << std::endl; //TEST
        delete[] intensityHistograms_Seq[0];
        intensityHistograms_Seq.erase(intensityHistograms_Seq.begin());
        //delete[] colorCodeHistograms_Seq[i];
        //delete[] intensityHistograms_CUDA[i];
        //delete[] colorCodeHistograms_CUDA[i];
    } */
    return 0;
}
