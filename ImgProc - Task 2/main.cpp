#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "functions.h"

using namespace cv;
using namespace std;

int main() {

	// load image
	Mat original = imread("D:\\images\\lorem_a4.png");
	// Mat original = imread("D:\\images\\lorem_small.png");
	imshow("Original", original);
	imwrite("D:\\images\\task2\\01-original.png", original);

	/*
	// binary image
	Mat binary;
	imageToBinary(original, binary);
	imshow("Binary", binary);
	*/

	// generate noise
	Mat noisy;
	generateNoise(original, noisy, 5);
	imshow("Noisy", noisy);
	imwrite("D:\\images\\task2\\02-noisy.png", noisy);

	// mean filter
	Mat mean;
	meanFilter(noisy, mean, 3);
	imshow("Mean - 3x3", mean);
	imwrite("D:\\images\\task2\\03-mean.png", mean);

	// median filter
	Mat median;
	medianFilter(noisy, median, 3);
	imshow("Median - 3x3", median);
	imwrite("D:\\images\\task2\\03-median.png", median);

	// histogram equalization
	Mat equalized;
	histogramEqualization(median, equalized);
	imshow("Histogram equalization", equalized);
	imwrite("D:\\images\\task2\\04-histogram_equalization.png", equalized);

	// wait
	waitKey();

	return 0;
}