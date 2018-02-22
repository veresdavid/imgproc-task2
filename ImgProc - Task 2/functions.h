#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <ctime>
#include <cstdlib>
#include <iostream>

using namespace cv;
using namespace std;

Vec3b BLACK(0, 0, 0);
Vec3b WHITE(255, 255, 255);

unsigned char relativeLuminance(const Vec3b& color) {

	double b = color[0];
	double g = color[1];
	double r = color[2];

	double lum = r * 0.2126 + g * 0.7152 + b * 0.0722;

	return round(lum);

}

void imageToBinary(const Mat& src, Mat& dst) {

	int width = src.cols;
	int height = src.rows;

	dst = src.clone();

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			Vec3b color = src.at<Vec3b>(i, j);

			unsigned char lum = relativeLuminance(color);

			if (lum < 128) {
				dst.at<Vec3b>(i, j) = BLACK;
			}
			else {
				dst.at<Vec3b>(i, j) = WHITE;
			}

		}
	}

}

void generateNoise(const Mat& src, Mat& dst, int percentage) {

	int width = src.cols;
	int height = src.rows;

	dst = src.clone();

	int numberOfIterations = width * height / 100.0 * percentage;

	srand(time(NULL));

	for (int i = 0; i < numberOfIterations; i++) {

		int x = rand() % height;
		int y = rand() % width;

		if (dst.at<Vec3b>(x, y) == BLACK) {
			dst.at<Vec3b>(x, y) = WHITE;
		}
		else {
			dst.at<Vec3b>(x, y) = BLACK;
		}

	}

}

void collectValuesFromMat(const Mat& src, vector<unsigned char>& bValues, vector<unsigned char>& gValues, vector<unsigned char>& rValues, int centerX, int centerY, int size) {

	bValues.clear();
	gValues.clear();
	rValues.clear();

	int startX = centerX - (size / 2);
	int endX = centerX + (size / 2);
	int startY = centerY - (size / 2);
	int endY = centerY + (size / 2);

	for (int i = startX; i < endX; i++) {
		for (int j = startY; j < endY; j++) {

			Vec3b color = src.at<Vec3b>(i, j);

			bValues.push_back(color[0]);
			gValues.push_back(color[1]);
			rValues.push_back(color[2]);

		}
	}

}

double meanOfVector(const vector<unsigned char>& values) {

	int sum = 0;

	for (int i = 0; i < values.size(); i++) {
		sum += values[i];
	}

	return sum * 1.0 / values.size();

}

void meanFilter(const Mat& src, Mat& dst, int size) {

	if (size % 2 == 0) {
		throw "Size should be odd!";
	}

	int width = src.cols;
	int height = src.rows;

	if (width < size || height < size) {
		throw "The image is too small!";
	}

	dst = src.clone();

	int borderSize = size / 2;

	for (int i = 0 + borderSize; i < height - borderSize; i++) {
		for (int j = 0 + borderSize; j < width - borderSize; j++) {

			// compute mean of src values
			vector<unsigned char> bValues;
			vector<unsigned char> gValues;
			vector<unsigned char> rValues;
			collectValuesFromMat(src, bValues, gValues, rValues, i, j, size);

			double bMean = meanOfVector(bValues);
			double gMean = meanOfVector(gValues);
			double rMean = meanOfVector(rValues);

			Vec3b mean(round(bMean), round(gMean), round(rMean));

			// store it in dst
			dst.at<Vec3b>(i, j) = mean;

		}
	}

}

unsigned char medianOfVector(const vector<unsigned char>& values) {

	vector<unsigned char> copy(values);

	sort(copy.begin(), copy.end());

	return copy[copy.size() / 2];

}

void medianFilter(const Mat& src, Mat& dst, int size) {

	if (size % 2 == 0) {
		throw "Size should be odd!";
	}

	int width = src.cols;
	int height = src.rows;

	if (width < size || height < size) {
		throw "The image is too small!";
	}

	dst = src.clone();

	int borderSize = size / 2;

	for (int i = 0 + borderSize; i < height - borderSize; i++) {
		for (int j = 0 + borderSize; j < width - borderSize; j++) {

			// compute mean of src values
			vector<unsigned char> bValues;
			vector<unsigned char> gValues;
			vector<unsigned char> rValues;
			collectValuesFromMat(src, bValues, gValues, rValues, i, j, size);

			unsigned char bMedian = medianOfVector(bValues);
			unsigned char gMedian = medianOfVector(gValues);
			unsigned char rMedian = medianOfVector(rValues);

			Vec3b median(bMedian, gMedian, rMedian);

			// store it in dst
			dst.at<Vec3b>(i, j) = median;

		}
	}

}

void histogramOfMat(const Mat& src, vector<int>& histogram) {

	histogram.clear();

	histogram.resize(256, 0);

	int width = src.cols;
	int height = src.rows;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			unsigned char value = src.at<uchar>(i, j);

			histogram[value]++;

		}
	}

}

void histogramOfVector(const vector<unsigned char>& values, vector<int>& histogram) {

	histogram.clear();

	histogram.resize(256, 0);

	for (int i = 0; i < values.size(); i++) {
		
		unsigned char value = values[i];

		histogram[value]++;

	}

}

void cumulativeFrequencyDistribution(const vector<int>& histogram, vector<int>& cuf) {

	cuf.clear();

	cuf.resize(256, 0);

	cuf[0] = histogram[0];

	for (int i = 1; i < 256; i++) {
		cuf[i] = cuf[i - 1] + histogram[i];
	}

}

void frequencyOfEqualizedHistogram(int numberOfPixels, vector<int>& feq) {

	feq.clear();

	int minimumNumberOfPelsInBin = numberOfPixels / 256;
	int remainder = numberOfPixels - 256 * minimumNumberOfPelsInBin;

	feq.resize(256, minimumNumberOfPelsInBin);

	int startIndex = (256 - remainder) / 2;

	for (int i = startIndex; i < startIndex + remainder; i++) {
		feq[i]++;
	}

}

void equalizeHistogram(const vector<int>& cuf, const vector<int>& cufeq, vector<int>& equalized) {

	equalized.clear();

	equalized.resize(256, 0);

	for (int i = 0; i < 256; i++) {

		int outputIndex = 0;
		int distance = abs(cuf[i] - cufeq[0]);

		for (int j = 1; j < 256; j++) {

			int currentDistance = abs(cuf[i] - cufeq[j]);

			if (currentDistance <= distance) {

				outputIndex = j;
				distance = currentDistance;

			}
			else {
				break;
			}

		}

		equalized[i] = outputIndex;

	}

}

// http://www.mee.tcd.ie/~ack/teaching/1e8/histogram_equalisation_slides.pdf
void histogramEqualization(const Mat& src, Mat& dst) {

	int width = src.cols;
	int height = src.rows;

	vector<unsigned char> bValues;
	vector<unsigned char> gValues;
	vector<unsigned char> rValues;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			Vec3b color = src.at<Vec3b>(i, j);

			bValues.push_back(color[0]);
			gValues.push_back(color[1]);
			rValues.push_back(color[2]);

		}
	}

	// get histogram
	vector<int> bHistogram;
	vector<int> gHistogram;
	vector<int> rHistogram;
	histogramOfVector(bValues, bHistogram);
	histogramOfVector(gValues, gHistogram);
	histogramOfVector(rValues, rHistogram);

	// get cumulative frequency distribution
	vector<int> bCuf;
	vector<int> gCuf;
	vector<int> rCuf;
	cumulativeFrequencyDistribution(bHistogram, bCuf);
	cumulativeFrequencyDistribution(gHistogram, gCuf);
	cumulativeFrequencyDistribution(rHistogram, rCuf);

	// get ideal histogram
	vector<int> feq;
	frequencyOfEqualizedHistogram(width*height, feq);

	// get cumulative frequency distribution of ideal histogram
	vector<int> cufeq;
	cumulativeFrequencyDistribution(feq, cufeq);

	// equalize histogram
	vector<int> bEqualized;
	vector<int> gEqualized;
	vector<int> rEqualized;
	equalizeHistogram(bCuf, cufeq, bEqualized);
	equalizeHistogram(gCuf, cufeq, gEqualized);
	equalizeHistogram(rCuf, cufeq, rEqualized);

	// the new image
	dst = src.clone();

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {

			Vec3b oldColor = src.at<Vec3b>(i, j);

			Vec3b newColor(bEqualized[oldColor[0]], gEqualized[oldColor[1]], rEqualized[oldColor[2]]);

			dst.at<Vec3b>(i, j) = newColor;

		}
	}

}