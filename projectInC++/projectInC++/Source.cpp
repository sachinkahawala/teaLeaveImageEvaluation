////#include <opencv2/opencv.hpp>
////#include <iostream>
////#include <conio.h>
//////using namespace std;
////int main() {
////	long long j = 0;
////	for (int i = 0; i < 10000000; i++) {
////		//std::cout << i;
////		//cout << "hello world!";
////		j += i;
////	}
////	std::cout << j;
////	std::cin.get();
////	return 0;
////
////}
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <opencv2\imgproc\imgproc.hpp>
#include <vector>
#include <numeric> 
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	String imageName("D://Projects//FYP//Tea Leave Image Evaluation//results//5-croped.jpeg"); // by default
	int iterations, k, minArea;
	iterations = 5, k = 3 , minArea =50;
	int values[3] = { 0, 125, 255};
	


	if (argc > 1)
	{
		imageName = argv[1];
	}
	Mat image, original_image;
	image = imread(imageName, IMREAD_COLOR); // Read the file
	original_image= image.clone();
	if (image.empty())                      // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		cin.get();
		return -1;
	}
	//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", image);                // Show our image inside it.
	
	int h, w;
	h = image.rows; w = image.cols;
	cout << h << " " << w << endl;

	Mat img_to_yuv;
	cvtColor(image, img_to_yuv, COLOR_BGR2YCrCb);
	//imshow("img_to_yuv", img_to_yuv);

	//Split the image into 3 channels; Y, Cr and Cb channels respectively and store it in a std::vector
	vector<Mat> vec_channels;
	split(img_to_yuv, vec_channels);

	//Equalize the histogram of only the Y channel 
	equalizeHist(vec_channels[0], vec_channels[0]);

	//Merge 3 channels in the vector to form the color image in YCrCB color space.
	merge(vec_channels, img_to_yuv);

	//Convert the histogram equalized image from YCrCb to BGR color space again
	Mat hist_equalization_result, hsi_applied_result;
	cvtColor(img_to_yuv, hist_equalization_result, COLOR_YCrCb2BGR);
	//imshow("hist_equalization_result", hist_equalization_result);
	cvtColor(hist_equalization_result, hsi_applied_result, COLOR_BGR2HLS);
	split(hsi_applied_result, vec_channels);
	image = vec_channels[2];

	 
	unsigned char clusterMat[1000][1000];

	for (int iteration = 0; iteration < iterations; iteration++) {
		uchar* p;
		for (int i = 0; i < h; ++i)
		{ 
			p = image.ptr<uchar>(i);
			for (int j = 0; j < w; ++j)
			{
				int cBest = 1231313123;
				for (int cluster = 0; cluster < k; cluster++) {
					int value = values[cluster];
					int imV = uchar(p[j]);
					int dist = (value - imV)*(value - imV);
					if (dist < cBest) {
						cBest = dist;
						clusterMat[i][j] = cluster;
					}


				}
			}
			
		}
		//cout << (int)clusterMat[0][0] << " " << (int)clusterMat[300][240] << endl;
		vector<vector<int> > vec(k);
		/*for (int i = 0; i < k; i++) {
			vec[i] = vector<int>(h*w);
		}*/
		for (int i = 0; i < h; ++i)
		{
			p = image.ptr<uchar>(i);
			for (int j = 0; j < w; ++j)
			{
				int imV = uchar(p[j]);
				vec[clusterMat[i][j]].push_back(imV);
			}
		}
		
		
		vector<float> newValues;
		
		for (int cluster = 0; cluster < k; cluster++) {
			if (vec[cluster].size() != 0) {
				float average = accumulate(vec[cluster].begin(), vec[cluster].end(), 0.0) / vec[cluster].size();
				newValues.push_back(average);
			}
			else {
				newValues.push_back(values[cluster]);
			}
			
		}
		for (int i = 0; i < k; i++) 
			values[i] = newValues[i];
		for (int i = 0; i < newValues.size(); ++i)
			cout << newValues[i] << ' ';
		cout << endl;
	
		
	}
	vector<float> colors(k);
	for (int cluster = 0; cluster < k; cluster++) {
		colors[cluster] = cluster * 255 / (k - 1);
	}
	
	for (int i = 0; i < h; ++i)
	{

		for (int j = 0; j < w; ++j)
		{
			float colorOfPixel = colors[clusterMat[i][j]];
			if (clusterMat[i][j] != k - 1) {
				original_image.at<Vec3b>(Point(j, i)) = Vec3b(0, 0, 0);
			}
		}
	}
	for (int i = 0; i < h; ++i)
	{

		for (int j = 0; j < w; ++j)
		{
			Vec3b intensity = original_image.at<Vec3b>(Point(j, i));
			if (intensity.val[1] < 150) {
				original_image.at<Vec3b>(Point(j, i)) = Vec3b(0, 0, 0);
			}
		}
	}

	Mat gray,binary;
	cvtColor(original_image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 127, 255, cv::THRESH_BINARY);

	Mat stats, centroids, labelImage;
	int nLabels = connectedComponentsWithStats(binary, labelImage, stats, centroids, 8, CV_32S);

	Mat mask(labelImage.size(), CV_8UC1, Scalar(0));
	Mat surfSup = stats.col(4) > minArea;
	for (int i = 1; i < nLabels; i++)
	{
		if (surfSup.at<uchar>(i, 0))
		{
			mask = mask | (labelImage == i);
		}
	}
	Mat r(original_image.size(), CV_8UC1, Scalar(0));
	original_image.copyTo(r, mask);
	imshow("Result", r);

	imshow("original_image", original_image);
	

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}