#include "opencv2/opencv.hpp"
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int main(int argc, char const *argv[])
{
    Mat img = imread("images/1.jpg");
	int height = img.size[0], width = img.size[1];
	
	// preprocess
	Mat gray, hist, th1, bin2;
	cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	equalizeHist(gray, hist);
	threshold(hist, th1, 127, 255, THRESH_BINARY);
	medianBlur(th1, th1, 5);
	adaptiveThreshold(th1, bin2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
	
	// Find lines and segmentation
	Mat hough;
	vector<Vec4i> linesP;
	HoughLinesP(bin2, linesP, 1, CV_PI/180, 5, 25, 5);
	int lineSize = linesP.size();
	if(lineSize == 0) return 0;
	
	vector<Point2f> angles;
	angles.reserve(lineSize + 4);
	angles.push_back(Point2f(cosf(CV_PI), sinf(CV_PI)));
	angles.push_back(Point2f(cosf(CV_2PI), sinf(CV_2PI)));
	for(auto l: linesP){
		float temp = 2 * (atan2f((l[1]-l[3]),(l[2]-l[0]))) + CV_PI;
		angles.push_back(Point2f(cosf(temp), sinf(temp)));
	}
	Mat labels, centers;
	kmeans(angles, 2, labels,
			TermCriteria( TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0),
			10, KMEANS_RANDOM_CENTERS, centers);
	
	// cout << centers.rows << " "<< centers.cols << endl;
	int H = labels.at<int>(0,0) == 0 ? 0 : 1;
	
	vector<float> resultAngle;
	for(int i =0; i< centers.rows; i++){
		resultAngle.push_back((atan2f(centers.at<float>(i,1), centers.at<float>(i,0)) + CV_PI) / 2);
	}
	cout << "H: " << resultAngle[H] << endl;
	cout << "V: " << resultAngle[H==0?1:0] << endl;

	// vector<Vec4i> linesH, linesV;
	// for( int i=0;i<labels.rows;i++){
	// 	cout << labels.at<int>(i,0) << endl;
	// }
	


	return 0;
}