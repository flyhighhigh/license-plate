#include "opencv2/opencv.hpp"
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

Mat tiltCorrection (Mat src){
	int height = src.size[0], width = src.size[1];
	
    int v = height/8;
	Point2f dstPt[4] = {Point2f(0,0), Point2f(width,0), Point2f(width,height), Point2f(0,height)};
    Point2f widerPt[4] = {Point2f(0,0), Point2f(width*2,0), Point2f(width*2,height), Point2f(0,height)};
    Point2f leftPt[4] = {Point2f(0,v), Point2f(width*2/3,-v), Point2f(width*2/3,height-v), Point2f(0,height+v)};
    Point2f rightPt[4] = {Point2f(width/3,-v), Point2f(width,v), Point2f(width,height+v), Point2f(width/3,height-v)};
    
	Mat W = getPerspectiveTransform(dstPt, widerPt);
    Mat L = getPerspectiveTransform(leftPt, dstPt);
    Mat R = getPerspectiveTransform(rightPt, dstPt);
    Mat new_wide, new_left, new_right, dst;

	warpPerspective(src, new_wide, W, Size(width*2,height), INTER_LINEAR);
    warpPerspective(src, new_left, L, Size(width,height), INTER_LINEAR);
    warpPerspective(src, new_right, R, Size(width,height), INTER_LINEAR);

    hconcat(new_left, new_right, new_left);
    vconcat(new_wide, new_left, dst);
	return dst;
	// imshow("warped", warped);
	// waitKey(0);
}
int main(int argc, char const *argv[])
{
	Mat img = imread("plate_my_image/IMG_20231009_144311.jpg");
	// imshow("img",img);
	Mat dst = tiltCorrection(img);
    resize(dst, dst, Size(img.size[1]/8,img.size[0]/8));
	imshow("new",dst);
	waitKey(0);
	return 0;
}