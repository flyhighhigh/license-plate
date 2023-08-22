#include "opencv2/opencv.hpp"
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

bool findIntersection(Point2f* line1, Point2f* line2, Point2f& resPt)
{
	Point2f pt1 = line1[0], pt2 = line1[1], pt3 = line2[0], pt4 = line2[1];
	Point2f a = pt1 - pt2, b = pt3 - pt4;
	float c = a.cross(b);
	// 兩線平行
	if (c == 0) return false;
	float t1 = pt1.x * pt2.y - pt1.y * pt2.x; // cross (pt1,pt2)
	float t2 = pt3.x * pt4.y - pt3.y * pt4.x; // cross (pt3,pt4)
	
	resPt.x = ( t1 * b.x - a.x * t2 ) / c;
	resPt.y = ( t1 * b.y - a.y * t2 ) / c;
	return true;
}

int main(int argc, char const *argv[])
{
    Mat img = imread("images/1.jpg");
	// imshow("img",img);
	int height = img.size[0], width = img.size[1];
	
	// preprocess
	Mat gray, hist, th1, bin2;
	cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	equalizeHist(gray, hist);
	threshold(hist, th1, 127, 255, THRESH_BINARY);
	medianBlur(th1, th1, 5);
	adaptiveThreshold(th1, bin2, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
	
	// Find lines
	Mat hough;
	vector<Vec4i> linesP;
	HoughLinesP(bin2, linesP, 1, CV_PI/180, 5, 25, 5);
	int lineSize = linesP.size();
	if(lineSize == 0) exit(0); // TODO: break
	
	// Lines' segmentation using angle (kmeans split to 2 set)
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
	
	// Find center angle of two line set
	vector<float> resultAngle;
	int H = labels.at<int>(0,0) == 0 ? 0 : 1; // horizonal index
	int V = H == 0 ? 1 : 0;
	for(int i =0; i< centers.rows; i++){
		resultAngle.push_back((atan2f(centers.at<float>(i,1), centers.at<float>(i,0)) + CV_PI) / 2);
	}
	// cout << "H: " << resultAngle[H] << endl;
	// cout << "V: " << resultAngle[V] << endl;

	float w = width / 2;
	float h = height / 2;
	Point2f lineTop[2], lineDwn[2], lineLft[2], lineRht[2];
	float cosVal = cosf(resultAngle[H]) * 1000;
	float sinVal = sinf(resultAngle[H]) * 1000;
	lineTop[0].x = w + cosVal; // x1
	lineTop[0].y = 0 - sinVal; // y1
	lineTop[1].x = w - cosVal; // x2
	lineTop[1].y = 0 + sinVal; // y2
	lineDwn[0].x = w + cosVal; // x1
	lineDwn[0].y = height - sinVal; // y1
	lineDwn[1].x = w - cosVal; // x2
	lineDwn[1].y = height + sinVal; // y2

	cosVal = cosf(resultAngle[V]) * 1000;
	sinVal = sinf(resultAngle[V]) * 1000;
	lineLft[0].x = 0 + cosVal; // x1
	lineLft[0].y = h - sinVal; // y1
	lineLft[1].x = 0 - cosVal; // x2
	lineLft[1].y = h + sinVal; // y2
	lineRht[0].x = width + cosVal; // x1
	lineRht[0].y = h - sinVal; // y1
	lineRht[1].x = width - cosVal; // x2
	lineRht[1].y = h + sinVal; // y2

	Point2f srcPt[4];
	Point2f dstPt[4] = {Point2f(0,0), Point2f(width,0), Point2f(width,height), Point2f(0,height)};
	bool b1, b2, b3, b4;
	b1 = findIntersection(lineLft, lineTop, srcPt[0]);
	b2 = findIntersection(lineTop, lineRht, srcPt[1]);
	b3 = findIntersection(lineRht, lineDwn, srcPt[2]);
	b4 = findIntersection(lineDwn, lineLft, srcPt[3]);
	// for(auto pt: srcPt){
	// 	cout << pt << endl;
	// }

	if(!(b1 && b2 && b3 && b4)) exit(0); // TODO: break

	Mat M = getPerspectiveTransform(srcPt, dstPt), warped;
	warpPerspective(img, warped, M, Size(width,height), INTER_LINEAR);
	// imshow("warped", warped);
	waitKey(0);
	return 0;
}