#include "opencv2/opencv.hpp"
#include "iostream"

using namespace std;

int main(int argc, char const *argv[])
{
    cv::Mat img = cv::imread("images/1.jpg");
	if (img.empty())
		cout << "image is empty or the path is invalid!" << endl;
	cv::imshow("Origin", img);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}