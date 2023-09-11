/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/dpu_task.hpp>
#include <vitis/ai/nnpp/yolov3.hpp>

#include<time.h>
#include<chrono>
#include<fstream>
#include<streambuf> 
using namespace std;
using namespace cv;

// The parameters of yolov3_voc, each value could be set as actual needs.
// Such format could be refer to the prototxts in /etc/dpu_model_param.d.conf/.
const string yolov3det_config = {
	"   name: \"yolov3tinydet-4\" \n"
	"   model_type : YOLOv3 \n"
	"   yolo_v3_param { \n"
	"     num_classes: 1 \n"
	"     anchorCnt: 3 \n"
	"     conf_threshold: 0.1 \n"
	"     nms_threshold: 0.45 \n"
	"     layer_name: \"10\" \n"
	"     layer_name: \"13\" \n"
	"     biases: 66 \n"
	"     biases: 35 \n"
	"     biases: 77 \n"
	"     biases: 60 \n"
	"     biases: 135 \n"
	"     biases: 59 \n"
	"     biases: 165 \n"
	"     biases: 96 \n"
	"     biases: 295 \n"
	"     biases: 142 \n"
	"     biases: 356 \n"
	"     biases: 287 \n"
	"     test_mAP: false \n"
	"   } \n" };
/* initiall anchor
const string yolov3rec_config = {
	"   name: \"yolov3tinyrec\" \n"
	"   model_type : YOLOv3 \n"
	"   yolo_v3_param { \n"
	"     num_classes: 36 \n"
	"     anchorCnt: 3 \n"
	"     conf_threshold: 0.3 \n"
	"     nms_threshold: 0.2 \n"
	"     layer_name: \"10\" \n"
	"     layer_name: \"13\" \n"
	"     biases: 10 \n"
	"     biases: 14 \n"
	"     biases: 23 \n"
	"     biases: 27 \n"
	"     biases: 37 \n"
	"     biases: 58 \n"
	"     biases: 81 \n"
	"     biases: 82 \n"
	"     biases: 135 \n"
	"     biases: 169 \n"
	"     biases: 344 \n"
	"     biases: 319 \n"
	"     test_mAP: false \n"
	"   } \n" };
*/
const string yolov3rec_config = {
	"   name: \"yolov3tinyrecanc\" \n"
	"   model_type : YOLOv3 \n"
	"   yolo_v3_param { \n"
	"     num_classes: 36 \n"
	"     anchorCnt: 3 \n"
	"     conf_threshold: 0.4 \n"
	"     nms_threshold: 0.2 \n"
	"     layer_name: \"22\" \n"
	"     layer_name: \"26\" \n"
	"     biases: 66 \n"
	"     biases: 35 \n"
	"     biases: 77 \n"
	"     biases: 60 \n"
	"     biases: 135 \n"
	"     biases: 59 \n"
	"     biases: 165 \n"
	"     biases: 96 \n"
	"     biases: 295 \n"
	"     biases: 142 \n"
	"     biases: 356 \n"
	"     biases: 287 \n"
	"     test_mAP: false \n"
	"   } \n" };
bool sortFunc(const vector<float>& p1, const vector<float>& p2) {
	return p1[0] > p2[0];
}

bool sortGFunc(const vector<float>&p1, const vector<float>& p2) {
	return p1[0] < p2[0];
}

class TreeNode
{
public:
	TreeNode *left;
	TreeNode *right;
	bool Class;
	bool state;
	vector<int> category;
	TreeNode(TreeNode *l, TreeNode *r, bool c, bool s, vector<int> &cate)
	{
		left = l;
		right = r;
		Class = c;
		state = s;
		category = cate;
	}
};

vector<string> classes{ "0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z" };

int main(int argc, char* argv[]) {
	// A kernel name, it should be samed as the dnnc result. e.g.
	// /usr/share/vitis_ai_library/models/yolov3_voc/yolov3_voc.elf
	auto kernel_det_name = argv[1];
	cout << "kernel_det_name: " << kernel_det_name << endl;
	auto kernel_rec_name = argv[2];
	cout << "kernel_rec_name: " << kernel_rec_name << endl;
	string video_name = argv[3];
	cout << "video_name: " << video_name << endl;
	// Read image from a path.
	/*
	vector<Mat> imgs;
	vector<string> imgs_names;

	for (int i = 2; i < argc; i++) {
	  // image file names.
	  auto img = cv::imread(argv[i]);
	  if (img.empty()) {
		std::cout << "Cannot load " << argv[i] << std::endl;
		continue;
	  }
	  imgs.push_back(img);
	  imgs_names.push_back(argv[i]);
	}

	if (imgs.empty()) {
	  std::cerr << "No image load success!" << std::endl;
	  abort();
	}
	*/
	// Read video
	VideoCapture cap;
	//cap.open(argv[3]);
	string gs = "filesrc location=" + video_name + " ! h264parse ! omxh264dec ! video/x-raw, width=1920, height=1080, format=NV12, framerate=30/1 ! appsink";
	cap.open(gs, CAP_GSTREAMER);

	//cap.open("rtspsrc location=rtsp://root:admin9369@140.123.102.158/live1s1.sdp ! rtph264depay ! h264parse ! omxh264dec ! video/x-raw, width=1920, height=1080, format=NV12, framerate=30/1 ! appsink", CAP_GSTREAMER);
	if (!cap.isOpened())
	{
		cout << "Could not read this video" << endl;
		abort();
	}
	//build the tree
	vector<int> empty;
	vector<int> four{ 4 };
	vector<int> eight{ 8 };
	vector<int> zeroTothree{ 0,1,2,3 };
	vector<int> zerotwo{ 0,2 };
	vector<int> zeroTofoureight{ 0,1,2,3,4,8 };
	vector<int> zeroone{ 0,1 };
	vector<int> fiveToseven{ 5,6,7 };
	vector<int> six{ 6 };
	vector<int> zeroTofour{ 0,1,2,3,4 };
	vector<int> zeroToeight{ 0,1,2,3,4,5,6,7,8 };

	TreeNode *root = new TreeNode(NULL, NULL, 0, 0, empty);

	TreeNode *TreeNode1 = new TreeNode(NULL, NULL, 1, 0, empty);
	TreeNode *TreeNode2 = new TreeNode(NULL, NULL, 0, 0, empty);

	TreeNode *TreeNode3 = new TreeNode(NULL, NULL, 1, 0, empty);
	TreeNode *TreeNode4 = new TreeNode(NULL, NULL, 0, 0, empty);
	TreeNode *TreeNode5 = new TreeNode(NULL, NULL, 1, 0, empty);
	TreeNode *TreeNode6 = new TreeNode(NULL, NULL, 0, 0, empty);

	TreeNode *TreeNode7 = new TreeNode(NULL, NULL, 1, 0, empty);
	TreeNode *TreeNode8 = new TreeNode(NULL, NULL, 0, 0, empty);
	TreeNode *TreeNode9 = new TreeNode(NULL, NULL, 1, 0, empty);
	TreeNode *TreeNode10 = new TreeNode(NULL, NULL, 0, 0, empty);
	TreeNode *TreeNode11 = new TreeNode(NULL, NULL, 1, 0, empty);
	TreeNode *TreeNode12 = new TreeNode(NULL, NULL, 0, 0, empty);
	TreeNode *TreeNode13 = new TreeNode(NULL, NULL, 1, 0, empty);
	TreeNode *TreeNode14 = new TreeNode(NULL, NULL, 0, 0, empty);

	TreeNode *TreeNode15 = new TreeNode(NULL, NULL, 1, 0, empty);
	TreeNode *TreeNode16 = new TreeNode(NULL, NULL, 0, 1, four);
	TreeNode *TreeNode17 = new TreeNode(NULL, NULL, 1, 1, four);
	TreeNode *TreeNode18 = new TreeNode(NULL, NULL, 0, 1, four);
	TreeNode *TreeNode19 = new TreeNode(NULL, NULL, 1, 0, empty);
	TreeNode *TreeNode20 = new TreeNode(NULL, NULL, 1, 0, empty);
	TreeNode *TreeNode21 = new TreeNode(NULL, NULL, 1, 0, empty);
	TreeNode *TreeNode22 = new TreeNode(NULL, NULL, 1, 0, empty);
	TreeNode *TreeNode23 = new TreeNode(NULL, NULL, 1, 1, eight);
	TreeNode *TreeNode24 = new TreeNode(NULL, NULL, 1, 1, four);

	TreeNode *TreeNode25 = new TreeNode(NULL, NULL, 1, 0, empty);
	TreeNode *TreeNode26 = new TreeNode(NULL, NULL, 0, 1, zeroTothree);
	TreeNode *TreeNode27 = new TreeNode(NULL, NULL, 1, 1, zeroTothree);
	TreeNode *TreeNode28 = new TreeNode(NULL, NULL, 0, 1, zeroTothree);
	TreeNode *TreeNode29 = new TreeNode(NULL, NULL, 1, 1, zerotwo);
	TreeNode *TreeNode30 = new TreeNode(NULL, NULL, 1, 0, empty);
	TreeNode *TreeNode31 = new TreeNode(NULL, NULL, 1, 1, zeroTothree);
	TreeNode *TreeNode32 = new TreeNode(NULL, NULL, 1, 0, empty);
	TreeNode *TreeNode33 = new TreeNode(NULL, NULL, 1, 1, zeroTofoureight);
	TreeNode *TreeNode34 = new TreeNode(NULL, NULL, 1, 1, zeroTothree);

	TreeNode *TreeNode35 = new TreeNode(NULL, NULL, 1, 1, zeroone);
	TreeNode *TreeNode36 = new TreeNode(NULL, NULL, 1, 1, zeroTothree);
	TreeNode *TreeNode37 = new TreeNode(NULL, NULL, 0, 1, zeroone);
	TreeNode *TreeNode38 = new TreeNode(NULL, NULL, 0, 1, fiveToseven);
	TreeNode *TreeNode39 = new TreeNode(NULL, NULL, 1, 1, zeroone);
	TreeNode *TreeNode40 = new TreeNode(NULL, NULL, 1, 1, six);
	TreeNode *TreeNode41 = new TreeNode(NULL, NULL, 1, 1, six);
	TreeNode *TreeNode42 = new TreeNode(NULL, NULL, 1, 1, six);
	TreeNode *TreeNode43 = new TreeNode(NULL, NULL, 1, 1, zeroone);
	TreeNode *TreeNode44 = new TreeNode(NULL, NULL, 1, 1, zeroTofour);

	TreeNode *TreeNode45 = new TreeNode(NULL, NULL, 1, 1, zeroone);
	TreeNode *TreeNode46 = new TreeNode(NULL, NULL, 1, 1, zeroToeight);


	root->left = TreeNode1;
	root->right = TreeNode2;

	TreeNode1->left = TreeNode3;
	TreeNode1->right = TreeNode4;
	TreeNode2->left = TreeNode5;
	TreeNode2->right = TreeNode6;

	TreeNode3->left = TreeNode7;
	TreeNode3->right = TreeNode8;
	TreeNode4->left = TreeNode9;
	TreeNode4->right = TreeNode10;
	TreeNode5->left = TreeNode11;
	TreeNode5->right = TreeNode12;
	TreeNode6->left = TreeNode13;
	TreeNode6->right = TreeNode14;

	TreeNode7->left = TreeNode15;
	TreeNode7->right = TreeNode16;
	TreeNode8->left = TreeNode17;
	TreeNode8->right = TreeNode18;
	TreeNode9->left = TreeNode19;
	TreeNode10->left = TreeNode20;
	TreeNode11->left = TreeNode21;
	TreeNode12->left = TreeNode22;
	TreeNode13->left = TreeNode23;
	TreeNode14->left = TreeNode24;

	TreeNode15->left = TreeNode25;
	TreeNode15->right = TreeNode26;
	TreeNode16->left = TreeNode27;
	TreeNode16->right = TreeNode28;
	TreeNode19->left = TreeNode29;
	TreeNode20->left = TreeNode30;
	TreeNode21->left = TreeNode31;
	TreeNode22->left = TreeNode32;
	TreeNode23->left = TreeNode33;
	TreeNode24->left = TreeNode34;

	TreeNode25->left = TreeNode35;
	TreeNode26->left = TreeNode36;
	TreeNode26->right = TreeNode37;
	TreeNode28->right = TreeNode38;
	TreeNode29->left = TreeNode39;
	TreeNode30->left = TreeNode40;
	TreeNode31->left = TreeNode41;
	TreeNode32->left = TreeNode42;
	TreeNode33->left = TreeNode43;
	TreeNode34->left = TreeNode44;

	TreeNode35->left = TreeNode45;
	TreeNode44->left = TreeNode46;

	//VideoWriter VW;
	//hdmi display
	//VW.open("appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw, format=NV12 ! kmssink driver-name=xlnx plane-id=39 sync=false fullscreen-overlay=true", CAP_GSTREAMER  , 0,(double)30,Size(1920,1080),true);
		//write .nv12 
	//VW.open("appsrc ! video/x-raw, format=BGR ! queue ! videoconvert ! video/x-raw, format=NV12 ! omxh264enc target-bitrate=3000 ! video/x-h264, alignment=au ! filesink location=./oooo.h264", CAP_GSTREAMER, 0, 30, Size(1920,1080),true);  
// Create a dpu task object.  (detect)
	auto task_det = vitis::ai::DpuTask::create(kernel_det_name);
	auto batch_det = task_det->get_input_batch(0, 0);
	// Set the mean values and scale values.
	task_det->setMeanScaleBGR({ 0.0f, 0.0f, 0.0f },
		{ 0.00390625f,0.00390625f, 0.00390625f });
	auto input_tensor_det = task_det->getInputTensor(0u);
	CHECK_EQ((int)input_tensor_det.size(), 1)
		<< " the dpu model must have only one input";
	auto width_det = input_tensor_det[0].width;
	auto height_det = input_tensor_det[0].height;
	auto size_det = cv::Size(width_det, height_det);
	// Create a config and set the correlating data to control post-process.
	vitis::ai::proto::DpuModelParam config_det;
	// Fill all the parameters.
	auto ok_det =
		google::protobuf::TextFormat::ParseFromString(yolov3det_config, &config_det);
	if (!ok_det) {
		cerr << "Set parameters failed!" << endl;
		abort();
	}

	// Create a dpu task object. (recognize)
	auto task_rec = vitis::ai::DpuTask::create(kernel_rec_name);
	auto batch_rec = task_rec->get_input_batch(0, 0);
	// Set the mean values and scale values.
	task_rec->setMeanScaleBGR({ 0.0f, 0.0f, 0.0f },
		{ 0.00390625f,0.00390625f, 0.00390625f });
	auto input_tensor_rec = task_rec->getInputTensor(0u);
	CHECK_EQ((int)input_tensor_rec.size(), 1)
		<< " the dpu model must have only one input";
	auto width_rec = input_tensor_rec[0].width;
	auto height_rec = input_tensor_rec[0].height;
	auto size_rec = cv::Size(width_rec, height_rec);
	// Create a config and set the correlating data to control post-process.
	vitis::ai::proto::DpuModelParam config_rec;
	// Fill all the parameters.

	auto ok_rec =
		google::protobuf::TextFormat::ParseFromString(yolov3rec_config, &config_rec);
	if (!ok_rec) {
		cerr << "Set parameters failed!" << endl;
	}



// --- start detection ---

	Mat frame_det, image_det;
	vector<Mat> inputs_det;
	vector<int> input_cols_det, input_rows_det;
	int frameCount = 1;
	string plateStr = "";
	int flag = 0;
	string showStr = "";
	string mainPlate = "";
	int mainPlateCount = 0;

	//write CSV
	ofstream oFile;
	oFile.open("result.csv", ios::out | ios::trunc);
	oFile << "frame" << "," << "detected" << "," << "recognized" << endl;
	ofstream detFile;
	detFile.open("detected.txt", ios::out | ios::trunc);

	auto t_start = chrono::high_resolution_clock::now();
	while (cap.read(frame_det))
	{
		oFile << frameCount << ",";
		cout << "Frame: " << frameCount << endl;
		cvtColor(frame_det, frame_det, COLOR_YUV2BGR_NV12);
		plateStr = "";
		showStr = "";
		flag = 0;
		Mat plate;
		//input_cols_det.push_back(frame_det.cols);
		//input_rows_det.push_back(frame_det.rows);
		resize(frame_det, image_det, size_det); // frame_det原圖 image_det縮放後 size_det模型tensor的size

		//cout<<"cols: "<<frame.cols_det<<", rows: "<<frame.rows_det<<endl;
		//inputs_det.push_back(image_det);
		
		task_det->setImageBGR(image_det); // 將image資訊放入detection task
		task_det->run(0u);
		auto output_tensor_det = task_det->getOutputTensor(0u);
		// Execute the yolov3 post-processing.
		auto results_det = vitis::ai::yolov3_post_process(
			input_tensor_det, output_tensor_det, config_det, frame_det.cols, frame_det.rows);
		//cout<<"size: "<<inputs.size()<<endl;
		/* Print the results */
		
		// Convert coordinate and draw boxes at originimage.
		vector<vector<float> > sortConf_det;
		//push all result in the vector
		for (auto& box : results_det.bboxes) { // 所有結果的batch box
			int label = box.label;
			float xmin = box.x * frame_det.cols + 1;
			float ymin = box.y * frame_det.rows + 1;
			float xmax = xmin + box.width * frame_det.cols;
			float ymax = ymin + box.height * frame_det.rows;
			if (xmin < 0.) xmin = 1.;
			if (ymin < 0.) ymin = 1.;
			if (xmax > frame_det.cols) xmax = frame_det.cols;
			if (ymax > frame_det.rows) ymax = frame_det.rows;
			float confidence = box.score;
			vector<float> v;
			v.push_back(confidence);
			v.push_back(xmin);
			v.push_back(ymin);
			v.push_back(xmax);
			v.push_back(ymax);
			sortConf_det.push_back(v);
		}
		//sort by confidence 如果車牌偵測到兩個 則依信心程度排名
		if (sortConf_det.size() > 1)
			sort(sortConf_det.begin(), sortConf_det.end(), sortFunc);
		int plateValid = 0;
		int iouX, iouY, iouW, iouH;

		//get the plate area in initial image 取得原本圖片中的車牌部分
		for (int i = 0; i < sortConf_det.size(); i++)
		{
			float confidence = sortConf_det[0][0]; // 取最大信心的 或是唯一一個
			int xmin = int(sortConf_det[0][1]);
			int ymin = int(sortConf_det[0][2]);
			int xmax = int(sortConf_det[0][3]);
			int ymax = int(sortConf_det[0][4]);
			//use aspect ratio, width, height to avoid the error judge
			if (((xmax - xmin) / (ymax - ymin) < 1) || ((xmax - xmin) / (ymax - ymin) > 3) || (xmax - xmin > 200) || (ymax - ymin > 100))
				continue;
			//widen the region 取得車牌部分時，多取一點 以避免不小心切掉
			xmin = max(xmin - 5, 0);
			xmax = min(xmax + 5, frame_det.cols);
			ymin = max(ymin - 5, 0);
			ymax = min(ymax + 5, frame_det.rows);
			//cout << "RESULT: " << "plate" << "\t" << xmin << "\t" << ymin << "\t" << xmax << "\t" << ymax << "\t" << confidence << "\n";
			plate = frame_det(Rect(xmin, ymin, xmax - xmin, ymax - ymin));
			iouX = xmin; iouY = ymin; iouW = xmax - xmin; iouH = ymax - ymin;
			
			// my code here
			resize(plate, plate, Size(512, 224));

			//rectangle(frame_det, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0), 2, 1, 0);

			plateValid = 1;
			break;
		}
		if (plateValid){
			detFile << "1 " << iouX << " " << iouY << " " << iouW << " " << iouH << endl; 
		}else{
			detFile << "0" << endl;
		}
		if (plateValid) // 如果上面的code有偵測到車牌
		{
			//cout<<"col: "<<plate.cols<<", row: "<<plate.rows<<endl;
			vector<Mat> inputs_rec;
			vector<int> input_cols_rec, input_rows_rec;
			input_cols_rec.push_back(plate.cols);
			input_rows_rec.push_back(plate.rows);
			inputs_rec.push_back(plate);
			task_rec->setImageBGR(inputs_rec);
			task_rec->run(0u);
			auto output_tensor_rec = task_rec->getOutputTensor(0u);
			// Execute the yolov3 post-processing.
			auto results_rec = vitis::ai::yolov3_post_process(
				input_tensor_rec, output_tensor_rec, config_rec, input_cols_rec, input_rows_rec);
			//cout<<"size: "<<inputs.size()<<endl;
				/* Print the results */
				// Convert coordinate and draw boxes at originimage.

			//cout << "image_name " << frameCount << endl;
			vector<vector<float> > sortX_rec;

			//push all result in the vector 辨識出每一個字 並且裝進vector
			for (auto& box : results_rec[0].bboxes) {
				int label = box.label;
				float xmin = box.x * input_cols_rec[0] + 1;
				float ymin = box.y * input_rows_rec[0] + 1;
				float xmax = xmin + box.width * input_cols_rec[0];
				float ymax = ymin + box.height * input_rows_rec[0];
				if (xmin < 0.) xmin = 1.;
				if (ymin < 0.) ymin = 1.;
				if (xmax > frame_det.cols) xmax = input_cols_rec[0];
				if (ymax > frame_det.rows) ymax = input_rows_rec[0];
				float confidence = box.score;
				vector<float> v;
				//v.push_back(confidence);
				v.push_back(xmin);
				v.push_back(ymin);
				v.push_back(xmax);
				v.push_back(ymax);
				v.push_back(confidence);
				v.push_back(float(label));
				sortX_rec.push_back(v);
			}
			//sort by X location, to get plate result
			if (sortX_rec.size() > 1) //把 每一個字依照x軸做sort
				sort(sortX_rec.begin(), sortX_rec.end(), sortGFunc);

			//string plateStr="";
			for (int i = 0; i < sortX_rec.size(); i++)
			{
				float confidence = sortX_rec[i][4];
				int xmin = int(sortX_rec[i][0]);
				int ymin = int(sortX_rec[i][1]);
				int xmax = int(sortX_rec[i][2]);
				int ymax = int(sortX_rec[i][3]);
				string word = classes[int(sortX_rec[i][5])];
				//int boxw=xmax-xmin;
				//int boxh=ymax-ymin;
			//use aspect ratio, width, height to avoid the error judge
			//if( ((xmax-xmin)/(ymax-ymin)<1) || ((xmax-xmin)/(ymax-ymin)>3) || (xmax-xmin>200) || (ymax-ymin>100) )
				//continue;
			//widen the region
			//xmin=max(xmin-5,0);
			//xmax=min(xmax+5,frame_det.cols);
			//ymin=max(ymin-5,0);
			//ymax=min(ymax+5,frame_det.rows);
				//cout << "RESULT_STR: " << word << "\t" << xmin << "\t" << ymin << "\t" << xmax << "\t" << ymax << "\t" << confidence << "\n";
				//plate=frame_det(Rect(xmin,ymin,xmax-xmin,ymax-ymin));
			//resize(plate,plate,Size(512,224));
			//imwrite("./Plate/"+to_string(frameCount)+"_result.jpg", plate);

				//rectangle(plate, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 255, 0), 1, 1, 0);
				//putText(plate, word, Point(xmin, ymin - 10), 0, 1.1, Scalar(255, 0, 0), 2, 1, 0);
				plateStr += word;

				//xmin=xmin * frame_det.cols+1;
				//ymin=ymin * frame_det.rows+1;
				//xmax = xmin + boxw*frame_det.cols;
				//ymax = ymin + boxh*frame_det.rows;
				//cout<<xmin<<" "<<xmax<<" "<<ymin<<" "<<ymax<<endl;
				//rectangle(frame_det,Point(xmin,ymin), Point(xmax, ymax),Scalar(0,255,0), 1, 1, 0);
				//putText(frame_det,word, Point(xmin,ymin-10),0, 1.1,Scalar(255,0,0),2,1,0);
			}
			oFile<<1<<",";
			oFile<<plateStr<<endl;
			//imwrite("./Result_plate/"+to_string(frameCount)+"_"+plateStr+"_result.jpg",plate);
		}
		else
		{
			oFile<<0<<",";
			oFile<<""<<endl;
		}
		//Verify license plate number is valid or not
		/*
		TreeNode *temp = root;
		if (plateStr == "")
			showStr = "None";

		else if (plateStr.size() < 4 || plateStr.size() > 7)
		{
			showStr += ": invalid";
			plateStr="";
		}
		else
		{
			//travel the tree
			temp = root;
			for (auto c : plateStr)
			{
				if (temp == NULL)
					break;
				if (isdigit(c))
					temp = temp->left;
				else
					temp = temp->right;
			}
			if (temp == NULL || temp->state != 1)
			{
				plateStr = "";
				showStr += ": invalid";
			}
			else
			{
				showStr += ": valid, category: ";
				for (auto s : temp->category)
					showStr += to_string(s) + " ";
			}
		}

		//vote to choice plateStr result
		if (mainPlate == plateStr && mainPlateCount < 5 && plateStr != "")
		{
			mainPlateCount ++;
			flag = 1;
		}
		else
			mainPlateCount--;

		if (mainPlateCount < 1)
		{
			mainPlateCount = 1;
			mainPlate = plateStr;
		}
		else
		{
			//if (!flag)
				//plateStr = mainPlate + "(remain)";
			//else
				plateStr = mainPlate;
		}

		//4 lines means vote
		if(plateStr=="")
			oFile<<0<<",";
		else
			oFile<<1<<",";
		*/
		//oFile<<plateStr<<endl;
		//putText(frame_det, plateStr + showStr, Point(20, 40), 0, 1.3, Scalar(255, 0, 0), 2, 1, 0);
		//if(frameCount%50==0)
		//imwrite("./Result/"+to_string(frameCount) + "_result.jpg", frame_det);
		//VW.write(frame_det);

		inputs_det.clear();
		input_cols_det.clear();
		input_rows_det.clear();
		//inputs_rec.clear();
		//input_cols_rec.clear();
		//input_rows_rec.clear();
		//j = -1;
		//cout<<"end"<<endl;
		frameCount++;

		//if (frameCount > 200)
			//break;
	}
	cap.release();
	//VW.release();
	destroyAllWindows();
	auto t_end = chrono::high_resolution_clock::now();
	double diff = chrono::duration<double, milli>(t_end - t_start).count();
	//double diff = ((double)(end-start))/CLOCKS_PER_SEC;
	cout << diff << "s" << endl;
	cout << "fps:" << frameCount / diff * 1000 << endl;

	detFile.close(); // new 最底下
	
	return 0;
}


