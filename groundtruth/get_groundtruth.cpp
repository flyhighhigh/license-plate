#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

int main(){
    vector<string> filenames = {"groundtruth_641.txt","groundtruth_2411.txt"};
    for(auto file:filenames){
        ifstream ifs(file);
        vector<vector<double>> groundtruths = { {0} }; 
        if(!ifs.is_open())continue;
        string line;
        string newline;
        
        while(getline(ifs, line)){
            stringstream ss("");
            int flag;
            double vals;
            ss << line << endl;
            ss >> flag;
            vector<double> v;
            v.push_back(flag);
            if(flag!=0){
                while(ss >> vals){
                    v.push_back(vals);
                }
            }
            groundtruths.push_back(v);
        }
        for(auto v: groundtruths){
            for(auto f: v){
                cout << f << " ";
            }
            cout << endl;
        }
        break;
    }
    
}