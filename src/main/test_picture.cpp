/*
  Copyright (C) 2017 Open Intelligent Machines Co.,Ltd

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "mtcnn.hpp"
#include "utils.hpp"

int main(int argc, char * argv[])
{
    std::string type = "mxnet";
    std::string fpath = "test.jpg";
    std::string model_dir = "../models"; 
    std::string out_dir = "../outputs";    
       
    int save_chop = 0;

    int res;

    while ((res = getopt(argc,argv,"f:m:o:t:s")) != -1) {
        switch (res) {
            case 'f':
                fpath = std::string(optarg);
                break;
            case 'm':
                model_dir = std::string(optarg);
                break;
            case 'o':
                out_dir = std::string(optarg);
                break;
            case 't':
                type = std::string(optarg);
                break;
            case 's':
                save_chop = 1;
                break;
            default:
                break;
        }
    }


    //cv::namedWindow(fpath, cv::WINDOW_AUTOSIZE);



    Mtcnn * p_mtcnn = MtcnnFactory::CreateDetector(type);

    if (p_mtcnn == nullptr) {
        std::cerr << type << " is not supported" << std::endl;
        std::cerr << "supported types: ";
        std::vector<std::string> type_list = MtcnnFactory::ListDetectorType();

        for (unsigned int i = 0; i < type_list.size(); i++)
            std::cerr << " " << type_list[i];

        std::cerr << std::endl;
        exit(1);
    }

    p_mtcnn->LoadModule(model_dir);

    // read image
    cv::Mat frame = cv::imread(fpath);
    if (!frame.data) {
        std::cerr << "failed to read image file: " << fpath << std::endl;
        exit(1);
    }

    int cycle = 0;
    while( cycle++ < 1) {

        std::vector<face_box> face_info;    
        unsigned long start_time = get_cur_time();
        p_mtcnn->Detect(frame,face_info);
        unsigned long end_time = get_cur_time();

        for(unsigned int i = 0; i < face_info.size(); i++) {
            face_box& box = face_info[i];
            std::ostringstream oss;
            oss << "face id: " << i << ". box: " << "(" << box.x0 << ", " << box.y0 << ")";
            std::cout << oss.str() << std::endl;
            printf("face %d: x0,y0 %2.5f %2.5f  x1,y1 %2.5f  %2.5f conf: %2.5f\n",i,
                    box.x0,box.y0,box.x1,box.y1, box.score);
            printf("landmark: ");

            for(unsigned int j = 0; j < 5; j++)
                printf(" (%2.5f %2.5f)",box.landmark.x[j], box.landmark.y[j]);

            printf("\n");

            if (save_chop) {
                cv::Mat corp_img = frame(cv::Range(box.y0, box.y1), cv::Range(box.x0, box.x1));
                auto outputs = str_split(fpath, '/');
                std::string fname = outputs.back();
                std::ostringstream oname;
                oname << out_dir << "/" << "chop_" << i << "_" << fname;
                if (!cv::imwrite(oname.str(), corp_img)) {
                    std::cerr << "can't save chopped image: " << oname.str() << std::endl;
                }
            }

            // draw box
            cv::rectangle(frame, cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1), cv::Scalar(0, 255, 0), 2);

            // draw landmark  scalar: BGR
            for (int l = 0; l < 5; l++) {
                cv::circle(frame, cv::Point(box.landmark.x[l],box.landmark.y[l]), 2, cv::Scalar(0, 0, 255), 2);
            }
        }

        std::cout << "total detected: " << face_info.size() << " faces. used "
             << (end_time-start_time) << " us" << std::endl;

    }
 
    cv::imshow(fpath, frame);
    cv::waitKey(0);

    return 0;
}
