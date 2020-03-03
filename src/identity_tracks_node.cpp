#include<ros/ros.h>
#include <algorithm>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
string video_path = "/home/esoman/cpp_test/src/identity_tracks/output.mp4";

class tracks{
    public:
        tracks(){
            pts.push_back(Point(530,443));
            pts.push_back(Point(730,443));
            pts.push_back(Point(730,712));
            pts.push_back(Point(530,712));
            contour.push_back(pts);
        }
        void find_base_pre();
        Point2i find_base(int);
        bool slide_window(Point2i pix,bool flag);
        VideoCapture capture;
        Mat frame;
        Mat frame_gray;//灰度图
        Mat frame_gauss;//高斯平滑1图
        Mat frame_sobel,frame_sobel2;//sobel算子
        Mat roi;
        vector<vector<Point> > contour;
        vector<Point> pts;
        Mat frame_roi;//框处roi区域
        Mat frame_lone_dil;//膨胀图像
        Mat frame_lone;
        Mat frame_warp;
        bool flag = 0;
        int pix_num = 0;
        Point2d line_pix;
        vector<Point2d> edge_pix;
        bool img_init = false;
        int left_base_warp,right_base_warp;//投影后的左右线基点
        int left_base_pre,right_base_pre;//投影前的左右边基点
        Point2i left_base,right_base;
        const int warp_left = 630;
        const int warp_right = 650;
        const bool left_flag = true;
        const bool right_flag = false;
        vector<Point2i> left_edge,right_edge;
};
/*搜索轨道起始点*/
void tracks::find_base_pre(){
       // if(img_init == false){
            vector<int> left_white_pix,right_white_pix;
            left_white_pix.clear();
            right_white_pix.clear();
            cout << "left:" << endl;
            for(int i = 630;i < 640;i++){
                int left_num = 0;
                for(int j = 680; j<712 ;j++){
                    if(frame_warp.at<uchar>(j,i) == 255){
                        left_num++;
                    }
                }
                cout << "i="<<i <<"->"<< left_num << " ";
                left_white_pix.push_back(left_num);
            }
            cout << endl;
            cout << "right:"<< endl;
            for(int i = 640;i < 650;i++){
                int right_num = 0;
                for(int j = 680; j<712 ;j++){
                    if(frame_warp.at<uchar>(j,i) == 255){
                        right_num++;
                    }
                }
                cout << "i=" <<i<<"->"<<right_num;
                right_white_pix.push_back(right_num);
            }
            cout << endl;
            std::vector<int>::iterator left_biggest = std::max_element(std::begin(left_white_pix),std::end(left_white_pix));
            std::vector<int>::iterator right_biggest = std::max_element(std::begin(right_white_pix),std::end(right_white_pix));
            img_init = true;
            left_base_warp = 630+std::distance(left_white_pix.begin(),left_biggest);
            right_base_warp = 640+std::distance(right_white_pix.begin(),right_biggest);
            left_base_pre = pts[0].x + (left_base_warp - warp_left)*(pts[1].x - pts[0].x)/(warp_right-warp_left);
            right_base_pre = pts[1].x - (warp_right - right_base_warp )*(pts[1].x - pts[0].x)/(warp_right-warp_left);
       //  }
        //circle(frame_lone,Point2d(left_base_pre,710),1,Scalar(255,255,0));
        //circle(frame_lone,Point2d(right_base_pre,710),1,Scalar(255,255,0));

}
/*找基点*/
Point2i tracks::find_base(int prebase_pix){
        vector<Point2d> pp;
        Point2d base,sum;
        for(int y = -20;y < 20;y++){
            for(int x = -20;x < 20;x++){
                Point2d ppt(prebase_pix + x,705 + y);
                if(frame_lone_dil.at<uchar>(ppt.y,ppt.x) == 255){
                    pp.push_back(ppt);
                }
            }
        }
        for(int i = 0;i < pp.size();i++){
            sum += pp[i];
        }
        base.x = sum.x / pp.size();
        base.y = sum.y / pp.size();
        return base;
}
/*滑动窗口巡线*/
bool tracks::slide_window(Point2i pix,bool flag){
    vector<Point2i> ppp;
    Point2i basep(0,0),sump(0,0);
    ppp.clear();
    if(pix.y < 500) return false;
    for(int a =0;a <=4;a++){
        for(int b = -10;b <=10;b++){
            Point2i new_pix(pix.x+b,pix.y - a);
            if(frame_lone.at<uchar>(new_pix.y,new_pix.x) == 255){
                ppp.push_back(new_pix);
            }
        }
    }
    if(ppp.size() == 0) return false;//如果窗口内无白点则返回
    for(int i = 0;i < ppp.size();i++){
        sump += ppp[i];
    }
    basep.x = sump.x / ppp.size();
    basep.y = pix.y -4;
    circle(frame,basep,1,Scalar(255,255,255));
    if(flag == left_flag)
    left_edge.push_back(basep);
    else 
    right_edge.push_back(basep);

    slide_window(basep,flag);
    return 0;
}
int main(int argc,char *argv[])
{
    ros::init(argc,argv,"identity_tracks");
    ros::NodeHandle n;   
    tracks idol;
    idol.frame = idol.capture.open(video_path);
    if(!idol.capture.isOpened()){
        cout << "the video can not open" << endl;
    }
    namedWindow("video_src",CV_WINDOW_AUTOSIZE);
    namedWindow("video_deal",CV_WINDOW_AUTOSIZE);
    namedWindow("video_lone",CV_WINDOW_AUTOSIZE);
    namedWindow("warp",CV_WINDOW_AUTOSIZE);
    ros::Rate rate(30);
    while(idol.capture.read(idol.frame) && ros::ok() ){
        /*转为灰度图*/
        cvtColor(idol.frame,idol.frame_gray,COLOR_RGB2GRAY);
        /*高斯滤波*/
        GaussianBlur(idol.frame_gray,idol.frame_gauss,cv::Size(3,3),5);
        /*边缘检测*/
        Sobel(idol.frame_gauss,idol.frame_sobel,idol.frame_gauss.depth(),1,0);
        cv::inRange(idol.frame_sobel,90,255,idol.frame_sobel2);
       // Mat frame_canny;
       // Canny(frame_sobel,frame_canny,100,200,3);
        /*选取roi区域*/
        idol.roi = Mat::zeros(idol.frame_sobel.size(),CV_8U);
        drawContours(idol.roi,idol.contour,0,Scalar::all(255),-1);   
        idol.frame_roi = Mat::zeros(idol.roi.size(),CV_8U);;
        idol.frame_sobel2.copyTo(idol.frame_roi,idol.roi);
        /*去除孤点*/
        idol.frame_lone = Mat::zeros(idol.frame_roi.size(),CV_8U);
        int n;
        medianBlur(idol.frame_roi,idol.frame_lone,3);
        for(int j = 443;j < 712;j++){
            for(int i = 530;i < 730;i++){
                n = 0;
                for (int a = -3 / 2; a <= 3 / 2; a++){
                for (int b = -3 / 2; b <= 3 / 2; b++)
                {
                 if(idol.frame_lone.at<uchar>(j, i) == idol.frame_lone.at<uchar>(j + a, i + b)){
                     if(!(a ==0&&b==0)) 
                     n++; //如果九宫格中有与中心处像素相等则加一
                 }
                }
             }
                if(n <= 2)//如果九宫格内中心处与其他处都不同则判断为噪声
            {
                idol.frame_lone.at<uchar>(j, i) = idol.frame_lone.at<uchar>(j-1, i-1);//将第一个像素赋值给这个噪声点
            }
            }
        }
        /*膨胀操作*/
        Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
        dilate(idol.frame_lone, idol.frame_lone_dil, element);
        /*逆透视*/
        Point2f srcQuad[4],dstQuad[4];
        srcQuad[0].x = idol.pts[0].x;
        srcQuad[0].y = idol.pts[0].y;
        srcQuad[1].x = idol.pts[1].x;
        srcQuad[1].y = idol.pts[1].y;
        srcQuad[2].x = idol.pts[3].x;
        srcQuad[2].y = idol.pts[3].y;
        srcQuad[3].x = idol.pts[2].x;
        srcQuad[3].y = idol.pts[2].y;

        dstQuad[0].x = idol.pts[0].x;
        dstQuad[0].y = idol.pts[0].y;
        dstQuad[1].x = idol.pts[1].x;
        dstQuad[1].y = idol.pts[1].y;
        dstQuad[2].x = idol.warp_left;
        dstQuad[2].y = idol.pts[3].y;
        dstQuad[3].x = idol.warp_right;
        dstQuad[3].y = idol.pts[2].y;

        Mat warp_matrix = cv::getPerspectiveTransform(srcQuad,dstQuad);
        idol.frame_warp = cv::Mat::zeros(idol.frame_lone.size(),idol.frame_lone.type());
        warpPerspective(idol.frame_lone_dil,idol.frame_warp,warp_matrix,idol.frame_warp.size());
        imshow("warp",idol.frame_warp);
        /*搜索轨道预起始点*/
        idol.find_base_pre();
        /*搜索轨道起始点*/
        idol.left_base = idol.find_base(idol.left_base_pre);
        idol.right_base = idol.find_base(idol.right_base_pre);
        circle(idol.frame_lone,idol.left_base,3,Scalar(255,255,0));
        circle(idol.frame_lone,idol.right_base,3,Scalar(255,255,0));
        /*滑动窗口巡线*/
        idol.left_edge.clear();
        idol.right_edge.clear();
        idol.slide_window(idol.left_base,idol.left_flag);
        idol.slide_window(idol.right_base,idol.right_flag);
        cout << "left edge size:" << idol.left_edge.size()
             << "right edge size:" << idol.right_edge.size()
             << endl;
        /*显示原始图像*/
        imshow("video_src",idol.frame);
        /*显示roi区域图像*/
        imshow("video_lone",idol.frame_roi);
        /*显示最终图像*/
        imshow("video_deal",idol.frame_lone);
        waitKey(1);
        rate.sleep();
    }
    idol.capture.release();

    return 0;
}
