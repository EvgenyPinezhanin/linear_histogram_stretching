#include<opencv2/opencv.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    string image_path = samples::findFile("resource/cat.jpg");
    Mat src = imread(image_path, IMREAD_COLOR);
    if (!src.data) {
        std::cerr << "No image data" << std::endl;
        return -1;
    }

    cvtColor(src, src, COLOR_BGR2GRAY);

    Mat dst;
    equalizeHist(src, dst);

    imshow("Source image", src);
    imshow("Equalized Image", dst);
    waitKey();

    return 0;
}