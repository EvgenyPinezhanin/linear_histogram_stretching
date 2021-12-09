#include<mpi.h>
#include<opencv2/opencv.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

std::vector<uchar> LinearHistogramStretchingParallel(std::vector<uchar> image) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int* sendCounts = new int[size];
    int* displs = new int[size];
    int n;

    n = image.size();
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < size; i++) {
        sendCounts[i] = n / size;
    }
    for (int i = 0; i < n % size; i++) {
        sendCounts[i] += 1;
    }
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + sendCounts[i - 1];
    }

    uchar *part_image = new uchar[sendCounts[rank]];
    MPI_Scatterv(image.data(), sendCounts, displs, MPI_UNSIGNED_CHAR,
                 part_image, sendCounts[rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    int imin = 255, imax = 0;
    int res_imin, res_imax;
    double a = 0.0, b = 0.0;
    for (int i = 0; i < sendCounts[rank]; i++) {
        if (part_image[i] < imin) imin = part_image[i];
        if (part_image[i] > imax) imax = part_image[i];
    }
    MPI_Reduce(&imin, &res_imin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&imax, &res_imax, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        if (res_imin == res_imax) {
            a = 0.0; b = 0.0;
        } else {
            if (res_imin != 0) {
                b = 255.0 / (1.0 - static_cast<double>(res_imax) / res_imin);
            } else {
                b = 0.0;
            }
            a = -255.0 / (static_cast<double>(res_imin) - res_imax);
        }
    }
    MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < sendCounts[rank]; i++) {
        part_image[i] = round(a * part_image[i] + b);
    }
    MPI_Gatherv(part_image, sendCounts[rank], MPI_UNSIGNED_CHAR,
                image.data(), sendCounts, displs, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    return image;
}

std::vector<uchar> LinearHistogramStretchingSequential(std::vector<uchar> image) {
    double a, b;
    int imin = 255, imax = 0;
    int size = image.size();
    for (int i = 0; i < size; i++) {
        if (image[i] < imin) imin = image[i];
        if (image[i] > imax) imax = image[i];
    }
    if (imin == imax) {
        a = 0.0; b = 0.0;
    } else {
        if (imin != 0) {
            b = 255.0 / (1.0 - static_cast<double>(imax) / imin);
        } else {
            b = 0.0;
        }
        a = -255.0 / (static_cast<double>(imin) - imax);
    }
    for (int i = 0; i < size; i++) {
        image[i] = round(a * image[i] + b);
    }
    return image;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int current_image;
    vector<uchar> matrix_par, matrix_seq;
    Mat src, dst_par, dst_seq, dst_cv;
    int height, width;

    if (rank == 0) {
        current_image = 1;
        namedWindow("Source Image", WINDOW_NORMAL | WINDOW_KEEPRATIO);
        namedWindow("Linear Histogram Stratche Image(parallel)", WINDOW_NORMAL | WINDOW_KEEPRATIO);
        namedWindow("Linear Histogram Stratche Image(sequential)", WINDOW_NORMAL | WINDOW_KEEPRATIO);
        namedWindow("Linear Histogram Stratche Image(opencv)", WINDOW_NORMAL | WINDOW_KEEPRATIO);

        string image_path = samples::findFile("resource/file" + to_string(current_image) + ".jpg");
        src = imread(image_path, IMREAD_COLOR);
        cvtColor(src, src, COLOR_BGR2GRAY);
        height = src.rows;
        width = src.cols;
        matrix_par.resize(height * width);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                matrix_par[i * width + j] = src.at<uchar>(i, j);
            }
        }
        matrix_seq = matrix_par;

        imshow("Source Image", src);
    }

    matrix_par = LinearHistogramStretchingParallel(matrix_par);

    if (rank == 0) {
        dst_par.create(height, width, CV_8UC1);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                dst_par.at<uchar>(i, j) = matrix_par[i * width + j];
            }
        }
        imshow("Linear Histogram Stratche Image(parallel)", dst_par);

        matrix_seq = LinearHistogramStretchingSequential(matrix_seq);
        dst_seq.create(height, width, CV_8UC1);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                dst_seq.at<uchar>(i, j) = matrix_seq[i * width + j];
            }
        }
        imshow("Linear Histogram Stratche Image(sequential)", dst_seq);

        normalize(src, dst_cv, 0, 255, NORM_MINMAX);
        imshow("Linear Histogram Stratche Image(opencv)", dst_cv);
    }

    while(true) {
        if (rank == 0) {
            char c = waitKey();
            
            if (c == 'Q' && current_image != 1) {
                current_image--;
            }

            if (c == 'S' && current_image != 6) {
                current_image++;
            }

            if (c == -1) {
                break;
            }

            string image_path = samples::findFile("resource/file" + to_string(current_image) + ".jpg");
            src = imread(image_path, IMREAD_COLOR);
            cvtColor(src, src, COLOR_BGR2GRAY);
            height = src.rows;
            width = src.cols;
            matrix_par.resize(height * width);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    matrix_par[i * width + j] = src.at<uchar>(i, j);
                }
            }
            matrix_seq = matrix_par;
            imshow("Source Image", src);
        }

        matrix_par = LinearHistogramStretchingParallel(matrix_par);

        if (rank == 0) {
            dst_par.create(height, width, CV_8UC1);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    dst_par.at<uchar>(i, j) = matrix_par[i * width + j];
                }
            }
            imshow("Linear Histogram Stratche Image(parallel)", dst_par);

            matrix_seq = LinearHistogramStretchingSequential(matrix_seq);
            dst_seq.create(height, width, CV_8UC1);
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    dst_seq.at<uchar>(i, j) = matrix_seq[i * width + j];
                }
            }
            imshow("Linear Histogram Stratche Image(sequential)", dst_seq);

            normalize(src, dst_cv, 0, 255, NORM_MINMAX);
            imshow("Linear Histogram Stratche Image(opencv)", dst_cv);
        }
    }
    MPI_Finalize();
    return 0;
}