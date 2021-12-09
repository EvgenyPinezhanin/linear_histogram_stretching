#include<mpi.h>
#include<opencv2/opencv.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<random>

using namespace cv;
using namespace std;

void fillMatrixRandomElem(vector<uchar> &matrix, int height, int width, int a, int b) {
    std::random_device dev;
    std::mt19937 gen(dev());
    matrix.resize(height * width);
    for (int i = 0; i < height * width; i++) {
        matrix[i] = gen() % (b - a) + a;
    }
}

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
    int height, width;
    vector<uchar> matrix_par, matrix_seq;
    Mat src, dst_par, dst_seq, dst_cv;
    double startwtime_p, endwtime_p;
    double startwtime_s, endwtime_s;
    double startwtime_cv, endwtime_cv;

    if (rank == 0) {
        height = 1000; width = 1000;
        src.create(height, width, CV_8UC1);
        dst_par.create(height, width, CV_8UC1);
        dst_seq.create(height, width, CV_8UC1);

        fillMatrixRandomElem(matrix_par, height, width, 23, 195);
        matrix_seq = matrix_par;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                src.at<uchar>(i, j) = matrix_par[i * width + j];
            }
        }
    }

    startwtime_p = MPI_Wtime();
    matrix_par = LinearHistogramStretchingParallel(matrix_par);
    endwtime_p = MPI_Wtime();

    if (rank == 0) {
        startwtime_s = MPI_Wtime();
        matrix_seq = LinearHistogramStretchingSequential(matrix_seq);
        endwtime_s = MPI_Wtime();

        startwtime_cv = MPI_Wtime();
        normalize(src, dst_cv, 0, 255, NORM_MINMAX);
        endwtime_cv = MPI_Wtime();

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                dst_par.at<uchar>(i, j) = matrix_par[i * width + j];
            }
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                dst_seq.at<uchar>(i, j) = matrix_seq[i * width + j];
            }
        }


        namedWindow("Source Image", WINDOW_NORMAL | WINDOW_KEEPRATIO);
        namedWindow("Linear Histogram Stratche Image(parallel) " + to_string(endwtime_p - startwtime_p), WINDOW_NORMAL | WINDOW_KEEPRATIO);
        namedWindow("Linear Histogram Stratche Image(sequential) " + to_string(endwtime_s - startwtime_s), WINDOW_NORMAL | WINDOW_KEEPRATIO);
        namedWindow("Linear Histogram Stratche Image(opencv) " + to_string(endwtime_cv - startwtime_cv), WINDOW_NORMAL | WINDOW_KEEPRATIO);

        imshow("Source Image", src);
        imshow("Linear Histogram Stratche Image(parallel) " + to_string(endwtime_p - startwtime_p), dst_par);
        imshow("Linear Histogram Stratche Image(sequential) " + to_string(endwtime_s - startwtime_s), dst_seq);
        imshow("Linear Histogram Stratche Image(opencv) " + to_string(endwtime_cv - startwtime_cv), dst_cv);
        waitKey();
    }

    MPI_Finalize();
    return 0;
}