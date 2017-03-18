#include "robust_matting.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <exception>

using namespace std;
using namespace cv;

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

using namespace Eigen;

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        cout << "Usage: RobustMatting.exe input.png trimap.png output.png" << endl;
        return -1;
    }

    string img_filename = argv[1];
    string trimap_filename = argv[2];
    string output_filename = argv[3];
    Mat img = imread(img_filename, CV_LOAD_IMAGE_COLOR);
    Mat trimap = imread(trimap_filename, CV_LOAD_IMAGE_GRAYSCALE);

    RobustMatting rm(img, trimap);
    rm.Run();

    Mat rst = rm.GetFinalMat();
    imwrite(output_filename, rst);

    return 0;
}