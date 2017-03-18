#include "robust_matting.h"
#include <queue>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>

using namespace std;
using namespace cv;

int RMTable3x9[3][9] =
{
    { 0, 1, 2, 3, 4, 5, 6, 7, 8 },
    { 0, 0, 0, 1, 1, 1, 2, 2, 2 },
    { 0, 1, 2, 0, 1, 2, 0, 1, 2 }
};

RobustMatting::RobustMatting(const cv::Mat& img, const cv::Mat& trimap)
{
    img.convertTo(m_image, CV_32FC3);
    m_trimap = trimap;
    if (m_trimap.channels() > 1)
        cvtColor(m_trimap, m_trimap, CV_BGR2GRAY);
    m_result = Mat(m_image.size(), CV_8UC1);
}

RobustMatting::~RobustMatting()
{
}

void RobustMatting::Run()
{
    int maxiter = 3;
    for (int iter = 0; iter < maxiter; iter++)
    {
        init();
        EstimateAlpha();
        BuildMatrix();
        Solve();
        m_result.copyTo(m_trimap);
    }
}

void RobustMatting::EstimateAlpha()
{
    m_initial_alpha = Mat(m_image.size(), CV_32FC1, Scalar(0.0));
    m_initial_confd = Mat(m_image.size(), CV_32FC1, Scalar(1.0));

    int height = m_image.rows, width = m_image.cols;
    float* pa = (float*)m_initial_alpha.data;
    float* pc = (float*)m_initial_confd.data;
    Vec3f* pimg = (Vec3f*)m_image.data;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = i * width + j;
            if (m_ukn_map.data[index] == 255)
            {
                pair<float, float> alpha_confd = compute_alpha_confd(pimg[index]);
                pa[index] = alpha_confd.first;
                pc[index] = alpha_confd.second;
            }
            else
                pa[index] = m_fgd_map.data[index] == 255 ? 1.0 : pa[index];
        }
    }
}

void RobustMatting::BuildMatrix()
{
    typedef Eigen::SparseMatrix<double> SpMat;

    SpMat Lu(m_num_ukn, m_num_ukn);
    Lu.reserve(Eigen::VectorXi::Constant(m_num_ukn, 35));
    SpMat Rt0(m_num_ukn, m_num_akn);
    Rt0.reserve(Eigen::VectorXi::Constant(m_num_akn, 35));
    Eigen::MatrixXd Rt1(m_num_ukn, 2);

    int height = m_image.rows, width = m_image.cols;
    Mat erode_map;
    erode(m_akn_map, erode_map, Mat());

    Matx19d mean_1x9 = Matx19d::all(1.0 / 9.0);
    const float eps = 1.0e-7, gamma = 1.0e-7;
    for (int i = 1; i < height - 1; i++)
    {
        for (int j = 1; j < width - 1; j++)
        {
            if (erode_map.at<unsigned char>(i, j) == 255)
                continue;
            Mat win_flag_idx = m_flag_idx_map(Range(i - 1, i + 2), Range(j - 1, j + 2));
            Mat win_img = m_image(Range(i - 1, i + 2), Range(j - 1, j + 2));
            
            Matx93d A9x3 = build_A9x3(win_img);
            Matx13d win_mu = mean_1x9 * A9x3;
            Matx93d M9x3 = build_M9x3(win_mu);
            Matx33d win_var = A9x3.t() * A9x3 * (1.0 / 9.0) - win_mu.t() * win_mu + eps / 9.0 * Matx33d::eye(); 
            Matx33d inv_win_var = win_var.inv();
            Matx99d T9x9 = (A9x3 - M9x3) * inv_win_var * (A9x3 - M9x3).t();
            T9x9 = 1.0 / 9.0 * (T9x9 + Matx99d::ones());

            for (int ii = 1; ii < 9; ii++)
            {
                for (int jj = 0; jj < ii; jj++)
                {
                    int idx_i = win_flag_idx.at<int>(RMTable3x9[1][ii], RMTable3x9[2][ii]);
                    int idx_j = win_flag_idx.at<int>(RMTable3x9[1][jj], RMTable3x9[2][jj]);
                    float v = T9x9(ii, jj);
                    if (idx_i > 0 && idx_j > 0) // both known
                        continue;
                    else if (idx_i < 0 && idx_j < 0) // both unknown
                    {
                        int ridx_i = -idx_i - 1, ridx_j = -idx_j - 1;
                        Lu.coeffRef(ridx_i, ridx_j) += -v;
                        Lu.coeffRef(ridx_j, ridx_i) += -v;
                        Lu.coeffRef(ridx_i, ridx_i) += v;
                        Lu.coeffRef(ridx_j, ridx_j) += v;
                    }
                    else if (idx_i < 0 && idx_j > 0) // ukn ~ kn
                    {
                        int ridx_i = -idx_i - 1, ridx_j = idx_j - 1;
                        Rt0.coeffRef(ridx_i, ridx_j) += -v;
                        Lu.coeffRef(ridx_i, ridx_i) += v;
                    }
                    else // kn ~ ukn
                    {
                        int ridx_i = idx_i - 1, ridx_j = -idx_j - 1;
                        Rt0.coeffRef(ridx_j, ridx_i) += -v;
                        Lu.coeffRef(ridx_j, ridx_j) += v;
                    }
                }
            }
        }
    }
    Eigen::VectorXd Ak = Eigen::VectorXd::Zero(m_num_akn);
    Eigen::VectorXd Au(m_num_ukn);
    int* pidx = (int*)m_flag_idx_map.data;
    float* pa = (float*)m_initial_alpha.data;
    float* pc = (float*)m_initial_confd.data;
    int k1 = 0, k2 = 0;
    for (int i = 0; i < m_flag_idx_map.total(); i++)
    {
        int idx = pidx[i];
        if (idx > 0)
        {
            Ak(k1++) = m_fgd_map.data[i] == 255 ? 1.0 : 0.0;
            continue;
        }
        float alpha = pa[i], confd = pc[i];
        int ridx = -idx - 1;
        Lu.coeffRef(ridx, ridx) += gamma;
        double x1 = -gamma * (confd * alpha + 1.0 - confd);
        double x2 = -gamma * confd * alpha;
        double x3 = -gamma * confd * (1.0 - alpha);
        double x4 = -gamma * (confd * (1.0 - alpha) + (1.0 - confd));
        Rt1(ridx, 0) = alpha > 0.5 ? x1 : x2;
        Rt1(ridx, 1) = alpha > 0.5 ? x3 : x4;
        Au(k2++) = alpha;
    }

    Eigen::VectorXd onezero(2);
    onezero << 1.0, 0.0;
    Eigen::VectorXd rhs = -(Rt0 * Ak + Rt1 * onezero);
    Eigen::ConjugateGradient<SpMat, Eigen::Upper | Eigen::Lower> cg;

    static int itime = 0;
    cg.setMaxIterations(max(30, 300 - itime * 30));
    itime++;

    cg.compute(Lu);
    Au = cg.solveWithGuess(rhs, Au);

    for (int i = 0; i < Au.size(); i++)
        Au(i) = Au(i) < 0.02 ? 0 : Au(i) > 0.98 ? 1.0 : Au(i);
    
    pidx = (int*)m_flag_idx_map.data;
    pa = (float*)m_initial_alpha.data;
    for (int i = 0; i < m_flag_idx_map.total(); i++)
    {
        int idx = pidx[i];
        m_result.data[i] = idx < 0 ? (unsigned char)(255.0 * Au(-idx - 1)) : (unsigned char)(255.0 * pa[i]);
    }
}

void RobustMatting::Solve()
{

}

void RobustMatting::init()
{
    m_fgd_map = m_trimap == 255;
    m_bgd_map = m_trimap == 0;
    m_akn_map = m_fgd_map + m_bgd_map;
    m_ukn_map = 255 - m_akn_map;
    
    m_flag_idx_map = Mat(m_image.size(), CV_32SC1);
    m_idx_map = Mat(m_image.size(), CV_32SC1);
    int uknidx = -1, aknidx = 1;
    int *ptr1 = (int*)(m_flag_idx_map.data);
    int *ptr2 = (int*)(m_idx_map.data);
    for (int i = 0; i < m_flag_idx_map.total(); i++)
    {
        ptr1[i] = m_ukn_map.data[i] == 255 ? uknidx-- : aknidx++;
        ptr2[i] = i;
    }

    Mat temp1, temp2, temp3;
    erode(m_fgd_map, temp1, Mat());
    erode(m_bgd_map, temp2, Mat());
    erode(m_ukn_map, temp3, Mat());
    m_fgd_bdy = m_fgd_map - temp1;
    m_bgd_bdy = m_bgd_map - temp2;
    m_ukn_bdy = m_ukn_map - temp3;

    m_fgd_seeds = select_seeds(m_fgd_bdy);
    m_bgd_seeds = select_seeds(m_bgd_bdy);

    m_num_fgd = countNonZero(m_fgd_map);
    m_num_bgd = countNonZero(m_bgd_map);
    m_num_ukn = countNonZero(m_ukn_map);
    m_num_akn = m_num_fgd + m_num_bgd;

}

std::vector<Seed> RobustMatting::select_seeds(const cv::Mat& bdy, int nseeds /*= 20*/)
{
    int nPixels = countNonZero(bdy);
    vector<int> flags(nPixels, 0);
    nseeds = min(nseeds, nPixels);
    fill(flags.begin(), flags.begin() + nseeds, 1);
    random_shuffle(flags.begin(), flags.end());
    vector<Seed> seeds(nseeds);
    int k = 0, m = 0;
    for (int idx = 0; idx < bdy.total(); idx++)
    {
        if (bdy.data[idx] == 255 && flags[k++])
        {
            int i = idx / bdy.cols, j = idx % bdy.cols;
            seeds[m++] = Seed(i, j, idx, m_image.at<Vec3f>(i, j));
        }
    }
    return seeds;
}

std::pair<float, float> RobustMatting::compute_alpha_confd(const cv::Vec3f& color) const
{
    int n_fgd_seeds = m_fgd_seeds.size();
    int n_bgd_seeds = m_bgd_seeds.size();
    int n_pairs = n_fgd_seeds * n_bgd_seeds;
    vector<pair<float, float>> alpha_confd_vec(n_pairs);
    double min_dist_fgd_sq = min_dist_fbgd_sq(color, m_fgd_seeds);
    double min_dist_bgd_sq = min_dist_fbgd_sq(color, m_bgd_seeds);
    int k = 0;
    for (int k1 = 0; k1 < n_fgd_seeds; k1++)
    {
        Vec3f fc = m_fgd_seeds[k1].color;
        for (int k2 = 0; k2 < n_bgd_seeds; k2++)
        {
            Vec3f bc = m_bgd_seeds[k2].color;
            float alpha = est_alpha(color, fc, bc);
            float rdval_sq = rd_sq(color, fc, bc, alpha);
            float wf = weight_fbgd(color, fc, min_dist_fgd_sq);
            float wb = weight_fbgd(color, bc, min_dist_bgd_sq);
            float confd = confidence(rdval_sq, wf, wb);

            confd = alpha < -0.05 || alpha > 1.05 ? 0.0 : confd;
            alpha = alpha >= -0.05 && alpha < 0.0 ? 0.0 : alpha <= 1.05 && alpha > 1.0 ? 1.0 : alpha;
            alpha_confd_vec[k++] = { alpha, confd };
        }
    }
    typedef pair<float, float> pff;
    int fixed_size = 3;
    float sum1 = 0.0, sum2 = 0.0;
    if (n_pairs <= fixed_size)
    {
        for (int k = 0; k < n_pairs; k++)
        {
            sum1 += alpha_confd_vec[k].first;
            sum2 += alpha_confd_vec[k].second;
        }
        return pff(sum1 / n_pairs, sum2 / n_pairs);
    }
    auto comp = [](pff a, pff b) { return a.second > b.second; };
    priority_queue < pff, vector<pff>, decltype(comp) > pq(alpha_confd_vec.begin(), alpha_confd_vec.begin() + fixed_size, comp);
    for (int k = fixed_size; k < n_pairs; k++)
    {
        if (alpha_confd_vec[k].second > pq.top().second)
        {
            pq.pop();
            pq.push(alpha_confd_vec[k]);
        }
    }
    while (!pq.empty())
    {
        sum1 += pq.top().first;
        sum2 += pq.top().second;
        pq.pop();
    }
    return pff(sum1 / fixed_size, sum2 / fixed_size);
}

float RobustMatting::min_dist_fbgd_sq(const cv::Vec3f& color, const std::vector<Seed>& seeds) const
{
    vector<float> dists(seeds.size(), 0.0);
    for (int i = 0; i < seeds.size(); i++)
        dists[i] = color_dist(color, seeds[i].color);
    return *min_element(dists.begin(), dists.end()) + 1e-5;
}

float RobustMatting::color_dist(const cv::Vec3f& c1, const cv::Vec3f& c2) const
{
    Vec3f diff = (Vec3f)c1 - (Vec3f)c2;
    return diff.dot(diff);
}

float RobustMatting::est_alpha(const cv::Vec3f& color, const cv::Vec3f& fc, const cv::Vec3f& bc) const
{
    Vec3f c_minus_b = (Vec3f)color - (Vec3f)bc;
    Vec3f f_minus_b = (Vec3f)fc - (Vec3f)bc;
    float temp1 = c_minus_b.dot(f_minus_b), temp2 = f_minus_b.dot(f_minus_b) + 1e-7;
    return temp1 / temp2;
}

float RobustMatting::rd_sq(const cv::Vec3f& color, const cv::Vec3f& fc, const cv::Vec3f& bc, float alpha) const
{
    Vec3f temp1 = color - alpha * fc - (1.0 - alpha) * bc;
    Vec3f temp2 = fc - bc;
    return temp1.dot(temp1) / (temp2.dot(temp2) + 1e-7);
}

float RobustMatting::weight_fbgd(const cv::Vec3f& color, const cv::Vec3f& fbc, float min_dist_fbgd_sq_val) const
{
    Vec3f fb_minus_c = fbc - color;
    float temp = fb_minus_c.dot(fb_minus_c) / min_dist_fbgd_sq_val;
    return std::exp(-temp);
}

float RobustMatting::confidence(float rdval_sq, float wf, float wb) const
{
    return std::exp(-rdval_sq * wf * wb * 100.0);
}

cv::Matx93d RobustMatting::build_A9x3(const cv::Mat& img3x3) const
{
    Matx93d A9x3;
    int k = 0;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            const Vec3f& c = img3x3.at<Vec3f>(i, j);
            A9x3(k, 0) = c[0];
            A9x3(k, 1) = c[1];
            A9x3(k, 2) = c[2];
            k++;
        }
    }
    return 1.0 / 255.0 * A9x3;
}

cv::Matx93d RobustMatting::build_M9x3(const cv::Matx13d mu) const
{
    Matx93d M9x3;
    for (int i = 0; i < 9; i++)
    {
        M9x3(i, 0) = mu(0);
        M9x3(i, 1) = mu(1);
        M9x3(i, 2) = mu(2);
    }
    return M9x3;
}
