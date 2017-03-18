#ifndef ROBUST_MATTING_H
#define ROBUST_MATTING_H

#include <opencv2/opencv.hpp>

namespace cv
{
    typedef Matx<double, 1, 9> Matx19d;
    typedef Matx<double, 9, 9> Matx99d;
    typedef Matx<double, 9, 3> Matx93d;
}

struct Seed
{
    Seed() {}
    Seed(int x1, int x2, int x3, cv::Vec3f c) :
        i(x1), j(x2), index(x3), color(c) {}
    int i, j, index;
    cv::Vec3f color;
};

class RobustMatting
{
public:
    RobustMatting(const cv::Mat& img, const cv::Mat& trimap);
    ~RobustMatting();

    void                        Run();
    cv::Mat                     GetFinalMat() const { return m_result; }

private:
    void                        EstimateAlpha();
    void                        BuildMatrix();

private:
    void                        init();
    std::vector<Seed>           select_seeds(const cv::Mat& bdy, int nseeds = 20);
    std::pair<float, float>     compute_alpha_confd(const cv::Vec3f& color) const;
    float                       min_dist_fbgd_sq(const cv::Vec3f& color, const std::vector<Seed>& seeds) const;
    float                       color_dist(const cv::Vec3f& c1, const cv::Vec3f& c2) const;
    float                       est_alpha(const cv::Vec3f& color, const cv::Vec3f& fc, const cv::Vec3f& bc) const;
    float                       rd_sq(const cv::Vec3f& color, const cv::Vec3f& fc, const cv::Vec3f& bc, float alpha) const;
    float                       weight_fbgd(const cv::Vec3f& color, const cv::Vec3f& fbc, float min_dist_fbgd_sq_val) const;
    float                       confidence(float rdval_sq, float wf, float wb) const;

    cv::Matx93d                 build_A9x3(const cv::Mat& img3x3) const;
    cv::Matx93d                 build_M9x3(const cv::Matx13d mu) const;

private:
    cv::Mat                     m_image;
    cv::Mat                     m_trimap;

    // internal data
    cv::Mat                     m_fgd_map;
    cv::Mat                     m_bgd_map;
    cv::Mat                     m_akn_map;
    cv::Mat                     m_ukn_map;
    cv::Mat                     m_flag_idx_map;
    cv::Mat                     m_idx_map;

    cv::Mat                     m_fgd_bdy;
    cv::Mat                     m_bgd_bdy;
    cv::Mat                     m_ukn_bdy;

    std::vector<Seed>           m_fgd_seeds;
    std::vector<Seed>           m_bgd_seeds;

    cv::Mat                     m_initial_alpha;
    cv::Mat                     m_initial_confd;

    int                         m_num_ukn;
    int                         m_num_fgd;
    int                         m_num_bgd;
    int                         m_num_akn;

    cv::Mat                     m_result;
};

#endif // !ROBUST_MATTING_H

