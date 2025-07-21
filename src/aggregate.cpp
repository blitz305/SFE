// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
#define ARMA_64BIT_WORD 1
#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <map>
#include <cmath>

enum class DiversityMetric { Shannon, Pielou };

// ---- 统一计算 Shannon 和 Pielou 多样性指数的 Worker ----
struct CountDiversityWorker : public RcppParallel::Worker {
    const arma::umat& index_matrix;
    const arma::vec& values_vector;
    arma::vec& output_metric;
    DiversityMetric metric;

    CountDiversityWorker(const arma::umat& im, const arma::vec& vv, arma::vec& om, DiversityMetric m)
        : index_matrix(im), values_vector(vv), output_metric(om), metric(m) {}

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            std::map<double, int> counts;
            int total_valid = 0;

            for (arma::uword j = 0; j < index_matrix.n_cols; ++j) {
                arma::uword idx = index_matrix(i, j);
                if (idx > 0 && idx <= values_vector.n_elem) {
                    double val = values_vector[idx - 1];  // 转0-based索引
                    counts[val]++;
                    total_valid++;
                }
            }

            if (total_valid < 2 || counts.empty()) {
                output_metric[i] = 0.0;
                continue;
            }

            double entropy = 0.0;
            for (auto& kv : counts) {
                double p = static_cast<double>(kv.second) / total_valid;
                entropy -= p * std::log2(p);
            }

            if (metric == DiversityMetric::Shannon) {
                output_metric[i] = entropy;
            } else if (metric == DiversityMetric::Pielou) {
                const size_t species = counts.size();
                output_metric[i] = (species > 1 && entropy > 0) ? entropy / std::log2(species) : 0.0;
            }
        }
    }
};

// ---- knn邻居细胞特征合并后计算信息熵的 Worker ----
struct ColumnMergerWorker : public RcppParallel::Worker {
    const arma::umat& index_matrix;   // knn索引矩阵，1-based索引
    const arma::sp_mat& x_sp_mat;     // 稀疏特征矩阵，行是细胞特征，列是细胞
    arma::vec& res;

    ColumnMergerWorker(const arma::umat& im, const arma::sp_mat& gsm, arma::vec& r)
        : index_matrix(im), x_sp_mat(gsm), res(r) {}

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            // 安全检查：构造列索引数组，过滤掉0或超出x_sp_mat列数的索引
            std::vector<arma::uword> valid_cols;
            for (arma::uword j = 0; j < index_matrix.n_cols; ++j) {
                arma::uword col_idx = index_matrix(i, j);
                if (col_idx > 0 && col_idx <= x_sp_mat.n_cols) {
                    valid_cols.push_back(col_idx - 1);  // 转0-based索引
                }
            }
            if (valid_cols.empty()) {
                res[i] = 0.0;
                continue;
            }

            // 取出这些列组成子矩阵
            arma::sp_mat sub_mat = x_sp_mat.cols(arma::uvec(valid_cols));
            // 对每一行（每个特征）求和，得到合并后的特征向量
            arma::sp_vec merged_feature_vec = arma::sum(sub_mat, 1);

            double total_sum = arma::accu(merged_feature_vec);

            if (total_sum <= 0 || merged_feature_vec.n_nonzero == 0) {
                res[i] = 0.0;
                continue;
            }

            double entropy = 0.0;
            for (arma::sp_vec::const_iterator it = merged_feature_vec.begin(); it != merged_feature_vec.end(); ++it) {
                double p = *it / total_sum;
                // 理论上不可能为0，因为sp_vec只迭代非零元素
                entropy -= p * std::log2(p);
            }

            res[i] = entropy;
        }
    }
};

// [[Rcpp::export]]
arma::vec calculate_diversity(const arma::umat& index_matrix, const arma::vec& values_vector, std::string metric_str) {
    if (index_matrix.n_rows == 0) return arma::vec();

    arma::vec result(index_matrix.n_rows, arma::fill::zeros);
    DiversityMetric metric;

    if (metric_str == "shannon") {
        metric = DiversityMetric::Shannon;
    } else if (metric_str == "pielou") {
        metric = DiversityMetric::Pielou;
    } else {
        Rcpp::stop("Unsupported metric: use 'shannon' or 'pielou'");
    }

    CountDiversityWorker worker(index_matrix, values_vector, result, metric);
    RcppParallel::parallelFor(0, index_matrix.n_rows, worker);
    return result;
}

// [[Rcpp::export]]
arma::vec merge_columns_entropy(const arma::umat& index_matrix, const arma::sp_mat& x_sp_mat) {
    if (index_matrix.n_rows == 0) return arma::vec();

    arma::vec result(index_matrix.n_rows, arma::fill::zeros);
    ColumnMergerWorker worker(index_matrix, x_sp_mat, result);
    RcppParallel::parallelFor(0, index_matrix.n_rows, worker);
    return result;
}
