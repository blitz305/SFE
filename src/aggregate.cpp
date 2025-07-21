// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
#define ARMA_64BIT_WORD 1
#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <map>
#include <cmath> 

// ---- 计算 Shannon 熵 Worker ----
struct CountShannonWorker : public RcppParallel::Worker {
    const arma::umat& index_matrix;
    const arma::vec& values_vector;
    arma::vec& output_entropy;

    CountShannonWorker(const arma::umat& im, const arma::vec& vv, arma::vec& oe)
        : index_matrix(im), values_vector(vv), output_entropy(oe) {}

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            std::map<double, int> row_counts;
            int total_valid_elements = 0;

            for (arma::uword j = 0; j < index_matrix.n_cols; ++j) {
                arma::uword r_index = index_matrix(i, j);
                if (r_index > 0 && r_index <= values_vector.n_elem) {
                    arma::uword c_index = r_index - 1;
                    row_counts[values_vector(c_index)]++;
                    total_valid_elements++;
                }
            }

            double current_entropy = 0.0;
            if (total_valid_elements > 1) {
                for (auto it = row_counts.begin(); it != row_counts.end(); ++it) {
                    int count = it->second;
                    if (count > 0) {
                        double p = static_cast<double>(count) / total_valid_elements;
                        current_entropy -= p * std::log(p) / std::log(2.0); // log base 2
                    }
                }
            }
            output_entropy(i) = current_entropy;
        }
    }
};

// ---- 计算 Pielou 多样性指数 Worker ----
struct CountPielouWorker : public RcppParallel::Worker {
    const arma::umat& index_matrix;
    const arma::vec& values_vector;
    arma::vec& output_pielou;

    CountPielouWorker(const arma::umat& im, const arma::vec& vv, arma::vec& op)
        : index_matrix(im), values_vector(vv), output_pielou(op) {}

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            std::map<double, int> row_counts;
            int total_valid_elements = 0;

            for (arma::uword j = 0; j < index_matrix.n_cols; ++j) {
                arma::uword r_index = index_matrix(i, j);
                if (r_index > 0 && r_index <= values_vector.n_elem) {
                    arma::uword c_index = r_index - 1;
                    row_counts[values_vector(c_index)]++;
                    total_valid_elements++;
                }
            }

            double current_entropy = 0.0;
            if (total_valid_elements > 1 && !row_counts.empty()) {
                for (auto it = row_counts.begin(); it != row_counts.end(); ++it) {
                    int count = it->second;
                    if (count > 0) {
                        double p = static_cast<double>(count) / total_valid_elements;
                        current_entropy -= p * std::log(p) / std::log(2.0);
                    }
                }
            }

            double pielou_index = 0.0;
            if (!row_counts.empty() && current_entropy > 0) {
                double s = static_cast<double>(row_counts.size());
                pielou_index = current_entropy / std::log2(s);
            }
            output_pielou(i) = pielou_index;
        }
    }
};

struct ColumnMergerWorker : public RcppParallel::Worker {
    const arma::umat index_matrix;
    const arma::sp_mat& x_sp_mat;
    arma::vec& res;

    ColumnMergerWorker(const arma::umat& im, const arma::sp_mat& gsm, arma::vec& res)
        : index_matrix(im), x_sp_mat(gsm), res(res) {}

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
           arma::sp_vec column_val = sum(x_sp_mat.cols(index_matrix.row(i)-1), 1);
           double col_sum = arma::accu(column_val);

           if (col_sum <= 0 || column_val.n_nonzero == 0) {
               res[i] = 0.0;
               continue;
           }

           double tmp_entropy = 0.0;

           for (arma::sp_vec::const_iterator it = column_val.begin(); it != column_val.end(); ++it) {
               double p = *it / col_sum;
               tmp_entropy -= p * std::log(p);
           }

           res[i] = tmp_entropy;
        }
    }
};

// [[Rcpp::export]]
arma::vec merge_columns_parallel(const arma::umat& index_matrix, const arma::sp_mat& x_sp_mat) {
    if (index_matrix.n_rows == 0) {
        return arma::vec();
    }

    arma::uword out_cols = index_matrix.n_rows;

    arma::vec result(out_cols);

    ColumnMergerWorker worker(index_matrix, x_sp_mat, result);

    RcppParallel::parallelFor(0, out_cols, worker);

    return result;
}

// [[Rcpp::export]]
arma::vec calculate_count_shannon(const arma::umat& index_matrix, const arma::vec& values_vector) {
    if (index_matrix.n_rows == 0) {
        return arma::vec();
    }
    
    arma::vec result_vector(index_matrix.n_rows, arma::fill::zeros);
    CountShannonWorker worker(index_matrix, values_vector, result_vector);
    RcppParallel::parallelFor(0, index_matrix.n_rows, worker);
    return result_vector;
}

// [[Rcpp::export]]
arma::vec calculate_count_pielou(const arma::umat& index_matrix, const arma::vec& values_vector) {
    if (index_matrix.n_rows == 0) {
        return arma::vec();
    }
    
    arma::vec result_vector(index_matrix.n_rows, arma::fill::zeros);
    CountPielouWorker worker(index_matrix, values_vector, result_vector);
    RcppParallel::parallelFor(0, index_matrix.n_rows, worker);
    return result_vector;
}
