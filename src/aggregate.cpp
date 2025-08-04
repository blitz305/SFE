// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
#define ARMA_64BIT_WORD 1
#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <cmath>
#include <map>
#include <vector>

using namespace Rcpp;

// --- Compute Shannon entropy ---
inline double compute_shannon(const std::vector<double>& frequencies) {
    double entropy = 0.0;
    for (double p : frequencies) {
        if (p > 0.0) {
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

// --- Compute Pielou's evenness ---
inline double compute_pielou(double shannon, std::size_t species_count) {
    return (species_count > 1 && shannon > 0.0) ? shannon / std::log2(species_count) : 0.0;
}

// --- Parallel computation worker ---
struct DiversityComputationWorker : public RcppParallel::Worker {
    const arma::umat& index_matrix;   // Index matrix
    const arma::vec& values_vector;   // Values vector
    const arma::sp_mat& x_sp_mat;     // Sparse matrix
    arma::mat& result;                // Output result matrix
    bool is_diversity_task;           // Task type flag

    DiversityComputationWorker(const arma::umat& im,
                               const arma::vec& vv,
                               const arma::sp_mat& gsm,
                               arma::mat& r,
                               bool is_div)
        : index_matrix(im), values_vector(vv), x_sp_mat(gsm),
          result(r), is_diversity_task(is_div) {}
 // Operator to process rows in parallel
    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            double shannon = 0.0, pielou = 0.0;

            if (is_diversity_task) {
                // Vector mode
                std::map<double, int> counts;
                int total_valid = 0;

                for (arma::uword j = 0; j < index_matrix.n_cols; ++j) {
                    arma::uword idx = index_matrix(i, j);
                    if (idx > 0 && idx <= values_vector.n_elem) {
                        double val = values_vector[idx - 1];
                        counts[val]++;
                        total_valid++;
                    }
                }

                if (total_valid >= 2 && !counts.empty()) {
                    std::vector<double> freqs;
                    for (auto& kv : counts) {
                        freqs.push_back(static_cast<double>(kv.second) / total_valid);
                    }
                    shannon = compute_shannon(freqs);
                    pielou  = compute_pielou(shannon, counts.size());
                }

            } else {
                // Calculate diversity by merging columns in sparse matrix
                std::vector<arma::uword> valid_cols;
                for (arma::uword j = 0; j < index_matrix.n_cols; ++j) {
                    arma::uword col_idx = index_matrix(i, j);
                    if (col_idx > 0 && col_idx <= x_sp_mat.n_cols) {
                        valid_cols.push_back(col_idx - 1);
                    }
                }

                if (!valid_cols.empty()) {
                    arma::sp_mat sub_mat = x_sp_mat.cols(arma::uvec(valid_cols));
                    arma::sp_vec merged_vec = arma::sum(sub_mat, 1);
                    double total = arma::accu(merged_vec);

                    if (total > 0 && merged_vec.n_nonzero > 0) {
                        std::vector<double> freqs;
                        for (arma::sp_vec::const_iterator it = merged_vec.begin(); it != merged_vec.end(); ++it) {
                            freqs.push_back(*it / total);
                        }
                        shannon = compute_shannon(freqs);
                        pielou  = compute_pielou(shannon, freqs.size());
                    }
                }
            }

            result(i, 0) = shannon;
            result(i, 1) = pielou;
        }
    }
};

// [[Rcpp::export]]
arma::mat calculate_metrics_from_vector(const arma::umat& index_matrix,
                                        const arma::vec& values_vector) {
    if (index_matrix.n_rows == 0 || values_vector.n_elem == 0)
        Rcpp::stop("index_matrix or values_vector is empty.");

    arma::sp_mat dummy; 
    arma::mat result(index_matrix.n_rows, 2, arma::fill::zeros);
    DiversityComputationWorker worker(index_matrix, values_vector, dummy, result, true);
    RcppParallel::parallelFor(0, index_matrix.n_rows, worker);
    return result;
}

// [[Rcpp::export]]
arma::mat calculate_metrics_from_spmat(const arma::umat& index_matrix,
                                       const arma::sp_mat& x_sp_mat) {
    if (index_matrix.n_rows == 0 || x_sp_mat.n_cols == 0 || x_sp_mat.n_rows == 0)
        Rcpp::stop("index_matrix or x_sp_mat is empty or invalid.");

    arma::vec dummy;  
    arma::mat result(index_matrix.n_rows, 2, arma::fill::zeros);
    DiversityComputationWorker worker(index_matrix, dummy, x_sp_mat, result, false);
    RcppParallel::parallelFor(0, index_matrix.n_rows, worker);
    return result;
}
