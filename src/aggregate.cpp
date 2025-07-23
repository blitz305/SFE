// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
#define ARMA_64BIT_WORD 1
#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <cmath>
#include <map>
#include <vector>

using namespace Rcpp;

// Compute Shannon entropy from frequencies
inline double compute_shannon(const std::vector<double>& frequencies) {
    double entropy = 0.0;
    for (double p : frequencies) {
        if (p > 0.0) {
            entropy -= p * std::log2(p);
        }
    }
    return entropy;
}

// Compute Pielou's evenness index
inline double compute_pielou(double shannon, std::size_t species_count) {
    return (species_count > 1 && shannon > 0.0) ? shannon / std::log2(species_count) : 0.0;
}

// ---- Parallel worker ----
struct DiversityComputationWorker : public RcppParallel::Worker {
    const arma::umat& index_matrix;
    const arma::vec& values_vector;
    const arma::sp_mat& x_sp_mat;
    arma::mat& result;
    bool is_diversity_task;

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
                // Calculate diversity based on values_vector and index_matrix
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
arma::mat calculate_auto_metrics(const arma::umat& index_matrix,
                                 const arma::vec& values_vector,
                                 const arma::sp_mat& x_sp_mat) {
    if (index_matrix.n_rows == 0)
        return arma::mat();

    bool use_vector = values_vector.n_elem > 0;
    bool use_spmat  = x_sp_mat.n_cols > 0 && x_sp_mat.n_rows > 0;

    if (!use_vector && !use_spmat)
        Rcpp::stop("Either values_vector or x_sp_mat must be non-empty.");
    if (use_vector && use_spmat)
        Rcpp::stop("Only one of values_vector or x_sp_mat should be provided.");

    arma::mat result(index_matrix.n_rows, 2, arma::fill::zeros);  // columns: [Shannon, Pielou]
    DiversityComputationWorker worker(index_matrix, values_vector, x_sp_mat, result, use_vector);
    RcppParallel::parallelFor(0, index_matrix.n_rows, worker);
    return result;
}
