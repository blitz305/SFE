// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppParallel)]]
#define ARMA_64BIT_WORD 1
#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <map>
#include <cmath>

// Enum for task types: Diversity and ColumnMerge
enum class TaskType { Diversity, ColumnMerge };

// Enum for diversity metrics: Shannon and Pielou
enum class DiversityMetric { Shannon, Pielou };

// ---- General worker for diversity and column merge entropy calculation ----
struct DiversityComputationWorker : public RcppParallel::Worker {
    const arma::umat& index_matrix;   
    const arma::vec& values_vector;   
    const arma::sp_mat& x_sp_mat;     
    arma::vec& result;                
    TaskType task;                     
    DiversityMetric diversity_metric; 

    DiversityComputationWorker(const arma::umat& im, const arma::vec& vv, const arma::sp_mat& gsm, arma::vec& r, TaskType t, DiversityMetric dm)
        : index_matrix(im), values_vector(vv), x_sp_mat(gsm), result(r), task(t), diversity_metric(dm) {}

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            if (task == TaskType::Diversity) {
                // Calculate Shannon or Pielou diversity index
                std::map<double, int> counts;
                int total_valid = 0;

                for (arma::uword j = 0; j < index_matrix.n_cols; ++j) {
                    arma::uword idx = index_matrix(i, j);
                    if (idx > 0 && idx <= values_vector.n_elem) {
                        double val = values_vector[idx - 1];  // Convert to 0-based index
                        counts[val]++;
                        total_valid++;
                    }
                }

                if (total_valid < 2 || counts.empty()) {
                    result[i] = 0.0;
                    continue;
                }

                double entropy = 0.0;
                for (auto& kv : counts) {
                    double p = static_cast<double>(kv.second) / total_valid;
                    entropy -= p * std::log2(p);
                }

                // Select the correct diversity metric (Shannon or Pielou)
                if (diversity_metric == DiversityMetric::Shannon) {
                    result[i] = entropy;
                } else if (diversity_metric == DiversityMetric::Pielou) {
                    const size_t species = counts.size();
                    result[i] = (species > 1 && entropy > 0) ? entropy / std::log2(species) : 0.0;
                }

            } else if (task == TaskType::ColumnMerge) {
                // Calculate entropy of merged columns
                std::vector<arma::uword> valid_cols;
                for (arma::uword j = 0; j < index_matrix.n_cols; ++j) {
                    arma::uword col_idx = index_matrix(i, j);
                    if (col_idx > 0 && col_idx <= x_sp_mat.n_cols) {
                        valid_cols.push_back(col_idx - 1);  // Convert to 0-based index
                    }
                }

                if (valid_cols.empty()) {
                    result[i] = 0.0;
                    continue;
                }

                // Extract valid columns and merge them into a sub-matrix
                arma::sp_mat sub_mat = x_sp_mat.cols(arma::uvec(valid_cols));
                // Sum each row (feature) to get the merged feature vector
                arma::sp_vec merged_feature_vec = arma::sum(sub_mat, 1);

                double total_sum = arma::accu(merged_feature_vec);

                if (total_sum <= 0 || merged_feature_vec.n_nonzero == 0) {
                    result[i] = 0.0;
                    continue;
                }

                double entropy = 0.0;
                for (arma::sp_vec::const_iterator it = merged_feature_vec.begin(); it != merged_feature_vec.end(); ++it) {
                    double p = *it / total_sum;
                    entropy -= p * std::log2(p);
                }

                result[i] = entropy;
            }
        }
    }
};

// [[Rcpp::export]]
arma::vec calculate_general_task(const arma::umat& index_matrix, 
                                  const arma::vec& values_vector, 
                                  const arma::sp_mat& x_sp_mat, 
                                  std::string task_str,
                                  std::string metric_str = "") {
    if (index_matrix.n_rows == 0) return arma::vec();

    arma::vec result(index_matrix.n_rows, arma::fill::zeros);
    TaskType task;
    DiversityMetric metric = DiversityMetric::Shannon;  // Default to Shannon

    // Choose task type
    if (task_str == "diversity") {
        task = TaskType::Diversity;

        // Choose diversity metric
        if (metric_str == "shannon") {
            metric = DiversityMetric::Shannon;
        } else if (metric_str == "pielou") {
            metric = DiversityMetric::Pielou;
        } else if (metric_str != "") {
            Rcpp::stop("Unsupported metric: use 'shannon' or 'pielou'");
        }

    } else if (task_str == "column_merge") {
        task = TaskType::ColumnMerge;
    } else {
        Rcpp::stop("Unsupported task: use 'diversity' or 'column_merge'");
    }

    // Run the selected task in parallel
    DiversityComputationWorker worker(index_matrix, values_vector, x_sp_mat, result, task, metric);
    RcppParallel::parallelFor(0, index_matrix.n_rows, worker);
    return result;
}
