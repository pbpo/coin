#pragma once

#include "core/tokenizer.hpp"
#include "utils/data_loader.hpp" // For CanDataLog and load_can_data
#include <string>
#include <vector>
#include <random>       // For std::mt19937, std::random_device, distributions
#include <algorithm>    // For std::shuffle, std::binary_search, std::sort

// Structure to hold a batch for Masked Language Modeling (MLM)
struct MLMBatch {
    std::vector<int> input_ids;      // Token IDs for the input sequence (potentially masked)
    std::vector<int> attention_mask; // Mask to indicate which tokens should be attended to (1) or not (0)
    std::vector<int> labels;         // Token IDs for the original tokens at masked positions, -100 otherwise
};

class MLMDataset {
public:
    // Constructor
    // file_path: Path to the CAN log file.
    // tokenizer: Reference to an initialized CANTokenizer.
    // seq_len: The fixed length of sequences to be generated.
    // dataset_type: Passed to load_can_data (e.g., "candump", "hcrl").
    // mask_prob: Probability of masking a token.
    MLMDataset(const std::string& file_path,
               const CANTokenizer& tokenizer,
               int seq_len,
               const std::string& dataset_type = "candump",
               float mask_prob = 0.15f);

    // Returns the total number of possible sequences that can be generated from the dataset.
    size_t size() const;

    // Gets a single data item (sequence) at the given index, applying dynamic masking.
    // idx: The starting index in the token_id_stream for the sequence.
    MLMBatch get_item(size_t idx) const; // Made const, rng needs to be mutable

private:
    const CANTokenizer& tokenizer_; // Store as reference
    int seq_len_;
    float mask_prob_;
    std::vector<int> token_id_stream_; // Concatenated stream of all token IDs from the log file

    // Masking related members, initialized from tokenizer
    int mask_token_id_;
    int pad_token_id_; // Though not explicitly used in Python's MLM masking logic for input_ids, good to have.
    int vocab_size_;
    std::vector<int> special_token_ids_sorted_; // For efficient checking during masking

    // Random number generator for dynamic masking.
    // `mutable` allows modification in `const` member functions like get_item.
    mutable std::mt19937 rng_;
};
