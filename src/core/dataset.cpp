#include "core/dataset.hpp"
#include <iostream>
#include <stdexcept> // For std::runtime_error
#include <numeric>   // For std::iota (potentially for candidate indices)
#include <algorithm> // For std::shuffle, std::sort, std::binary_search

MLMDataset::MLMDataset(const std::string& file_path,
                       const CANTokenizer& tokenizer,
                       int seq_len,
                       const std::string& dataset_type,
                       float mask_prob)
    : tokenizer_(tokenizer), // Initialize reference
      seq_len_(seq_len),
      mask_prob_(mask_prob),
      rng_(std::random_device{}()) // Seed the random number generator
{
    std::cout << "MLMDataset: Initializing..." << std::endl;
    std::cout << "MLMDataset: Loading and tokenizing data from file: " << file_path << " (type: " << dataset_type << ")" << std::endl;

    CanDataLog can_log;
    try {
        can_log = load_can_data(file_path, dataset_type);
    } catch (const std::exception& e) {
        throw std::runtime_error("MLMDataset: Failed to load CAN data: " + std::string(e.what()));
    }

    if (can_log.frames.empty()) {
        // It's not necessarily an error if the file is empty, but could be an issue.
        // The size() method will return 0.
        std::cout << "MLMDataset: Warning - No CAN frames loaded from " << file_path
                  << ". The dataset will be empty." << std::endl;
    }

    // Construct the token_id_stream by iterating through frames and their data
    // This follows the logic from the original Python's `pretrain.py` `main()` function
    // where it builds `token_stream` then encodes it.
    std::vector<std::string> string_token_stream;
    string_token_stream.reserve(can_log.frames.size() * (1 + 8)); // Pre-allocate (1 ID + avg 8 data bytes)

    for (const auto& frame : can_log.frames) {
        // Token for CAN ID: convert hex ID to decimal, add offset, then convert to string
        try {
            unsigned long can_id_dec = std::stoul(frame.id, nullptr, 16);
            string_token_stream.push_back(std::to_string(can_id_dec + CANTokenizer::ID_OFFSET));
        } catch (const std::exception& e) {
            // This can happen if frame.id is not a valid hex string, though load_can_data should filter most.
            std::cerr << "MLMDataset: Warning - Skipping invalid CAN ID format in log: " << frame.id << std::endl;
            continue;
        }

        // Tokens for data bytes (already hex strings '00'-'FF')
        for (const auto& byte_token : frame.data_bytes) {
            string_token_stream.push_back(byte_token);
        }
    }

    if (string_token_stream.empty() && !can_log.frames.empty()) {
         std::cout << "MLMDataset: Warning - String token stream is empty despite having CAN frames. Check ID processing." << std::endl;
    }


    this->token_id_stream_ = tokenizer_.encode(string_token_stream);

    // Initialize masking-related members from the tokenizer
    this->mask_token_id_ = tokenizer_.get_mask_token_id();
    this->pad_token_id_ = tokenizer_.get_pad_token_id(); // Store if needed
    this->vocab_size_ = tokenizer_.get_vocab_size();
    this->special_token_ids_sorted_ = tokenizer_.get_special_token_ids();
    std::sort(this->special_token_ids_sorted_.begin(), this->special_token_ids_sorted_.end());

    std::cout << "MLMDataset: Initialization complete. Token ID stream length: " << this->token_id_stream_.size() << std::endl;
    if (this->token_id_stream_.size() < static_cast<size_t>(this->seq_len_)) {
        std::cout << "MLMDataset: Warning - Token stream length (" << this->token_id_stream_.size()
                  << ") is less than sequence length (" << this->seq_len_
                  << "). Dataset will effectively be empty." << std::endl;
    }
}

size_t MLMDataset::size() const {
    if (token_id_stream_.size() < static_cast<size_t>(seq_len_)) {
        return 0; // Not enough tokens to form even one sequence
    }
    // Number of possible starting positions for a sequence of seq_len_
    return token_id_stream_.size() - seq_len_ + 1;
}

MLMBatch MLMDataset::get_item(size_t idx) const {
    if (idx + seq_len_ > token_id_stream_.size()) {
        throw std::out_of_range("MLMDataset::get_item: Index out of range.");
    }

    // Extract the original sequence segment
    std::vector<int> sequence_segment(token_id_stream_.begin() + idx,
                                      token_id_stream_.begin() + idx + seq_len_);

    std::vector<int> input_ids = sequence_segment; // Copy for modification
    std::vector<int> labels(seq_len_, -100);    // Initialize labels with -100 (ignore index for loss)

    // Identify candidate indices for masking (non-special tokens)
    std::vector<int> candidate_indices;
    for (int i = 0; i < seq_len_; ++i) {
        // Check if input_ids[i] is a special token
        if (!std::binary_search(special_token_ids_sorted_.begin(),
                                special_token_ids_sorted_.end(),
                                input_ids[i])) {
            candidate_indices.push_back(i);
        }
    }

    // Shuffle candidate indices to pick random ones for masking
    std::shuffle(candidate_indices.begin(), candidate_indices.end(), rng_);

    // Determine how many tokens to mask (approximately mask_prob_ * number of non-special tokens)
    int num_to_mask = static_cast<int>(candidate_indices.size() * mask_prob_);

    for (int i = 0; i < num_to_mask; ++i) {
        int masked_idx = candidate_indices[i];

        // Store the original token ID in labels
        labels[masked_idx] = input_ids[masked_idx];

        // Apply masking strategy (from Hugging Face Transformers)
        // 80% of the time, replace with [MASK] token
        // 10% of the time, replace with a random token
        // 10% of the time, keep the original token (no change to input_ids[masked_idx])
        std::uniform_real_distribution<> dist_prob(0.0, 1.0);
        float rand_val = dist_prob(rng_);

        if (rand_val < 0.8f) {
            input_ids[masked_idx] = mask_token_id_;
        } else if (rand_val < 0.9f) {
            // Replace with a random token (excluding special tokens if possible, or just any token)
            // For simplicity, pick any token from vocab.
            // Ensure the random token is not a special token, if that's desired.
            // The original Python code just picks from the whole vocab_size.
            std::uniform_int_distribution<> dist_token(0, vocab_size_ - 1);
            int random_token_id = dist_token(rng_);
            // Avoid picking another special token if the original wasn't special, could be refined.
            // For now, simple random token:
            input_ids[masked_idx] = random_token_id;
        } else {
            // 10% of the time, do nothing (keep original token)
            // input_ids[masked_idx] remains unchanged.
        }
    }

    // Attention mask is all 1s for CAN-BERT as per typical BERT unless padding is involved.
    // If sequences could be shorter than seq_len_ and padded, this would be different.
    // Here, all sequences are exactly seq_len_.
    std::vector<int> attention_mask(seq_len_, 1);

    return {input_ids, attention_mask, labels};
}
