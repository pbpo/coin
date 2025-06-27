#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "utils/data_loader.hpp" // For CanDataLog
#include <nlohmann/json.hpp>    // For JSON operations

class CANTokenizer {
public:
    // ID_OFFSET is used to distinguish CAN ID tokens from data byte tokens.
    // Data bytes are 0-255. Special tokens are at the beginning.
    // CAN ID tokens will be original_can_id_decimal + ID_OFFSET.
    static constexpr int ID_OFFSET = 260; // Python code used 260. Max data byte 255 + few special tokens.
                                       // Let's ensure special tokens are handled correctly.
                                       // <PAD>:0, <UNK>:1, <MASK>:2, <VOID>:3
                                       // Data bytes '00'-'FF': 4 - 259
                                       // CAN IDs: original_id_dec + ID_OFFSET (e.g. 260 + id_dec)

    CANTokenizer();

    // Build vocabulary from CAN data log
    // The CanDataLog provides unique CAN IDs. Data bytes '00'-'FF' are fixed.
    void build_vocab(const CanDataLog& can_log);

    // Save vocabulary to a JSON file
    void save_vocab(const std::string& file_path) const;

    // Load vocabulary from a JSON file
    void load_vocab(const std::string& file_path);

    // Encode a sequence of string tokens into integer IDs
    std::vector<int> encode(const std::vector<std::string>& tokens) const;

    // Decode a sequence of integer IDs back into string tokens
    std::vector<std::string> decode(const std::vector<int>& ids) const;

    // Get the total size of the vocabulary
    int get_vocab_size() const;

    // Get the ID for a specific token
    // Throws std::out_of_range if token is not found (like .at())
    int get_token_id(const std::string& token) const;

    // Get the token for a specific ID
    // Returns "<UNK>" if ID is not found or is invalid
    std::string get_id_token(int id) const;

    // Get IDs of special tokens like <PAD>, <MASK>, etc.
    std::vector<int> get_special_token_ids() const;

    // Getters for specific special token IDs
    int get_pad_token_id() const { return token_to_id.at("<PAD>"); }
    int get_unk_token_id() const { return token_to_id.at("<UNK>"); }
    int get_mask_token_id() const { return token_to_id.at("<MASK>"); }
    int get_void_token_id() const { return token_to_id.at("<VOID>"); }


private:
    void _add_token(const std::string& token, int explicit_id = -1);

    std::unordered_map<std::string, int> token_to_id;
    std::unordered_map<int, std::string> id_to_token;
    std::vector<std::string> special_tokens_list; // To maintain order and easy access

    // Current next ID to assign if not explicit
    // Start from 0 for special tokens, then data tokens, then CAN ID tokens
    int next_token_id_ = 0;
};
