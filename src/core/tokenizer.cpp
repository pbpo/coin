#include "core/tokenizer.hpp"
#include <fstream>
#include <iostream>
#include <algorithm> // For std::find
#include <stdexcept> // For std::runtime_error, std::out_of_range
#include <iomanip>   // For std::setw, std::setfill, std::hex, std::dec


CANTokenizer::CANTokenizer() : next_token_id_(0) {
    // Initialize special tokens first and ensure they get IDs 0, 1, 2, 3...
    // The order matters if specific IDs are assumed by the model (e.g., PAD=0)
    special_tokens_list = {"<PAD>", "<UNK>", "<MASK>", "<VOID>"};
    for (const auto& token : special_tokens_list) {
        _add_token(token); // Assigns next_token_id_ automatically
    }
}

void CANTokenizer::_add_token(const std::string& token, int explicit_id) {
    if (token_to_id.find(token) == token_to_id.end()) {
        int id_to_assign;
        if (explicit_id != -1) {
            // Ensure this explicit ID isn't already taken by another token
            if (id_to_token.count(explicit_id) && id_to_token.at(explicit_id) != token) {
                 throw std::runtime_error("Explicit ID " + std::to_string(explicit_id) +
                                         " is already assigned to token '" + id_to_token.at(explicit_id) +
                                         "'. Cannot assign to '" + token + "'.");
            }
            id_to_assign = explicit_id;
        } else {
            id_to_assign = next_token_id_;
        }

        token_to_id[token] = id_to_assign;
        id_to_token[id_to_assign] = token;

        if (explicit_id == -1) {
            next_token_id_++;
        } else {
            // If we added an explicit ID, we need to ensure next_token_id_ is beyond the max assigned ID
            next_token_id_ = std::max(next_token_id_, id_to_assign + 1);
        }
    }
}

void CANTokenizer::build_vocab(const CanDataLog& can_log) {
    // Special tokens are already added by the constructor.
    // next_token_id_ is currently after special tokens.

    // 1. Add data byte tokens ('00' to 'FF')
    // These should come right after special tokens.
    for (int i = 0; i < 256; ++i) {
        std::ostringstream oss;
        oss << std::hex << std::uppercase << std::setw(2) << std::setfill('0') << i;
        _add_token(oss.str()); // Assigns next_token_id_ automatically
    }

    // 2. Add CAN ID tokens (original_id_decimal + ID_OFFSET)
    // ID_OFFSET is designed to place these tokens after data byte tokens.
    // The Python code used `std.stol(can_id_hex, nullptr, 16) + ID_OFFSET` as the token string.
    // Let's verify if the *token itself* is "value+offset" or if the *ID assigned to token* is "value+offset".
    // The python code `_add_token(std::to_string(can_id_dec + ID_OFFSET))` suggests the token string IS "value+offset".
    // The ID assigned to this token string will be `next_token_id_`.

    // The provided C++ code for CANTokenizer had:
    // `std::string token = std::to_string(can_id_dec + ID_OFFSET); _add_token(token);`
    // This means the *string representation* of the CAN ID in the vocabulary is, for example, "260+CAN_ID_as_decimal".
    // And its vocabulary ID (the integer used in encoding) is determined by `_add_token`.

    // However, the initial Python code for `CANTokenizer` in `models/tokenizer.py` implies:
    // `token = str(int(can_id, 16) + self.ID_OFFSET)` and `self.token_to_id[token] = index`.
    // This means the *string* "260+CAN_ID" is the token, and its ID is `index`.
    // This seems consistent.

    auto unique_ids_hex = can_log.get_unique_can_ids();
    for (const auto& can_id_hex : unique_ids_hex) {
        try {
            unsigned long can_id_dec = std::stoul(can_id_hex, nullptr, 16);
            std::string token_representation = std::to_string(can_id_dec + ID_OFFSET);
            _add_token(token_representation); // Assigns next_token_id_ automatically
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Warning: Invalid CAN ID hex string for vocab building: " << can_id_hex << " - " << ia.what() << std::endl;
        } catch (const std::out_of_range& oor) {
            std::cerr << "Warning: CAN ID hex string out of range for vocab building: " << can_id_hex << " - " << oor.what() << std::endl;
        }
    }
}


void CANTokenizer::save_vocab(const std::string& file_path) const {
    // We need to save both token_to_id and id_to_token, or reconstruct one from the other.
    // Also save next_token_id_ and special_tokens_list for consistency.
    nlohmann::json j;
    j["token_to_id"] = token_to_id;
    j["id_to_token"] = id_to_token; // For easier decoding and verification
    j["special_tokens_list"] = special_tokens_list;
    j["next_token_id"] = next_token_id_; // To ensure consistency if vocab is extended later (though typically not done after training starts)
    j["id_offset"] = ID_OFFSET; // Store for reference

    std::ofstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + file_path);
    }
    file << j.dump(4); // Pretty print with indent 4
}

void CANTokenizer::load_vocab(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for reading: " + file_path);
    }
    nlohmann::json j;
    try {
        file >> j;
        token_to_id = j.at("token_to_id").get<std::unordered_map<std::string, int>>();
        id_to_token = j.at("id_to_token").get<std::unordered_map<int, std::string>>();
        special_tokens_list = j.at("special_tokens_list").get<std::vector<std::string>>();
        next_token_id_ = j.at("next_token_id").get<int>();

        // Verify ID_OFFSET consistency if needed, though it's static const
        if (j.contains("id_offset") && j.at("id_offset").get<int>() != ID_OFFSET) {
            std::cerr << "Warning: ID_OFFSET in loaded vocab (" << j.at("id_offset").get<int>()
                      << ") differs from compiled ID_OFFSET (" << ID_OFFSET << ")." << std::endl;
        }

    } catch (nlohmann::json::parse_error& e) {
        throw std::runtime_error("Failed to parse vocabulary file " + file_path + ": " + e.what());
    } catch (nlohmann::json::out_of_range& e) { // For missing keys
        throw std::runtime_error("Missing key in vocabulary file " + file_path + ": " + e.what());
    }
}

std::vector<int> CANTokenizer::encode(const std::vector<std::string>& tokens) const {
    std::vector<int> ids;
    ids.reserve(tokens.size());
    int unk_id = token_to_id.at("<UNK>"); // Assuming <UNK> is always present
    for (const auto& token : tokens) {
        auto it = token_to_id.find(token);
        if (it != token_to_id.end()) {
            ids.push_back(it->second);
        } else {
            // Attempt to handle numeric CAN ID tokens that might not have been explicitly added
            // if they follow the "value+ID_OFFSET" format. This is a fallback.
            // A more robust way is to ensure all expected tokens are in vocab.
            try {
                // Check if token is a number, it might be a CAN ID string like "360" (e.g. 100 + 260)
                // This part is tricky: is the input token "1A4" (hex CAN ID) or "360" (offsetted decimal string)?
                // The original python code's MLMDataset `__getitem__` generates tokens like:
                // `token_stream.append(str(int(can_df['CAN_ID'].iloc[i], 16) + self.tokenizer.ID_OFFSET))`
                // So, the tokens passed to encode() for CAN IDs are already in "value+offset" string format.
                // If such a string is not in token_to_id, it means it wasn't in the training data's unique_can_ids.
                ids.push_back(unk_id);
                 // std::cerr << "Token not found in vocab: " << token << ", using UNK." << std::endl;
            } catch (const std::exception&) {
                ids.push_back(unk_id); // Not a number or other issue
            }
        }
    }
    return ids;
}

std::vector<std::string> CANTokenizer::decode(const std::vector<int>& ids) const {
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    for (int id : ids) {
        auto it = id_to_token.find(id);
        if (it != id_to_token.end()) {
            tokens.push_back(it->second);
        } else {
            tokens.push_back("<UNK>"); // Or some other representation for unknown ID
        }
    }
    return tokens;
}

int CANTokenizer::get_vocab_size() const {
    return token_to_id.size();
}

int CANTokenizer::get_token_id(const std::string& token) const {
    try {
        return token_to_id.at(token);
    } catch (const std::out_of_range& oor) {
        // For robustness, one might return UNK_ID here instead of throwing,
        // but .at() behavior is to throw.
        // Let's re-throw to make it clear the token is missing.
        // Or, align with Python's `tokenizer.token_to_id.get(token, unk_id)`
        // For now, stick to `at()` semantics. If UNK is desired, caller handles exception or uses a find.
        throw std::out_of_range("Token '" + token + "' not found in vocabulary.");
    }
}

std::string CANTokenizer::get_id_token(int id) const {
    auto it = id_to_token.find(id);
    if (it != id_to_token.end()) {
        return it->second;
    }
    return "<UNK>"; // Default for unknown IDs
}

std::vector<int> CANTokenizer::get_special_token_ids() const {
    std::vector<int> ids;
    for(const auto& token_str : special_tokens_list){
        ids.push_back(get_token_id(token_str));
    }
    return ids;
}
