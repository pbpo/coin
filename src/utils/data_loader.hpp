#pragma once

#include <string>
#include <vector>
#include <stdexcept> // For std::runtime_error

// Structure to hold CAN data
struct CanFrame {
    std::string timestamp; // Original timestamp string or calculated time
    std::string can_interface; // e.g., "can0", "vcan0"
    std::string id;        // CAN ID (hex string, e.g., "18FF00F0")
    std::vector<std::string> data_bytes; // Data bytes (hex strings, e.g., "DE", "AD")
    size_t dlc; // Data Length Code (actual number of data bytes)

    // Helper to convert data bytes to a single string if needed
    std::string data_to_string() const {
        std::string s;
        for (const auto& byte : data_bytes) {
            s += byte;
        }
        return s;
    }
};

// Structure to hold all loaded CAN frames from a file
struct CanDataLog {
    std::vector<CanFrame> frames;
    // We can add metadata here if needed, e.g., source file name

    // Helper to get unique CAN IDs, similar to python df.unique_can_ids()
    std::vector<std::string> get_unique_can_ids() const;
};

// Function to load CAN data from a file
// Supports "candump" and "hcrl" (assumed to be similar to candump, possibly without interface name)
CanDataLog load_can_data(const std::string& file_path, const std::string& format_type = "candump");

// Helper function (declaration, definition in .cpp)
std::vector<std::string> split_string(const std::string& s, char delimiter);
std::string trim_string(const std::string& str);
bool is_hex_string(const std::string& s);
