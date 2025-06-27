#include "utils/data_loader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm> // For std::remove_if, std::sort, std::unique
#include <iomanip>   // For std::setw, std::setfill with std::hex
#include <cctype>    // For std::isspace, std::isxdigit

// Helper function to trim leading/trailing whitespace
std::string trim_string(const std::string& str) {
    const std::string whitespace = " \t\n\r\f\v";
    size_t start = str.find_first_not_of(whitespace);
    if (start == std::string::npos) {
        return ""; // Return empty string if only whitespace
    }
    size_t end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
}

// Helper function to split string by delimiter
std::vector<std::string> split_string(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Helper function to check if a string is a valid hex string
bool is_hex_string(const std::string& s) {
    if (s.empty()) return false;
    return std::all_of(s.begin(), s.end(), ::isxdigit);
}


// Implementation for CanDataLog::get_unique_can_ids
std::vector<std::string> CanDataLog::get_unique_can_ids() const {
    std::vector<std::string> ids;
    for (const auto& frame : frames) {
        ids.push_back(frame.id);
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}


CanDataLog load_can_data(const std::string& file_path, const std::string& format_type) {
    CanDataLog data_log;
    std::ifstream file(file_path);
    std::string line;

    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file " + file_path);
    }

    // Candump format typically: (timestamp) interface id##data
    // Example: (1609459200.000000) can0 123#DEADBEEF
    // HCRL format might be similar, or simplified. Assuming it's space-separated
    // and CAN ID is identifiable. Let's make it flexible.
    // Example HCRL (assumption): timestamp ID D0 D1 D2 D3 D4 D5 D6 D7 (or similar)
    // For HCRL, it's mentioned: ID_OFFSET (260) is added to CAN ID for tokens.
    // The raw log file itself contains hex CAN IDs.

    while (std::getline(file, line)) {
        line = trim_string(line);
        if (line.empty() || line[0] == '#') { // Skip empty lines or comments
            continue;
        }

        CanFrame frame;
        std::vector<std::string> parts;

        if (format_type == "candump") {
            // Example: (1609459200.000000) can0 123#DEADBEEF
            // Or:        1609459200.000000  can0 123#DEADBEEF (if parentheses are not strict)

            size_t time_end_pos = line.find(')');
            if (time_end_pos != std::string::npos) {
                frame.timestamp = trim_string(line.substr(1, time_end_pos - 1));
                line = trim_string(line.substr(time_end_pos + 1));
            } else {
                // Try to parse timestamp if it's just at the beginning without parens
                std::istringstream iss_first_part(line);
                std::string first_token;
                iss_first_part >> first_token;
                // A simple check: if it contains a dot and digits, assume it's a timestamp
                if (first_token.find('.') != std::string::npos &&
                    std::all_of(first_token.begin(), first_token.end(), [](char c){ return std::isdigit(c) || c == '.'; })) {
                    frame.timestamp = first_token;
                    line = trim_string(line.substr(first_token.length()));
                } else {
                    frame.timestamp = "0.0"; // Default or indicate missing
                }
            }

            parts = split_string(line, ' ');
            if (parts.size() < 2) continue; // Expecting at least interface and id#data

            frame.can_interface = parts[0];

            std::string id_data_part;
            // Handle cases where there might be extra spaces between interface and id#data
            if (parts.size() > 2) {
                 for(size_t i = 1; i < parts.size(); ++i) id_data_part += parts[i];
            } else {
                 id_data_part = parts[1];
            }

            std::vector<std::string> id_data_split = split_string(id_data_part, '#');
            if (id_data_split.size() != 2 && id_data_split.size() != 1) continue; // Expecting ID and Data, or just ID if no data (DLC 0)

            frame.id = trim_string(id_data_split[0]);
            if (!is_hex_string(frame.id)) continue; // Invalid CAN ID

            if (id_data_split.size() == 2) {
                std::string data_hex_str = trim_string(id_data_split[1]);
                if (data_hex_str.length() % 2 != 0) continue; // Hex data bytes must be in pairs

                for (size_t i = 0; i < data_hex_str.length(); i += 2) {
                    frame.data_bytes.push_back(data_hex_str.substr(i, 2));
                }
            }
            frame.dlc = frame.data_bytes.size();

        } else if (format_type == "hcrl" || format_type == "log") { // Assuming HCRL is similar to general log
            // Example: (timestamp) ID DataByte0 DataByte1 ...
            // Or:       ID DataByte0 DataByte1 ... (if no timestamp in this format)
            // Or from the original Python: it seems to parse lines like "123 DE AD BE EF..." for HCRL
            // Let's assume space-separated values, first is ID, rest are data bytes

            // Try to extract timestamp if present and enclosed in ()
            size_t time_end_pos = line.find(')');
            if (time_end_pos != std::string::npos && line[0] == '(') {
                frame.timestamp = trim_string(line.substr(1, time_end_pos - 1));
                line = trim_string(line.substr(time_end_pos + 1));
            } else {
                // No explicit timestamp or not in (): check if first token is a float-like string
                std::string potential_ts;
                std::istringstream iss_line(line);
                iss_line >> potential_ts;
                if (potential_ts.find('.') != std::string::npos &&
                    std::all_of(potential_ts.begin(), potential_ts.end(), [](char c){ return std::isdigit(c) || c == '.'; })) {
                    frame.timestamp = potential_ts;
                    // Consume the timestamp from the line
                    size_t first_space = line.find_first_of(" \t");
                    if(first_space != std::string::npos) {
                        line = trim_string(line.substr(first_space));
                    } else {
                        line = ""; // Only timestamp was on the line
                    }
                } else {
                     frame.timestamp = "0.0"; // Default or indicate missing
                }
            }

            frame.can_interface = "unknown"; // HCRL format might not specify interface

            std::istringstream iss(line);
            std::string token;

            // First token is CAN ID
            if (!(iss >> token)) continue;
            frame.id = trim_string(token);
            if (!is_hex_string(frame.id)) continue; // Skip if ID is not hex

            // Subsequent tokens are data bytes
            while (iss >> token) {
                token = trim_string(token);
                if (token.length() == 2 && is_hex_string(token)) {
                    frame.data_bytes.push_back(token);
                } else if (token.length() == 1 && is_hex_string(token)) { // handle single digit hex like "0"
                    frame.data_bytes.push_back("0" + token);
                } else if (!token.empty()) {
                    // Potentially malformed data byte, decide how to handle
                    // For now, we'll skip this frame if a data byte is malformed
                    // Or, break and take what we have so far. Let's be strict.
                    // std::cerr << "Warning: Malformed data byte '" << token << "' in line: " << line_copy_for_error << std::endl;
                    frame.data_bytes.clear(); // Invalidate data for this frame
                    break;
                }
            }
            if (frame.id.empty()) continue; // Skip if ID parsing failed earlier
            frame.dlc = frame.data_bytes.size();

        } else {
            throw std::runtime_error("Unsupported log format type: " + format_type);
        }

        // Normalize CAN ID to uppercase and ensure consistent length if desired (e.g. 3 or 8 chars for specific CAN types)
        // For now, just uppercase
        std::transform(frame.id.begin(), frame.id.end(), frame.id.begin(), ::toupper);
        for(auto& byte : frame.data_bytes) {
            std::transform(byte.begin(), byte.end(), byte.begin(), ::toupper);
        }

        data_log.frames.push_back(frame);
    }

    file.close();
    if (data_log.frames.empty() && file.eof() && !file_path.empty()) { // Check if file was read but no valid frames found
        std::cout << "Warning: No valid CAN frames loaded from " << file_path << ". File might be empty or in an unexpected format." << std::endl;
    }
    return data_log;
}
