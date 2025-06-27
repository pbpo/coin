#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem> // C++17 이상 필요
#include <stdexcept>  // For std::runtime_error

// For argument parsing (simple version)
#include <unordered_map>

namespace fs = std::filesystem;

// Simple command line argument parser
std::unordered_map<std::string, std::string> parse_args(int argc, char* argv[]) {
    std::unordered_map<std::string, std::string> args_map;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0) { // Starts with --
            std::string key = arg.substr(2);
            if (i + 1 < argc) {
                std::string value = argv[++i];
                if (value.rfind("--", 0) == 0) { // Next arg is another key
                    args_map[key] = "true"; // Treat as boolean flag
                    --i; // Decrement i to process the new key in next iteration
                } else {
                    args_map[key] = value;
                }
            } else {
                args_map[key] = "true"; // Boolean flag at the end
            }
        }
    }
    return args_map;
}

int main(int argc, char* argv[]) {
    auto args = parse_args(argc, argv);

    std::string source_dir_str;
    std::string output_file_str;

    if (args.count("source_dir")) {
        source_dir_str = args["source_dir"];
    } else {
        std::cerr << "오류: --source_dir 인자가 필요합니다." << std::endl;
        std::cerr << "사용법: aggregate_data --source_dir <입력_디렉토리> --output_file <출력_파일_경로>" << std::endl;
        return 1;
    }

    if (args.count("output_file")) {
        output_file_str = args["output_file"];
    } else {
        std::cerr << "오류: --output_file 인자가 필요합니다." << std::endl;
        std::cerr << "사용법: aggregate_data --source_dir <입력_디렉토리> --output_file <출력_파일_경로>" << std::endl;
        return 1;
    }

    fs::path source_dir(source_dir_str);
    fs::path output_file(output_file_str);

    if (!fs::exists(source_dir) || !fs::is_directory(source_dir)) {
        std::cerr << "오류: 소스 디렉토리 '" << source_dir_str << "'를 찾을 수 없거나 디렉토리가 아닙니다." << std::endl;
        return 1;
    }

    // Create output directory if it doesn't exist
    fs::path output_parent_dir = output_file.parent_path();
    if (!output_parent_dir.empty() && !fs::exists(output_parent_dir)) {
        try {
            fs::create_directories(output_parent_dir);
            std::cout << "생성된 디렉토리: " << output_parent_dir.string() << std::endl;
        } catch (const fs::filesystem_error& e) {
            std::cerr << "오류: 출력 디렉토리를 생성할 수 없습니다 - " << output_parent_dir.string() << ": " << e.what() << std::endl;
            return 1;
        }
    }


    std::vector<fs::path> log_files;
    try {
        for (const auto& entry : fs::recursive_directory_iterator(source_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".log") {
                log_files.push_back(entry.path());
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "오류: 소스 디렉토리를 순회하는 중 오류 발생 - " << source_dir_str << ": " << e.what() << std::endl;
        return 1;
    }


    if (log_files.empty()) {
        std::cerr << "경고: '" << source_dir_str << "'에서 .log 파일을 찾을 수 없습니다." << std::endl;
        // Create an empty output file if no source files are found
        std::ofstream empty_outfile(output_file);
        if (!empty_outfile.is_open()) {
            std::cerr << "오류: 빈 출력 파일을 생성할 수 없습니다 - " << output_file.string() << std::endl;
            return 1;
        }
        empty_outfile.close();
        std::cout << "빈 출력 파일이 생성되었습니다: " << output_file.string() << std::endl;
        return 0;
    }

    std::cout << "총 " << log_files.size() << "개 .log 파일 병합 시작 -> " << output_file.string() << std::endl;
    std::ofstream outfile(output_file, std::ios::binary); // Open in binary mode to preserve line endings
    if (!outfile.is_open()) {
        std::cerr << "오류: 출력 파일을 열 수 없습니다 - " << output_file.string() << std::endl;
        return 1;
    }

    char buffer[4096];
    for (const auto& filepath : log_files) {
        std::ifstream infile(filepath, std::ios::binary);
        if (infile.is_open()) {
            while (infile.read(buffer, sizeof(buffer))) {
                outfile.write(buffer, infile.gcount());
            }
            // Write any remaining part of the buffer
            outfile.write(buffer, infile.gcount());
        } else {
            std::cerr << "경고: 파일 열기 실패 건너뜀: " << filepath.string() << std::endl;
        }
    }
    outfile.close();
    std::cout << "병합 완료!" << std::endl;

    return 0;
}
