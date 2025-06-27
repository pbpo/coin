#include "utils/data_loader.hpp"
#include "core/tokenizer.hpp"
#include <iostream>
#include <string>
#include <stdexcept>
#include <filesystem> // C++17
#include <unordered_map> // For argument parsing

namespace fs = std::filesystem;

// Simple command line argument parser (can be reused or moved to a common util)
std::unordered_map<std::string, std::string> parse_args_vocab(int argc, char* argv[]) {
    std::unordered_map<std::string, std::string> args_map;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0) {
            std::string key = arg.substr(2);
            if (i + 1 < argc) {
                std::string value = argv[++i];
                 if (value.rfind("--", 0) == 0) {
                    args_map[key] = "true";
                    --i;
                } else {
                    args_map[key] = value;
                }
            } else {
                args_map[key] = "true";
            }
        }
    }
    return args_map;
}

int main(int argc, char* argv[]) {
    auto args = parse_args_vocab(argc, argv);

    std::string data_file_str;
    std::string vocab_path_str;
    std::string data_format_type = "hcrl"; // Default format type

    if (args.count("data_file")) {
        data_file_str = args["data_file"];
    } else {
        std::cerr << "오류: --data_file 인자가 필요합니다." << std::endl;
        std::cerr << "사용법: build_vocab --data_file <데이터_파일_경로> --vocab_path <어휘_파일_저장_경로> [--data_format <candump|hcrl>]" << std::endl;
        return 1;
    }

    if (args.count("vocab_path")) {
        vocab_path_str = args["vocab_path"];
    } else {
        std::cerr << "오류: --vocab_path 인자가 필요합니다." << std::endl;
         std::cerr << "사용법: build_vocab --data_file <데이터_파일_경로> --vocab_path <어휘_파일_저장_경로> [--data_format <candump|hcrl>]" << std::endl;
        return 1;
    }

    if (args.count("data_format")) {
        data_format_type = args["data_format"];
        if (data_format_type != "candump" && data_format_type != "hcrl" && data_format_type != "log") {
            std::cerr << "오류: 지원되지 않는 --data_format 값입니다: " << data_format_type << ". 'candump' 또는 'hcrl'/'log'를 사용하세요." << std::endl;
            return 1;
        }
    }


    fs::path data_file(data_file_str);
    fs::path vocab_path(vocab_path_str);

    if (!fs::exists(data_file) || !fs::is_regular_file(data_file)) {
        std::cerr << "오류: 데이터 파일 '" << data_file_str << "'를 찾을 수 없거나 파일이 아닙니다." << std::endl;
        return 1;
    }

    // Create output directory for vocab if it doesn't exist
    fs::path vocab_parent_dir = vocab_path.parent_path();
    if (!vocab_parent_dir.empty() && !fs::exists(vocab_parent_dir)) {
        try {
            fs::create_directories(vocab_parent_dir);
            std::cout << "생성된 디렉토리: " << vocab_parent_dir.string() << std::endl;
        } catch (const fs::filesystem_error& e) {
            std::cerr << "오류: 어휘 저장 디렉토리를 생성할 수 없습니다 - " << vocab_parent_dir.string() << ": " << e.what() << std::endl;
            return 1;
        }
    }

    try {
        std::cout << "데이터 로딩 중 어휘 구축 시작: " << data_file_str << " (형식: " << data_format_type << ")" << std::endl;
        CanDataLog can_log = load_can_data(data_file_str, data_format_type);

        if (can_log.frames.empty()) {
            std::cout << "경고: 로드된 CAN 프레임이 없습니다. 빈 어휘집이 생성될 수 있습니다." << std::endl;
        }


        CANTokenizer tokenizer;
        std::cout << "어휘 구축 중..." << std::endl;
        tokenizer.build_vocab(can_log); // Build vocab from loaded data

        std::cout << "어휘 저장 중 -> " << vocab_path_str << std::endl;
        tokenizer.save_vocab(vocab_path_str);

        std::cout << "어휘집 생성 완료! 총 " << tokenizer.get_vocab_size() << "개 토큰이 '" << vocab_path_str << "'에 저장되었습니다." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "어휘 구축 중 오류 발생: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
