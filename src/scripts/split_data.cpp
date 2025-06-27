#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <algorithm> // For std::shuffle
#include <random>    // For std::mt19937

namespace fs = std::filesystem;

// Simple command line argument parser
std::unordered_map<std::string, std::string> parse_args_split(int argc, char* argv[]) {
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

// Function to write lines to a file
void write_lines(const fs::path& file_path, const std::vector<std::string>& lines) {
    fs::create_directories(file_path.parent_path()); // Ensure directory exists
    std::ofstream outfile(file_path);
    if (!outfile.is_open()) {
        throw std::runtime_error("출력 파일을 열 수 없습니다: " + file_path.string());
    }
    for (const auto& line : lines) {
        outfile << line << '\n';
    }
    outfile.close();
}

int main(int argc, char* argv[]) {
    auto args = parse_args_split(argc, argv);

    std::string input_file_str;
    std::string output_dir_str;
    double train_ratio = 0.8;
    double val_ratio = 0.1;
    // test_ratio is 1.0 - train_ratio - val_ratio
    bool shuffle = false;
    unsigned int random_seed = std::random_device{}();

    if (args.count("input_file")) input_file_str = args["input_file"];
    else {
        std::cerr << "오류: --input_file 인자가 필요합니다." << std::endl; return 1;
    }
    if (args.count("output_dir")) output_dir_str = args["output_dir"];
    else {
        std::cerr << "오류: --output_dir 인자가 필요합니다." << std::endl; return 1;
    }
    if (args.count("train_ratio")) train_ratio = std::stod(args["train_ratio"]);
    if (args.count("val_ratio")) val_ratio = std::stod(args["val_ratio"]);
    if (args.count("shuffle") && args["shuffle"] == "true") shuffle = true;
    if (args.count("seed")) random_seed = std::stoul(args["seed"]);

    if (train_ratio < 0.0 || train_ratio > 1.0 || val_ratio < 0.0 || val_ratio > 1.0 || (train_ratio + val_ratio) > 1.0) {
        std::cerr << "오류: 비율은 0.0과 1.0 사이여야 하며, train_ratio + val_ratio <= 1.0 이어야 합니다." << std::endl;
        return 1;
    }

    fs::path input_file(input_file_str);
    fs::path output_dir(output_dir_str);

    if (!fs::exists(input_file) || !fs::is_regular_file(input_file)) {
        std::cerr << "오류: 입력 파일 '" << input_file_str << "'을(를) 찾을 수 없거나 파일이 아닙니다." << std::endl;
        return 1;
    }

    try {
        fs::create_directories(output_dir); // Create output directory if it doesn't exist
    } catch (const fs::filesystem_error& e) {
        std::cerr << "오류: 출력 디렉토리를 생성할 수 없습니다 - " << output_dir_str << ": " << e.what() << std::endl;
        return 1;
    }

    std::vector<std::string> all_lines;
    std::ifstream infile(input_file);
    if (!infile.is_open()) {
        std::cerr << "오류: 입력 파일을 열 수 없습니다 - " << input_file_str << std::endl;
        return 1;
    }
    std::string line;
    while (std::getline(infile, line)) {
        all_lines.push_back(line);
    }
    infile.close();

    if (all_lines.empty()) {
        std::cout << "경고: 입력 파일이 비어있습니다. 빈 분할 파일이 생성됩니다." << std::endl;
        write_lines(output_dir / "train.log", {});
        write_lines(output_dir / "val.log", {});
        write_lines(output_dir / "test.log", {});
        return 0;
    }

    if (shuffle) {
        std::cout << "데이터 셔플링 중 (시드: " << random_seed << ")" << std::endl;
        std::mt19937 g(random_seed);
        std::shuffle(all_lines.begin(), all_lines.end(), g);
    }

    size_t total_lines = all_lines.size();
    size_t train_count = static_cast<size_t>(total_lines * train_ratio);
    size_t val_count = static_cast<size_t>(total_lines * val_ratio);
    size_t test_count = total_lines - train_count - val_count;

    // Ensure test_count is not negative if ratios are slightly off due to rounding
    if (train_count + val_count > total_lines) {
        val_count = total_lines - train_count; // Adjust val_count
        test_count = 0;
    }
     if (val_count > total_lines - train_count) { // if train_ratio was 1.0
        val_count = total_lines - train_count;
    }
    if (train_count > total_lines) train_count = total_lines;


    std::vector<std::string> train_lines(all_lines.begin(), all_lines.begin() + train_count);
    std::vector<std::string> val_lines(all_lines.begin() + train_count, all_lines.begin() + train_count + val_count);
    std::vector<std::string> test_lines(all_lines.begin() + train_count + val_count, all_lines.end());
    // If test_count became negative from calculation, .end() handles it okay, giving empty vector.
    // But ensure test_lines are correctly assigned even if total_lines - train_count - val_count was negative.
    // The above calculation for test_count ensures it's >=0.

    fs::path train_file = output_dir / "train.log";
    fs::path val_file = output_dir / "val.log";
    fs::path test_file = output_dir / "test.log";

    std::cout << "훈련 세트 저장 중 (" << train_lines.size() << " 라인) -> " << train_file.string() << std::endl;
    write_lines(train_file, train_lines);

    std::cout << "검증 세트 저장 중 (" << val_lines.size() << " 라인) -> " << val_file.string() << std::endl;
    write_lines(val_file, val_lines);

    std::cout << "테스트 세트 저장 중 (" << test_lines.size() << " 라인) -> " << test_file.string() << std::endl;
    write_lines(test_file, test_lines);

    std::cout << "데이터 분할 완료!" << std::endl;

    return 0;
} catch (const std::exception& e) {
    std::cerr << "오류 발생: " << e.what() << std::endl;
    return 1;
}
