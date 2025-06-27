#include <iostream>
#include <string>
#include <vector>
#include <cstdlib> // For system()
#include <filesystem> // For path manipulation and checking executable existence

namespace fs = std::filesystem;

// Function to execute a command and check its status
int execute_command(const std::string& command, const std::string& executable_name_for_error = "") {
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "실행 중: " << command << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
    int status = std::system(command.c_str());
    if (status != 0) {
        std::cerr << "오류: 명령어 실행 실패 (종료 코드: " << status << ")" << std::endl;
        if (!executable_name_for_error.empty()) {
            std::cerr << executable_name_for_error << " 실행 파일이 현재 PATH에 있는지 또는 정확한 경로로 지정되었는지 확인하십시오." << std::endl;
        }
        // exit(status); // Exit if any command fails
    }
    std::cout << "명령어 완료 (종료 코드: " << status << ")" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
    return status;
}

// Helper to find executable relative to this script's likely build location
// This is a very basic helper and might need adjustment based on actual build structure.
std::string find_executable_path(const std::string& exec_name, const fs::path& base_path_hint) {
    // 1. Check if exec_name itself is a full path or in PATH
    fs::path exec_p(exec_name);
    if (fs::exists(exec_p) && fs::is_regular_file(exec_p)) return exec_name; // Already a path or found in current dir

    // Try to find it in PATH (system() will do this, but good to check)
    // This is non-trivial to do portably in C++. system() handles it.

    // 2. Try common relative paths from build directory structure
    // Assuming executables are in the same directory as run_full_training or one level up (e.g., build/scripts/ and build/)
    if (fs::exists(base_path_hint / exec_name)) return (base_path_hint / exec_name).string();
    if (fs::exists(base_path_hint.parent_path() / exec_name)) return (base_path_hint.parent_path() / exec_name).string();

    // Fallback to just using the name, relying on PATH
    return exec_name;
}


int main(int argc, char *argv[]) {
    std::cout << "전체 훈련 파이프라인 시작..." << std::endl;

    // Determine the path of the current executable to help find others if they are relative.
    fs::path current_executable_path;
    if (argc > 0 && argv[0] != nullptr) {
        current_executable_path = fs::path(argv[0]).parent_path();
    } else {
        current_executable_path = fs::current_path(); // Fallback
    }
    std::cout << "실행 파일 기본 검색 경로: " << current_executable_path.string() << std::endl;


    // --- Configuration (데이터 경로 등은 여기서 설정하거나 외부에서 인자로 받아야 함) ---
    // 이 예제에서는 하드코딩된 경로를 사용하지만, 실제로는 인자 파싱이 필요합니다.
    std::string base_data_dir = "data"; // Base for all data operations
    std::string raw_data_source_dir = "dataset/CAN-MIRGU(train)/Benign"; // Example source
    std::string aggregated_log_dir = base_data_dir + "/HCRL_dataset";
    std::string aggregated_log_file = aggregated_log_dir + "/train_aggregated.log";

    std::string checkpoints_dir = "checkpoints";
    std::string vocab_file = checkpoints_dir + "/vocab.json";

    std::string split_data_input_file = aggregated_log_file;
    std::string split_data_output_dir = base_data_dir + "/HCRL_dataset_split";
    std::string train_split_file = split_data_output_dir + "/train.log";
    // val_split_file, test_split_file... (pretrain은 현재 train_split_file만 사용)

    std::string pretrain_output_dir = checkpoints_dir + "/pretrain_output";

    // 실행 파일 이름 (CMake에서 정의한 타겟 이름과 동일해야 함)
    std::string agg_exec = find_executable_path("aggregate_data", current_executable_path);
    std::string vocab_exec = find_executable_path("build_vocab", current_executable_path);
    std::string split_exec = find_executable_path("split_data", current_executable_path);
    std::string pretrain_exec = find_executable_path("pretrain", current_executable_path);

    int status_sum = 0;

    // 1. 데이터 집계 (aggregate_data)
    std::string cmd_aggregate = agg_exec +
                                " --source_dir " + raw_data_source_dir +
                                " --output_file " + aggregated_log_file;
    status_sum += execute_command(cmd_aggregate, "aggregate_data");
    if (status_sum !=0 && !fs::exists(aggregated_log_file)) {
         std::cerr << "데이터 집계 실패 또는 출력 파일이 생성되지 않았습니다. 중단합니다." << std::endl; return 1;
    }


    // 2. 어휘 구축 (build_vocab)
    std::string cmd_build_vocab = vocab_exec +
                                  " --data_file " + aggregated_log_file +
                                  " --vocab_path " + vocab_file +
                                  " --data_format hcrl"; // 또는 candump 등
    status_sum += execute_command(cmd_build_vocab, "build_vocab");
     if (status_sum !=0 && !fs::exists(vocab_file)) {
         std::cerr << "어휘 구축 실패 또는 출력 파일이 생성되지 않았습니다. 중단합니다." << std::endl; return 1;
    }

    // 3. 데이터 분할 (split_data)
    std::string cmd_split_data = split_exec +
                                 " --input_file " + split_data_input_file +
                                 " --output_dir " + split_data_output_dir +
                                 " --train_ratio 0.8 --val_ratio 0.1 --shuffle true --seed 42";
    status_sum += execute_command(cmd_split_data, "split_data");
    if (status_sum !=0 && !fs::exists(train_split_file)) { // Check one of the expected outputs
         std::cerr << "데이터 분할 실패 또는 출력 파일이 생성되지 않았습니다. 중단합니다." << std::endl; return 1;
    }

    // 4. 사전 훈련 (pretrain)
    std::string cmd_pretrain = pretrain_exec +
                               " --data_path " + train_split_file + // 훈련 분할 사용
                               " --vocab_path " + vocab_file +
                               " --output_dir " + pretrain_output_dir +
                               " --seq_len 126 --batch_size 32 --epochs 3 " + // 배치 크기 줄임 (예시)
                               " --num_layers 2 --hidden_size 128 --num_heads 1 " + // 모델 크기 줄임 (예시)
                               " --device_id 0";
    status_sum += execute_command(cmd_pretrain, "pretrain");


    if (status_sum == 0) {
        std::cout << "전체 훈련 파이프라인 성공적으로 완료!" << std::endl;
    } else {
        std::cout << "전체 훈련 파이프라인 중 오류 발생. 로그를 확인하십시오." << std::endl;
    }

    std::cout << "참고: 이 스크립트는 각 단계의 실행 파일이 현재 PATH에 있거나, " << std::endl;
    std::cout << "      빌드 구조에 따라 상대적으로 접근 가능하다고 가정합니다." << std::endl;
    std::cout << "      실제 환경에서는 경로를 정확히 지정해야 할 수 있습니다." << std::endl;


    return status_sum == 0 ? 0 : 1;
}
