#include "core/tokenizer.hpp"
#include "core/dataset.hpp"
#include "models/teacher.hpp" // Includes HIP/rocBLAS checks and GpuTensor
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>    // For timing epochs
#include <iomanip>   // For std::fixed, std::setprecision
#include <limits>    // For std::numeric_limits
#include <filesystem> // For creating output directory

// Simple command line argument parser (could be refactored into a common utility)
#include <unordered_map>
namespace fs = std::filesystem;

std::unordered_map<std::string, std::string> parse_args_pretrain(int argc, char* argv[]) {
    std::unordered_map<std::string, std::string> args_map;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0) {
            std::string key = arg.substr(2);
            if (i + 1 < argc) {
                std::string value = argv[++i];
                if (value.rfind("--", 0) == 0) { // Next arg is another key
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

// Helper to convert MLMBatch (vector<int>) to GpuTensor (vector<float> then to GPU)
// THIS IS A MAJOR HACK due to GpuTensor being float-only.
// In a real system, GpuTensor would be templated or have int specializations.
void batch_to_gpu_tensor_float_hack(const std::vector<int>& cpu_int_vector, GpuTensor& gpu_tensor, const std::string& name_for_error) {
    if (cpu_int_vector.size() != gpu_tensor.num_elements()) {
        // This can happen if batch size changes and tensor wasn't re-allocated properly for the new actual batch size
        // For example, last batch might be smaller.
        // The calling code needs to ensure gpu_tensor is already sized for cpu_int_vector.
         throw std::runtime_error("Mismatch in size for " + name_for_error + ": CPU vector " +
                                 std::to_string(cpu_int_vector.size()) + ", GPU tensor " +
                                 std::to_string(gpu_tensor.num_elements()) +
                                 ". Ensure GPU tensor is resized for partial batches if any.");
    }
    if (gpu_tensor.num_elements() == 0) return; // Nothing to copy

    std::vector<float> temp_float_vector(cpu_int_vector.size());
    for (size_t i = 0; i < cpu_int_vector.size(); ++i) {
        temp_float_vector[i] = static_cast<float>(cpu_int_vector[i]);
    }
    gpu_tensor.to_gpu(temp_float_vector);
}


void train_one_epoch(CANBertForMaskedLM& model, MLMDataset& dataset, int epoch_num, int total_epochs,
                     int config_batch_size, int seq_len, const BertConfig& model_config, const std::string& output_dir) {
    std::cout << "--- Epoch " << epoch_num + 1 << "/" << total_epochs << " ---" << std::endl;

    size_t num_data_items = dataset.size();
    if (num_data_items == 0) {
        std::cout << "데이터셋이 비어있어 에폭을 건너뜁니다." << std::endl;
        return;
    }
    // Effective number of batches, handling partial last batch by iterating up to num_data_items
    // size_t num_batches = (num_data_items + config_batch_size - 1) / config_batch_size;

    double total_epoch_loss_sum = 0.0; // Sum of sum_losses from each batch
    long long total_valid_labels_epoch = 0; // Total number of non-ignored tokens in the epoch

    // Prepare GPU tensors. These will be resized if the last batch is smaller.
    GpuTensor gpu_input_ids("gpu_input_ids");
    GpuTensor gpu_attention_mask("gpu_attention_mask");
    GpuTensor gpu_labels("gpu_labels");
    GpuTensor gpu_prediction_scores("gpu_pred_scores");
    GpuTensor gpu_batch_loss_sum({1}, "gpu_batch_loss_sum"); // To hold the sum of losses for the batch

    auto epoch_start_time = std::chrono::high_resolution_clock::now();
    size_t batches_processed_count = 0;

    for (size_t current_item_idx = 0; current_item_idx < num_data_items; ) {
        auto batch_start_time = std::chrono::high_resolution_clock::now();

        size_t current_batch_actual_size = std::min((size_t)config_batch_size, num_data_items - current_item_idx);
        if (current_batch_actual_size == 0) break;

        std::vector<int> batch_input_ids_cpu; batch_input_ids_cpu.reserve(current_batch_actual_size * seq_len);
        std::vector<int> batch_attention_mask_cpu; batch_attention_mask_cpu.reserve(current_batch_actual_size * seq_len);
        std::vector<int> batch_labels_cpu; batch_labels_cpu.reserve(current_batch_actual_size * seq_len);
        long long num_valid_labels_in_batch = 0;

        for(size_t j=0; j < current_batch_actual_size; ++j) {
            MLMBatch batch_item = dataset.get_item(current_item_idx + j);
            batch_input_ids_cpu.insert(batch_input_ids_cpu.end(), batch_item.input_ids.begin(), batch_item.input_ids.end());
            batch_attention_mask_cpu.insert(batch_attention_mask_cpu.end(), batch_item.attention_mask.begin(), batch_item.attention_mask.end());
            batch_labels_cpu.insert(batch_labels_cpu.end(), batch_item.labels.begin(), batch_item.labels.end());
            for(int lbl : batch_item.labels) if(lbl != -100) num_valid_labels_in_batch++;
        }

        // Resize GPU tensors if current batch size differs from their last configured size
        if (gpu_input_ids.dims().empty() || gpu_input_ids.dims()[0] != (int)current_batch_actual_size) {
            gpu_input_ids.reshape({(int)current_batch_actual_size, seq_len}); gpu_input_ids.allocate();
            gpu_attention_mask.reshape({(int)current_batch_actual_size, seq_len}); gpu_attention_mask.allocate();
            gpu_labels.reshape({(int)current_batch_actual_size, seq_len}); gpu_labels.allocate();
            gpu_prediction_scores.reshape({(int)current_batch_actual_size, seq_len, model_config.vocab_size}); gpu_prediction_scores.allocate();
        }

        try {
            batch_to_gpu_tensor_float_hack(batch_input_ids_cpu, gpu_input_ids, "input_ids");
            batch_to_gpu_tensor_float_hack(batch_attention_mask_cpu, gpu_attention_mask, "attention_mask");
            batch_to_gpu_tensor_float_hack(batch_labels_cpu, gpu_labels, "labels");
        } catch (const std::exception& e) {
            std::cerr << "오류: 배치 " << batches_processed_count << " GPU로 데이터 복사 중 오류: " << e.what() << std::endl;
            current_item_idx += current_batch_actual_size; // Move to next batch items
            continue;
        }

        model.forward(gpu_input_ids, gpu_attention_mask, gpu_prediction_scores);
        model.compute_loss(gpu_prediction_scores, gpu_labels, gpu_batch_loss_sum);

        std::vector<float> batch_loss_cpu_vec = gpu_batch_loss_sum.to_cpu();
        float current_batch_sum_loss_val = batch_loss_cpu_vec[0];

        total_epoch_loss_sum += current_batch_sum_loss_val;
        total_valid_labels_epoch += num_valid_labels_in_batch;
        batches_processed_count++;

        auto batch_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> batch_duration = batch_end_time - batch_start_time;
        float current_batch_avg_loss = (num_valid_labels_in_batch > 0) ? (current_batch_sum_loss_val / num_valid_labels_in_batch) : 0.0f;

        if ((batches_processed_count) % 10 == 0 || (current_item_idx + current_batch_actual_size >= num_data_items) ) {
            size_t estimated_total_batches = (num_data_items + config_batch_size - 1) / config_batch_size;
            std::cout << "에폭 [" << epoch_num + 1 << "/" << total_epochs << "], 배치 [" << batches_processed_count << "/" << estimated_total_batches << "] | "
                      << "평균 배치 손실: " << std::fixed << std::setprecision(4) << current_batch_avg_loss << " | "
                      << "배치 처리 시간: " << std::fixed << std::setprecision(2) << batch_duration.count() << "s"
                      << std::endl;
        }
        current_item_idx += current_batch_actual_size;
    }
    auto epoch_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> epoch_duration = epoch_end_time - epoch_start_time;

    double average_epoch_loss = (total_valid_labels_epoch > 0 && batches_processed_count > 0) ? (total_epoch_loss_sum / total_valid_labels_epoch) : 0.0;
    std::cout << "--- 에폭 " << epoch_num + 1 << " 완료 ---" << std::endl;
    std::cout << "평균 에폭 손실: " << std::fixed << std::setprecision(4) << average_epoch_loss << std::endl;
    std::cout << "총 에폭 시간: " << std::fixed << std::setprecision(2) << epoch_duration.count() << "s" << std::endl;
}


int main(int argc, char* argv[]) {
    auto args = parse_args_pretrain(argc, argv);

    std::string data_path_str = "data/HCRL_dataset/train_aggregated.log";
    std::string vocab_path_str = "checkpoints/vocab.json";
    std::string output_dir_str = "checkpoints/pretrain_run";
    std::string data_format = "hcrl";
    int seq_len = 126;
    int batch_size = 64;
    int epochs = 5;
    int num_layers = 4;
    int hidden_size = 256;
    int num_heads = 1;
    int intermediate_size_multiplier = 2;
    int device_id = 0;

    if (args.count("data_path")) data_path_str = args["data_path"];
    if (args.count("vocab_path")) vocab_path_str = args["vocab_path"];
    if (args.count("output_dir")) output_dir_str = args["output_dir"];
    if (args.count("data_format")) data_format = args["data_format"];
    if (args.count("seq_len")) seq_len = std::stoi(args["seq_len"]);
    if (args.count("batch_size")) batch_size = std::stoi(args["batch_size"]);
    if (args.count("epochs")) epochs = std::stoi(args["epochs"]);
    if (args.count("num_layers")) num_layers = std::stoi(args["num_layers"]);
    if (args.count("hidden_size")) hidden_size = std::stoi(args["hidden_size"]);
    if (args.count("num_heads")) num_heads = std::stoi(args["num_heads"]);
    if (args.count("intermediate_size_multiplier")) intermediate_size_multiplier = std::stoi(args["intermediate_size_multiplier"]);
    if (args.count("device_id")) device_id = std::stoi(args["device_id"]);

    std::cout << "===== 훈련 설정 =====" << std::endl;
    std::cout << "데이터 경로: " << data_path_str << " (형식: " << data_format << ")" << std::endl;
    std::cout << "어휘 경로: " << vocab_path_str << std::endl;
    std::cout << "출력 디렉토리: " << output_dir_str << std::endl;
    std::cout << "시퀀스 길이: " << seq_len << std::endl;
    std::cout << "배치 크기: " << batch_size << std::endl;
    std::cout << "에폭 수: " << epochs << std::endl;
    std::cout << "모델 레이어: " << num_layers << std::endl;
    std::cout << "히든 사이즈: " << hidden_size << std::endl;
    std::cout << "어텐션 헤드 수: " << num_heads << std::endl;
    std::cout << "중간 FF 크기 배수: " << intermediate_size_multiplier << std::endl;
    std::cout << "GPU 장치 ID: " << device_id << std::endl;
    std::cout << "=====================" << std::endl;


    try {
        fs::create_directories(output_dir_str);

        HIP_CHECK(hipSetDevice(device_id));
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, device_id));
        std::cout << "사용 중인 GPU: " << props.name << " (ID: " << device_id << ")" << std::endl;

        CANTokenizer tokenizer;
        std::cout << "토크나이저 로딩 중: " << vocab_path_str << std::endl;
        tokenizer.load_vocab(vocab_path_str);
        std::cout << "토크나이저 로드 완료. 어휘 크기: " << tokenizer.get_vocab_size() << std::endl;

        BertConfig model_config;
        model_config.vocab_size = tokenizer.get_vocab_size();
        model_config.max_position_embeddings = seq_len;
        model_config.num_hidden_layers = num_layers;
        model_config.hidden_size = hidden_size;
        model_config.num_attention_heads = num_heads;
        model_config.intermediate_size = hidden_size * intermediate_size_multiplier;

        std::cout << "모델 초기화 중..." << std::endl;
        CANBertForMaskedLM model(model_config);
        model.initialize_parameters();
        std::cout << "모델 초기화 완료." << std::endl;

        std::cout << "데이터셋 로딩 중: " << data_path_str << std::endl;
        MLMDataset dataset(data_path_str, tokenizer, seq_len, data_format);
        if (dataset.size() == 0) {
            std::cerr << "오류: 데이터셋이 비어있습니다. 훈련을 진행할 수 없습니다." << std::endl;
            return 1;
        }
        std::cout << "데이터셋 로드 완료. 총 아이템 수: " << dataset.size() << std::endl;

        std::cout << "훈련 시작..." << std::endl;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            train_one_epoch(model, dataset, epoch, epochs, batch_size, seq_len, model_config, output_dir_str);
        }

    } catch (const std::exception& e) {
        std::cerr << "훈련 중 심각한 오류 발생: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "훈련 완료." << std::endl;
    return 0;
}

```
