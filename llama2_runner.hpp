#pragma once

#include <Arduino.h>
#include <FS.h>
#include <functional>
#include <stdint.h>

struct Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};

struct QuantizedTensor {
    int8_t* q = nullptr;
    float* s = nullptr;
};

struct TransformerWeights {
    QuantizedTensor q_tokens{};
    float* token_embedding_table = nullptr;
    float* rms_att_weight = nullptr;
    float* rms_ffn_weight = nullptr;
    QuantizedTensor* wq = nullptr;
    QuantizedTensor* wk = nullptr;
    QuantizedTensor* wv = nullptr;
    QuantizedTensor* wo = nullptr;
    QuantizedTensor* w1 = nullptr;
    QuantizedTensor* w2 = nullptr;
    QuantizedTensor* w3 = nullptr;
    float* rms_final_weight = nullptr;
    QuantizedTensor* wcls = nullptr;
    bool shared_classifier = false;
};

struct RunState {
    float* x = nullptr;
    float* xb = nullptr;
    float* xb2 = nullptr;
    float* hb = nullptr;
    float* hb2 = nullptr;
    QuantizedTensor xq{};
    QuantizedTensor hq{};
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* att = nullptr;
    float* logits = nullptr;
    float* key_cache = nullptr;
    float* value_cache = nullptr;
};

class Llama2Engine {
public:
    using TokenCallback = std::function<void(const char*)>;

    Llama2Engine() = default;
    ~Llama2Engine();

    bool init(fs::FS& fs, const char* model_path, const char* tokenizer_path);
    void generate(const char* prompt, TokenCallback callback, int steps = 100, float temperature = 0.8f);
    void free_buffers();
    bool ready() const { return initialized; }

private:
    Config config{};
    TransformerWeights weights{};
    RunState state{};

    uint8_t* model_blob = nullptr;
    size_t model_blob_size = 0;
    int group_size = 0;
    char** vocab = nullptr;
    float* vocab_scores = nullptr;
    int vocab_capacity = 0;
    int max_token_length = 0;
    bool initialized = false;

    bool allocate_run_state();
    void free_run_state();
    bool load_model(fs::FS& fs, const char* model_path);
    bool load_tokenizer(fs::FS& fs, const char* tokenizer_path);
    void transformer(int token, int pos);
    int sample(float* logits, float temperature);
    int encode(const char* text, int* tokens, int max_tokens);
    const char* decode(int token) const;
    int lookup_token(const char* piece) const;
    String sanitize_prompt(const char* prompt) const;
};
