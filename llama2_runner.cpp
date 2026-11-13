#include "llama2_runner.hpp"

#include <esp_heap_caps.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

namespace {

constexpr uint32_t kMagic = 0x616b3432;  // "ak42"
constexpr int kVersionQ8 = 2;
constexpr int kHeaderSize = 256;
constexpr int kBosToken = 1;
constexpr int kEosToken = 2;
constexpr float kRmsEps = 1e-5f;

void* alloc_bytes(size_t byte_count) {
    void* ptr = heap_caps_malloc(byte_count, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!ptr) {
        ptr = heap_caps_malloc(byte_count, MALLOC_CAP_8BIT);
    }
    return ptr;
}

float* alloc_floats(size_t count) {
    return static_cast<float*>(alloc_bytes(count * sizeof(float)));
}

int8_t* alloc_int8(size_t count) {
    return static_cast<int8_t*>(alloc_bytes(count * sizeof(int8_t)));
}

void rmsnorm(float* out, const float* x, const float* weight, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; ++i) {
        ss += x[i] * x[i];
    }
    ss = 1.0f / sqrtf(ss / size + kRmsEps);
    for (int i = 0; i < size; ++i) {
        out[i] = weight[i] * (ss * x[i]);
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    if (sum == 0.0f) {
        return;
    }

    for (int i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}

void accum(float* x, const float* y, int size) {
    for (int i = 0; i < size; ++i) {
        x[i] += y[i];
    }
}

float silu(float x) {
    return x / (1.0f + expf(-x));
}

bool token_is_byte_piece(const char* text) {
    return text &&
           strlen(text) == 6 &&
           text[0] == '<' &&
           text[1] == '0' &&
           text[2] == 'x' &&
           text[5] == '>';
}

char hex_digit(char c) {
    if (c >= '0' && c <= '9') return static_cast<char>(c - '0');
    if (c >= 'A' && c <= 'F') return static_cast<char>(c - 'A' + 10);
    if (c >= 'a' && c <= 'f') return static_cast<char>(c - 'a' + 10);
    return 0;
}

QuantizedTensor init_quantized_tensor(uint8_t*& ptr, int size_each, int group_size) {
    QuantizedTensor qt;
    qt.q = reinterpret_cast<int8_t*>(ptr);
    ptr += size_each * sizeof(int8_t);
    qt.s = reinterpret_cast<float*>(ptr);
    ptr += (size_each / group_size) * sizeof(float);
    return qt;
}

QuantizedTensor* init_quantized_tensor_array(uint8_t*& ptr, int count, int size_each, int group_size) {
    QuantizedTensor* items = static_cast<QuantizedTensor*>(alloc_bytes(static_cast<size_t>(count) * sizeof(QuantizedTensor)));
    if (!items) {
        return nullptr;
    }

    for (int i = 0; i < count; ++i) {
        items[i] = init_quantized_tensor(ptr, size_each, group_size);
    }
    return items;
}

void dequantize(const QuantizedTensor& qx, float* out, int n, int group_size) {
    for (int i = 0; i < n; ++i) {
        out[i] = static_cast<float>(qx.q[i]) * qx.s[i / group_size];
    }
}

void quantize(QuantizedTensor& qx, const float* x, int n, int group_size) {
    const int num_groups = n / group_size;
    const float qmax = 127.0f;

    for (int group = 0; group < num_groups; ++group) {
        float wmax = 0.0f;
        const int offset = group * group_size;
        for (int i = 0; i < group_size; ++i) {
            const float val = fabsf(x[offset + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        float scale = wmax / qmax;
        if (scale == 0.0f) {
            scale = 1.0f;
        }
        qx.s[group] = scale;

        for (int i = 0; i < group_size; ++i) {
            const float quant_value = x[offset + i] / scale;
            qx.q[offset + i] = static_cast<int8_t>(roundf(quant_value));
        }
    }
}

void matmul(float* out, const QuantizedTensor& x, const QuantizedTensor& w, int n, int d, int group_size) {
    for (int i = 0; i < d; ++i) {
        float val = 0.0f;
        int32_t ival = 0;
        const int base = i * n;

        for (int j = 0; j <= n - group_size; j += group_size) {
            for (int k = 0; k < group_size; ++k) {
                ival += static_cast<int32_t>(x.q[j + k]) * static_cast<int32_t>(w.q[base + j + k]);
            }
            val += static_cast<float>(ival) * w.s[(base + j) / group_size] * x.s[j / group_size];
            ival = 0;
        }

        out[i] = val;
    }
}

}  // namespace

Llama2Engine::~Llama2Engine() {
    free_buffers();
}

bool Llama2Engine::allocate_run_state() {
    const int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    const int xq_groups = config.dim / group_size;
    const int hq_groups = config.hidden_dim / group_size;

    Serial.printf("[SYSTEM] Alloc run state: dim=%d hidden=%d heads=%d kv_heads=%d seq=%d kv_dim=%d group=%d\n",
                  config.dim, config.hidden_dim, config.n_heads, config.n_kv_heads, config.seq_len, kv_dim, group_size);

    state.x = alloc_floats(config.dim);
    state.xb = alloc_floats(config.dim);
    state.xb2 = alloc_floats(config.dim);
    state.hb = alloc_floats(config.hidden_dim);
    state.hb2 = alloc_floats(config.hidden_dim);
    state.xq.q = alloc_int8(config.dim);
    state.xq.s = alloc_floats(xq_groups);
    state.hq.q = alloc_int8(config.hidden_dim);
    state.hq.s = alloc_floats(hq_groups);
    state.q = alloc_floats(config.dim);
    state.k = alloc_floats(kv_dim);
    state.v = alloc_floats(kv_dim);
    state.att = alloc_floats(static_cast<size_t>(config.n_heads) * config.seq_len);
    state.logits = alloc_floats(config.vocab_size);
    state.key_cache = alloc_floats(static_cast<size_t>(config.n_layers) * config.seq_len * kv_dim);
    state.value_cache = alloc_floats(static_cast<size_t>(config.n_layers) * config.seq_len * kv_dim);

    if (!state.x || !state.xb || !state.xb2 || !state.hb || !state.hb2 ||
        !state.xq.q || !state.xq.s || !state.hq.q || !state.hq.s ||
        !state.q || !state.k || !state.v || !state.att || !state.logits ||
        !state.key_cache || !state.value_cache) {
        Serial.printf("[ERROR] Failed to allocate run state. Free heap=%u free PSRAM=%u\n",
                      static_cast<unsigned>(ESP.getFreeHeap()),
                      static_cast<unsigned>(ESP.getFreePsram()));
        return false;
    }

    memset(state.key_cache, 0, static_cast<size_t>(config.n_layers) * config.seq_len * kv_dim * sizeof(float));
    memset(state.value_cache, 0, static_cast<size_t>(config.n_layers) * config.seq_len * kv_dim * sizeof(float));
    return true;
}

void Llama2Engine::free_run_state() {
    free(state.x);
    free(state.xb);
    free(state.xb2);
    free(state.hb);
    free(state.hb2);
    free(state.xq.q);
    free(state.xq.s);
    free(state.hq.q);
    free(state.hq.s);
    free(state.q);
    free(state.k);
    free(state.v);
    free(state.att);
    free(state.logits);
    free(state.key_cache);
    free(state.value_cache);
    state = {};
}

bool Llama2Engine::load_model(fs::FS& fs, const char* model_path) {
    File model_file = fs.open(model_path, FILE_READ);
    if (!model_file && model_path[0] == '/') {
        model_file = fs.open(model_path + 1, FILE_READ);
    }
    if (!model_file) {
        Serial.printf("[ERROR] Failed to open model file: %s\n", model_path);
        return false;
    }

    uint32_t magic = 0;
    int version = 0;
    if (model_file.read(reinterpret_cast<uint8_t*>(&magic), sizeof(uint32_t)) != sizeof(uint32_t) ||
        model_file.read(reinterpret_cast<uint8_t*>(&version), sizeof(int)) != sizeof(int)) {
        Serial.println("[ERROR] Failed to read quantized model header.");
        model_file.close();
        return false;
    }

    if (magic != kMagic || version != kVersionQ8) {
        Serial.printf("[ERROR] Unsupported model format. magic=0x%08lx version=%d\n",
                      static_cast<unsigned long>(magic), version);
        model_file.close();
        return false;
    }

    if (model_file.read(reinterpret_cast<uint8_t*>(&config), sizeof(Config)) != sizeof(Config)) {
        Serial.println("[ERROR] Failed to read model config.");
        model_file.close();
        return false;
    }

    uint8_t shared_classifier = 0;
    if (model_file.read(&shared_classifier, sizeof(uint8_t)) != sizeof(uint8_t) ||
        model_file.read(reinterpret_cast<uint8_t*>(&group_size), sizeof(int)) != sizeof(int)) {
        Serial.println("[ERROR] Failed to read model flags.");
        model_file.close();
        return false;
    }
    weights.shared_classifier = shared_classifier != 0;

    Serial.printf("[SYSTEM] Model header: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq=%d group=%d\n",
                  config.dim, config.hidden_dim, config.n_layers, config.n_heads,
                  config.n_kv_heads, config.vocab_size, config.seq_len, group_size);

    if (group_size <= 0 || config.dim <= 0 || config.hidden_dim <= 0 || config.n_layers <= 0 ||
        config.n_heads <= 0 || config.n_kv_heads <= 0 || config.vocab_size <= 0 || config.seq_len <= 0) {
        Serial.println("[ERROR] Invalid quantized model header values.");
        model_file.close();
        return false;
    }

    if ((config.dim % group_size) != 0 || (config.hidden_dim % group_size) != 0) {
        Serial.println("[ERROR] Model dimensions are not divisible by quantization group size.");
        model_file.close();
        return false;
    }

    model_blob_size = model_file.size();
    model_blob = static_cast<uint8_t*>(alloc_bytes(model_blob_size));
    if (!model_blob) {
        Serial.println("[ERROR] Out of memory loading quantized model blob.");
        model_file.close();
        return false;
    }

    model_file.seek(0);
    if (model_file.read(model_blob, model_blob_size) != static_cast<int>(model_blob_size)) {
        Serial.println("[ERROR] Failed to read full quantized model.");
        model_file.close();
        return false;
    }
    model_file.close();

    const int head_size = config.dim / config.n_heads;
    const int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    uint8_t* ptr = model_blob + kHeaderSize;

    float* fptr = reinterpret_cast<float*>(ptr);
    weights.rms_att_weight = fptr;
    fptr += config.n_layers * config.dim;
    weights.rms_ffn_weight = fptr;
    fptr += config.n_layers * config.dim;
    weights.rms_final_weight = fptr;
    fptr += config.dim;

    ptr = reinterpret_cast<uint8_t*>(fptr);
    weights.q_tokens = init_quantized_tensor(ptr, config.vocab_size * config.dim, group_size);
    weights.token_embedding_table = alloc_floats(static_cast<size_t>(config.vocab_size) * config.dim);
    if (!weights.token_embedding_table) {
        Serial.println("[ERROR] Failed to allocate dequantized token embeddings.");
        return false;
    }
    dequantize(weights.q_tokens, weights.token_embedding_table, config.vocab_size * config.dim, group_size);

    weights.wq = init_quantized_tensor_array(ptr, config.n_layers, config.dim * (config.n_heads * head_size), group_size);
    weights.wk = init_quantized_tensor_array(ptr, config.n_layers, config.dim * kv_dim, group_size);
    weights.wv = init_quantized_tensor_array(ptr, config.n_layers, config.dim * kv_dim, group_size);
    weights.wo = init_quantized_tensor_array(ptr, config.n_layers, config.dim * (config.n_heads * head_size), group_size);
    weights.w1 = init_quantized_tensor_array(ptr, config.n_layers, config.dim * config.hidden_dim, group_size);
    weights.w2 = init_quantized_tensor_array(ptr, config.n_layers, config.hidden_dim * config.dim, group_size);
    weights.w3 = init_quantized_tensor_array(ptr, config.n_layers, config.dim * config.hidden_dim, group_size);

    if (!weights.wq || !weights.wk || !weights.wv || !weights.wo || !weights.w1 || !weights.w2 || !weights.w3) {
        Serial.println("[ERROR] Failed to map quantized weights.");
        return false;
    }

    if (weights.shared_classifier) {
        weights.wcls = &weights.q_tokens;
    } else {
        weights.wcls = init_quantized_tensor_array(ptr, 1, config.dim * config.vocab_size, group_size);
        if (!weights.wcls) {
            Serial.println("[ERROR] Failed to map classifier weights.");
            return false;
        }
    }

    return true;
}

bool Llama2Engine::load_tokenizer(fs::FS& fs, const char* tokenizer_path) {
    File tok_file = fs.open(tokenizer_path, FILE_READ);
    if (!tok_file && tokenizer_path[0] == '/') {
        tok_file = fs.open(tokenizer_path + 1, FILE_READ);
    }
    if (!tok_file) {
        Serial.printf("[ERROR] Failed to open tokenizer file: %s\n", tokenizer_path);
        return false;
    }

    if (tok_file.read(reinterpret_cast<uint8_t*>(&max_token_length), sizeof(int)) != sizeof(int)) {
        Serial.println("[ERROR] Failed to read tokenizer header.");
        tok_file.close();
        return false;
    }

    vocab_capacity = config.vocab_size;
    vocab = static_cast<char**>(alloc_bytes(static_cast<size_t>(vocab_capacity) * sizeof(char*)));
    vocab_scores = alloc_floats(vocab_capacity);
    if (!vocab || !vocab_scores) {
        Serial.println("[ERROR] Failed to allocate tokenizer tables.");
        tok_file.close();
        return false;
    }
    memset(vocab, 0, static_cast<size_t>(vocab_capacity) * sizeof(char*));

    for (int i = 0; i < vocab_capacity; ++i) {
        if (tok_file.read(reinterpret_cast<uint8_t*>(&vocab_scores[i]), sizeof(float)) != sizeof(float)) {
            Serial.println("[ERROR] Failed to read tokenizer scores.");
            tok_file.close();
            return false;
        }

        int len = 0;
        if (tok_file.read(reinterpret_cast<uint8_t*>(&len), sizeof(int)) != sizeof(int)) {
            Serial.println("[ERROR] Failed to read tokenizer token length.");
            tok_file.close();
            return false;
        }

        if (len < 0 || len > 1024) {
            Serial.printf("[ERROR] Suspicious tokenizer token length at id=%d: %d\n", i, len);
            tok_file.close();
            return false;
        }

        vocab[i] = static_cast<char*>(malloc(static_cast<size_t>(len) + 1));
        if (!vocab[i]) {
            Serial.printf("[ERROR] Failed to allocate tokenizer token at id=%d len=%d. Free heap=%u free PSRAM=%u\n",
                          i, len,
                          static_cast<unsigned>(ESP.getFreeHeap()),
                          static_cast<unsigned>(ESP.getFreePsram()));
            tok_file.close();
            return false;
        }

        if (tok_file.read(reinterpret_cast<uint8_t*>(vocab[i]), len) != len) {
            Serial.println("[ERROR] Failed to read tokenizer token bytes.");
            tok_file.close();
            return false;
        }
        vocab[i][len] = '\0';
    }

    tok_file.close();
    return true;
}

bool Llama2Engine::init(fs::FS& fs, const char* model_path, const char* tokenizer_path) {
    free_buffers();

    if (!load_model(fs, model_path)) {
        free_buffers();
        return false;
    }

    if (!load_tokenizer(fs, tokenizer_path)) {
        free_buffers();
        return false;
    }

    if (!allocate_run_state()) {
        free_buffers();
        return false;
    }

    initialized = true;
    Serial.printf("[SYSTEM] model ready: dim=%d layers=%d vocab=%d seq=%d max_gen=%d\n",
                  config.dim, config.n_layers, config.vocab_size, config.seq_len, config.seq_len - 1);
    return true;
}

void Llama2Engine::transformer(int token, int pos) {
    const int dim = config.dim;
    const int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    const int kv_mul = config.n_heads / config.n_kv_heads;
    const int head_size = dim / config.n_heads;

    memcpy(state.x, weights.token_embedding_table + static_cast<size_t>(token) * dim, dim * sizeof(float));

    for (int layer = 0; layer < config.n_layers; ++layer) {
        rmsnorm(state.xb, state.x, weights.rms_att_weight + static_cast<size_t>(layer) * dim, dim);
        quantize(state.xq, state.xb, dim, group_size);

        matmul(state.q, state.xq, weights.wq[layer], dim, dim, group_size);
        matmul(state.k, state.xq, weights.wk[layer], dim, kv_dim, group_size);
        matmul(state.v, state.xq, weights.wv[layer], dim, kv_dim, group_size);

        for (int i = 0; i < dim; i += 2) {
            const int head_dim = i % head_size;
            const float freq = 1.0f / powf(10000.0f, head_dim / static_cast<float>(head_size));
            const float angle = pos * freq;
            const float fcr = cosf(angle);
            const float fci = sinf(angle);
            const int rotn = i < kv_dim ? 2 : 1;

            for (int vec_idx = 0; vec_idx < rotn; ++vec_idx) {
                float* vec = vec_idx == 0 ? state.q : state.k;
                const float v0 = vec[i];
                const float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        const int layer_offset = layer * config.seq_len * kv_dim;
        float* key_cache_row = state.key_cache + layer_offset + pos * kv_dim;
        float* value_cache_row = state.value_cache + layer_offset + pos * kv_dim;
        memcpy(key_cache_row, state.k, kv_dim * sizeof(float));
        memcpy(value_cache_row, state.v, kv_dim * sizeof(float));

        for (int head = 0; head < config.n_heads; ++head) {
            float* q = state.q + head * head_size;
            float* att = state.att + static_cast<size_t>(head) * config.seq_len;

            for (int t = 0; t <= pos; ++t) {
                float* k = state.key_cache + layer_offset + t * kv_dim + (head / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; ++i) {
                    score += q[i] * k[i];
                }
                att[t] = score / sqrtf(static_cast<float>(head_size));
            }

            softmax(att, pos + 1);

            float* xb = state.xb + head * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; ++t) {
                float* v = state.value_cache + layer_offset + t * kv_dim + (head / kv_mul) * head_size;
                const float a = att[t];
                for (int i = 0; i < head_size; ++i) {
                    xb[i] += a * v[i];
                }
            }
        }

        quantize(state.xq, state.xb, dim, group_size);
        matmul(state.xb2, state.xq, weights.wo[layer], dim, dim, group_size);
        accum(state.x, state.xb2, dim);

        rmsnorm(state.xb, state.x, weights.rms_ffn_weight + static_cast<size_t>(layer) * dim, dim);
        quantize(state.xq, state.xb, dim, group_size);
        matmul(state.hb, state.xq, weights.w1[layer], dim, config.hidden_dim, group_size);
        matmul(state.hb2, state.xq, weights.w3[layer], dim, config.hidden_dim, group_size);

        for (int i = 0; i < config.hidden_dim; ++i) {
            state.hb[i] = silu(state.hb[i]) * state.hb2[i];
        }

        quantize(state.hq, state.hb, config.hidden_dim, group_size);
        matmul(state.xb, state.hq, weights.w2[layer], config.hidden_dim, dim, group_size);
        accum(state.x, state.xb, dim);
    }

    rmsnorm(state.x, state.x, weights.rms_final_weight, dim);
    quantize(state.xq, state.x, dim, group_size);
    matmul(state.logits, state.xq, *weights.wcls, dim, config.vocab_size, group_size);
}

int Llama2Engine::sample(float* logits, float temperature) {
    if (temperature <= 0.0f) {
        int max_i = 0;
        float max_p = logits[0];
        for (int i = 1; i < config.vocab_size; ++i) {
            if (logits[i] > max_p) {
                max_p = logits[i];
                max_i = i;
            }
        }
        return max_i;
    }

    for (int i = 0; i < config.vocab_size; ++i) {
        logits[i] /= temperature;
    }
    softmax(logits, config.vocab_size);

    const float r = static_cast<float>(random(1000000)) / 1000000.0f;
    float cdf = 0.0f;
    for (int i = 0; i < config.vocab_size; ++i) {
        cdf += logits[i];
        if (r < cdf) {
            return i;
        }
    }
    return config.vocab_size - 1;
}

int Llama2Engine::lookup_token(const char* piece) const {
    if (!piece || !vocab) {
        return -1;
    }

    for (int i = 0; i < vocab_capacity; ++i) {
        if (vocab[i] && strcmp(vocab[i], piece) == 0) {
            return i;
        }
    }
    return -1;
}

String Llama2Engine::sanitize_prompt(const char* prompt) const {
    String text = prompt ? String(prompt) : String();
    text.trim();
    if (text.startsWith("Story:")) {
        text = text.substring(6);
        text.trim();
        text = "Once upon a time, " + text;
    } else if (text.length() > 0) {
        if (!text.endsWith(".") && !text.endsWith("!") && !text.endsWith("?")) {
            text += ".";
        }
        text += " ";
    }
    return text;
}

int Llama2Engine::encode(const char* text, int* tokens, int max_tokens) {
    if (!tokens || max_tokens <= 0) {
        return 0;
    }

    int n_tokens = 0;
    tokens[n_tokens++] = kBosToken;

    const String prompt = sanitize_prompt(text);
    if (prompt.isEmpty()) {
        return n_tokens;
    }

    const int byte_count = prompt.length();
    if (byte_count <= 0) {
        return n_tokens;
    }

    int pieces[512];
    int piece_count = 0;

    for (int i = 0; i < byte_count && piece_count < 512; ++i) {
        char ch = prompt[i];
        char one_char[2] = { ch, '\0' };
        int id = lookup_token(one_char);
        if (id >= 0) {
            pieces[piece_count++] = id;
            continue;
        }

        char byte_piece[7];
        snprintf(byte_piece, sizeof(byte_piece), "<0x%02X>", static_cast<unsigned char>(ch));
        id = lookup_token(byte_piece);
        if (id < 0) {
            continue;
        }
        pieces[piece_count++] = id;
    }

    bool merged = true;
    while (merged) {
        merged = false;
        float best_score = -1e10f;
        int best_index = -1;
        int best_token = -1;

        for (int i = 0; i < piece_count - 1; ++i) {
            const char* left = vocab[pieces[i]];
            const char* right = vocab[pieces[i + 1]];
            if (!left || !right) {
                continue;
            }

            char merged_piece[64];
            snprintf(merged_piece, sizeof(merged_piece), "%s%s", left, right);
            const int token = lookup_token(merged_piece);
            if (token >= 0 && vocab_scores[token] > best_score) {
                best_score = vocab_scores[token];
                best_index = i;
                best_token = token;
            }
        }

        if (best_index >= 0) {
            pieces[best_index] = best_token;
            for (int j = best_index + 1; j < piece_count - 1; ++j) {
                pieces[j] = pieces[j + 1];
            }
            --piece_count;
            merged = true;
        }
    }

    for (int i = 0; i < piece_count && n_tokens < max_tokens; ++i) {
        tokens[n_tokens++] = pieces[i];
    }

    return n_tokens;
}

const char* Llama2Engine::decode(int token) const {
    static char byte_piece[2];

    if (!vocab || token < 0 || token >= vocab_capacity || !vocab[token]) {
        return "";
    }

    if (token_is_byte_piece(vocab[token])) {
        const char high = hex_digit(vocab[token][3]);
        const char low = hex_digit(vocab[token][4]);
        byte_piece[0] = static_cast<char>((high << 4) | low);
        byte_piece[1] = '\0';
        return byte_piece;
    }

    if (token == kBosToken || token == kEosToken) {
        return "";
    }

    return vocab[token];
}

void Llama2Engine::generate(const char* prompt, TokenCallback callback, int steps, float temperature) {
    if (!initialized || !callback) {
        return;
    }

    int prompt_tokens[512];
    int num_prompt_tokens = encode(prompt, prompt_tokens, 512);
    if (num_prompt_tokens <= 0) {
        return;
    }

    const int max_steps = steps > 0 ? min(steps, config.seq_len - 1) : config.seq_len - 1;
    int token = prompt_tokens[0];
    int pos = 0;

    while (pos < max_steps) {
        transformer(token, pos);

        int next_token = 0;
        if (pos < num_prompt_tokens - 1) {
            next_token = prompt_tokens[pos + 1];
        } else {
            next_token = sample(state.logits, temperature);
        }

        if (next_token == kBosToken || next_token == kEosToken) {
            break;
        }

        const char* piece = decode(next_token);
        if (piece[0] != '\0') {
            callback(piece);
        }

        token = next_token;
        ++pos;
    }
}

void Llama2Engine::free_buffers() {
    free_run_state();

    free(weights.token_embedding_table);
    weights.token_embedding_table = nullptr;
    free(weights.wq);
    free(weights.wk);
    free(weights.wv);
    free(weights.wo);
    free(weights.w1);
    free(weights.w2);
    free(weights.w3);
    if (weights.wcls && !weights.shared_classifier) {
        free(weights.wcls);
    }
    weights = {};

    free(model_blob);
    model_blob = nullptr;
    model_blob_size = 0;
    group_size = 0;

    if (vocab) {
        for (int i = 0; i < vocab_capacity; ++i) {
            free(vocab[i]);
        }
    }
    free(vocab);
    free(vocab_scores);
    vocab = nullptr;
    vocab_scores = nullptr;
    vocab_capacity = 0;
    max_token_length = 0;
    initialized = false;
}
