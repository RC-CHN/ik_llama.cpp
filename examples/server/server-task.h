#pragma once
#include "common.h"
#include "llama.h"

#include <string>
#include <unordered_set>
#include <list>
// TODO: prevent including the whole server-common.h as we only use server_tokens
#include "server-common.h"

#include <vector>

using json = nlohmann::ordered_json;

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};



enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_SET_LORA,
    SERVER_TASK_TYPE_LOAD_CONTROL_VECTOR,
    SERVER_TASK_TYPE_UNLOAD_CONTROL_VECTOR,
    SERVER_TASK_TYPE_SET_CONTROL_VECTOR,
};

enum oaicompat_type {
    OAICOMPAT_TYPE_NONE,
    OAICOMPAT_TYPE_CHAT,
    OAICOMPAT_TYPE_COMPLETION,
    OAICOMPAT_TYPE_EMBEDDING,
    OAICOMPAT_TYPE_ANTHROPIC,
    OAICOMPAT_TYPE_RESP,
};


struct slot_params {
    bool stream = true;
    bool include_usage = false;
    bool cache_prompt = true; // remember the prompt to avoid reprocessing all prompt

    int32_t  n_keep = 0; // number of tokens to keep from initial prompt
    int32_t  n_discard = 0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t  n_predict = -1; // new tokens to predict

    thinking_tokens think_tokens;

    std::vector<std::string> antiprompt;

    bool timings_per_token = false;
    bool post_sampling_probs = false;
    json input_prefix;
    json input_suffix;

    // speculative decoding parameters
    struct common_params_speculative speculative;

    // OAI-compat fields
    oaicompat_type        oaicompat = OAICOMPAT_TYPE_NONE;
    std::string           oaicompat_model;
    std::string           oaicompat_cmpl_id;
    common_chat_parser_params           chat_parser_params;

    // Embeddings
    int32_t embd_normalize = 2; // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)
};

struct server_task {
    int id = -1; // to be filled by server_queue
    int id_multi = -1;

    int index = -1; // used when there are multiple prompts (batch request)

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;
    int id_slot = -1;

    // used by SERVER_TASK_TYPE_INFERENCE
    struct slot_params params;
    server_tokens tokens;

    server_task_type type;
    json data;

    bool infill = false;
    bool embedding = false;

    server_task() = default;
    server_task(server_task_type type) : type(type) {}

    int32_t n_tokens() const {
        return tokens.size();
    }

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task>& tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
        }
        return ids;
    }
};

struct result_timings {
    int32_t prompt_n = -1;
    double prompt_ms;
    double prompt_per_token_ms;
    double prompt_per_second;

    int32_t predicted_n = -1;
    double predicted_ms;
    double predicted_per_token_ms;
    double predicted_per_second;
    int32_t n_ctx = 0;
    int32_t n_past = 0;

    // Optional speculative metrics - only included when > 0
    int32_t draft_n = 0;
    int32_t draft_n_accepted = 0;
    std::vector<int32_t> draft_n_by_depth;
    std::vector<int32_t> draft_n_accepted_by_depth;

    json to_json() const;
};

struct server_task_result {
    int id = -1;
    int id_multi = -1;

    json data;

    bool stop;
    bool error;
    bool final_result = false;
    result_timings timings;
    // OAI-compat fields
    //bool                  verbose = false;
    oaicompat_type        oaicompat = OAICOMPAT_TYPE_NONE;
    std::string           oaicompat_model;
    std::string           oaicompat_cmpl_id;
    common_chat_format    oaicompat_chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_chat_msg    oaicompat_msg;
    std::vector<common_chat_msg_diff> oaicompat_msg_diffs;

    int index = 0;

    std::string content;
    std::vector<llama_token> tokens;

    bool stream;
    bool include_usage;
    std::string prompt;
    //slot_params generation_params;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;
    int32_t n_tokens_cached;
    bool has_new_line;
    std::string stopping_word;

    bool post_sampling_probs = false;
    std::vector<completion_token_output> probs_output;
    std::vector<std::string>  response_fields;

    //slot_params generation_params;

    bool                  verbose = false;

    virtual bool is_error() {
        // only used by server_task_result_error
        return false;
    }

    virtual bool is_stop() {
        // only used by server_task_result_cmpl_*
        // in stream mode, final responses are considered stop
        return true;
    }

    virtual json to_json() {
        return {};
    };

    int get_index() {
        return index;
    }

};

struct server_task_result_cmpl_partial : server_task_result {
    bool anthropic_has_reasoning = false;
    bool anthropic_thinking_block_started = false;
    bool anthropic_text_block_started = false;

    bool oai_resp_thinking_block_started = false;
    bool oai_resp_text_block_started = false;

    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;

    virtual bool is_stop() override {
        return false; // in stream mode, partial responses are not considered stop
    }

    json to_json_non_oaicompat_partial();

    json to_json_oaicompat_partial();

    json to_json_anthropic_partial();

    json to_json_oaicompat_chat_partial();

    json to_json_oaicompat_resp_partial();

    virtual json to_json() override {
        switch (oaicompat) {
        case OAICOMPAT_TYPE_NONE:
            return to_json_non_oaicompat_partial();
        case OAICOMPAT_TYPE_COMPLETION:
            return to_json_oaicompat_partial();
        case OAICOMPAT_TYPE_CHAT:
            return to_json_oaicompat_chat_partial();
        case OAICOMPAT_TYPE_ANTHROPIC:
            return to_json_anthropic_partial();
        case OAICOMPAT_TYPE_RESP:
            return to_json_oaicompat_resp_partial();
        default:
            GGML_ASSERT(false && "Invalid oaicompat_type");
        };
    }
};

struct server_task_result_cmpl_final : server_task_result {
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;

    bool anthropic_thinking_block_started = false;
    bool anthropic_text_block_started = false;

    virtual bool is_stop() override {
        return true;
    }

    json to_json_non_oaicompat_final();

    json usage_json_oaicompat();

    json to_json_oaicompat_final();

    json to_json_oaicompat_chat_final();

    json to_json_anthropic_final();

    json to_json_anthropic_stream();

    json to_json_oaicompat_chat_stream();

    json to_json_oaicompat_resp_final();

    json to_json_oaicompat_resp_stream();

    virtual json to_json() override {
        switch (oaicompat) {
        case OAICOMPAT_TYPE_NONE:
            return to_json_non_oaicompat_final();
        case OAICOMPAT_TYPE_COMPLETION:
            return to_json_oaicompat_final();
        case OAICOMPAT_TYPE_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat_final();
        case OAICOMPAT_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic_final();
        case OAICOMPAT_TYPE_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp_final();
        default:
            GGML_ASSERT(false && "Invalid oaicompat_type");
        }
    }
};

struct server_task_result_error : server_task_result {
    int index = 0;
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
    int32_t n_prompt_tokens = 0;
    int32_t n_ctx = 0;

    virtual bool is_error() override {
        return true;
    }

    virtual json to_json() override {
        json res = format_error_response(err_msg, err_type);
        return res;
    }
};

struct server_task_result_embd : server_task_result {
    int index = 0;
    std::vector<std::vector<float>> embedding;

    int32_t n_tokens;

    // OAI-compat fields
    oaicompat_type oaicompat = OAICOMPAT_TYPE_NONE;

    virtual json to_json() override {
        return oaicompat == OAICOMPAT_TYPE_EMBEDDING
            ? to_json_oaicompat()
            : to_json_non_oaicompat();
    }

    json to_json_non_oaicompat() {
        return json{
            {"index",     index},
            {"embedding", embedding},
        };
    }

    json to_json_oaicompat() {
        return json{
            {"index",            index},
            {"embedding",        embedding[0]},
            {"tokens_evaluated", n_tokens},
        };
    }
};


// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct server_prompt_checkpoint {
    llama_pos pos_min;
    llama_pos pos_max;
    llama_pos pos_min_prompt;
    llama_pos pos_max_prompt;

    int64_t n_tokens;

    std::vector<uint8_t> data;

    // Optional full sequence snapshot captured at the checkpoint position.
    // Hybrid recurrent prompt caching uses this instead of reconstructing a
    // past state from a later snapshot plus partial recurrent data.
    std::string state_file;
    size_t state_file_size = 0;

    size_t size() const {
        return data.size();
    }

    json to_json() const {
        json j;
        j["pos_min"] = pos_min;
        j["pos_max"] = pos_max;
        j["pos_min_prompt"] = pos_min_prompt;
        j["pos_max_prompt"] = pos_max_prompt;
        j["n_tokens"] = n_tokens;
        return j;
    }

    void from_json(const json & j) {
        pos_min = j.value<llama_pos>("pos_min", 0);
        pos_max = j.value<llama_pos>("pos_max", 0);
        pos_min_prompt = j.value<llama_pos>("pos_min_prompt", 0);
        pos_max_prompt = j.value<llama_pos>("pos_max_prompt", 0);
        n_tokens = j.value<int64_t>("n_tokens", 0);
    }
};


struct server_prompt {
    server_tokens tokens;
    int n_kept_prompt;
    int n_discarded_prompt;
    thinking_tokens think_tokens;

    std::vector<uint8_t> data;

    // Exactly one of data or state_file is populated. Disk-backed entries keep
    // only token/similarity metadata in memory; recurrent checkpoints spill to
    // their own sidecar as well.
    std::string state_file;
    size_t state_file_size = 0;
    std::string checkpoint_file;
    size_t checkpoint_file_size = 0;

    // Number of cached tokens and prompt position represented by the serialized
    // state. Hybrid recurrent models may deliberately save a self-consistent
    // prompt-tail state while retaining the longer token list for matching.
    int64_t state_n_tokens = -1;
    llama_pos state_pos_max_prompt = -1;

    std::list<server_prompt_checkpoint> checkpoints;

    size_t size() const;

    size_t ram_size() const;

    size_t disk_size() const {
        return state_file_size + checkpoint_file_size;
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt{
            tokens.clone(),
            n_kept_prompt,
            n_discarded_prompt,
            think_tokens,
            data,
            state_file,
            state_file_size,
            checkpoint_file,
            checkpoint_file_size,
            state_n_tokens,
            state_pos_max_prompt,
            checkpoints
        };
    }

    json to_json() const
    {
        json j;
        j["tokens"] = tokens.to_json();
        j["n_kept_prompt"] = n_kept_prompt;
        j["n_discarded_prompt"] = n_discarded_prompt;
        return j;
    }

    void from_json(const json & j) {
        tokens.from_json(j.at("tokens"));
        n_kept_prompt = j.value<llama_pos>("n_kept_prompt", 0);
        n_discarded_prompt = j.value<llama_pos>("n_discarded_prompt", 0);
    }
};

struct server_prompt_cache {
    server_prompt_cache(
            llama_context * ctx,
                   int32_t   limit_ram_mib,
                   int32_t   limit_disk_mib,
                    size_t   limit_tokens,
               std::string   disk_root = {},
                    bool     rewind_to_checkpoint_on_save = false);

    ~server_prompt_cache();

    std::list<server_prompt> states;

    bool ram_enabled = false;

    // In bytes, 0 means no limit for an enabled tier.
    size_t limit_ram_size = 0;
    size_t limit_disk_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;
    llama_context* ctx;

    // Empty when no disk spill tier is configured. Otherwise this is a private
    // per-process directory below the user-selected cache root.
    std::string disk_directory;
    uint64_t disk_file_id = 0;
    bool rewind_to_checkpoint_on_save = false;

    bool has_disk_tier() const {
        return !disk_directory.empty();
    }

    size_t size() const;

    size_t ram_size() const;

    size_t disk_size() const;

    size_t n_tokens() const;

    server_prompt* alloc(server_prompt & prompt, size_t state_size);

    bool save(server_prompt & prompt, int32_t id_slot);

    bool stage_checkpoint(server_prompt_checkpoint & checkpoint,
            const server_tokens & tokens, int32_t id_slot);

    bool spill_to_disk(server_prompt & prompt);

    bool save_to_disk(server_prompt & prompt, int32_t id_slot);

    void make_ram_room(size_t incoming_size);

    bool make_disk_room(size_t incoming_size);

    bool load(server_prompt& prompt, const server_tokens& tokens_new, llama_context* ctx, int32_t id_slot, float min_reusable_fraction);

    void update();

    void clear();
};
