#include "server-task.h"
#include "server-chat.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <limits>
#include <system_error>

#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

json result_timings::to_json() const {
    json base = {
        {"prompt_n",               prompt_n},
        {"prompt_ms",              prompt_ms},
        {"prompt_per_token_ms",    prompt_per_token_ms},
        {"prompt_per_second",      prompt_per_second},

        {"predicted_n",            predicted_n},
        {"predicted_ms",           predicted_ms},
        {"predicted_per_token_ms", predicted_per_token_ms},
        {"predicted_per_second",   predicted_per_second},

        {"n_ctx",           n_ctx},
        {"n_past",           n_past},
    };

    if (draft_n > 0) {
        base["draft_n"] = draft_n;
        base["draft_n_accepted"] = draft_n_accepted;
        if (!draft_n_by_depth.empty()) {
            json by_depth = json::array();
            for (size_t i = 0; i < draft_n_by_depth.size(); ++i) {
                if (draft_n_by_depth[i] <= 0) {
                    continue;
                }
                const int32_t accepted = i < draft_n_accepted_by_depth.size()
                    ? draft_n_accepted_by_depth[i] : 0;
                by_depth.push_back({
                    {"depth",              (int32_t) i + 1},
                    {"draft_n",            draft_n_by_depth[i]},
                    {"draft_n_accepted",   accepted},
                });
            }
            if (!by_depth.empty()) {
                base["draft_by_depth"] = by_depth;
            }
        }
    }

    return base;
}


//json server_task_result_cmpl_partial::to_json_non_oaicompat_partial() {
//    // non-OAI-compat JSON
//    json res = json{
//        {"index",            index},
//        {"content",          content},
//        {"tokens",           tokens},
//        {"stop",             false},
//        {"id_slot",          id_multi},
//        {"tokens_predicted", n_decoded},
//        {"tokens_evaluated", n_prompt_tokens},
//    };
//    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
//    if (timings.prompt_n > 0) {
//        res.push_back({ "timings", timings.to_json() });
//    }
//    if (!probs_output.empty()) {
//        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
//    }
//    return res;
//}

//json server_task_result_cmpl_final::to_json_non_oaicompat_final() {
//    json res = json{
//        {"index",               index},
//        {"content",             stream ? "" : content}, // in stream mode, content is already in last partial chunk
//        {"tokens",              stream ? std::vector<llama_token> {} : tokens},
//        {"id_slot",             id_multi},
//        {"stop",                true},
//        {"model",               oaicompat_model},
//        {"tokens_predicted",    n_decoded},
//        {"tokens_evaluated",    n_prompt_tokens},
//        //{"generation_settings", default_generation_settings_for_props.to_json()},
//        {"prompt",              prompt},
//        {"has_new_line",        has_new_line},
//        {"truncated",           truncated},
//        //{"stop_type",           stop_type_to_str(STOP_TYPE_EOS)},
//        {"stopping_word",       stopping_word},
//        {"tokens_cached",       n_tokens_cached},
//        {"timings",             timings.to_json()},
//    };
//    if (!stream && !probs_output.empty()) {
//        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
//    }
//    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
//}

json server_task_result_cmpl_partial::to_json_non_oaicompat_partial() {
    // non-OAI-compat JSON
    return data;
}

json server_task_result_cmpl_final::to_json_non_oaicompat_final() {
    // non-OAI-compat JSON
    return data;
}

json server_task_result_cmpl_partial::to_json_oaicompat_partial() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json res = json{
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat_partial();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({ "timings", timings.to_json() });
    }

    return res;
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json{
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}


json server_task_result_cmpl_final::to_json_oaicompat_final() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json{
        {"choices",            json::array({
            json{
                {"text",          stream ? "" : content}, // in stream mode, content is already in last partial chunk
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat_final();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({ "timings", timings.to_json() });
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat_partial() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json& delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", 0},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"object", "chat.completion.chunk"},
            });
    };
    // We have to send an initial update to conform to openai behavior
    if (first) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
            });
    }

    for (const auto& diff : oaicompat_msg_diffs) {        
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        GGML_ASSERT(deltas[deltas.size() - 1].at("choices").size() >= 1);

        if (probs_output.size() > 0) {
            deltas[deltas.size() - 1].at("choices").at(0)["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
            };
        }

        if (timings.prompt_n >= 0) {
            deltas[deltas.size() - 1].push_back({ "timings", timings.to_json() });
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp_partial() {
    std::vector<json> events;

    if (n_decoded == 1) {
        events.push_back(json{
            {"event", "response.created"},
            {"data", json{
                {"type", "response.created"},
                {"response", json{
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json{
            {"event", "response.in_progress"},
            {"data", json{
                {"type", "response.in_progress"},
                {"response", json{
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const auto& diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!oai_resp_thinking_block_started) {
                events.push_back(json{
                    {"event", "response.output_item.added"},
                    {"data", json{
                        {"type", "response.output_item.added"},
                        {"item", json{
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                oai_resp_thinking_block_started = true;
            }
            events.push_back(json{
                {"event", "response.reasoning_text.delta"},
                {"data", json{
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!oai_resp_text_block_started) {
                events.push_back(json{
                    {"event", "response.output_item.added"},
                    {"data", json{
                        {"type", "response.output_item.added"},
                        {"item", json{
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json{
                    {"event", "response.content_part.added"},
                    {"data", json{
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json{
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                oai_resp_text_block_started = true;
            }
            events.push_back(json{
                {"event", "response.output_text.delta"},
                {"data", json{
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json{
                {"event", "response.output_item.added"},
                {"data", json{
                    {"type",  "response.output_item.added"},
                    {"item", json{
                        {"arguments", ""},
                        {"call_id",   "fc_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json{
                {"event", "response.function_call_arguments.delta"},
                {"data", json{
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }

    return events;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_final() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    }
    else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }


    json choice{
        {"finish_reason", finish_reason},
        {"index", 0},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json{
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat_final();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({ "timings", timings.to_json() });
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop) {
        //if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto& diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", 0},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"object", "chat.completion.chunk"},
            });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", 0},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"object",             "chat.completion.chunk"},
        });
    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
            });
    }
    if (timings.prompt_n >= 0) {
        deltas.back().push_back({ "timings", timings.to_json() });
    }
    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat_final();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_final() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    }
    else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (!msg.reasoning_content.empty()) {
        output.push_back(json{
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({json{
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (!msg.content.empty()) {
        output.push_back(json{
            {"content", json::array({json{
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     oai_resp_message_id},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const auto& tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json{
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "fc_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> events;
    std::vector<json> output;

    if (!oaicompat_msg.reasoning_content.empty()) {
        const json output_item = json{
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({json{
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        events.push_back(json{
            {"event", "response.output_item.done"},
            {"data", json{
                {"type", "response.output_item.done"},
                {"item", output_item},
            }},
        });
        output.push_back(output_item);
    }

    if (!oaicompat_msg.content.empty()) {
        events.push_back(json{
            {"event", "response.output_text.done"},
            {"data", json{
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content},
            }},
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content},
        };

        events.push_back(json{
            {"event", "response.content_part.done"},
            {"data", json{
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part},
            }},
        });

        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"},
        };

        events.push_back(json{
            {"event", "response.output_item.done"},
            {"data", json{
                {"type", "response.output_item.done"},
                {"item", output_item},
            }},
        });
        output.push_back(output_item);
    }

    for (const auto& tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "fc_" + tool_call.id},
            {"name",      tool_call.name},
        };
        events.push_back(json{
            {"event", "response.output_item.done"},
            {"data", json{
                {"type", "response.output_item.done"},
                {"item", output_item},
            }},
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    events.push_back(json{
        {"event", "response.completed"},
        {"data", json{
            {"type", "response.completed"},
            {"response", json{
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }},
    });

    return events;
}

json server_task_result_cmpl_final::to_json_anthropic_final() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    }
    else {
        msg.role = "assistant";
        msg.content = content;
    }

    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
            });
    }

    for (const auto& tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        }
        catch (const std::exception&) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    size_t thinking_block_index = 0;
    size_t text_block_index = anthropic_thinking_block_started ? 1 : 0;

    bool thinking_block_started = anthropic_thinking_block_started;
    bool text_block_started = anthropic_text_block_started;
    std::set<size_t> tool_calls_started;

    for (const auto& diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                    });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
                });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                    });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
                });
        }

        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (thinking_block_started ? 1 : 0) + (text_block_started ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto& full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                    });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                    });
            }
        }
    }

    if (thinking_block_started) {
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
            });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
            });
    }

    if (text_block_started) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
            });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (thinking_block_started ? 1 : 0) + (text_block_started ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
            });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
        });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
        });

    // extra fields for debugging purposes
    if (verbose && !events.empty()) {
        events.front()["data"]["__verbose"] = to_json_non_oaicompat_final();
    }
    // Don't add timings for Anthropic API (breaks spec compliance)
    if (oaicompat != OAICOMPAT_TYPE_ANTHROPIC && timings.prompt_n >= 0 && !events.empty()) {
        events.back()["data"]["timings"] = timings.to_json();
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_anthropic_partial() {
    json events = json::array();
    bool first = n_decoded == 1;

    size_t thinking_block_index = 0;
    size_t text_block_index = anthropic_has_reasoning ? 1 : 0;

    bool thinking_started = anthropic_thinking_block_started;
    bool text_started = anthropic_text_block_started;

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
            });
    }

    for (const auto& diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                    });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
                });
        }

        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                    });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
                });
        }

        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                    });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                    });
            }
        }
    }

    if (verbose && !events.empty() && first) {
        events.front()["data"]["__verbose"] = to_json_non_oaicompat_partial();
    }

    if (timings.prompt_n >= 0 && !events.empty()) {
        events.back()["data"]["timings"] = timings.to_json();
    }

    //if (is_progress && !events.empty()) {
    //    events.back()["data"]["prompt_progress"] = progress.to_json();
    //}

    return events;
}


size_t server_prompt::size() const {
    return ram_size() + disk_size();
}

size_t server_prompt::ram_size() const {
    size_t res = data.size();

    for (const auto& checkpoint : checkpoints) {
        res += checkpoint.size();
    }

    return res;
}

server_prompt_cache::server_prompt_cache(
        llama_context * ctx,
               int32_t   limit_ram_mib,
               int32_t   limit_disk_mib,
                size_t   limit_tokens,
           std::string   disk_root,
                bool     rewind_to_checkpoint_on_save) {
    this->ctx = ctx;
    this->ram_enabled = limit_ram_mib != 0;
    this->limit_ram_size = 1024ull * 1024ull * (limit_ram_mib < 0 ? 0 : limit_ram_mib);
    this->limit_disk_size = 1024ull * 1024ull * (limit_disk_mib < 0 ? 0 : limit_disk_mib);
    this->limit_tokens = limit_tokens;
    this->rewind_to_checkpoint_on_save = rewind_to_checkpoint_on_save;

    if (limit_disk_mib != 0) {
        if (disk_root.empty()) {
            throw std::runtime_error("disk prompt caching requires a cache root");
        }
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(disk_root, ec);
        if (ec) {
            throw std::runtime_error("failed to create disk prompt-cache root '" + disk_root + "': " + ec.message());
        }

        const uint64_t stamp = (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
        for (uint32_t attempt = 0; attempt < 1024; ++attempt) {
            const fs::path candidate = fs::path(disk_root) /
                ("ik-llama-prompt-cache-" + std::to_string(stamp) + "-" + std::to_string(attempt));
            ec.clear();
            if (fs::create_directory(candidate, ec)) {
                fs::permissions(candidate, fs::perms::owner_all,
                        fs::perm_options::replace, ec);
                if (ec) {
                    std::error_code remove_ec;
                    fs::remove(candidate, remove_ec);
                    throw std::runtime_error("failed to make disk prompt-cache directory private below '" +
                            disk_root + "': " + ec.message());
                }
                disk_directory = candidate.string();
                break;
            }
            if (ec) {
                throw std::runtime_error("failed to create private disk prompt-cache directory below '" +
                        disk_root + "': " + ec.message());
            }
        }
        if (disk_directory.empty()) {
            throw std::runtime_error("failed to allocate a unique disk prompt-cache directory below '" + disk_root + "'");
        }
    }
}

server_prompt_cache::~server_prompt_cache() {
    clear();
    if (!disk_directory.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(disk_directory, ec);
        if (ec) {
            LLAMA_LOG_WARN("failed to remove disk prompt-cache directory '%s': %s\n",
                    disk_directory.c_str(), ec.message().c_str());
        }
    }
}

static void server_prompt_remove_state_file(server_prompt & prompt) {
    for (const std::string * path : { &prompt.state_file, &prompt.checkpoint_file }) {
        if (path->empty()) {
            continue;
        }
        std::error_code ec;
        std::filesystem::remove(*path, ec);
        if (ec) {
            LLAMA_LOG_WARN("failed to remove disk prompt-cache file '%s': %s\n",
                    path->c_str(), ec.message().c_str());
        }
    }
    prompt.state_file.clear();
    prompt.state_file_size = 0;
    prompt.checkpoint_file.clear();
    prompt.checkpoint_file_size = 0;
}

static void server_prompt_remove_checkpoint_state_files(server_prompt & prompt) {
    for (auto & checkpoint : prompt.checkpoints) {
        if (checkpoint.state_file.empty()) {
            continue;
        }
        std::error_code ec;
        std::filesystem::remove(checkpoint.state_file, ec);
        if (ec) {
            LLAMA_LOG_WARN("failed to remove staged prompt checkpoint '%s': %s\n",
                    checkpoint.state_file.c_str(), ec.message().c_str());
        }
        checkpoint.state_file.clear();
        checkpoint.state_file_size = 0;
    }
}

static void server_prompt_remove_all_files(server_prompt & prompt) {
    server_prompt_remove_state_file(prompt);
    server_prompt_remove_checkpoint_state_files(prompt);
}

static bool server_prompt_load_checkpoint_file(server_prompt & prompt);

static void server_prompt_release_file_pages(const std::string & path, bool flush_dirty) {
#if defined(__linux__)
    const int flags = (flush_dirty ? O_RDWR : O_RDONLY) | O_CLOEXEC;
    const int fd = open(path.c_str(), flags);
    if (fd < 0) {
        LLAMA_LOG_WARN("failed to open disk prompt-cache file for page-cache release '%s': errno %d\n",
                path.c_str(), errno);
        return;
    }
    if (flush_dirty && fdatasync(fd) != 0) {
        LLAMA_LOG_WARN("failed to flush disk prompt-cache file '%s': errno %d\n",
                path.c_str(), errno);
    }
    const int rc = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    if (rc != 0) {
        LLAMA_LOG_WARN("failed to release disk prompt-cache pages for '%s': error %d\n",
                path.c_str(), rc);
    }
    close(fd);
#else
    (void) path;
    (void) flush_dirty;
#endif
}

size_t server_prompt_cache::size() const {
    return ram_size() + disk_size();
}

size_t server_prompt_cache::ram_size() const {
    size_t res = 0;

    for (const auto& state : states) {
        res += state.ram_size();
    }

    return res;
}

size_t server_prompt_cache::disk_size() const {
    // Count the private directory rather than only cached LRU entries. A
    // loaded hybrid snapshot remains leased by its live slot so it can be
    // re-adopted without a multi-GiB rewrite, and still belongs to this tier's
    // capacity budget while it is active.
    if (!disk_directory.empty()) {
        namespace fs = std::filesystem;
        std::error_code ec;
        size_t res = 0;
        fs::directory_iterator it(disk_directory, ec);
        const fs::directory_iterator end;
        while (!ec && it != end) {
            if (it->is_regular_file(ec) && !ec) {
                const uintmax_t file_size = it->file_size(ec);
                if (!ec) {
                    res += file_size > std::numeric_limits<size_t>::max() - res
                        ? std::numeric_limits<size_t>::max() - res
                        : (size_t) file_size;
                }
            }
            if (!ec) {
                it.increment(ec);
            }
        }
        if (!ec) {
            return res;
        }
    }

    size_t res = 0;
    for (const auto & state : states) {
        res += state.disk_size();
    }
    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto& state : states) {
        res += state.n_tokens();
    }
    return res;

}

bool server_prompt_cache::load(server_prompt& prompt, const server_tokens& tokens_new, llama_context* ctx, int32_t id_slot, float min_reusable_fraction) {
    thinking_tokens think_tokens;
    for (auto it = states.begin(); it != states.end(); ++it) {
        think_tokens = it->think_tokens;
        break;
    }
    server_tokens prompt_tokens;
    server_tokens tokens_new_ex;
    if (think_tokens.exclude) {
        prompt_tokens = prompt.tokens.get_tokens_exclude_think(ctx, think_tokens);
        tokens_new_ex = tokens_new.get_tokens_exclude_think(ctx, think_tokens);
    }
    else {
        prompt_tokens = prompt.tokens.clone();
        tokens_new_ex = tokens_new.clone();
    }
    const auto lcp_best_fuzzy = prompt_tokens.get_common_prefix(ctx, tokens_new_ex);
    size_t lcp_best = rewind_to_checkpoint_on_save
        ? prompt_tokens.get_common_prefix_exact(tokens_new_ex)
        : lcp_best_fuzzy.second;
    float f_keep_best = prompt_tokens.empty() ? 0.0f : float(lcp_best) / prompt_tokens.size();
    float sim_best = prompt_tokens.get_tokens_similarity(ctx, tokens_new_ex, prompt.n_kept_prompt, prompt.n_discarded_prompt);
    LLAMA_LOG_INFO(" - looking for better prompt, base lcp = %zu, f_keep = %.3f, sim = %.3f, n_keep = %d, n_discarded_prompt = %d\n",
            lcp_best, f_keep_best, sim_best, prompt.n_kept_prompt, prompt.n_discarded_prompt);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        server_tokens tokens;
        if (think_tokens.exclude) {
            tokens = it->tokens.get_tokens_exclude_think(ctx, think_tokens);
        }
        else {
            tokens = it->tokens.clone();
        }
        const auto lcp_cur_fuzzy = tokens.get_common_prefix(ctx, tokens_new_ex);
        const size_t lcp_cur = rewind_to_checkpoint_on_save
            ? tokens.get_common_prefix_exact(tokens_new_ex)
            : lcp_cur_fuzzy.first;
        const float f_keep_cur = tokens.empty() ? 0.0f : float(lcp_cur) / tokens.size();
        if (f_keep_cur < min_reusable_fraction) {
            continue;
        }
        const float sim_cur = tokens.get_tokens_similarity(ctx, tokens_new_ex, it->n_kept_prompt, it->n_discarded_prompt);
        const bool better = rewind_to_checkpoint_on_save
            ? (lcp_cur > lcp_best || (lcp_cur == lcp_best && sim_cur > sim_best))
            : sim_best < sim_cur;
        if (better) {
            lcp_best = lcp_cur;
            f_keep_best = f_keep_cur;
            sim_best = sim_cur;
            it_best = it;
        }
    }

    if (it_best != states.end()) {
        LLAMA_LOG_INFO(" - found better prompt with lcp = %zu, f_keep = %.3f, sim = %.3f, n_keep = %d, n_discarded_prompt = %d\n",
                lcp_best, f_keep_best, sim_best, it_best->n_kept_prompt, it_best->n_discarded_prompt);
        if (!it_best->state_file.empty()) {
            make_ram_room(it_best->checkpoint_file_size);
            if (!server_prompt_load_checkpoint_file(*it_best)) {
                return false;
            }
            const size_t state_token_count = it_best->state_n_tokens >= 0
                ? (size_t) it_best->state_n_tokens : it_best->tokens.size();
            llama_tokens state_tokens(state_token_count);
            size_t token_count = 0;
            const size_t n = llama_state_seq_load_file(ctx, it_best->state_file.c_str(), id_slot,
                    state_tokens.data(), state_tokens.size(), &token_count);
            server_prompt_release_file_pages(it_best->state_file, false);
            if (n == 0 || token_count != state_tokens.size()) {
                LLAMA_LOG_INFO("failed to restore disk prompt-cache state '%s' (%zu of %zu tokens)\n",
                        it_best->state_file.c_str(), token_count, state_tokens.size());
                std::list<server_prompt_checkpoint>().swap(it_best->checkpoints);
                return false;
            }

            // A hybrid prompt-tail snapshot is immutable and remains a valid
            // future cache entry after it has been loaded. Transfer its file
            // into the live slot's checkpoint list instead of deleting and
            // immediately streaming the same multi-GiB state back to disk.
            // save() will adopt this file again when the slot is displaced.
            if (rewind_to_checkpoint_on_save && it_best->state_n_tokens > 0) {
                server_prompt_checkpoint retained = {};
                retained.pos_min = llama_kv_cache_seq_pos_min(ctx, id_slot);
                retained.pos_max = llama_kv_cache_seq_pos_max(ctx, id_slot);
                const llama_pos prompt_offset = it_best->state_pos_max_prompt >= 0
                    ? it_best->state_pos_max_prompt - retained.pos_max : 0;
                retained.pos_min_prompt = retained.pos_min + prompt_offset;
                retained.pos_max_prompt = retained.pos_max + prompt_offset;
                retained.n_tokens = it_best->state_n_tokens;
                retained.state_file = std::move(it_best->state_file);
                retained.state_file_size = it_best->state_file_size;
                it_best->state_file_size = 0;
                it_best->checkpoints.push_back(std::move(retained));
                LLAMA_LOG_INFO("retained loaded hybrid prompt snapshot for zero-copy re-adoption\n");
            }
            server_prompt_remove_state_file(*it_best);
        } else {
            const size_t size = it_best->data.size();
            const size_t n = llama_state_seq_set_data(ctx, it_best->data.data(), size, id_slot, 0);
            if (n != size) {
                LLAMA_LOG_INFO("failed to restore state with size %zu\n", size);
                return false;
            }
        }

        it_best->data.clear();
        it_best->data.shrink_to_fit();

        // The destination may own a staged snapshot from its current live
        // sequence. It is about to be replaced, so release that file now
        // instead of retaining it until process teardown.
        server_prompt_remove_all_files(prompt);
        prompt = std::move(*it_best);

        states.erase(it_best);
    }

    return true;
}

server_prompt* server_prompt_cache::alloc(server_prompt & prompt, size_t state_size) {
    for (auto it = states.begin(); it != states.end();) {
        auto tokens_ctx_shift = prompt.tokens.clone();  // copy cache tokens
        tokens_ctx_shift.discard_n_tokens(prompt.n_kept_prompt, prompt.n_discarded_prompt);
        const auto prefix_fuzzy = it->tokens.get_common_prefix(ctx, tokens_ctx_shift);
        const size_t len = rewind_to_checkpoint_on_save
            ? it->tokens.get_common_prefix_exact(tokens_ctx_shift)
            : prefix_fuzzy.first;
        const size_t len_prompt = rewind_to_checkpoint_on_save ? len : prefix_fuzzy.second;
        // first check if the current state is contained fully in the cache
        if (len_prompt == tokens_ctx_shift.size()) {
            LLAMA_LOG_INFO("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
        // next, remove any cached prompts that are fully contained in the current prompt
        else if (len == it->tokens.size()) {
            LLAMA_LOG_INFO(" - removing obsolete cached prompt with length %d\n", (int)len);
            server_prompt_remove_all_files(*it);
            it = states.erase(it);
        }
        else {
            ++it;
        }
    }

    std::vector<uint8_t> state_data;

    if (state_size > 0) {
        state_data.resize(state_size);
    }

    // TODO: for some reason we can't copy server_tokens, so we have to do this workaround
    auto& cur = states.emplace_back();
    cur = {
        /*.tokens          =*/ prompt.tokens.clone(),
        /*.n_keep          =*/ prompt.n_kept_prompt,
        /*.n_discarded_prompt     =*/ prompt.n_discarded_prompt,
        /*.think_tokens                   =*/ prompt.think_tokens,
        /*.data            =*/ std::move(state_data),
        /*.state_file      =*/ {},
        /*.state_file_size =*/ 0,
        /*.checkpoint_file =*/ {},
        /*.checkpoint_file_size =*/ 0,
        /*.state_n_tokens =*/ prompt.state_n_tokens,
        /*.state_pos_max_prompt =*/ prompt.state_pos_max_prompt,
        /*.checkpoints     =*/ std::move(prompt.checkpoints),
    };

    return &cur;
}

static bool server_prompt_commit_partial(const std::string & partial_path, const std::string & final_path);

static size_t server_prompt_checkpoint_file_size(const server_prompt & prompt) {
    if (prompt.checkpoints.empty()) {
        return 0;
    }

    size_t size = 3*sizeof(uint32_t);
    for (const auto & checkpoint : prompt.checkpoints) {
        size += sizeof(checkpoint.pos_min) + sizeof(checkpoint.pos_max) +
            sizeof(checkpoint.pos_min_prompt) + sizeof(checkpoint.pos_max_prompt) +
            sizeof(checkpoint.n_tokens) + sizeof(uint64_t) + checkpoint.data.size();
    }
    return size;
}

static std::pair<std::string, std::string> server_prompt_cache_next_paths(server_prompt_cache & cache) {
    namespace fs = std::filesystem;
    const std::string base = "prompt-" + std::to_string(cache.disk_file_id++);
    const fs::path final_path = fs::path(cache.disk_directory) / (base + ".bin");
    const fs::path partial_path = fs::path(cache.disk_directory) / (base + ".partial");
    return { final_path.string(), partial_path.string() };
}

bool server_prompt_cache::stage_checkpoint(
        server_prompt_checkpoint & checkpoint,
            const server_tokens & tokens,
                      int32_t      id_slot,
                       size_t      replaceable_disk_size) {
    if (!has_disk_tier() || !checkpoint.state_file.empty() || checkpoint.n_tokens <= 0 ||
            checkpoint.n_tokens > tokens.n_tokens()) {
        return false;
    }

    llama_tokens state_tokens = tokens.tokens_data();
    state_tokens.resize((size_t) checkpoint.n_tokens);
    const size_t state_size = llama_state_seq_get_size(ctx, id_slot, 0);
    const size_t estimated_file_size = 3*sizeof(uint32_t) +
        state_tokens.size()*sizeof(llama_token) + state_size;
    if (state_size == 0 || !make_disk_room(estimated_file_size, replaceable_disk_size)) {
        LLAMA_LOG_WARN("cannot stage hybrid prompt checkpoint: %.3f MiB state does not fit the disk tier budget\n",
                estimated_file_size / (1024.0 * 1024.0));
        return false;
    }

    const auto paths = server_prompt_cache_next_paths(*this);
    const int64_t t_start = ggml_time_us();
    const size_t written = llama_state_seq_save_file(ctx, paths.second.c_str(), id_slot,
            state_tokens.data(), state_tokens.size());
    if (written == 0 || !server_prompt_commit_partial(paths.second, paths.first)) {
        std::error_code ec;
        std::filesystem::remove(paths.second, ec);
        return false;
    }
    server_prompt_release_file_pages(paths.first, true);

    checkpoint.state_file = paths.first;
    checkpoint.state_file_size = written;
    LLAMA_LOG_INFO(" - staged self-consistent hybrid checkpoint at token %" PRId64
            ": %.3f MiB in %.2f ms\n", checkpoint.n_tokens,
            written/(1024.0*1024.0), (ggml_time_us() - t_start)/1000.0);
    return true;
}

static bool server_prompt_commit_partial(const std::string & partial_path, const std::string & final_path) {
    std::error_code ec;
    std::filesystem::rename(partial_path, final_path, ec);
    if (!ec) {
        return true;
    }
    std::error_code remove_ec;
    std::filesystem::remove(partial_path, remove_ec);
    LLAMA_LOG_WARN("failed to commit disk prompt-cache state '%s': %s\n",
            final_path.c_str(), ec.message().c_str());
    return false;
}

static bool server_prompt_save_checkpoint_file(server_prompt & prompt) {
    if (prompt.checkpoints.empty()) {
        return true;
    }
    GGML_ASSERT(!prompt.state_file.empty());

    const std::string final_path = prompt.state_file + ".checkpoints";
    const std::string partial_path = final_path + ".partial";
    std::ofstream file(partial_path, std::ios::binary | std::ios::trunc);
    const uint32_t magic = LLAMA_STATE_SEQ_MAGIC;
    const uint32_t version = LLAMA_STATE_SEQ_VERSION;
    const uint32_t count = (uint32_t) prompt.checkpoints.size();
    size_t file_size = sizeof(magic) + sizeof(version) + sizeof(count);
    file.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char *>(&version), sizeof(version));
    file.write(reinterpret_cast<const char *>(&count), sizeof(count));
    for (const auto & checkpoint : prompt.checkpoints) {
        const uint64_t data_size = checkpoint.data.size();
        file.write(reinterpret_cast<const char *>(&checkpoint.pos_min), sizeof(checkpoint.pos_min));
        file.write(reinterpret_cast<const char *>(&checkpoint.pos_max), sizeof(checkpoint.pos_max));
        file.write(reinterpret_cast<const char *>(&checkpoint.pos_min_prompt), sizeof(checkpoint.pos_min_prompt));
        file.write(reinterpret_cast<const char *>(&checkpoint.pos_max_prompt), sizeof(checkpoint.pos_max_prompt));
        file.write(reinterpret_cast<const char *>(&checkpoint.n_tokens), sizeof(checkpoint.n_tokens));
        file.write(reinterpret_cast<const char *>(&data_size), sizeof(data_size));
        file.write(reinterpret_cast<const char *>(checkpoint.data.data()), checkpoint.data.size());
        file_size += sizeof(checkpoint.pos_min) + sizeof(checkpoint.pos_max) +
            sizeof(checkpoint.pos_min_prompt) + sizeof(checkpoint.pos_max_prompt) +
            sizeof(checkpoint.n_tokens) + sizeof(data_size) + checkpoint.data.size();
    }
    file.close();
    if (!file) {
        std::error_code ec;
        std::filesystem::remove(partial_path, ec);
        LLAMA_LOG_WARN("failed to write prompt-cache checkpoints to '%s'\n", partial_path.c_str());
        return false;
    }
    if (!server_prompt_commit_partial(partial_path, final_path)) {
        return false;
    }
    server_prompt_release_file_pages(final_path, true);

    prompt.checkpoint_file = final_path;
    prompt.checkpoint_file_size = file_size;
    return true;
}

static bool server_prompt_load_checkpoint_file(server_prompt & prompt) {
    if (prompt.checkpoint_file.empty()) {
        return true;
    }

    std::ifstream file(prompt.checkpoint_file, std::ios::binary);
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t count = 0;
    file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    file.read(reinterpret_cast<char *>(&count), sizeof(count));
    if (!file || magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION || count > 4096) {
        LLAMA_LOG_WARN("invalid disk prompt-cache checkpoint file '%s'\n", prompt.checkpoint_file.c_str());
        return false;
    }

    std::list<server_prompt_checkpoint> checkpoints;
    for (uint32_t i = 0; i < count; ++i) {
        server_prompt_checkpoint checkpoint = {};
        uint64_t data_size = 0;
        file.read(reinterpret_cast<char *>(&checkpoint.pos_min), sizeof(checkpoint.pos_min));
        file.read(reinterpret_cast<char *>(&checkpoint.pos_max), sizeof(checkpoint.pos_max));
        file.read(reinterpret_cast<char *>(&checkpoint.pos_min_prompt), sizeof(checkpoint.pos_min_prompt));
        file.read(reinterpret_cast<char *>(&checkpoint.pos_max_prompt), sizeof(checkpoint.pos_max_prompt));
        file.read(reinterpret_cast<char *>(&checkpoint.n_tokens), sizeof(checkpoint.n_tokens));
        file.read(reinterpret_cast<char *>(&data_size), sizeof(data_size));
        if (!file || data_size > std::numeric_limits<size_t>::max()) {
            LLAMA_LOG_WARN("truncated disk prompt-cache checkpoint file '%s'\n", prompt.checkpoint_file.c_str());
            return false;
        }
        try {
            checkpoint.data.resize((size_t) data_size);
        } catch (const std::bad_alloc &) {
            LLAMA_LOG_WARN("cannot allocate %.3f MiB for a restored context checkpoint\n",
                    data_size / (1024.0 * 1024.0));
            return false;
        }
        file.read(reinterpret_cast<char *>(checkpoint.data.data()), checkpoint.data.size());
        if (!file) {
            LLAMA_LOG_WARN("truncated disk prompt-cache checkpoint file '%s'\n", prompt.checkpoint_file.c_str());
            return false;
        }
        checkpoints.push_back(std::move(checkpoint));
    }

    server_prompt_release_file_pages(prompt.checkpoint_file, false);
    prompt.checkpoints = std::move(checkpoints);
    return true;
}

bool server_prompt_cache::save_to_disk(server_prompt & prompt, int32_t id_slot) {
    GGML_ASSERT(has_disk_tier());
    llama_tokens state_tokens = prompt.tokens.tokens_data();
    if (prompt.state_n_tokens >= 0 &&
            (size_t) prompt.state_n_tokens < state_tokens.size()) {
        state_tokens.resize((size_t) prompt.state_n_tokens);
    }
    const size_t state_size = llama_state_seq_get_size(ctx, id_slot, 0);
    const size_t estimated_file_size = 3*sizeof(uint32_t) +
        state_tokens.size()*sizeof(llama_token) + state_size +
        server_prompt_checkpoint_file_size(prompt);
    if (state_size == 0 || !make_disk_room(estimated_file_size)) {
        LLAMA_LOG_WARN("cannot stream %.3f MiB prompt state: disk tier budget is exhausted\n",
                estimated_file_size / (1024.0 * 1024.0));
        return false;
    }

    const auto paths = server_prompt_cache_next_paths(*this);
    const size_t n = llama_state_seq_save_file(ctx, paths.second.c_str(), id_slot,
            state_tokens.data(), state_tokens.size());
    if (n == 0) {
        std::error_code ec;
        std::filesystem::remove(paths.second, ec);
        LLAMA_LOG_WARN("failed to stream prompt-cache state to '%s'\n", paths.second.c_str());
        return false;
    }
    if (!server_prompt_commit_partial(paths.second, paths.first)) {
        return false;
    }
    server_prompt_release_file_pages(paths.first, true);

    prompt.state_file = paths.first;
    prompt.state_file_size = n;
    if (!server_prompt_save_checkpoint_file(prompt)) {
        server_prompt_remove_state_file(prompt);
        return false;
    }
    std::list<server_prompt_checkpoint>().swap(prompt.checkpoints);
    LLAMA_LOG_INFO(" - streamed prompt cache to disk: %s, %.3f MiB\n",
            prompt.state_file.c_str(), prompt.disk_size() / (1024.0 * 1024.0));
    return true;
}

bool server_prompt_cache::spill_to_disk(server_prompt & prompt) {
    GGML_ASSERT(has_disk_tier());
    if (prompt.data.empty()) {
        return !prompt.state_file.empty();
    }

    llama_tokens state_tokens = prompt.tokens.tokens_data();
    if (prompt.state_n_tokens >= 0 &&
            (size_t) prompt.state_n_tokens < state_tokens.size()) {
        state_tokens.resize((size_t) prompt.state_n_tokens);
    }
    if (state_tokens.size() > std::numeric_limits<uint32_t>::max()) {
        LLAMA_LOG_WARN("cannot spill prompt cache with %zu tokens to disk\n", state_tokens.size());
        return false;
    }

    const size_t estimated_file_size = 3*sizeof(uint32_t) +
        state_tokens.size()*sizeof(llama_token) + prompt.data.size() +
        server_prompt_checkpoint_file_size(prompt);
    if (!make_disk_room(estimated_file_size)) {
        LLAMA_LOG_WARN("cannot spill %.3f MiB prompt state: disk tier budget is exhausted\n",
                estimated_file_size / (1024.0 * 1024.0));
        return false;
    }

    const auto paths = server_prompt_cache_next_paths(*this);
    std::ofstream file(paths.second, std::ios::binary | std::ios::trunc);
    const uint32_t magic = LLAMA_STATE_SEQ_MAGIC;
    const uint32_t version = LLAMA_STATE_SEQ_VERSION;
    const uint32_t token_count = (uint32_t) state_tokens.size();
    file.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char *>(&version), sizeof(version));
    file.write(reinterpret_cast<const char *>(&token_count), sizeof(token_count));
    file.write(reinterpret_cast<const char *>(state_tokens.data()), state_tokens.size() * sizeof(llama_token));
    file.write(reinterpret_cast<const char *>(prompt.data.data()), prompt.data.size());
    const size_t file_size = sizeof(magic) + sizeof(version) + sizeof(token_count) +
        state_tokens.size() * sizeof(llama_token) + prompt.data.size();
    file.close();
    if (!file) {
        std::error_code ec;
        std::filesystem::remove(paths.second, ec);
        LLAMA_LOG_WARN("failed to spill RAM prompt-cache state to '%s'\n", paths.second.c_str());
        return false;
    }
    if (!server_prompt_commit_partial(paths.second, paths.first)) {
        return false;
    }
    server_prompt_release_file_pages(paths.first, true);

    prompt.state_file = paths.first;
    prompt.state_file_size = file_size;
    if (!server_prompt_save_checkpoint_file(prompt)) {
        server_prompt_remove_state_file(prompt);
        return false;
    }
    std::vector<uint8_t>().swap(prompt.data);
    std::list<server_prompt_checkpoint>().swap(prompt.checkpoints);
    LLAMA_LOG_INFO(" - spilled cached prompt to disk: %s, %.3f MiB\n",
            prompt.state_file.c_str(), prompt.disk_size() / (1024.0 * 1024.0));
    return true;
}

void server_prompt_cache::make_ram_room(size_t incoming_size) {
    if (!ram_enabled || limit_ram_size == 0) {
        return;
    }

    while (ram_size() + incoming_size > limit_ram_size) {
        auto it = std::find_if(states.begin(), states.end(), [](const server_prompt & state) {
            return !state.data.empty();
        });
        if (it == states.end()) {
            break;
        }

        if (has_disk_tier() && spill_to_disk(*it)) {
            continue;
        }

        LLAMA_LOG_INFO(" - RAM prompt-cache limit reached, dropping oldest in-memory entry (%.3f MiB)\n",
                it->ram_size() / (1024.0 * 1024.0));
        server_prompt_remove_all_files(*it);
        states.erase(it);
    }
}

bool server_prompt_cache::make_disk_room(size_t incoming_size, size_t replaceable_size) {
    if (!has_disk_tier()) {
        return false;
    }

    // A live hybrid slot can carry an immutable disk snapshot that is about
    // to be superseded by this write.  Keep the old file until the new one is
    // atomically committed, but do not evict an unrelated LRU entry merely
    // because both generations coexist for those few seconds.
    bool space_query_warned = false;
    while (true) {
        const size_t current_size = disk_size();
        const size_t reclaimable = std::min(current_size, replaceable_size);
        const size_t retained_size = current_size - reclaimable;

        const bool budget_fits = limit_disk_size == 0 ||
            (incoming_size <= limit_disk_size &&
             retained_size <= limit_disk_size - incoming_size);

        // replaceable_size is useful for the logical cache budget, but the old
        // snapshot deliberately remains present until the new .partial file is
        // committed.  The backing filesystem therefore needs room for both
        // generations.  Query its live headroom instead of assuming that the
        // configured cache budget is the whole filesystem.
        std::error_code space_ec;
        const auto space = std::filesystem::space(disk_directory, space_ec);
        const bool physical_fits = space_ec || incoming_size <= space.available;
        if (space_ec && !space_query_warned) {
            LLAMA_LOG_WARN("failed to query disk prompt-cache filesystem headroom for '%s': %s\n",
                    disk_directory.c_str(), space_ec.message().c_str());
            space_query_warned = true;
        }

        if (budget_fits && physical_fits) {
            break;
        }
        auto it = std::find_if(states.begin(), states.end(), [](const server_prompt & state) {
            return state.disk_size() > 0;
        });
        if (it == states.end()) {
            return false;
        }
        LLAMA_LOG_INFO(" - making disk prompt-cache room, removing oldest spilled entry "
                "(%.3f MiB, reason: %s%s%s, filesystem available: %.3f MiB, incoming: %.3f MiB)\n",
                it->disk_size() / (1024.0 * 1024.0),
                budget_fits ? "" : "tier budget",
                !budget_fits && !physical_fits ? " + " : "",
                physical_fits ? "" : "atomic-write headroom",
                space_ec ? -1.0 : space.available / (1024.0 * 1024.0),
                incoming_size / (1024.0 * 1024.0));
        server_prompt_remove_all_files(*it);
        states.erase(it);
    }
    return true;
}

static bool server_prompt_read_state_payload(
        const std::string & path,
                  uint32_t   expected_tokens,
     std::vector<uint8_t> & data) {
    std::ifstream file(path, std::ios::binary);
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t token_count = 0;
    file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    file.read(reinterpret_cast<char *>(&token_count), sizeof(token_count));
    if (!file || magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION ||
            token_count != expected_tokens) {
        return false;
    }
    file.seekg((std::streamoff) token_count*sizeof(llama_token), std::ios::cur);
    file.read(reinterpret_cast<char *>(data.data()), data.size());
    const bool result = file && file.peek() == std::ifstream::traits_type::eof();
    file.close();
    server_prompt_release_file_pages(path, false);
    return result;
}

bool server_prompt_cache::save(server_prompt & prompt, int32_t id_slot) {
    if (rewind_to_checkpoint_on_save) {
        const auto checkpoint = std::find_if(prompt.checkpoints.rbegin(), prompt.checkpoints.rend(),
                [&](const server_prompt_checkpoint & cur) {
                    return cur.n_tokens < prompt.tokens.n_tokens() && !cur.state_file.empty();
                });
        if (checkpoint == prompt.checkpoints.rend()) {
            LLAMA_LOG_WARN("hybrid prompt cache has no self-consistent staged tail state; "
                    "skipping this cache entry\n");
            return false;
        }

        std::string staged_file = checkpoint->state_file;
        const size_t staged_file_size = checkpoint->state_file_size;
        prompt.state_n_tokens = checkpoint->n_tokens;
        prompt.state_pos_max_prompt = checkpoint->pos_max_prompt;
        checkpoint->state_file.clear();
        checkpoint->state_file_size = 0;
        for (const auto & cur : prompt.checkpoints) {
            if (!cur.state_file.empty()) {
                std::error_code ec;
                std::filesystem::remove(cur.state_file, ec);
            }
        }
        std::list<server_prompt_checkpoint>().swap(prompt.checkpoints);

        const size_t header_size = 3*sizeof(uint32_t) +
            (size_t) prompt.state_n_tokens*sizeof(llama_token);
        if (staged_file_size <= header_size) {
            std::error_code ec;
            std::filesystem::remove(staged_file, ec);
            return false;
        }
        const size_t state_size = staged_file_size - header_size;
        make_ram_room(state_size);
        const bool state_fits_ram = ram_enabled &&
            (limit_ram_size == 0 || ram_size() + state_size <= limit_ram_size);

        server_prompt * cur = nullptr;
        if (state_fits_ram) {
            try {
                cur = alloc(prompt, state_size);
            } catch (const std::bad_alloc &) {
                cur = nullptr;
            }
            if (cur != nullptr && server_prompt_read_state_payload(staged_file,
                        (uint32_t) prompt.state_n_tokens, cur->data)) {
                std::error_code ec;
                std::filesystem::remove(staged_file, ec);
                LLAMA_LOG_INFO(" - promoted staged hybrid prompt cache to RAM: %.3f MiB\n",
                        state_size/(1024.0*1024.0));
                update();
                return true;
            }
            if (cur != nullptr) {
                states.pop_back();
            }
        }

        cur = alloc(prompt, 0);
        if (cur == nullptr) {
            std::error_code ec;
            std::filesystem::remove(staged_file, ec);
            return false;
        }
        cur->state_file = std::move(staged_file);
        cur->state_file_size = staged_file_size;
        LLAMA_LOG_INFO(" - adopted staged hybrid prompt cache on disk: %s, %.3f MiB\n",
                cur->state_file.c_str(), cur->disk_size()/(1024.0*1024.0));
        update();
        return true;
    }
    if (prompt.state_n_tokens < 0) {
        prompt.state_n_tokens = prompt.tokens.n_tokens();
        prompt.state_pos_max_prompt = prompt.tokens.pos_next() - 1;
    }

    const size_t state_size = llama_state_seq_get_size(ctx, id_slot, 0);
    if (state_size == 0) {
        LLAMA_LOG_WARN("failed to query prompt-cache state size for sequence %d\n", id_slot);
        return false;
    }

    make_ram_room(state_size + prompt.ram_size());
    const bool state_fits_ram = ram_enabled &&
        (limit_ram_size == 0 || state_size + prompt.ram_size() <= limit_ram_size);

    if (!state_fits_ram && has_disk_tier()) {
        server_prompt * cur = alloc(prompt, 0);
        if (cur == nullptr) {
            return false;
        }
        if (!save_to_disk(*cur, id_slot)) {
            prompt.checkpoints = std::move(cur->checkpoints);
            states.pop_back();
            return false;
        }
        update();
        return true;
    }

    if (!ram_enabled) {
        LLAMA_LOG_WARN("prompt cache has no enabled storage tier\n");
        return false;
    }

    server_prompt * cur = nullptr;
    try {
        cur = alloc(prompt, state_size);
    } catch (const std::bad_alloc & e) {
        LLAMA_LOG_WARN("failed to allocate %.3f MiB for RAM prompt cache: %s\n",
                state_size / (1024.0 * 1024.0), e.what());
        if (!has_disk_tier()) {
            return false;
        }
        cur = alloc(prompt, 0);
        if (cur == nullptr || !save_to_disk(*cur, id_slot)) {
            if (cur != nullptr) {
                prompt.checkpoints = std::move(cur->checkpoints);
                states.pop_back();
            }
            return false;
        }
        update();
        return true;
    }
    if (cur == nullptr) {
        return false;
    }

    const size_t n = llama_state_seq_get_data(ctx, cur->data.data(), state_size, id_slot, 0);
    if (n != state_size) {
        LLAMA_LOG_WARN("failed to save RAM prompt-cache state: expected %zu bytes, wrote %zu\n", state_size, n);
        prompt.checkpoints = std::move(cur->checkpoints);
        states.pop_back();
        return false;
    }
    update();
    return true;
}

void server_prompt_cache::update() {
    make_ram_room(0);

    if (has_disk_tier() && limit_disk_size > 0) {
        while (disk_size() > limit_disk_size) {
            auto it = std::find_if(states.begin(), states.end(), [](const server_prompt & state) {
                return !state.state_file.empty();
            });
            if (it == states.end()) {
                break;
            }
            LLAMA_LOG_INFO(" - disk prompt-cache limit reached, removing oldest spilled entry (%.3f MiB)\n",
                    it->disk_size() / (1024.0 * 1024.0));
            server_prompt_remove_all_files(*it);
            states.erase(it);
        }
    }

    const float size_per_token = std::max<float>(1.0f, float(size()) / std::max<size_t>(1, n_tokens()));
    const size_t combined_limit = limit_ram_size + limit_disk_size;
    const size_t limit_tokens_cur = combined_limit > 0
        ? std::max<size_t>(limit_tokens, combined_limit / size_per_token)
        : limit_tokens;

    LLAMA_LOG_INFO(" - tiered prompt cache: %zu prompts, RAM %.3f/%.3f MiB, disk %.3f/%.3f MiB (%zu token limit, %zu est)\n",
        states.size(), ram_size() / (1024.0 * 1024.0), limit_ram_size / (1024.0 * 1024.0),
        disk_size() / (1024.0 * 1024.0), limit_disk_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto& state : states) {
        LLAMA_LOG_INFO("   - prompt %p: %7d tokens, %7d discarded, checkpoints: %2zu, RAM %9.3f MiB, disk %9.3f MiB\n",
            (const void*)&state, state.n_tokens(), state.n_discarded_prompt, state.checkpoints.size(),
            state.ram_size() / (1024.0 * 1024.0), state.disk_size() / (1024.0 * 1024.0));
    }
}

void server_prompt_cache::clear() {
    for (auto & state : states) {
        server_prompt_remove_all_files(state);
    }
    states.clear();
}
