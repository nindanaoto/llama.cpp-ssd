#include "llama.h"
#include "llama-ssd.h"

#include "ggml.h"
#include "gguf.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

static size_t parse_size_bytes(const std::string & value) {
    if (value.empty()) {
        throw std::invalid_argument("size must not be empty");
    }

    char * end = nullptr;
    errno = 0;
    unsigned long long n = std::strtoull(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str()) {
        throw std::invalid_argument("invalid size: " + value);
    }

    size_t mul = 1;
    if (*end != '\0') {
        char suffix = (char)std::tolower((unsigned char)*end++);
        if (*end != '\0') {
            throw std::invalid_argument("invalid size suffix: " + value);
        }
        switch (suffix) {
            case 'k': mul = 1024ULL; break;
            case 'm': mul = 1024ULL * 1024ULL; break;
            case 'g': mul = 1024ULL * 1024ULL * 1024ULL; break;
            default:
                throw std::invalid_argument("invalid size suffix: " + value);
        }
    }

    if (n > SIZE_MAX / mul) {
        throw std::invalid_argument("size is too large: " + value);
    }
    return (size_t)n * mul;
}

static size_t align_up_size(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static std::vector<std::string> split_csv(const std::string & value) {
    std::vector<std::string> result;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}

static std::string path_join(const std::string & dir, const std::string & name) {
    if (dir.empty() || dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

static std::string path_basename(const std::string & path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

static std::string default_stripe_name_for_model(const std::string & path) {
    std::string name = path_basename(path);
    const std::string suffix = ".gguf";
    if (name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        name.resize(name.size() - suffix.size());
    }
    return name;
}

struct file_stat {
    size_t size = 0;
    int64_t mtime = 0;
};

static file_stat stat_file(const std::string & path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        throw std::runtime_error("failed to stat file: " + path);
    }
    return { (size_t) st.st_size, (int64_t) st.st_mtime };
}

static std::vector<std::string> get_split_paths(const std::string & first_path, gguf_context * ctx) {
    int n_split = 1;
    const int split_count_key = gguf_find_key(ctx, "split.count");
    if (split_count_key >= 0) {
        n_split = gguf_get_val_u16(ctx, split_count_key);
    }
    if (n_split <= 1) {
        return { first_path };
    }

    int split_no = 0;
    const int split_no_key = gguf_find_key(ctx, "split.no");
    if (split_no_key >= 0) {
        split_no = gguf_get_val_u16(ctx, split_no_key);
    }
    if (split_no != 0) {
        throw std::runtime_error("--model must point to the first GGUF split");
    }

    std::vector<char> prefix(first_path.size() + 64);
    int ret = llama_split_prefix(prefix.data(), prefix.size(), first_path.c_str(), split_no, n_split);
    if (ret <= 0) {
        throw std::runtime_error("failed to derive split prefix from: " + first_path);
    }

    std::vector<std::string> result;
    result.reserve(n_split);
    for (int i = 0; i < n_split; ++i) {
        std::vector<char> split_path(first_path.size() + 64);
        ret = llama_split_path(split_path.data(), split_path.size(), prefix.data(), i, n_split);
        if (ret <= 0) {
            throw std::runtime_error("failed to derive split path");
        }
        result.emplace_back(split_path.data());
    }
    return result;
}

struct params {
    std::string model;
    std::string out_dirs_arg;
    std::string name;
    size_t chunk_size = 1024 * 1024;
};

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s --model FNAME --out-dirs DIRS [--name NAME] [--chunk-size N]\n"
        "\n"
        "  --model FNAME       first GGUF split path\n"
        "  --out-dirs DIRS     comma-separated SSD stripe directories\n"
        "  --name NAME         sidecar name (default: model filename without .gguf)\n"
        "  --chunk-size N      chunk size, accepts K/M/G suffixes (default: 1M)\n",
        argv0);
}

static params parse_args(int argc, char ** argv) {
    params p;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const char * opt) -> std::string {
            if (++i >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + opt);
            }
            return argv[i];
        };

        if (arg == "--model") {
            p.model = need_value("--model");
        } else if (arg == "--out-dirs") {
            p.out_dirs_arg = need_value("--out-dirs");
        } else if (arg == "--name") {
            p.name = need_value("--name");
        } else if (arg == "--chunk-size") {
            p.chunk_size = parse_size_bytes(need_value("--chunk-size"));
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (p.model.empty()) {
        throw std::invalid_argument("--model is required");
    }
    if (p.out_dirs_arg.empty()) {
        throw std::invalid_argument("--out-dirs is required");
    }
    if (p.chunk_size == 0 || p.chunk_size % 4096 != 0) {
        throw std::invalid_argument("--chunk-size must be a non-zero multiple of 4096");
    }
    if (p.name.empty()) {
        p.name = default_stripe_name_for_model(p.model);
    }
    return p;
}

static void pack(const params & p) {
    std::vector<std::string> out_dirs = split_csv(p.out_dirs_arg);
    if (out_dirs.empty()) {
        throw std::runtime_error("no output directories parsed from --out-dirs");
    }

    struct ggml_context * first_meta = nullptr;
    gguf_init_params first_params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &first_meta,
    };
    gguf_context * first_gguf = gguf_init_from_file(p.model.c_str(), first_params);
    if (!first_gguf) {
        throw std::runtime_error("failed to open GGUF: " + p.model);
    }

    std::vector<std::string> split_paths = get_split_paths(p.model, first_gguf);
    gguf_free(first_gguf);

    std::vector<std::string> stripe_paths;
    std::vector<std::ofstream> stripe_outs;
    std::vector<size_t> stripe_offsets(out_dirs.size(), 0);
    stripe_paths.reserve(out_dirs.size());
    stripe_outs.reserve(out_dirs.size());
    for (const auto & dir : out_dirs) {
        std::string path = path_join(dir, p.name + ".stripe");
        stripe_paths.push_back(path);
        stripe_outs.emplace_back(path, std::ios::binary | std::ios::trunc);
        if (!stripe_outs.back()) {
            throw std::runtime_error("failed to open stripe output: " + path);
        }
    }

    nlohmann::ordered_json manifest;
    manifest["version"] = 1;
    manifest["name"] = p.name;
    manifest["chunk_size"] = p.chunk_size;
    manifest["stripe_count"] = out_dirs.size();
    manifest["model_files"] = nlohmann::ordered_json::array();
    manifest["stripe_files"] = nlohmann::ordered_json::array();
    manifest["tensors"] = nlohmann::ordered_json::object();

    for (const auto & path : split_paths) {
        const auto st = stat_file(path);
        manifest["model_files"].push_back({
            {"basename", path_basename(path)},
            {"size", st.size},
            {"mtime", st.mtime},
        });
    }

    std::vector<uint8_t> buffer(p.chunk_size);
    size_t next_stripe = 0;
    size_t n_tensors = 0;
    size_t n_experts = 0;
    size_t bytes_written = 0;

    for (const auto & split_path : split_paths) {
        struct ggml_context * meta = nullptr;
        gguf_init_params gguf_params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };
        gguf_context * ctx = gguf_init_from_file(split_path.c_str(), gguf_params);
        if (!ctx) {
            throw std::runtime_error("failed to open GGUF split: " + split_path);
        }

        std::ifstream input(split_path, std::ios::binary);
        if (!input) {
            gguf_free(ctx);
            throw std::runtime_error("failed to open GGUF data: " + split_path);
        }

        for (ggml_tensor * t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t)) {
            const char * name = ggml_get_name(t);
            if (!llama_ssd_manager::is_expert_weight(name)) {
                continue;
            }

            const int tensor_idx = gguf_find_tensor(ctx, name);
            if (tensor_idx < 0) {
                gguf_free(ctx);
                throw std::runtime_error(std::string("tensor not found in GGUF metadata: ") + name);
            }

            const size_t tensor_offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, tensor_idx);
            const int n_expert = (int)t->ne[2];
            const size_t slice_size = t->nb[2];

            nlohmann::ordered_json jt;
            jt["n_expert"] = n_expert;
            jt["slice_size"] = slice_size;
            jt["experts"] = nlohmann::ordered_json::array();

            for (int e = 0; e < n_expert; ++e) {
                nlohmann::ordered_json je = nlohmann::ordered_json::array();
                size_t remaining = slice_size;
                size_t src_offset = tensor_offset + (size_t)e * slice_size;
                size_t dst_offset = 0;

                while (remaining > 0) {
                    const size_t n = std::min(remaining, p.chunk_size);
                    input.clear();
                    input.seekg((std::streamoff)src_offset, std::ios::beg);
                    input.read((char *)buffer.data(), (std::streamsize)n);
                    if ((size_t)input.gcount() != n) {
                        gguf_free(ctx);
                        throw std::runtime_error(std::string("short read from GGUF split: ") + split_path);
                    }

                    const size_t stripe_idx = next_stripe++ % stripe_outs.size();
                    const size_t aligned_offset = align_up_size(stripe_offsets[stripe_idx], 4096);
                    if (aligned_offset != stripe_offsets[stripe_idx]) {
                        static const uint8_t zero_pad[4096] = {};
                        const size_t pad = aligned_offset - stripe_offsets[stripe_idx];
                        stripe_outs[stripe_idx].write((const char *)zero_pad, (std::streamsize)pad);
                        if (!stripe_outs[stripe_idx]) {
                            gguf_free(ctx);
                            throw std::runtime_error("failed writing stripe padding: " + stripe_paths[stripe_idx]);
                        }
                        stripe_offsets[stripe_idx] = aligned_offset;
                    }
                    const size_t out_offset = stripe_offsets[stripe_idx];
                    stripe_outs[stripe_idx].write((const char *)buffer.data(), (std::streamsize)n);
                    if (!stripe_outs[stripe_idx]) {
                        gguf_free(ctx);
                        throw std::runtime_error("failed writing stripe output: " + stripe_paths[stripe_idx]);
                    }
                    stripe_offsets[stripe_idx] += n;

                    je.push_back({
                        {"stripe", stripe_idx},
                        {"file_offset", out_offset},
                        {"dst_offset", dst_offset},
                        {"size", n},
                    });

                    remaining -= n;
                    src_offset += n;
                    dst_offset += n;
                    bytes_written += n;
                }

                jt["experts"].push_back(std::move(je));
                n_experts++;
            }

            manifest["tensors"][std::string(name)] = std::move(jt);
            n_tensors++;
        }

        gguf_free(ctx);
    }

    for (auto & out : stripe_outs) {
        out.flush();
        out.close();
    }
    for (const auto & path : stripe_paths) {
        const auto st = stat_file(path);
        manifest["stripe_files"].push_back({
            {"basename", path_basename(path)},
            {"size", st.size},
        });
    }

    const std::string manifest_text = manifest.dump(2);
    for (const auto & dir : out_dirs) {
        const std::string manifest_path = path_join(dir, p.name + ".manifest.json");
        std::ofstream fout(manifest_path, std::ios::binary | std::ios::trunc);
        if (!fout) {
            throw std::runtime_error("failed to write manifest: " + manifest_path);
        }
        fout << manifest_text << "\n";
    }

    fprintf(stderr, "packed %zu tensors, %zu experts, %.2f GiB into %zu stripes\n",
        n_tensors, n_experts, (double)bytes_written / (1024.0 * 1024.0 * 1024.0), out_dirs.size());
}

int main(int argc, char ** argv) {
    try {
        pack(parse_args(argc, argv));
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n\n", e.what());
        print_usage(argv[0]);
        return 1;
    }
}
