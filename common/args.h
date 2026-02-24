#pragma once

#include <charconv>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

// All utilities live in the cuda_learning namespace to avoid polluting global scope.
namespace cuda_learning {

// ---------------------------------------------------------------------------
// ArgParser — lightweight C++17 command-line argument parser
//
// Supported syntax (both forms are accepted):
//   --key value
//   --key=value
//
// Usage:
//   ArgParser args;
//   args.add("N",      "16777216", "number of elements")
//       .add("block",  "256",      "thread block size")
//       .add("device", "0",        "CUDA device id");
//
//   if (!args.parse(argc, argv)) return 0;   // --help was requested
//
//   int  N      = args.get<int>("N");
//   int  block  = args.get<int>("block");
//   int  device = args.get<int>("device");
// ---------------------------------------------------------------------------

class ArgParser {
   public:
    /// Register an argument with a default value and optional description.
    ArgParser& add(std::string key, std::string default_val,
                   std::string description = "") {
        defaults_[key] = std::move(default_val);
        descriptions_[key] = std::move(description);
        return *this;
    }

    /// Parse argv.  Returns false when --help / -h is requested (caller should exit).
    bool parse(int argc, char** argv) {
        prog_name_ = argv[0];
        values_ = defaults_;  // start from defaults

        for (int i = 1; i < argc; ++i) {
            std::string_view arg = argv[i];

            if (arg == "--help" || arg == "-h") {
                print_help();
                return false;
            }

            // Accept --key=value
            if (arg.rfind("--", 0) == 0) {
                const auto eq = arg.find('=');
                if (eq != std::string_view::npos) {
                    values_[std::string(arg.substr(2, eq - 2))] =
                        std::string(arg.substr(eq + 1));
                } else if (i + 1 < argc) {
                    // Accept --key value
                    values_[std::string(arg.substr(2))] = argv[++i];
                }
            }
        }
        return true;
    }

    /// Retrieve a parsed value, converting it to type T.
    /// Supported T: std::string, integral types, floating-point types.
    template <typename T>
    T get(const std::string& key) const {
        const auto it = values_.find(key);
        if (it == values_.end()) {
            throw std::runtime_error("ArgParser: unknown key \"" + key + "\"");
        }
        const std::string& s = it->second;

        if constexpr (std::is_same_v<T, std::string>) {
            return s;
        } else if constexpr (std::is_integral_v<T>) {
            T val{};
            const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), val);
            if (ec != std::errc{}) {
                throw std::runtime_error("ArgParser: cannot convert \"" + s +
                                         "\" to integer for key \"" + key + "\"");
            }
            return val;
        } else if constexpr (std::is_floating_point_v<T>) {
            return static_cast<T>(std::stod(s));
        }
    }

    void print_help() const {
        std::cout << "Usage: " << prog_name_ << " [--key value | --key=value] ...\n\n";
        std::cout << "Options:\n";
        for (const auto& [k, desc] : descriptions_) {
            std::cout << "  --" << k;
            const auto& def = defaults_.at(k);
            if (!def.empty()) std::cout << " (default: " << def << ")";
            if (!desc.empty()) std::cout << "  " << desc;
            std::cout << "\n";
        }
        std::cout << "  --help, -h   Show this message\n";
    }

   private:
    std::string prog_name_;
    std::unordered_map<std::string, std::string> defaults_;
    std::unordered_map<std::string, std::string> descriptions_;
    std::unordered_map<std::string, std::string> values_;
};

}  // namespace cuda_learning
