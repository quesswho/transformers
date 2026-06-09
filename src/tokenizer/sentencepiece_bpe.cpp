#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <iostream>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

// ▁ (U+2581) as a hex literal — avoids MSVC source-encoding issues.
static const std::string SPIECE_UNDERLINE = "\xe2\x96\x81";
static const std::string UNK_TOKEN = "<unk>";
static const std::string BOS_TOKEN = "<s>";
static const std::string EOS_TOKEN = "</s>";
static const int UNK_ID = 0;
static const int NUM_SPECIAL = 3;

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

// Returns each Unicode scalar value as its own std::string (raw UTF-8 bytes).
static std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> result;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        int len;
        if      ((c & 0x80) == 0x00) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        else                          len = 1; // invalid lead byte — emit as-is
        if (i + len > s.size()) len = static_cast<int>(s.size() - i);
        result.push_back(s.substr(i, len));
        i += len;
    }
    return result;
}

// Split a string by a multi-byte delimiter (exact byte match).
static std::vector<std::string> split_by(const std::string& s, const std::string& delim) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t pos;
    while ((pos = s.find(delim, start)) != std::string::npos) {
        parts.push_back(s.substr(start, pos - start));
        start = pos + delim.size();
    }
    parts.push_back(s.substr(start));
    return parts;
}

// Replace every occurrence of `from` inside `s` with `to`.
static std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

// Flatten a word (vector of UTF-8 char strings) to a single key using
// null-byte separators — null bytes cannot appear in valid UTF-8 text.
static std::string word_key(const std::vector<std::string>& word) {
    std::string key;
    for (size_t i = 0; i < word.size(); ++i) {
        if (i) key += '\0';
        key += word[i];
    }
    return key;
}

// ---------------------------------------------------------------------------
// Pair hash — Boost-style combine avoids XOR symmetry collision.
// ---------------------------------------------------------------------------

using Pair = std::pair<std::string, std::string>;

struct PairHash {
    std::size_t operator()(const Pair& p) const {
        std::size_t h1 = std::hash<std::string>{}(p.first);
        std::size_t h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9u + (h1 << 6) + (h1 >> 2));
    }
};

using PairMap   = std::unordered_map<Pair, int, PairHash>;
using PairToSet = std::unordered_map<Pair, std::unordered_set<int>, PairHash>;

// ---------------------------------------------------------------------------
// SentencePieceBPE
// ---------------------------------------------------------------------------

class SentencePieceBPE {
public:
    // ------------------------------------------------------------------
    // Training
    // ------------------------------------------------------------------

    void train(const std::string& text, int vocab_size) {
        encode_cache_.clear();

        // Build word frequency table — stream one word at a time so we never
        // materialise the full O(total_words) vector for large corpora.
        std::unordered_map<std::string, int> word_freq;
        for_each_word(text, [&](const std::vector<std::string>& word) {
            word_freq[word_key(word)]++;
        });

        // Collect unique words and their frequencies.
        std::vector<std::vector<std::string>> words;
        std::vector<int> freqs;
        for (auto& [k, f] : word_freq) {
            words.push_back(key_to_word(k));
            freqs.push_back(f);
        }

        // Initial vocab: specials + sorted unique characters.
        vocab_.clear();
        vocab_[UNK_TOKEN] = 0;
        vocab_[BOS_TOKEN] = 1;
        vocab_[EOS_TOKEN] = 2;

        std::unordered_set<std::string> seen_chars;
        for (auto& w : words)
            for (auto& ch : w)
                seen_chars.insert(ch);

        std::vector<std::string> chars(seen_chars.begin(), seen_chars.end());
        std::sort(chars.begin(), chars.end());
        for (auto& ch : chars)
            if (!vocab_.count(ch))
                vocab_[ch] = static_cast<int>(vocab_.size());

        std::cout << "Initial vocab size: " << vocab_.size()
                  << " (including " << NUM_SPECIAL << " special tokens)\n"
                  << std::flush;

        // Build pair_counts and pair_to_words.
        PairMap   pair_counts;
        PairToSet pair_to_words;

        for (int wid = 0; wid < static_cast<int>(words.size()); ++wid) {
            const auto& word = words[wid];
            int freq = freqs[wid];
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                Pair p{word[i], word[i + 1]};
                pair_counts[p] += freq;
                pair_to_words[p].insert(wid);
            }
        }

        // Max-heap via negated counts: min-heap of (-count, pair).
        using HeapEntry = std::pair<int, Pair>;
        std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;
        for (auto& [p, cnt] : pair_counts)
            heap.push({-cnt, p});

        merges_.clear();
        int num_merges = vocab_size - static_cast<int>(vocab_.size());

        for (int step = 0; step < num_merges; ++step) {
            // Lazy-pop: find a heap entry whose stored count is still current.
            Pair best_pair;
            bool found = false;
            while (!heap.empty()) {
                auto [neg_cnt, candidate] = heap.top();
                heap.pop();
                auto it = pair_counts.find(candidate);
                if (it != pair_counts.end() && it->second == -neg_cnt) {
                    best_pair = candidate;
                    found = true;
                    break;
                }
            }
            if (!found) break;

            std::string new_token = best_pair.first + best_pair.second;
            vocab_[new_token] = static_cast<int>(vocab_.size());
            merges_.push_back(best_pair);

            auto affected_it = pair_to_words.find(best_pair);
            std::unordered_set<int> affected;
            if (affected_it != pair_to_words.end()) {
                affected = std::move(affected_it->second);
                pair_to_words.erase(affected_it);
            }
            pair_counts.erase(best_pair);

            for (int wid : affected) {
                auto& word = words[wid];
                int freq = freqs[wid];

                // Remove old pair contributions.
                for (size_t i = 0; i + 1 < word.size(); ++i) {
                    Pair p{word[i], word[i + 1]};
                    if (p == best_pair) continue;
                    auto it = pair_counts.find(p);
                    if (it != pair_counts.end()) {
                        it->second -= freq;
                        if (it->second <= 0) pair_counts.erase(it);
                    }
                    auto sit = pair_to_words.find(p);
                    if (sit != pair_to_words.end()) {
                        sit->second.erase(wid);
                        if (sit->second.empty()) pair_to_words.erase(sit);
                    }
                }

                // Apply merge in-place.
                std::vector<std::string> new_word;
                new_word.reserve(word.size());
                size_t i = 0;
                while (i < word.size()) {
                    if (i + 1 < word.size() &&
                        word[i] == best_pair.first &&
                        word[i + 1] == best_pair.second) {
                        new_word.push_back(new_token);
                        i += 2;
                    } else {
                        new_word.push_back(word[i]);
                        ++i;
                    }
                }
                word = std::move(new_word);

                // Add new pair contributions.
                for (size_t j = 0; j + 1 < word.size(); ++j) {
                    Pair p{word[j], word[j + 1]};
                    pair_counts[p] += freq;
                    pair_to_words[p].insert(wid);
                    heap.push({-pair_counts[p], p});
                }
            }

            if ((step + 1) % 200 == 0 || step == num_merges - 1)
                std::cout << "  merge " << (step + 1) << "/" << num_merges
                          << "  vocab=" << vocab_.size() << "\n" << std::flush;
        }

        // Build derived lookups.
        rebuild_derived();
    }

    // ------------------------------------------------------------------
    // Encode / Decode
    // ------------------------------------------------------------------

    py::array_t<int32_t> encode(const std::string& text) {
        std::vector<int32_t> ids;
        {
            py::gil_scoped_release release;
            for_each_word(text, [&](const std::vector<std::string>& word) {
                std::string k = word_key(word);
                auto it = encode_cache_.find(k);
                if (it == encode_cache_.end()) {
                    auto merged = merge_word(word);
                    std::vector<int> word_ids;
                    word_ids.reserve(merged.size());
                    for (auto& tok : merged) {
                        auto vit = vocab_.find(tok);
                        word_ids.push_back(vit != vocab_.end() ? vit->second : UNK_ID);
                    }
                    it = encode_cache_.emplace(k, std::move(word_ids)).first;
                }
                for (int id : it->second)
                    ids.push_back(static_cast<int32_t>(id));
            });
        }
        py::array_t<int32_t> arr(static_cast<py::ssize_t>(ids.size()));
        std::copy(ids.begin(), ids.end(), arr.mutable_data());
        return arr;
    }

    std::string decode(const std::vector<int>& ids) {
        std::string result;
        for (int id : ids) {
            auto it = vocab_inv_.find(id);
            result += (it != vocab_inv_.end() ? it->second : UNK_TOKEN);
        }
        result = replace_all(result, SPIECE_UNDERLINE, " ");
        // lstrip leading space
        if (!result.empty() && result[0] == ' ')
            result.erase(result.begin());
        return result;
    }

    // ------------------------------------------------------------------
    // Serialisation (delegates JSON I/O to Python)
    // ------------------------------------------------------------------

    py::dict to_dict() const {
        py::dict vocab_dict;
        for (auto& [tok, id] : vocab_)
            vocab_dict[py::str(tok)] = py::int_(id);

        py::list merges_list;
        for (auto& [a, b] : merges_)
            merges_list.append(py::make_tuple(py::str(a), py::str(b)));

        return py::dict("vocab"_a = vocab_dict, "merges"_a = merges_list);
    }

    static SentencePieceBPE from_dict(py::dict data) {
        SentencePieceBPE tok;
        py::dict vocab_d = data["vocab"].cast<py::dict>();
        for (auto item : vocab_d)
            tok.vocab_[item.first.cast<std::string>()] = item.second.cast<int>();

        py::list merges_l = data["merges"].cast<py::list>();
        for (auto m : merges_l) {
            // json.load gives list-of-lists; py::sequence handles both list and tuple.
            py::sequence seq = m.cast<py::sequence>();
            tok.merges_.push_back({seq[0].cast<std::string>(), seq[1].cast<std::string>()});
        }

        tok.rebuild_derived();
        return tok;
    }

    void save(const std::string& path) const {
        py::module_ json    = py::module_::import("json");
        py::object  open_fn = py::module_::import("builtins").attr("open");
        py::object  f       = open_fn(path, "w", "encoding"_a = "utf-8");
        json.attr("dump")(to_dict(), f, "ensure_ascii"_a = false, "indent"_a = 2);
        f.attr("close")();
    }

    static SentencePieceBPE load(const std::string& path) {
        py::module_ json    = py::module_::import("json");
        py::object  open_fn = py::module_::import("builtins").attr("open");
        py::object  f       = open_fn(path, "r", "encoding"_a = "utf-8");
        py::dict    data    = json.attr("load")(f).cast<py::dict>();
        f.attr("close")();
        return from_dict(data);
    }

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------

    int vocab_size() const { return static_cast<int>(vocab_.size()); }

    std::unordered_map<std::string, int> get_vocab()  const { return vocab_; }
    std::vector<Pair>                    get_merges() const { return merges_; }

private:
    std::unordered_map<std::string, int>  vocab_;
    std::unordered_map<int, std::string>  vocab_inv_;
    std::vector<Pair>                     merges_;
    PairMap                               merge_rank_;
    std::unordered_map<std::string, std::vector<int>> encode_cache_;

    void rebuild_derived() {
        vocab_inv_.clear();
        for (auto& [s, id] : vocab_) vocab_inv_[id] = s;

        merge_rank_.clear();
        for (int r = 0; r < static_cast<int>(merges_.size()); ++r)
            merge_rank_[merges_[r]] = r;
    }

    // ------------------------------------------------------------------
    // Streaming word iterator — O(1) extra memory per word
    // ------------------------------------------------------------------

    // Calls callback(word) for each space-delimited token in text.
    // Non-first tokens are prepended with SPIECE_UNDERLINE, matching the
    // old mark_spaces + split_into_words behaviour without materialising
    // the entire word list.
    template<typename F>
    static void for_each_word(const std::string& text, F callback) {
        size_t n   = text.size();
        size_t pos = 0;
        bool first = true;

        while (true) {
            size_t sp  = text.find(' ', pos);
            size_t end = (sp == std::string::npos) ? n : sp;

            std::vector<std::string> word;
            if (!first) word.push_back(SPIECE_UNDERLINE);

            size_t j = pos;
            while (j < end) {
                unsigned char c = static_cast<unsigned char>(text[j]);
                int len = (c & 0x80) == 0x00 ? 1
                        : (c & 0xE0) == 0xC0 ? 2
                        : (c & 0xF0) == 0xE0 ? 3
                        : (c & 0xF8) == 0xF0 ? 4 : 1;
                if (j + static_cast<size_t>(len) > end)
                    len = static_cast<int>(end - j);
                word.push_back(text.substr(j, len));
                j += len;
            }

            if (!word.empty()) callback(word);

            first = false;
            if (sp == std::string::npos) break;
            pos = sp + 1;
        }
    }

    static std::vector<std::string> key_to_word(const std::string& k) {
        std::vector<std::string> word;
        size_t start = 0;
        size_t pos;
        while ((pos = k.find('\0', start)) != std::string::npos) {
            word.push_back(k.substr(start, pos - start));
            start = pos + 1;
        }
        word.push_back(k.substr(start));
        return word;
    }

    // ------------------------------------------------------------------
    // _merge_word — O(L log L) linked-list approach
    // ------------------------------------------------------------------

    std::vector<std::string> merge_word(const std::vector<std::string>& word) const {
        int n = static_cast<int>(word.size());
        if (n < 2) return word;

        std::vector<std::optional<std::string>> symbols(word.begin(), word.end());
        std::vector<int> prev(n), nxt(n);
        for (int i = 0; i < n; ++i) { prev[i] = i - 1; nxt[i] = i + 1; }

        int total_merges = static_cast<int>(merges_.size());
        // Min-heap on (rank, position).
        using HE = std::pair<int, int>;
        std::priority_queue<HE, std::vector<HE>, std::greater<HE>> heap;

        for (int i = 0; i < n - 1; ++i) {
            auto it = merge_rank_.find({*symbols[i], *symbols[i + 1]});
            if (it != merge_rank_.end())
                heap.push({it->second, i});
        }

        while (!heap.empty()) {
            auto [rank, i] = heap.top();
            heap.pop();

            if (!symbols[i].has_value()) continue;
            int j = nxt[i];
            if (j >= n || !symbols[j].has_value()) continue;

            Pair p{*symbols[i], *symbols[j]};
            auto it = merge_rank_.find(p);
            if (it == merge_rank_.end() || it->second != rank) {
                // Stale — re-push with correct rank if still mergeable.
                if (it != merge_rank_.end() && it->second < total_merges)
                    heap.push({it->second, i});
                continue;
            }

            std::string new_tok = *symbols[i] + *symbols[j];
            symbols[i] = new_tok;
            symbols[j] = std::nullopt;

            // Splice j out of the linked list.
            int nxt_j = nxt[j];
            nxt[i] = nxt_j;
            if (nxt_j < n) prev[nxt_j] = i;

            // Check left neighbour.
            int pi = prev[i];
            if (pi >= 0 && symbols[pi].has_value()) {
                auto lit = merge_rank_.find({*symbols[pi], new_tok});
                if (lit != merge_rank_.end())
                    heap.push({lit->second, pi});
            }
            // Check right neighbour.
            int ni = nxt[i];
            if (ni < n && symbols[ni].has_value()) {
                auto rit = merge_rank_.find({new_tok, *symbols[ni]});
                if (rit != merge_rank_.end())
                    heap.push({rit->second, i});
            }
        }

        std::vector<std::string> result;
        result.reserve(n);
        for (auto& s : symbols)
            if (s.has_value())
                result.push_back(*s);
        return result;
    }
};

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_sentencepiece_bpe, m) {
    m.doc() = "SentencePiece BPE tokenizer (C++ backend)";

    py::class_<SentencePieceBPE>(m, "SentencePieceBPE")
        .def(py::init<>())
        .def("train", &SentencePieceBPE::train,
             py::arg("text"), py::arg("vocab_size"),
             py::call_guard<py::gil_scoped_release>())
        .def("encode",      &SentencePieceBPE::encode,    py::arg("text"))
        .def("decode",      &SentencePieceBPE::decode,    py::arg("ids"))
        .def("save",        &SentencePieceBPE::save,      py::arg("path"))
        .def_static("load", &SentencePieceBPE::load,      py::arg("path"))
        .def("to_dict",     &SentencePieceBPE::to_dict)
        .def_static("from_dict", &SentencePieceBPE::from_dict, py::arg("data"))
        .def_property_readonly("vocab_size", &SentencePieceBPE::vocab_size)
        .def_property_readonly("vocab", [](const SentencePieceBPE& self) {
            py::dict d;
            for (auto& [tok, id] : self.get_vocab())
                d[py::str(tok)] = py::int_(id);
            return d;
        })
        .def_property_readonly("merges", [](const SentencePieceBPE& self) {
            py::list l;
            for (auto& [a, b] : self.get_merges())
                l.append(py::make_tuple(py::str(a), py::str(b)));
            return l;
        })
        .def_property_readonly_static("SPECIAL_TOKENS", [](py::object) {
            return py::make_tuple(
                py::str("<unk>"), py::str("<s>"), py::str("</s>"));
        });
}
