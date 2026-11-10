// Image input on the chat routes: the DeepSeek renderer's placeholders become spans of the image token sized by the
// engine's prepare hook, in prompt order; OpenAI image_url, Anthropic image blocks (user content and tool results) and
// Responses input_image all reach the engine as PromptImage entries; limits and malformed inputs are refused with 400;
// without image support the parts are refused as before.
#include "strix/serve/api.hpp"
#include "strix/serve/deepseek_chat.hpp"
#include "strix/serve/json.hpp"

#include <algorithm>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

using namespace strix::serve;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

std::shared_ptr<const BpeTokenizer> byte_tokenizer() {
    std::vector<std::string> tokens;
    std::vector<std::int32_t> types;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        int cp = b;
        const bool direct = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        if (!direct) cp = 256 + extra++;
        std::string s;
        if (cp < 0x80) s.push_back(static_cast<char>(cp));
        else {
            s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        tokens.push_back(s);
        types.push_back(1);
    }
    tokens.push_back("<|eos|>");
    types.push_back(3);
    return BpeTokenizer::from_parts(std::move(tokens), std::move(types), {}, "qwen35");
}

class RecordingEngine final : public GenerationEngine {
public:
    explicit RecordingEngine(std::uint32_t eos) : eos_(eos) {}
    GenerationStats generate(GenerationJob& job) override {
        {
            const std::lock_guard lock(mu);
            prompts.push_back(job.prompt);
            images.push_back(job.images);
        }
        const std::uint32_t t[1] = {eos_};
        job.on_tokens(t);
        GenerationStats s;
        s.finish = "stop";
        return s;
    }
    std::uint64_t context_tokens() const override { return 100000; }
    std::mutex mu;
    std::vector<std::vector<std::uint32_t>> prompts;
    std::vector<std::vector<PromptImage>> images;

private:
    std::uint32_t eos_;
};

struct Writer final : HttpResponseWriter {
    int status = 0;
    std::string body;
    void send(int s, std::string_view, std::string_view b, const std::vector<std::pair<std::string, std::string>>&) override {
        status = s;
        body = b;
    }
    bool begin_stream(int s, std::string_view, const std::vector<std::pair<std::string, std::string>>&) override {
        status = s;
        return true;
    }
    bool write(std::string_view d) override {
        body += d;
        return true;
    }
    void end_stream() override {}
    bool client_gone() const override { return false; }
};

Writer post(ApiService& api, const std::string& path, const std::string& body) {
    HttpRequest r;
    r.method = "POST";
    r.target = r.path = path;
    r.version = "HTTP/1.1";
    r.body = body;
    r.headers.push_back({"content-type", "application/json"});
    Writer w;
    api.handle(r, w);
    return w;
}

constexpr std::uint32_t kImageToken = 300;
const std::string kPlaceholder = "<\xEF\xBD\x9C" "deepseek_image" "\xEF\xBD\x9C>";

// base64 data URL of `n` bytes of value v (the fake prepare reads the picture's size from its byte count)
std::string data_url(std::size_t n, char v) {
    static const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string raw(n, v), out;
    for (std::size_t i = 0; i < raw.size(); i += 3) {
        const auto b0 = static_cast<unsigned char>(raw[i]);
        const auto b1 = i + 1 < raw.size() ? static_cast<unsigned char>(raw[i + 1]) : 0U;
        const auto b2 = i + 2 < raw.size() ? static_cast<unsigned char>(raw[i + 2]) : 0U;
        const unsigned v24 = (b0 << 16U) | (b1 << 8U) | b2;
        out += alphabet[(v24 >> 18U) & 63U];
        out += alphabet[(v24 >> 12U) & 63U];
        out += i + 1 < raw.size() ? alphabet[(v24 >> 6U) & 63U] : '=';
        out += i + 2 < raw.size() ? alphabet[v24 & 63U] : '=';
    }
    return "data:image/png;base64," + out;
}
std::string b64_of(const std::string& url) { return url.substr(url.find(',') + 1U); }

}  // namespace

int main() try {
    const auto tok = byte_tokenizer();
    RecordingEngine engine(tok->find("<|eos|>"));
    GenerationQueue queue;
    std::thread worker([&] { queue.run(engine); });
    ApiConfig config;
    config.model_id = "m";
    config.chat_renderer = render_deepseek_v41_chat;
    config.output_parser = parse_deepseek_v41_output;
    config.stop_tokens = {tok->find("<|eos|>")};
    ApiConfig plain_config = config;
    config.image_input = ApiConfig::ImageInput{};
    config.image_input->placeholder = kPlaceholder;
    config.image_input->token_id = kImageToken;
    config.image_input->max_images = 3;
    config.image_input->max_tokens = 40;
    config.image_input->prepare = [](std::span<const std::uint8_t> bytes) {
        if (bytes.empty() || bytes[0] == 'X') throw std::runtime_error("image input: cannot identify image file");
        PromptImage p;
        p.tokens = static_cast<std::uint32_t>(bytes.size());
        p.digest[0] = bytes[0];
        return p;
    };
    ApiService api(config, tok, queue, 100000, false);
    ApiService plain(plain_config, tok, queue, 100000, false);
    auto session_config = config;
    session_config.sessions_enabled = true;
    ApiService sessions(session_config, tok, queue, 100000, false);

    // chat_prompt: spans at the placeholders, sized by prepare, in prompt order, text around them as the plain encode
    {
        const auto body = parse_json(R"({"messages":[{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":")" +
                                     data_url(5, 'a') + R"("}},{"type":"text","text":"and"},{"type":"image_url","image_url":")" + data_url(7, 'b') +
                                     R"("}]}],"enable_thinking":false})");
        const auto ids = api.chat_prompt(body);
        ChatTemplateInput in;
        in.messages = *body.find("messages");
        in.enable_thinking = false;
        std::vector<Json> parts;
        in.images = &parts;
        const auto text = render_deepseek_v41_chat(in);
        const auto p1 = text.find(kPlaceholder), p2 = text.find(kPlaceholder, p1 + 1U);
        std::vector<std::uint32_t> want = tok->encode(text.substr(0, p1));
        want.insert(want.end(), 5, kImageToken);
        const auto mid = tok->encode(text.substr(p1 + kPlaceholder.size(), p2 - p1 - kPlaceholder.size()));
        want.insert(want.end(), mid.begin(), mid.end());
        want.insert(want.end(), 7, kImageToken);
        const auto tail = tok->encode(text.substr(p2 + kPlaceholder.size()));
        want.insert(want.end(), tail.begin(), tail.end());
        check(ids == want, "chat_prompt expands each placeholder to its span");
    }
    // OpenAI chat route: the job carries the images with their spans
    {
        const auto w = post(api, "/v1/chat/completions",
                            R"({"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":")" + data_url(4, 'c') +
                                R"("}},{"type":"text","text":"what is it?"}]}],"max_tokens":4})");
        check(w.status == 200, "chat with an image: 200 (" + std::to_string(w.status) + " " + w.body.substr(0, 200) + ")");
        const std::lock_guard lock(engine.mu);
        const auto& imgs = engine.images.back();
        const auto& prompt = engine.prompts.back();
        check(imgs.size() == 1U && imgs[0].tokens == 4U && imgs[0].digest[0] == 'c' && imgs[0].encoded && imgs[0].encoded->size() == 4U, "chat job image");
        if (!imgs.empty())
            check(imgs[0].start + 4U <= prompt.size() &&
                      std::all_of(prompt.begin() + imgs[0].start, prompt.begin() + imgs[0].start + 4, [](std::uint32_t t) { return t == kImageToken; }) &&
                      std::count(prompt.begin(), prompt.end(), kImageToken) == 4,
                  "chat job span tokens");
    }
    // Anthropic: image in user content and in a tool result, prompt order kept (tool result, then user picture)
    {
        const auto w = post(api, "/v1/messages", R"({"model":"m","max_tokens":4,"messages":[
            {"role":"user","content":"take a screenshot"},
            {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"shot","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"done"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":")" +
                                                            b64_of(data_url(6, 'd')) + R"("}}]},
                                      {"type":"text","text":"and this"},{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":")" +
                                                            b64_of(data_url(3, 'e')) + R"("}}]}]})");
        check(w.status == 200, "anthropic with images: 200 (" + std::to_string(w.status) + " " + w.body.substr(0, 200) + ")");
        const std::lock_guard lock(engine.mu);
        const auto& imgs = engine.images.back();
        check(imgs.size() == 2U && imgs[0].digest[0] == 'd' && imgs[1].digest[0] == 'e' && imgs[0].start < imgs[1].start, "anthropic images in order");
    }
    // Responses input_image
    {
        const auto w = post(api, "/v1/responses", R"({"model":"m","max_output_tokens":4,"input":[{"role":"user","content":[{"type":"input_text","text":"hi"},{"type":"input_image","image_url":")" +
                                                      data_url(2, 'f') + R"("}]}]})");
        check(w.status == 200, "responses with an image: 200 (" + std::to_string(w.status) + " " + w.body.substr(0, 200) + ")");
        const std::lock_guard lock(engine.mu);
        check(engine.images.back().size() == 1U && engine.images.back()[0].digest[0] == 'f', "responses image");
    }
    // refusals: count, token budget, undecodable picture, non-data URL, URL source, image support off
    const auto refused = [&](ApiService& a, const std::string& path, const std::string& body, const std::string& needle, const std::string& what) {
        const auto w = post(a, path, body);
        check(w.status == 400 && w.body.find(needle) != std::string::npos, what + " (" + std::to_string(w.status) + " " + w.body.substr(0, 160) + ")");
    };
    const std::string img1 = R"({"type":"image_url","image_url":")" + data_url(2, 'g') + R"("})";
    refused(api, "/v1/chat/completions", R"({"model":"m","messages":[{"role":"user","content":[)" + img1 + "," + img1 + "," + img1 + "," + img1 + R"(]}]})",
            "at most 3 pictures", "too many pictures");
    refused(api, "/v1/chat/completions",
            R"({"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":")" + data_url(41, 'h') + R"("}]}]})", "tokens",
            "token budget");
    refused(api, "/v1/chat/completions",
            R"({"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":")" + data_url(3, 'X') + R"("}]}]})", "cannot identify",
            "undecodable picture");
    refused(api, "/v1/chat/completions", R"({"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":"https://x/y.png"}]}]})",
            "data:", "https URL");
    refused(api, "/v1/messages",
            R"({"model":"m","max_tokens":4,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://x"}}]}]})", "base64",
            "anthropic URL source");
    refused(plain, "/v1/chat/completions", R"({"model":"m","messages":[{"role":"user","content":[)" + img1 + R"(]}]})", "not supported",
            "image support off");
    refused(sessions, "/v1/chat/completions", R"({"model":"m","session":"s1","messages":[{"role":"user","content":[)" + img1 + R"(]}]})", "image input",
            "session + image");
    // capabilities advertise image input
    {
        HttpRequest r;
        r.method = "GET";
        r.target = r.path = "/v1/capabilities";
        Writer w;
        api.handle(r, w);
        Writer w2;
        plain.handle(r, w2);
        check(w.body.find("\"image_input\":true") != std::string::npos && w2.body.find("\"image_input\":false") != std::string::npos,
              "capabilities image_input (" + w.body.substr(0, 120) + ")");
    }
    queue.shutdown();
    worker.join();
    if (failures) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "serve image input tests passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
}
