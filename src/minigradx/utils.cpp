#include <minigradx/utils.hpp>
#include <cctype>

std::string to_snake_case(const std::string& input) {
    std::string out;
    out.reserve(input.size() * 2);

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        if (std::isupper(static_cast<unsigned char>(c))) {
            if (i > 0) {
                out.push_back('_');
            }
            out.push_back(static_cast<char>(std::tolower(c)));
        } else {
            out.push_back(c);
        }
    }
    return out;
}