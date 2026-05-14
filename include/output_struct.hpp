#include <optional>
#include <string>

struct OutputReport {
    int object_id;
    std::optional<float> temp;
    std::string direction;
    std::optional<float> distance;
};