#pragma once

#include <cstdint>
#include <string>
#include <string_view>

inline constexpr const char* COCO_CLASSES[80] = {
    "person",        "bicycle",       "car",           "motorcycle",
    "airplane",      "bus",           "train",         "truck",
    "boat",          "traffic light", "fire hydrant",  "stop sign",
    "parking meter", "bench",         "bird",          "cat",
    "dog",           "horse",         "sheep",         "cow",
    "elephant",      "bear",          "zebra",         "giraffe",
    "backpack",      "umbrella",      "handbag",       "tie",
    "suitcase",      "frisbee",       "skis",          "snowboard",
    "sports ball",   "kite",          "baseball bat",  "baseball glove",
    "skateboard",    "surfboard",     "tennis racket", "bottle",
    "wine glass",    "cup",           "fork",          "knife",
    "spoon",         "bowl",          "banana",        "apple",
    "sandwich",      "orange",        "broccoli",      "carrot",
    "hot dog",       "pizza",         "donut",         "cake",
    "chair",         "couch",         "potted plant",  "bed",
    "dining table",  "toilet",        "tv",            "laptop",
    "mouse",         "remote",        "keyboard",      "cell phone",
    "microwave",     "oven",          "toaster",       "sink",
    "refrigerator",  "book",          "clock",         "vase",
    "scissors",      "teddy bear",    "hair drier",    "toothbrush"
};

inline uint8_t coco_id_for_word(const std::string& word)
{
    for (int i = 0; i < 80; ++i) {
        if (word == COCO_CLASSES[i]) {
            return static_cast<uint8_t>(i);
        }
    }

    return 255;
}

inline const char* coco_word_for_id(uint8_t id)
{
    if (id < 80) {
        return COCO_CLASSES[id];
    }

    return "object";
}