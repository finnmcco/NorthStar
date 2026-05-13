#pragma once

#include <cstdint>
#include <string>

// Maps a plain English COCO class name to its model index (0-79).
// Returns 255 if the word is not a recognised COCO class name.
//
// Used by main.cpp to build Pipeline::Config::object_ids from command-line
// words, and by test_pipeline_logic to verify the mapping table.

inline uint8_t coco_id_for_word(const std::string& word) {
    static const char* const CLASSES[80] = {
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
    for (int i = 0; i < 80; ++i)
        if (word == CLASSES[i]) return static_cast<uint8_t>(i);
    return 255;
}
