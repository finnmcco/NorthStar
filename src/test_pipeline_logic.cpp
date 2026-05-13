/*
    test_pipeline_logic.cpp
    ════════════════════════
    Pure-logic unit tests for the Pipeline word→object_id mapping and callback
    chain.  No ALSA, no Vosk, no cameras, no Hailo — runs on any machine.

    Tests the following independently of hardware:

      resolve_object_id  (the private Pipeline helper, logic replicated here)
        1.  Word found at correct index returns the right id
        2.  Word not present in object_list returns 0
        3.  object_ids shorter than object_list returns 0 for the missing slots
        4.  Empty object_ids always returns 0
        5.  First word in list resolved correctly
        6.  Last word in list resolved correctly
        7.  Duplicate words: first match wins

      fire() callback chain  (the lambda inside inference_loop, logic replicated)
        8.  on_detection receives the resolved object_id
        9.  on_intent receives the same object_id as on_detection
       10.  on_intent not called when left as nullptr (null-safe)
       11.  on_detection fires before on_intent

      coco_id_for_word  (from coco_lookup.hpp, used to build object_ids in main)
       12.  Known first class ("person") returns 0
       13.  Known mid class ("cat") returns 15
       14.  Known last class ("toothbrush") returns 79
       15.  Unknown word returns 255
       16.  Empty string returns 255
       17.  Case-sensitive: "Cat" is not "cat"

    Build (no extra libs needed beyond standard C++17):
        g++ -std=c++17 -I include src/test_pipeline_logic.cpp \
            -o build/test_pipeline_logic
    Run:
        ./build/test_pipeline_logic
*/

#include "coco_lookup.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  Test harness
// ─────────────────────────────────────────────────────────────────────────────

static int g_run = 0, g_pass = 0;

static void report(const char* name, bool pass, const char* detail = "") {
    ++g_run;
    if (pass) {
        ++g_pass;
        std::printf("  [PASS]  %s\n", name);
    } else {
        std::printf("  [FAIL]  %s%s%s\n", name,
                    detail[0] ? "  — " : "", detail);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Replicated logic under test
//
//  These are exact copies of the private helpers in pipeline.cpp.
//  Keeping them here (rather than exposing private methods or using friend
//  classes) lets the test stay self-contained and compile without Vosk/ALSA.
// ─────────────────────────────────────────────────────────────────────────────

// Mirrors Pipeline::resolve_object_id
static uint8_t resolve_object_id(const std::vector<std::string>& object_list,
                                  const std::vector<uint8_t>&     object_ids,
                                  const std::string&              word)
{
    auto it = std::find(object_list.begin(), object_list.end(), word);
    if (it == object_list.end()) return 0;
    const std::size_t idx = static_cast<std::size_t>(
        std::distance(object_list.begin(), it));
    if (idx < object_ids.size()) return object_ids[idx];
    return 0;
}

// Mirrors the fire() lambda inside Pipeline::inference_loop.
// Returns the object_id that was resolved and passed to both callbacks.
struct DetectionResult {
    std::string word;
    uint8_t     object_id = 0;
    bool        is_final  = true;
};

static uint8_t fire(const std::vector<std::string>&        object_list,
                    const std::vector<uint8_t>&             object_ids,
                    const std::string&                      word,
                    std::function<void(const DetectionResult&)> on_detection,
                    std::function<void(uint8_t)>            on_intent)
{
    DetectionResult result;
    result.word     = word;
    result.is_final = true;
    result.object_id = resolve_object_id(object_list, object_ids, word);

    on_detection(result);
    if (on_intent) on_intent(result.object_id);

    return result.object_id;
}

// ─────────────────────────────────────────────────────────────────────────────
//  resolve_object_id tests
// ─────────────────────────────────────────────────────────────────────────────

static void test_01_found_returns_correct_id() {
    std::vector<std::string> list = {"cat", "dog", "person"};
    std::vector<uint8_t>     ids  = {15,    16,    0};
    report("01 word found returns correct id",
           resolve_object_id(list, ids, "dog") == 16);
}

static void test_02_not_in_list_returns_0() {
    std::vector<std::string> list = {"cat", "dog"};
    std::vector<uint8_t>     ids  = {15, 16};
    report("02 word not in list returns 0",
           resolve_object_id(list, ids, "elephant") == 0);
}

static void test_03_ids_shorter_than_list_returns_0() {
    std::vector<std::string> list = {"cat", "dog", "person"};
    std::vector<uint8_t>     ids  = {15};   // only covers "cat"
    // "person" is at index 2, ids has no entry there
    report("03 ids shorter than list returns 0 for missing slot",
           resolve_object_id(list, ids, "person") == 0);
}

static void test_04_empty_ids_always_returns_0() {
    std::vector<std::string> list = {"cat"};
    std::vector<uint8_t>     ids  = {};
    report("04 empty object_ids returns 0",
           resolve_object_id(list, ids, "cat") == 0);
}

static void test_05_first_word_resolved() {
    std::vector<std::string> list = {"cat", "dog"};
    std::vector<uint8_t>     ids  = {15, 16};
    report("05 first word in list resolved correctly",
           resolve_object_id(list, ids, "cat") == 15);
}

static void test_06_last_word_resolved() {
    std::vector<std::string> list = {"cat", "dog", "cup"};
    std::vector<uint8_t>     ids  = {15, 16, 41};
    report("06 last word in list resolved correctly",
           resolve_object_id(list, ids, "cup") == 41);
}

static void test_07_duplicate_word_first_match_wins() {
    std::vector<std::string> list = {"cat", "cat"};
    std::vector<uint8_t>     ids  = {15, 99};
    report("07 duplicate words: first match wins",
           resolve_object_id(list, ids, "cat") == 15);
}

// ─────────────────────────────────────────────────────────────────────────────
//  fire() callback chain tests
// ─────────────────────────────────────────────────────────────────────────────

static void test_08_on_detection_receives_resolved_id() {
    std::vector<std::string> list = {"cat"};
    std::vector<uint8_t>     ids  = {15};
    uint8_t received_id = 255;

    fire(list, ids, "cat",
         [&](const DetectionResult& r) { received_id = r.object_id; },
         nullptr);

    report("08 on_detection receives resolved object_id",
           received_id == 15);
}

static void test_09_on_intent_receives_same_id() {
    std::vector<std::string> list = {"dog"};
    std::vector<uint8_t>     ids  = {16};
    uint8_t detection_id = 255, intent_id = 254;

    fire(list, ids, "dog",
         [&](const DetectionResult& r) { detection_id = r.object_id; },
         [&](uint8_t id)               { intent_id    = id; });

    report("09 on_intent receives same id as on_detection",
           detection_id == 16 && intent_id == 16);
}

static void test_10_on_intent_null_safe() {
    std::vector<std::string> list = {"cat"};
    std::vector<uint8_t>     ids  = {15};
    bool detection_fired = false;

    // on_intent is nullptr — must not crash
    fire(list, ids, "cat",
         [&](const DetectionResult&) { detection_fired = true; },
         nullptr);

    report("10 on_intent nullptr does not crash",
           detection_fired);
}

static void test_11_on_detection_fires_before_on_intent() {
    std::vector<std::string> list = {"cat"};
    std::vector<uint8_t>     ids  = {15};

    int detection_order = -1, intent_order = -1, counter = 0;

    fire(list, ids, "cat",
         [&](const DetectionResult&) { detection_order = counter++; },
         [&](uint8_t)                { intent_order    = counter++; });

    report("11 on_detection fires before on_intent",
           detection_order == 0 && intent_order == 1);
}

// ─────────────────────────────────────────────────────────────────────────────
//  coco_id_for_word tests
// ─────────────────────────────────────────────────────────────────────────────

static void test_12_person_is_0() {
    report("12 coco_id_for_word: person == 0",
           coco_id_for_word("person") == 0);
}

static void test_13_cat_is_15() {
    report("13 coco_id_for_word: cat == 15",
           coco_id_for_word("cat") == 15);
}

static void test_14_toothbrush_is_79() {
    report("14 coco_id_for_word: toothbrush == 79",
           coco_id_for_word("toothbrush") == 79);
}

static void test_15_unknown_word_returns_255() {
    report("15 coco_id_for_word: unknown word returns 255",
           coco_id_for_word("unicorn") == 255);
}

static void test_16_empty_string_returns_255() {
    report("16 coco_id_for_word: empty string returns 255",
           coco_id_for_word("") == 255);
}

static void test_17_case_sensitive() {
    report("17 coco_id_for_word: case-sensitive (\"Cat\" != \"cat\")",
           coco_id_for_word("Cat") == 255);
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::printf("══════════════════════════════════════════\n");
    std::printf("  TEST SUITE: Pipeline logic\n");
    std::printf("══════════════════════════════════════════\n\n");

    test_01_found_returns_correct_id();
    test_02_not_in_list_returns_0();
    test_03_ids_shorter_than_list_returns_0();
    test_04_empty_ids_always_returns_0();
    test_05_first_word_resolved();
    test_06_last_word_resolved();
    test_07_duplicate_word_first_match_wins();
    test_08_on_detection_receives_resolved_id();
    test_09_on_intent_receives_same_id();
    test_10_on_intent_null_safe();
    test_11_on_detection_fires_before_on_intent();
    test_12_person_is_0();
    test_13_cat_is_15();
    test_14_toothbrush_is_79();
    test_15_unknown_word_returns_255();
    test_16_empty_string_returns_255();
    test_17_case_sensitive();

    std::printf("\n══════════════════════════════════════════\n");
    std::printf("  Results: %d / %d passed\n", g_pass, g_run);
    std::printf("══════════════════════════════════════════\n");

    return (g_pass == g_run) ? 0 : 1;
}
