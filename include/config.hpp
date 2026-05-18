#pragma once

/*
 * NorthStar project-wide config.
 *
 * Run location assumption:
 *   You normally run the binary from:
 *
 *     ~/NSJamie/NorthStar/build
 *
 * Therefore "../model_inf/..." resolves to:
 *
 *     ~/NSJamie/NorthStar/model_inf/...
 *
 * Put your Hailo .hef files in:
 *
 *     ~/NSJamie/NorthStar/model_inf/
 *
 * Example:
 *     model_inf/yolov8n.hef
 *     model_inf/yolov8m.hef
 *     model_inf/yolov8s.hef
 */

// -----------------------------------------------------------------------------
// Hailo model selection
// -----------------------------------------------------------------------------

#define NORTHSTAR_MODEL_DIR "../model_inf"

#define NORTHSTAR_MODEL_FILENAME "yolov8m.hef"
#define DEFAULT_HEF_PATH NORTHSTAR_MODEL_DIR "/" NORTHSTAR_MODEL_FILENAME

#define NORTHSTAR_HAILO_BGR 1
#define MIN_DURATION_MS 2000
#define MAX_DURATION_MS 10000