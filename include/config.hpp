#pragma once

//  Drop .hef files into model_inf/ at the repo root, then change the line
//  below to the filename you want.  After editing, just run


#define NORTHSTAR_MODEL_FILENAME "yolov8s.hef"

#define DEFAULT_HEF_PATH NORTHSTAR_MODEL_DIR "/" NORTHSTAR_MODEL_FILENAME

#define NORTHSTAR_HAILO_BGR 1
#define MIN_DURATION_MS 2000
#define MAX_DURATION_MS 10000
