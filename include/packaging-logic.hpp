#pragma once
#include "sensor-fusion-structs.hpp"
#include "direction_estimator.hpp"
#include "distance_estimator.hpp"
#include "temp_estimator.hpp"
#include "inference_packet.hpp"
#include "ir_capture.hpp"

class PackagingLogic {
public:
    PackagingLogic(DirectionEstimator& direction, TempEstimator& temperature, DistanceEstimator& distance, IRCapture& ir_capture)
        :   direction_(direction), distance_(distance), temperature_(temperature), ir_capture_(ir_capture)
    {
    }

    std::optional<ObjectReport> process(const FilteredInferencePair pair){
        ObjectReport report;
        report.object_id = pair.object_id;
        if (pair.cam0_detections.empty()){
            if (pair.cam1_detections.empty()){
                //00: can't find item
                return std::nullopt;
            }
            else {
                //01
                report.direction = direction_.compute(pair.cam1_detections[0].box);
                return report;
            }
        }
        //by now we know that cam0 has a detection
        else if (pair.cam1_detections.empty()){
            //10
            report.direction = direction_.compute(pair.cam0_detections[0].box);
            return report;
        }
        else {
            //11
            report.direction = direction_.compute(pair.cam0_detections[0].box);
            report.temp = temperature_.compute(ir_capture_.get_latest());
            report.distance = distance_.compute(pair.cam0_detections[0].box, pair.cam1_detections[0].box);
            return report;
        }
    }

private:
    DirectionEstimator& direction_;
    TempEstimator& temperature_;
    DistanceEstimator& distance_;
    IRCapture& ir_capture_;
};