import copy
from dataclasses import dataclass
from typing import List

from src.models import DetectedKeypoint, Keypoint


@dataclass
class ClassifiedKeypoint(DetectedKeypoint):
    treshold_distance: float
    true_positive: bool


def keypoint_classification(
    detected_keypoints: List[DetectedKeypoint],
    ground_truth_keypoints: List[Keypoint],
    treshold_distance,
) -> List[ClassifiedKeypoint]:
    classified_keypoints: List[ClassifiedKeypoint] = []
    for detected_keypoint in sorted(detected_keypoints, key=lambda x: x.probability, reverse=True):
        matched = False
        for gt_keypoint in ground_truth_keypoints:
            distance = detected_keypoint.l2_distance(gt_keypoint)
            if distance < treshold_distance:
                classified_keypoint = ClassifiedKeypoint(
                    detected_keypoint.u,
                    detected_keypoint.v,
                    detected_keypoint.probability,
                    treshold_distance,
                    True,
                )
                matched = True
                # remove keypoint from gt to avoid muliple matching
                ground_truth_keypoints.remove(gt_keypoint)
                break
        if not matched:
            classified_keypoint = ClassifiedKeypoint(
                detected_keypoint.u,
                detected_keypoint.v,
                detected_keypoint.probability,
                treshold_distance,
                False,
            )
        classified_keypoints.append(classified_keypoint)

    return classified_keypoints


def calculate_precision_recall(classified_keypoints: List[ClassifiedKeypoint], total_ground_truth_keypoints: int):
    """
    Note that this function is tailored towards a Detector, not a Classifier. For classifiers, the outputs contain both TP, FP and FN. Whereas for a Detector the
    outputs only define the TP and the FP; the FN are not contained in the output as the point is exactly that the detector did not detect this event.

    A detector is a ROI finder + classifier and the ROI finder could miss certain regions, which results in FNs that are hence never passed to the classifier.

    This also explains why the scikit average_precision function states it is for Classification tasks only. Since it takes "total_gt_events" to be the # of positive_class labels.
    The function can however be used by using as label (TP = 1, FP = 0) and by then multiplying the result with TP/(TP + FN) since the recall values are then corrected
    to take the unseen events (FN's) into account as well. They do not matter for precision calcultations.
    """
    precision = [1.0]
    recall = [0.0]

    true_positives = 0
    false_positives = 0

    for keypoint in sorted(classified_keypoints, key=lambda x: x.probability, reverse=True):
        if keypoint.true_positive:
            true_positives += 1
        else:
            false_positives += 1

        precision.append(true_positives / (true_positives + false_positives))
        recall.append(true_positives / total_ground_truth_keypoints)

    precision.append(0.0)
    recall.append(1.0)

    return precision, recall


def calculate_ap_from_pr(precision, recall):
    # https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
    # AUC AP.

    smoothened_precision = copy.deepcopy(precision)

    for i in range(len(smoothened_precision) - 2, 0, -1):
        smoothened_precision[i] = max(smoothened_precision[i], smoothened_precision[i + 1])

    ap = 0
    for i in range(len(recall) - 1):
        ap += (recall[i + 1] - recall[i]) * smoothened_precision[i + 1]

    return ap


# TODO: (benchmark against sklearn ap and if it's faster -> switch)

# TODO: integrate in Pl.metrics module to update keypoints @ each batch and compute final metrics after doing the whole validation dataset
