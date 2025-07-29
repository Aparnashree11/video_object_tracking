import motmetrics as mm
import os

def evaluate(gt_path, pred_path):
    acc = mm.MOTAccumulator(auto_id=True)

    gt_file = os.path.join(gt_path)
    pred_file = os.path.join(pred_path)

    with open(gt_file) as f_gt, open(pred_file) as f_pred:
        gt_lines = f_gt.readlines()
        pred_lines = f_pred.readlines()

    gt_by_frame = {}
    pred_by_frame = {}

    for line in gt_lines:
        frame, obj_id, x, y, w, h, *_ = map(float, line.strip().split(','))
        gt_by_frame.setdefault(int(frame), []).append((int(obj_id), [x, y, w, h]))

    for line in pred_lines:
        frame, obj_id, x, y, w, h, *_ = map(float, line.strip().split(','))
        pred_by_frame.setdefault(int(frame), []).append((int(obj_id), [x, y, w, h]))

    frames = sorted(set(gt_by_frame.keys()) & set(pred_by_frame.keys()))

    for frame_id in frames:
        gt_objs = gt_by_frame[frame_id]
        pred_objs = pred_by_frame[frame_id]

        gt_ids = [obj_id for obj_id, _ in gt_objs]
        gt_boxes = [box for _, box in gt_objs]
        pred_ids = [obj_id for obj_id, _ in pred_objs]
        pred_boxes = [box for _, box in pred_objs]

        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='Evaluation')
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
