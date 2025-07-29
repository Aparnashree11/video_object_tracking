from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)

    def update_tracks(self, detections, frame):
        return self.tracker.update_tracks(detections, frame=frame)
