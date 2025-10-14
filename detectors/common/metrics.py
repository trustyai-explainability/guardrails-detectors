def update_detection_rates(instruments, detector_kind, detector_name):
    """Update detection, pass, and error rate gauges accordingly."""

    # update all rate functions
    det = float(instruments["detections"].labels(detector_kind, detector_name)._value.get())
    req = float(instruments["requests"].labels(detector_kind, detector_name)._value.get())
    err = float(instruments["errors"].labels(detector_kind, detector_name)._value.get())

    detection_rate = det / req if req else 0.0
    error_rate = err / req if req else 0.0
    pass_rate = (req - det - err) / req if req else 0.0

    instruments["pass_rate"].labels(detector_kind, detector_name).set(pass_rate)
    instruments["error_rate"].labels(detector_kind, detector_name).set(error_rate)
    instruments["detection_rate"].labels(detector_kind, detector_name).set(detection_rate)

