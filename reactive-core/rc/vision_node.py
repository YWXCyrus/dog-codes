"""Camera detector sidecar: best.pt (YOLO26n: 0 curtain, 1 human, 2 stairs) -> /rc/detections.

A separate process on purpose: YOLO on the dog's CPU is slow and must never stall the driver (the driver
only reads the latest detections, rc/curtain.py). Needs ultralytics + torch on whatever machine runs it.

    python3 -m rc.vision_node --model ~/best.pt            (on the dog, domain 61: publishes /rc/detections)
    python3 -m rc.vision_node --model best.pt --image test.jpg --dry   (no ROS: print detections of one image)

Message (std_msgs/String, JSON):
    {"t": <unix s>, "dets": [{"cls": "curtain", "conf": 0.91, "bearing": 0.12, "w": 0.35}, ...]}
bearing = angle of the box centre from the camera axis, rad, + = LEFT (like the lanes); w = box width / image
width. The camera axis is ASSUMED to point along the body's x axis and cam_hfov_deg is a guess: measure both
on the dog before trusting the bearing (a curtain 1 m ahead on the centre line must give bearing ~ 0).
"""
import argparse
import json
import math
import sys
import time


def pixel_bearing(u, width, hfov_deg):
    """Bearing (rad, + = left) of image column u in an image `width` px wide with horizontal FOV hfov_deg."""
    half = width / 2.0
    return math.atan(((half - u) / half) * math.tan(math.radians(hfov_deg) / 2.0))


def boxes_to_detections(boxes, names, width, hfov_deg, min_conf=0.0):
    """boxes: iterable of (x1, y1, x2, y2, conf, cls_id) in pixels -> detection dicts."""
    out = []
    for x1, y1, x2, y2, conf, cid in boxes:
        if conf < min_conf:
            continue
        out.append({'cls': names[int(cid)], 'conf': round(float(conf), 3),
                    'bearing': round(pixel_bearing(0.5 * (x1 + x2), width, hfov_deg), 4),
                    'w': round(float(x2 - x1) / width, 3)})
    return out


def image_to_rgb(msg):
    """sensor_msgs/Image -> HxWx3 uint8 RGB (no cv_bridge); None for an unsupported encoding."""
    import numpy as np
    enc = msg.encoding.lower()
    if enc in ('rgb8', 'bgr8'):
        a = np.frombuffer(msg.data, dtype=np.uint8)
        a = a.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return a if enc == 'rgb8' else a[:, :, ::-1].copy()
    if enc == 'mono8':
        a = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
        return np.repeat(a[:, :, None], 3, axis=2)
    return None


def run_model(model, rgb, conf, hfov):
    res = model.predict(rgb, conf=conf, verbose=False)[0]
    boxes = [(*map(float, b.xyxy[0]), float(b.conf[0]), int(b.cls[0])) for b in res.boxes]
    return boxes_to_detections(boxes, res.names, rgb.shape[1], hfov)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='best.pt')
    ap.add_argument('--topic', default='/camera/color/image_raw')
    ap.add_argument('--out', default='/rc/detections')
    ap.add_argument('--hfov', type=float, default=70.0)
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--hz', type=float, default=5.0)
    ap.add_argument('--image', help='run on one image file and print (with --dry)')
    ap.add_argument('--dry', action='store_true', help='no ROS')
    a = ap.parse_args(argv)
    from ultralytics import YOLO
    model = YOLO(a.model)
    print('classes:', model.names, flush=True)
    if a.dry:
        import numpy as np
        from PIL import Image
        rgb = np.asarray(Image.open(a.image).convert('RGB'))
        print(json.dumps(run_model(model, rgb, a.conf, a.hfov), indent=1))
        return 0
    import rclpy
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
    rclpy.init()
    node = rclpy.create_node('rc_vision')
    pub = node.create_publisher(String, a.out, 10)
    last = [0.0]
    warned = [False]

    def cb(m):
        now = time.monotonic()
        if now - last[0] < 1.0 / a.hz:
            return
        last[0] = now
        rgb = image_to_rgb(m)
        if rgb is None:
            if not warned[0]:
                node.get_logger().warn('unsupported image encoding %r' % m.encoding)
                warned[0] = True
            return
        t0 = time.monotonic()
        dets = run_model(model, rgb, a.conf, a.hfov)
        out = String()
        out.data = json.dumps({'t': time.time(), 'dets': dets}, separators=(',', ':'))
        pub.publish(out)
        node.get_logger().info('%d det(s) in %.0f ms %s' % (len(dets), (time.monotonic() - t0) * 1000.0,
                                                          [(d['cls'], d['conf']) for d in dets]),
                               throttle_duration_sec=2.0)

    node.create_subscription(Image, a.topic, cb,
                             QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                        history=HistoryPolicy.KEEP_LAST, depth=1))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
