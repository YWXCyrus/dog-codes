#!/usr/bin/env python3
"""QR test bench for the challenge (perception only - never sends anything to the dog).

Uses the SAME code as the mission (frontier-fastlio/dog_codes): vision.decode_qr (OpenCV locator +
our pure-python decoder) and reporter.validate (booklet v6.3 section E JSON). Nothing in that
folder or in reactive-core is changed or started.

  python3 qr_test.py selftest                       # synthetic codes: size / angle / blur limits
  python3 qr_test.py image photo1.png photo2.jpg     # decode saved pictures
  python3 qr_test.py live                            # dog camera -> decoded text + the JSON payload
  python3 qr_test.py live --post --group-id group02_1234   # ... and really POST it (scoring server)

live: needs ROS sourced and the camera running; read-only subscriber on --image-topic.
Every decode is logged to qr-test/out/<time>/ (payloads.jsonl + a snapshot of the frame).
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MISSION = os.path.join(HERE, '..', 'frontier-fastlio', 'dog_codes')
sys.path.insert(0, MISSION)
import vision as V                                       # noqa: E402  (mission QR pipeline)

REPORT_BASE = 'http://192.168.10.141:8001/task_report_'   # + A (left arena) / B (right), organisers 2026-11-25


def quiet_opencv():
    """OpenCV 4.5.4 prints 'Library QUIRC is not linked' (on stdout, from C++) on every decode attempt."""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass


class _Mute:
    """Silence C-level stdout/stderr around one call (Python prints are flushed first)."""

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.null = os.open(os.devnull, os.O_WRONLY)
        self.saved = [os.dup(1), os.dup(2)]
        os.dup2(self.null, 1)
        os.dup2(self.null, 2)

    def __exit__(self, *exc):
        os.dup2(self.saved[0], 1)
        os.dup2(self.saved[1], 2)
        for fd in self.saved + [self.null]:
            os.close(fd)
        return False


def decode(img):
    """Mission QR pipeline (vision.decode_qr) without the QUIRC noise."""
    with _Mute():
        return V.decode_qr(img)


def validate(rep, group_id):
    """The mission reporter's own check + payload builder (imported, so the format is identical)."""
    import reporter                                   # needs rclpy importable (ROS sourced)
    return reporter.validate(rep, group_id)


def qr_report(text, task_id, scene=''):
    return {'task_id': task_id, 'task_type': 'qr', 'scene_type': scene, 'label': '', 'pos': None,
            'message': text}


def scene_hint(text):
    """Same rule as perception.do_qr (scene_client.scene_from_text) -> M05 target room."""
    import scene_client
    return scene_client.scene_from_text(text)


def post(url, payload, timeout=6.0):
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf8'), method='POST',
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode('utf8', 'replace')
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode('utf8', 'replace')
    except Exception as e:
        return 0, str(e)


# ------------------------------------------------------------------ selftest
def place(code_img, width_px, angle_deg=0.0, tilt=0.0, blur=0.0, noise=4.0, seed=0):
    """Put a rendered code (grey) into a 640x480 camera-like BGR frame."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(90, 170, (480, 640), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (0, 0), 3)
    s = width_px / code_img.shape[1]
    small = cv2.resize(code_img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    c = np.float32([320, 240])
    a = np.radians(angle_deg)
    R = np.float32([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = (src - [w / 2, h / 2]) @ R.T
    pts[:, 0] *= np.where(pts[:, 1] < 0, 1.0 - tilt, 1.0)   # top edge shorter: seen from below
    dst = (pts + c).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(small, M, (640, 480), borderValue=0)
    mask = cv2.warpPerspective(np.full_like(small, 255), M, (640, 480), borderValue=0)
    frame[mask > 128] = warped[mask > 128]
    if blur > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
    frame = np.clip(frame + rng.normal(0, noise, frame.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def selftest(args):
    import qr_encode as E
    texts = args.text or ['Survivor in the kitchen', 'M05: clinic, room 3', 'HELLO-123']
    widths = [60, 80, 100, 130, 160, 220, 300]
    cases = [('straight', 0, 0.0, 0.0), ('rotated 20 deg', 20, 0.0, 0.0),
             ('seen from below (tilt)', 0, 0.25, 0.0), ('blurred', 0, 0.0, 1.2),
             ('rotated + tilt + blur', 15, 0.2, 1.0)]
    print('decoder = mission vision.decode_qr; frame 640x480 like /camera/color/image_raw')
    print('width = QR side incl. quiet zone, in pixels. "ok" = decoded text matches exactly\n')
    total = bad = 0
    for text in texts:
        m = E.encode(text, args.level)
        img = E.render(m)
        print('text %r  (version %d, level %s, %d modules)' % (text, (m.shape[0] - 17) // 4, args.level, m.shape[0]))
        print('  %-24s' % 'case' + ''.join('%7d' % w for w in widths))
        for name, ang, tilt, blur in cases:
            row = ''
            for w in widths:
                t0 = time.time()
                r = decode(place(img, w, ang, tilt, blur, seed=w))
                ok = bool(r and r[0] == text)
                wrong = bool(r and r[0] and r[0] != text)
                total += 1
                bad += wrong
                row += '%7s' % ('ok' if ok else ('WRONG' if wrong else '-'))
            print('  %-24s' % name + row)
        print()
    print('wrong text returned: %d of %d attempts (must be 0)' % (bad, total))
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        p = os.path.join(args.save, 'selftest_qr.png')
        cv2.imwrite(p, E.render(E.encode(texts[0], args.level), module_px=10))
        print('saved %s - scan it with a phone to check the generator against a real reader' % p)
    return 1 if bad else 0


# ------------------------------------------------------------------ images
def images(args):
    rc = 1
    for path in args.files:
        img = cv2.imread(path)
        if img is None:
            print('%s: cannot read' % path)
            continue
        r = decode(img)
        if r and r[0]:
            rc = 0
            payload, err = validate(qr_report(r[0], args.task_id), args.group_id)
            print('%s: DECODED %r (QR %.0f px wide)' % (path, r[0], r[1]))
            print('  payload: %s' % (json.dumps(payload) if payload else 'INVALID: ' + err))
            hint = scene_hint(r[0])
            if hint:
                print('  scene hint for M05: %s' % hint)
        elif r:
            print('%s: QR-like shape located (%.0f px) but NOT decoded (false detection, or too small/blurred)' % (path, r[1]))
        else:
            print('%s: no QR found' % path)
    return rc


# ------------------------------------------------------------------ live
def live(args):
    import threading
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    import common as C

    if args.url is None:
        args.url = REPORT_BASE + args.arena
    if args.post and ('XXXX' in args.group_id or not args.group_id):
        print('REFUSED: --post needs the registered --group-id (not the placeholder %r)' % args.group_id)
        return 2
    out = os.path.join(HERE, 'out', time.strftime('%Y%m%d_%H%M%S'))      # created on the first decode
    print('camera %s, decoding at %.0f Hz, ROS_DOMAIN_ID=%s, %s. Ctrl+C to stop. Log: %s'
          % (args.image_topic, args.hz, os.environ.get('ROS_DOMAIN_ID', '0'),
             ('POSTING to %s' % args.url) if args.post else 'dry run (nothing sent)', out))

    rclpy.init()
    node = Node('qr_test')
    latest = {'img': None, 't': 0.0, 'n': 0}
    lock = threading.Lock()

    def on_image(msg):
        try:
            img = C.image_to_bgr(msg)
        except Exception as e:
            node.get_logger().warn('image: %s' % e, throttle_duration_sec=5.0)
            return
        with lock:
            latest['img'], latest['t'] = img, time.monotonic()
            latest['n'] += 1

    node.create_subscription(Image, args.image_topic, on_image, C.SENSOR_QOS)
    def _spin():
        try:
            rclpy.spin(node)
        except Exception:
            pass                                       # context shut down at Ctrl+C

    spin = threading.Thread(target=_spin, daemon=True)
    spin.start()

    sent, last_status, t_start, n_dec, n_seen, last_seen_print = {}, 0.0, time.monotonic(), 0, 0, 0.0
    try:
        while True:
            time.sleep(1.0 / args.hz)
            with lock:
                img, age, nimg = latest['img'], time.monotonic() - latest['t'], latest['n']
            now = time.monotonic()
            if now - last_status > 5.0:
                last_status = now
                print('[%5.0fs] camera frames %d%s, decoded %d, located-not-decoded %d'
                      % (now - t_start, nimg, '' if img is not None and age < 1.0 else ' (NO RECENT IMAGE)',
                         n_dec, n_seen))
            if img is None or age > 1.0:
                continue
            r = decode(img)
            if not r:
                continue
            text, w, c = r
            if not text:
                n_seen += 1
                if now - last_seen_print > 2.0:
                    last_seen_print = now
                    print('  QR-like shape %.0f px wide at x=%.0f, not readable (a false detection on '
                          'panels/frames, or a real code too small/blurred: get closer)' % (w, c[0]))
                continue
            n_dec += 1
            payload, err = validate(qr_report(text, args.task_id), args.group_id)
            if text in sent:
                continue
            sent[text] = payload
            os.makedirs(out, exist_ok=True)
            snap = os.path.join(out, 'qr_%d.jpg' % len(sent))
            cv2.imwrite(snap, img)
            print('\nDECODED %r  (QR %.0f px wide, centre x=%.0f)' % (text, w, c[0]))
            hint = scene_hint(text)
            if hint:
                print('  scene hint for M05: %s' % hint)
            if payload is None:
                print('  payload INVALID: %s' % err)
                rec = {'text': text, 'invalid': err}
            else:
                print('  payload: %s' % json.dumps(payload))
                rec = {'text': text, 'payload': payload, 'w_px': round(w, 1), 'snapshot': snap}
                if args.post:
                    code, body = post(args.url, payload)
                    print('  POST -> http %s: %s' % (code, body[:300]))
                    rec.update(http=code, response=body[:2000])
            with open(os.path.join(out, 'payloads.jsonl'), 'a') as f:
                f.write(json.dumps(dict(t=time.time(), **rec)) + '\n')
            print()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    print('stopped. decoded texts: %s' % (list(sent) or 'none'))
    return 0 if sent else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    gid = 'group02_XXXX'
    try:
        gid = open(os.path.expanduser('~/.dog_group_id')).read().strip() or gid   # outside the public repo
    except OSError:
        pass
    ap.add_argument('--group-id', default=gid, help='registered group id (default: ~/.dog_group_id)')
    ap.add_argument('--task-id', default='M04', help='task id of the QR task (M04 in config/mission.yaml)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('selftest')
    s.add_argument('--text', action='append')
    s.add_argument('--level', default='M', choices='LMQH')
    s.add_argument('--save', default=None, help='folder to save a test QR picture (scan it with a phone)')
    i = sub.add_parser('image')
    i.add_argument('files', nargs='+')
    lv = sub.add_parser('live')
    lv.add_argument('--image-topic', default='/camera/color/image_raw')
    lv.add_argument('--hz', type=float, default=6.0, help='decode rate (mission: qr_hz 6)')
    lv.add_argument('--post', action='store_true', help='really POST each new decoded QR to --url')
    lv.add_argument('--arena', default='A', choices='AB', help='A = left arena, B = right')
    lv.add_argument('--url', default=None, help='overrides --arena')
    args = ap.parse_args(argv)
    quiet_opencv()
    return {'selftest': selftest, 'image': images, 'live': live}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
