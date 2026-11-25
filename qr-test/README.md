# qr-test — QR reading and JSON report test bench

Tests the challenge QR task (M04) with the **same code the mission uses**
(`frontier-fastlio/dog_codes/vision.py` + `qr_decode.py` to read, `reporter.validate` to build the JSON).
Nothing in `frontier-fastlio/` or `reactive-core/` is changed or started, and nothing is ever sent to the dog.

```bash
source /opt/ros/humble/setup.bash
cd ~/dog-codes/qr-test

python3 qr_test.py selftest                      # synthetic QR codes: which sizes / angles decode
python3 qr_test.py image a.png b.jpg             # decode saved pictures
export ROS_DOMAIN_ID=1                           # the dog (read-only camera subscriber)
python3 qr_test.py live                          # dog camera: hold a QR code in front of it
```

`live` prints every newly decoded text and the JSON that would be sent, and saves it with a snapshot
in `out/<time>/` (`payloads.jsonl`, `qr_N.jpg`). It is a **dry run** unless you add `--post`:

```bash
python3 qr_test.py live --post                   # really POSTs to the scoring server
```
`--post` refuses the placeholder group id. Only the latest submission per task id counts on the
server, so do not post test codes on competition day.

## JSON sent for a QR (reporter.py, booklet v6.3 section E)
```json
{"group_id": "<registered id>", "task_id": "M04", "task_type": "qr", "scene_type": "",
 "label": "", "pos": null, "message": "<decoded text, exactly>"}
```
POST to `http://192.168.10.141:8001/task_report_A` (left arena) or `_B` (right; `--arena B`) -
organisers 2026-11-25, both accepted during the competition. If the text names a room (one of the 8 scene types of booklet v6.3, or a
synonym: 'gym' -> fitness_centre) the mission also uses it as the M05 survivor room.
**Check this format against the booklet** - it is taken from the existing reporter code.

## How close the dog must be (selftest, 2026-11-23)
Camera 640x480, fx = 462.6 px (`/camera/color/camera_info`). Decoding is reliable when the QR's
black square is about 165 px or more in the picture, works sometimes from about 100 px, never
below about 80 px. Distance for reliable reading ~= 2.8 x the printed square's side:

| printed square | reliable within | sometimes up to |
|---|---|---|
| 10 cm | 0.28 m | 0.46 m |
| 15 cm | 0.42 m | 0.69 m |
| 20 cm | 0.56 m | 0.92 m |

No wrong text was ever returned (0 of 105): the decoder either reads the code or reports nothing.
OpenCV's locator also "finds" QR-like shapes on panels and frames; those are reported as
"not readable" and never produce a payload.

`qr_encode.py` is a small QR generator used only by `selftest` (the dog has no QR generator).
