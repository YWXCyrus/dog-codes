"""HTTP client + response parser for the organisers' scene-understanding server (no ROS, urllib only).

Known from the TA stress-test logs: server http://192.168.10.141:8000, one request in flight per
group (429 group_busy + Retry-After), reply envelope
  {"success": true, "group_id": ..., "request_id": ..., "result": "<JSON string>", ...}
where result is a JSON object with exactly: scene_description, relevant_to_mission, evidence,
confidence. The exact route and multipart field names are NOT documented in our files, so the
client first reads the server's /openapi.json (FastAPI-style) to discover them, falling back to
the configured defaults.
"""
import json
import mimetypes
import re
import time
import urllib.error
import urllib.request
import uuid

SCENE_SYNONYMS = {
    'kitchen': ['kitchen', 'stove', 'oven', 'sink', 'cooking', 'microwave', 'cooker'],
    'market': ['market', 'shop', 'store', 'grocery', 'stall', 'supermarket', 'vendor', 'shelves of goods'],
    'workshop': ['workshop', 'workbench', 'garage', 'tool shed', 'tools', 'hardware'],
    'clinic': ['clinic', 'medical', 'hospital', 'first aid', 'first-aid', 'pharmacy', 'doctor', 'nurse', 'patient', 'treatment room'],
    'cleaning_room': ['cleaning', 'janitor', 'mop', 'broom', 'laundry', 'detergent', 'bucket', 'custodial'],
}
OBJECT_SYNONYMS = {
    'medical_kit': ['medical kit', 'first aid', 'first-aid', 'medical supplies', 'medicine', 'bandage', 'medical_kit', 'med kit'],
    'hand_tools': ['hand tool', 'hand_tools', 'hammer', 'wrench', 'screwdriver', 'pliers', 'saw', 'toolbox', 'tools'],
    'food': ['food', 'fruit', 'vegetable', 'bread', 'snack', 'groceries', 'canned', 'meal', 'produce'],
    'water': ['water', 'water bottle', 'bottled water', 'jug', 'drinking water'],
    'utensils': ['utensil', 'cutlery', 'fork', 'spoon', 'plate', 'ladle', 'kitchenware', 'cookware', 'pot', 'pan'],
}
HAZARD_SYNONYMS = {
    'high_voltage': ['high voltage', 'high_voltage', 'electric', 'lightning', 'electrical hazard'],
    'radioactive': ['radioactive', 'radiation', 'trefoil'],
    'chemical': ['chemical', 'toxic', 'corrosive', 'poison'],
    'biological': ['biohazard', 'biological', 'bio hazard', 'bio-hazard'],
    'flammable': ['flammable', 'flame', 'fire hazard', 'inflammable'],
}
SURVIVOR_WORDS = ['survivor', 'person', 'human', 'people', 'mannequin', 'doll', 'victim',
                  'figure', 'man ', 'woman', 'child', 'injured']


def _count(text, words):
    n = 0
    for w in words:
        if w in text:
            n += 1
    return n


def parse_result(res):
    """res: dict with the 4 keys (or str). Return normalised dict."""
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except Exception:
            m = re.search(r'\{.*\}', res, re.S)
            res = json.loads(m.group(0)) if m else {}
    desc = str(res.get('scene_description', '')).lower()
    ev = res.get('evidence', [])
    if isinstance(ev, str):
        ev = [ev]
    evs = ' | '.join(str(e) for e in ev).lower()
    text = desc + ' | ' + evs
    conf = res.get('confidence', 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    # scene: exact vocabulary word in the description first, then synonym counts
    scene, sbest = '', 0
    for k in SCENE_SYNONYMS:
        if k.replace('_', ' ') in desc or k in desc:
            scene, sbest = k, 99
            break
    if not scene:
        for k, ws in SCENE_SYNONYMS.items():
            n = _count(text, ws)
            if n > sbest:
                scene, sbest = k, n
    objects = [k for k, ws in OBJECT_SYNONYMS.items() if _count(text, ws)]
    hazards = [k for k, ws in HAZARD_SYNONYMS.items() if _count(text, ws)]
    survivor = _count(text, SURVIVOR_WORDS) > 0
    # explicit tokens requested in our prompt: "survivor" token in evidence
    return {'scene': scene, 'objects': objects, 'hazards': hazards, 'survivor': survivor,
            'conf': conf, 'relevant': bool(res.get('relevant_to_mission', False)),
            'desc': desc[:160], 'evidence': ev[:3], 'raw': res}


def build_prompt(mission_text=''):
    return (
        "Analyze the attached image for this mission.\n\n"
        "COMPETITION CONTEXT:\nThis image comes from a simulated post-disaster robotics competition.\n"
        "The environment may use miniature, toy, cardboard, or printed props to represent real objects, "
        "rooms, equipment, supplies etc. Interpret these props according to the real-world objects or "
        "locations they clearly represent. Do not mark a scene irrelevant merely because the objects are "
        "toys, miniature, or nonfunctional props.\n\n"
        "MISSION: Identify the room type visible, supplies/objects, any survivor (person, doll or "
        "mannequin) and any hazard warning sign. " + mission_text + "\n\n"
        "Return exactly one JSON object containing exactly these four keys:\n"
        "\"scene_description\", \"relevant_to_mission\", \"evidence\", \"confidence\".\n\n"
        "Definitions:\n"
        "- scene_description: the room type, using one of: kitchen, market, workshop, clinic, "
        "cleaning_room (or \"unknown\" / \"corridor\"), followed by a few words.\n"
        "- relevant_to_mission: true if a survivor, hazard sign or mission supplies are clearly visible.\n"
        "- evidence: an array of at most three short strings naming what is visible, using these exact "
        "words where they apply: medical_kit, hand_tools, food, water, utensils, survivor, high_voltage, "
        "radioactive, chemical, biological, flammable.\n"
        "- confidence: a number from 0.0 to 1.0 expressing your confidence that the scene_description is "
        "correct.\n\n"
        "Output valid JSON only. Do not include additional keys, Markdown, code fences, explanations, "
        "or text outside the JSON. Do not invent robot localisation coordinates.\n")


CURTAIN_PROMPT = (
    "Analyze the attached image for this mission.\n\n"
    "MISSION: Decide whether the robot is looking at a hanging fabric curtain or cloth strips that a "
    "robot could push through, as opposed to a solid wall or panel.\n\n"
    "Return exactly one JSON object containing exactly these four keys:\n"
    "\"scene_description\", \"relevant_to_mission\", \"evidence\", \"confidence\".\n"
    "- scene_description: short description of what blocks the view.\n"
    "- relevant_to_mission: true ONLY if a hanging curtain / cloth / strip barrier is clearly visible.\n"
    "- evidence: array of at most three short strings.\n- confidence: number 0.0 to 1.0.\n"
    "Output valid JSON only, no Markdown.\n")


class SceneClient:

    def __init__(self, server, group, path='/analyze', image_field='image',
                 prompt_field='prompt', group_field='group_id', timeout=(5, 75), logger=None):
        self.server = server.rstrip('/')
        self.group = group
        self.path = path
        self.image_field = image_field
        self.prompt_field = prompt_field
        self.group_field = group_field
        self.timeout = timeout
        self.log = logger or (lambda *a: None)
        self.discovered = False
        self.busy_until = 0.0
        self.mode = 'multipart'

    # ---- discovery ---------------------------------------------------------
    def discover(self):
        try:
            with urllib.request.urlopen(self.server + '/openapi.json', timeout=self.timeout[0]) as r:
                spec = json.loads(r.read().decode('utf8'))
        except Exception as e:
            self.log('openapi discovery failed: %s (using %s)' % (e, self.path))
            return False
        best = None
        for path, item in spec.get('paths', {}).items():
            post = item.get('post')
            if not post:
                continue
            body = post.get('requestBody', {}).get('content', {})
            for ctype, cdef in body.items():
                props = {}
                sch = cdef.get('schema', {})
                if '$ref' in sch:
                    ref = sch['$ref'].split('/')[-1]
                    sch = spec.get('components', {}).get('schemas', {}).get(ref, {})
                props = sch.get('properties', {})
                score = sum(1 for k in props if any(w in k.lower() for w in
                                                    ('image', 'file', 'photo', 'prompt', 'group')))
                if score and (best is None or score > best[0]):
                    best = (score, path, ctype, props)
        if best:
            _, path, ctype, props = best
            self.path = path
            self.mode = 'multipart' if 'multipart' in ctype else 'json'
            for k in props:
                kl = k.lower()
                if any(w in kl for w in ('image', 'file', 'photo')):
                    self.image_field = k
                elif 'prompt' in kl or 'mission' in kl:
                    self.prompt_field = k
                elif 'group' in kl:
                    self.group_field = k
            self.log('discovered scene API: %s %s image=%s prompt=%s group=%s' % (
                self.mode, self.path, self.image_field, self.prompt_field, self.group_field))
            self.discovered = True
            return True
        return False

    # ---- request -----------------------------------------------------------
    def _multipart(self, fields, files):
        boundary = uuid.uuid4().hex
        body = b''
        for k, v in fields.items():
            body += ('--%s\r\nContent-Disposition: form-data; name="%s"\r\n\r\n%s\r\n'
                     % (boundary, k, v)).encode('utf8')
        for k, (fn, data, ctype) in files.items():
            body += ('--%s\r\nContent-Disposition: form-data; name="%s"; filename="%s"\r\n'
                     'Content-Type: %s\r\n\r\n' % (boundary, k, fn, ctype)).encode('utf8')
            body += data + b'\r\n'
        body += ('--%s--\r\n' % boundary).encode('utf8')
        return body, 'multipart/form-data; boundary=' + boundary

    def analyse(self, jpeg_bytes, prompt):
        """Return (parsed_result_dict or None, info str)."""
        if time.time() < self.busy_until:
            return None, 'cooldown'
        if self.mode == 'multipart':
            body, ctype = self._multipart({self.prompt_field: prompt, self.group_field: self.group},
                                          {self.image_field: ('frame.jpg', jpeg_bytes, 'image/jpeg')})
        else:
            import base64
            body = json.dumps({self.prompt_field: prompt, self.group_field: self.group,
                               self.image_field: base64.b64encode(jpeg_bytes).decode()}).encode()
            ctype = 'application/json'
        req = urllib.request.Request(self.server + self.path, data=body,
                                     headers={'Content-Type': ctype}, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=self.timeout[1]) as r:
                env = json.loads(r.read().decode('utf8'))
        except urllib.error.HTTPError as e:
            retry = 0.0
            try:
                retry = float(e.headers.get('Retry-After', '0'))
            except Exception:
                pass
            self.busy_until = time.time() + max(retry, 1.0 if e.code == 429 else 0.0)
            detail = ''
            try:
                detail = e.read().decode('utf8')[:200]
            except Exception:
                pass
            return None, 'http %d %s' % (e.code, detail)
        except Exception as e:
            return None, 'error %s' % e
        res = env.get('result', env)
        try:
            return parse_result(res), 'ok'
        except Exception as e:
            return None, 'parse error %s' % e
