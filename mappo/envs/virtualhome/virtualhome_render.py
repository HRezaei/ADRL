from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOM_LAYOUT = {
    "kitchen":    {"x": 10,  "y": 50,  "w": 380, "h": 270, "color": "#FFF8E1", "label": "Kitchen"},
    "livingroom": {"x": 410, "y": 50,  "w": 380, "h": 270, "color": "#E8F5E9", "label": "Living Room"},
    "bathroom":   {"x": 10,  "y": 340, "w": 380, "h": 270, "color": "#E3F2FD", "label": "Bathroom"},
    "bedroom":    {"x": 410, "y": 340, "w": 380, "h": 270, "color": "#F3E5F5", "label": "Bedroom"},
}

CHARACTER_ROOM_POS = {
    "kitchen":    (200, 185),
    "livingroom": (600, 185),
    "bathroom":   (200, 475),
    "bedroom":    (600, 475),
}

def _get_font(size=14):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except (IOError, OSError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except (IOError, OSError):
            return ImageFont.load_default()


def _draw_rounded_rect(draw, xy, radius=8, fill=None, outline=None, width=1):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def render_v1(graph, action_text=None, goal_text=None, seed=None):
    img = Image.new("RGB", (800, 640), "#F5F5F5")
    draw = ImageDraw.Draw(img)
    font = _get_font(13)
    font_small = _get_font(11)
    font_goal = _get_font(14)

    # goal banner
    _draw_rounded_rect(draw, (4, 4, 796, 36), radius=6, fill="#E8EAF6", outline="#3F51B5", width=2)
    if goal_text:
        draw.text((12, 11), f"Goal: {goal_text}", fill="#1A237E", font=font_goal)
    if seed is not None:
        seed_label = f"seed={seed}"
        seed_w = draw.textlength(seed_label, font=font_small)
        draw.text((790 - seed_w, 13), seed_label, fill="#5C6BC0", font=font_small)

    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]

    char_id = None
    room_ids = {}
    obj_ids = {}
    for nid, node in nodes.items():
        name = node["class_name"]
        if name == "character":
            char_id = nid
        elif name in ROOM_LAYOUT:
            room_ids[name] = nid
        elif name == "pancake":
            obj_ids["pancake"] = nid
        elif name == "microwave":
            obj_ids["microwave"] = nid

    char_room = None
    for edge in edges:
        if edge["from_id"] == char_id and edge["relation_type"] == "INSIDE":
            for rname, rid in room_ids.items():
                if rid == edge["to_id"]:
                    char_room = rname
                    break

    close_to_ids = set()
    hold_ids = set()
    for edge in edges:
        if edge["from_id"] == char_id:
            if edge["relation_type"] == "CLOSE":
                close_to_ids.add(edge["to_id"])
            if "HOLDS" in edge["relation_type"]:
                hold_ids.add(edge["to_id"])

    pancake_in_microwave = False
    for edge in edges:
        if edge["relation_type"] == "INSIDE" and edge.get("from_id") == obj_ids.get("pancake") and edge.get("to_id") == obj_ids.get("microwave"):
            pancake_in_microwave = True

    microwave_open = False
    if "microwave" in obj_ids:
        mid = obj_ids["microwave"]
        if "OPEN" in nodes.get(mid, {}).get("states", []):
            microwave_open = True

    for rname, layout in ROOM_LAYOUT.items():
        is_current = (rname == char_room)
        fill = layout["color"]
        outline_color = "#FF6F00" if is_current else "#BDBDBD"
        outline_width = 3 if is_current else 1
        _draw_rounded_rect(draw, (layout["x"], layout["y"], layout["x"] + layout["w"], layout["y"] + layout["h"]),
                           radius=10, fill=fill, outline=outline_color, width=outline_width)
        draw.text((layout["x"] + 10, layout["y"] + 6), layout["label"], fill="#424242", font=font)

    if char_room and char_id is not None:
        cx, cy = CHARACTER_ROOM_POS[char_room]
        draw.ellipse([cx - 18, cy - 18, cx + 18, cy + 18], fill="#1565C0", outline="#0D47A1", width=2)
        draw.text((cx - 6, cy - 7), "A", fill="white", font=font)

        if hold_ids:
            held_names = []
            for nid in hold_ids:
                if nid in nodes:
                    held_names.append(nodes[nid]["class_name"])
            if held_names:
                draw.text((cx - 25, cy - 38), "Holding: " + ", ".join(held_names), fill="#C62828", font=font_small)

    if "pancake" in obj_ids:
        px, py = (120, 140) if char_room == "kitchen" else (80, 80)
        pancake_color = "#FFB300"
        if obj_ids["pancake"] in hold_ids:
            if char_room:
                px, py = CHARACTER_ROOM_POS[char_room][0] + 35, CHARACTER_ROOM_POS[char_room][1] - 25
        elif pancake_in_microwave:
            pancake_color = "#FFE082"
        _draw_rounded_rect(draw, (px - 22, py - 22, px + 22, py + 22), radius=6, fill=pancake_color, outline="#FF8F00", width=2)
        draw.text((px - 7, py - 7), "P", fill="#5D4037", font=font)
        if pancake_in_microwave:
            draw.text((px - 18, py - 40), "in microwave", fill="#E65100", font=font_small)

    if "microwave" in obj_ids:
        mx, my = (250, 160) if char_room == "kitchen" else (180, 100)
        mw, mh = 60, 40
        mfill = "#90CAF9" if microwave_open else "#78909C"
        moutline = "#1565C0" if microwave_open else "#546E7A"
        _draw_rounded_rect(draw, (mx - mw // 2, my - mh // 2, mx + mw // 2, my + mh // 2), radius=5, fill=mfill, outline=moutline, width=2)
        draw.text((mx - 25, my - 7), "Micr.", fill="white", font=font_small)
        if not microwave_open:
            draw.text((mx - 15, my - 28), "[closed]", fill="#757575", font=font_small)

        if obj_ids.get("pancake") in hold_ids:
            pass

    if char_room:
        room_center_x = ROOM_LAYOUT[char_room]["x"] + ROOM_LAYOUT[char_room]["w"] // 2
        draw.text((room_center_x - 60, ROOM_LAYOUT[char_room]["y"] + 100),
                  "Close to: " + ", ".join(nodes[nid]["class_name"] for nid in close_to_ids if nid in nodes) if close_to_ids else "",
                  fill="#424242", font=font_small)

    if action_text:
        draw.text((10, 615), f"Action: {action_text}", fill="#1B5E20", font=font)

    return np.array(img)


def render_v2(graph, action_text=None, goal_text=None, seed=None):
    img = Image.new("RGB", (800, 640), "#F5F5F5")
    draw = ImageDraw.Draw(img)
    font = _get_font(13)
    font_small = _get_font(11)
    font_goal = _get_font(14)

    # goal banner
    _draw_rounded_rect(draw, (4, 4, 796, 36), radius=6, fill="#E8EAF6", outline="#3F51B5", width=2)
    if goal_text:
        draw.text((12, 11), f"Goal: {goal_text}", fill="#1A237E", font=font_goal)
    if seed is not None:
        seed_label = f"seed={seed}"
        seed_w = draw.textlength(seed_label, font=font_small)
        draw.text((790 - seed_w, 13), seed_label, fill="#5C6BC0", font=font_small)

    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]

    char_id = None
    room_ids = {}
    obj_ids = {}
    for nid, node in nodes.items():
        name = node["class_name"]
        if name == "character":
            char_id = nid
        elif name in ROOM_LAYOUT:
            room_ids[name] = nid
        elif name in ("chips", "milk", "tv", "sofa", "coffeetable"):
            obj_ids[name] = nid

    char_room = None
    for edge in edges:
        if edge["from_id"] == char_id and edge["relation_type"] == "INSIDE":
            for rname, rid in room_ids.items():
                if rid == edge["to_id"]:
                    char_room = rname
                    break

    close_to_ids = set()
    hold_ids = set()
    for edge in edges:
        if edge["from_id"] == char_id and "HOLDS" in edge["relation_type"]:
            hold_ids.add(edge["to_id"])
        if edge["from_id"] == char_id and edge["relation_type"] == "CLOSE":
            close_to_ids.add(edge["to_id"])

    chips_on_table = False
    milk_on_table = False
    for edge in edges:
        if edge["relation_type"] == "ON":
            if edge.get("from_id") == obj_ids.get("chips") and edge.get("to_id") == obj_ids.get("coffeetable"):
                chips_on_table = True
            if edge.get("from_id") == obj_ids.get("milk") and edge.get("to_id") == obj_ids.get("coffeetable"):
                milk_on_table = True

    tv_on = False
    if "tv" in obj_ids:
        tv_id = obj_ids["tv"]
        if "ON" in nodes.get(tv_id, {}).get("states", []):
            tv_on = True

    sitting = False
    for edge in edges:
        if edge["from_id"] == char_id and edge["relation_type"] == "ON" and edge.get("to_id") == obj_ids.get("sofa"):
            sitting = True

    for rname, layout in ROOM_LAYOUT.items():
        is_current = (rname == char_room)
        fill = layout["color"]
        outline_color = "#FF6F00" if is_current else "#BDBDBD"
        outline_width = 3 if is_current else 1
        _draw_rounded_rect(draw, (layout["x"], layout["y"], layout["x"] + layout["w"], layout["y"] + layout["h"]),
                           radius=10, fill=fill, outline=outline_color, width=outline_width)
        draw.text((layout["x"] + 10, layout["y"] + 6), layout["label"], fill="#424242", font=font)

    if char_room and char_id is not None:
        cx, cy = CHARACTER_ROOM_POS[char_room]
        draw.ellipse([cx - 18, cy - 18, cx + 18, cy + 18], fill="#1565C0", outline="#0D47A1", width=2)
        draw.text((cx - 6, cy - 7), "A", fill="white", font=font)

        if sitting:
            draw.text((cx - 30, cy - 36), "[sitting]", fill="#6A1B9A", font=font_small)

        if hold_ids:
            held_names = [nodes[nid]["class_name"] for nid in hold_ids if nid in nodes]
            if held_names:
                draw.text((cx - 30, cy - 50), "Holding: " + ", ".join(held_names), fill="#C62828", font=font_small)

    obj_config = {
        "chips":       {"room": "kitchen",    "pos": (120, 120),  "color": "#F44336", "label": "C"},
        "milk":        {"room": "kitchen",    "pos": (250, 160),  "color": "#FAFAFA", "label": "M"},
        "tv":          {"room": "livingroom", "pos": (550, 90),   "color": "#37474F", "label": "TV"},
        "sofa":        {"room": "livingroom", "pos": (650, 180),  "color": "#795548", "label": "S"},
        "coffeetable": {"room": "livingroom", "pos": (530, 180),  "color": "#8D6E63", "label": "T"},
    }

    for oname, cfg in obj_config.items():
        if oname not in obj_ids:
            continue
        oid = obj_ids[oname]
        base_x, base_y = cfg["pos"]
        if oid in hold_ids and char_room:
            base_x, base_y = CHARACTER_ROOM_POS[char_room][0] + 35, CHARACTER_ROOM_POS[char_room][1] - 25

        color = cfg["color"]
        label = cfg["label"]

        if oname == "tv":
            if tv_on:
                color = "#FFF9C4"
                outline = "#F9A825"
            else:
                outline = "#1A237E"
            _draw_rounded_rect(draw, (base_x - 20, base_y - 15, base_x + 20, base_y + 15), radius=4, fill=color, outline=outline, width=2)
            draw.text((base_x - 5, base_y - 6), label, fill="#1A237E" if not tv_on else "#E65100", font=font)
            if tv_on:
                draw.text((base_x - 16, base_y - 32), "ON", fill="#E65100", font=font_small)
        elif oname == "coffeetable":
            _draw_rounded_rect(draw, (base_x - 28, base_y - 12, base_x + 28, base_y + 12), radius=4, fill=color, outline="#5D4037", width=2)
            draw.text((base_x - 5, base_y - 6), label, fill="white", font=font)
            table_top = base_y - 12
            if chips_on_table:
                draw.text((base_x - 10, table_top - 18), "chips", fill="#C62828", font=font_small)
            if milk_on_table:
                draw.text((base_x + 5, table_top - 18), "milk", fill="#1A237E", font=font_small)
        elif oname == "sofa":
            _draw_rounded_rect(draw, (base_x - 30, base_y - 14, base_x + 30, base_y + 14), radius=6, fill=color, outline="#3E2723", width=2)
            draw.text((base_x - 4, base_y - 6), label, fill="white", font=font)
        else:
            _draw_rounded_rect(draw, (base_x - 18, base_y - 18, base_x + 18, base_y + 18), radius=8, fill=color, outline="#B71C1C" if oname == "chips" else "#0D47A1", width=2)
            draw.text((base_x - 5, base_y - 7), label, fill="#B71C1C" if oname == "chips" else "#0D47A1", font=font)

    if char_room and close_to_ids:
        close_names = [nodes[nid]["class_name"] for nid in close_to_ids if nid in nodes]
        if close_names:
            room_layout = ROOM_LAYOUT[char_room]
            draw.text((room_layout["x"] + 10, room_layout["y"] + room_layout["h"] - 40),
                      "Near: " + ", ".join(close_names), fill="#424242", font=font_small)

    if action_text:
        draw.text((10, 615), f"Action: {action_text}", fill="#1B5E20", font=font)

    return np.array(img)
