import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openai

BASE_URL = os.environ.get("PVEP_OPENAI_BASE_URL", "https://api.bianxie.ai/v1")
MODEL_NAME = os.environ.get("PVEP_OPENAI_MODEL", "gpt-5.2-2025-12-11")

ROOT = Path(__file__).resolve().parent
PROMPT_EXAMPLE_DIR = ROOT / "prompt_example"
PART_TYPES_REF_IMAGE = PROMPT_EXAMPLE_DIR / "part_types.png"
SLOT_MAP_REF_IMAGE = PROMPT_EXAMPLE_DIR / "slot_map.png"
LEFT_BINS_EXAMPLE_PATH = PROMPT_EXAMPLE_DIR / "left_bins_example.png"
RIGHT_BINS_EXAMPLE_PATH = PROMPT_EXAMPLE_DIR / "right_bins_example.png"

EXAMPLE_PDDL_TEXT = """
    (part_on blue_pump_1 bin5_1)
    (part_on blue_pump_2 bin5_3)
    (part_on blue_pump_3 bin5_5)
    (part_on blue_pump_4 bin5_7)
    (part_on blue_pump_5 bin5_9)
    (part_on green_regulator_1 bin6_1)
    (part_on green_regulator_2 bin6_3)
    (part_on green_regulator_3 bin6_5)
    (part_on green_regulator_4 bin6_7)
    (part_on green_regulator_5 bin6_9)
    (part_on purple_sensor_1 bin1_1)
    (part_on purple_sensor_2 bin1_3)
    (part_on purple_sensor_3 bin1_5)
    (part_on purple_sensor_4 bin1_7)
    (part_on purple_sensor_5 bin1_9)
    (part_on red_battery_1 bin2_1)
    (part_on red_battery_2 bin2_3)
    (part_on red_battery_3 bin2_5)
    (part_on red_battery_4 bin2_7)
    (part_on red_battery_5 bin2_9)
"""



# -------------------------------
# Utility helpers
# -------------------------------
def load_api_key() -> str:
    key = os.environ.get("PVEP_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    key = key.strip()
    if not key:
        raise RuntimeError("Set PVEP_OPENAI_API_KEY or OPENAI_API_KEY before calling the VLM helpers.")
    return key


def encode_image_to_base64(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/png"
    data = image_path.read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def parse_pddl_from_text(text: str) -> Optional[str]:
    """从文本中提取最外层 (define ...) PDDL 片段。"""
    text = text.strip()
    if "(define" not in text:
        return None

    start_idx = text.find("(define")
    depth = 0
    for i in range(start_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1]
    return None


def _extract_block(tag: str, text: str) -> Tuple[int, int]:
    start = text.find(f"(:{tag}")
    if start < 0:
        raise ValueError(f"找不到 '(:{tag}'")
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return start, idx + 1
    raise ValueError(f"'(:{tag}' 括号不平衡")


def normalize_part_name(name: str) -> str:
    # 去掉尾部数字或下划线数字
    cleaned = re.sub(r"[_-]?\d+$", "", name)
    return cleaned


def parse_part_on(pddl_text: str) -> List[Tuple[str, str]]:
    """从 PDDL 的 (:init) 中抽取 (part_on <part> <bin_slot>) 并做名称规范化。"""
    try:
        start, end = _extract_block("init", pddl_text)
        init_block = pddl_text[start:end]
        pairs: List[Tuple[str, str]] = []
        for raw in re.findall(r"\([^\(\)]*\)", init_block):
            stripped = raw.strip()
            if not stripped.startswith("(part_on"):
                continue
            parts = stripped[1:-1].split()
            if len(parts) < 3:
                continue
            # For reconstruction, we want the raw part name (e.g. blue_battery) without ID stripping yet,
            # but the helper normalize_part_name strips numbers.
            # The prompt asks LLM to output {color}_{type} without numbers, but "If multiple identical parts exist, repeat the same name multiple times."
            # So the LLM output might look like (part_on blue_battery bin3_2) (part_on blue_battery bin3_3).
            # The previous logic used normalize_part_name to strip accidental suffixes.
            # We will use normalize_part_name as is, which handles cases like blue_battery_1 -> blue_battery.
            part_raw = normalize_part_name(parts[1])
            location = parts[2]
            pairs.append((part_raw, location))
        return pairs
    except ValueError:
        return []


def parse_goals(pddl_text: str) -> List[str]:
    """从 PDDL 的 (:goal) 中抽取 (submitted <order_id>)。"""
    try:
        start, end = _extract_block("goal", pddl_text)
        goal_block = pddl_text[start:end]
        # remove "and" if present
        goal_content = goal_block.replace("(:goal", "").strip()
        if goal_content.startswith("(and"):
             # Simple heuristic to strip (and ... ) outer wrapper
             # But _extract_block returned the whole (:goal ...) block.
             # Let's just regex search for (submitted ...)
             pass
        
        submitted_orders = []
        for raw in re.findall(r"\(submitted\s+([^\s\)]+)\)", goal_block):
            submitted_orders.append(raw.strip())
        return submitted_orders
    except ValueError:
        return []


def build_prompt(orders_info: str = "") -> str:
    prompt = (
        f"# Task: Identify parts and their bin slots from images, then output PDDL\n"
        f"Problem: Realtime_Trial\n\n"
        "You will receive FOUR images in this order:\n"
        "A) Reference Image - Part Types: uncolored 3D CAD models with labels (Battery, Pump, Sensor, Regulator).\n"
        "B) Reference Image - Slot Map: 3x3 grid showing slot numbers 1..9.\n"
        "C) Trial Image - left_bins (bins 8/7/5/6).\n"
        "D) Trial Image - right_bins (bins 3/4/2/1).\n\n"
        "IMPORTANT:\n"
        "- Use images A and B only as references to avoid confusing part types and slot numbering.\n"
        "- Use images C and D to detect the actual parts and positions.\n\n"
        "Bin layout (top-down):\n"
        "- left_bins image (C): top-left bin8, top-right bin7, bottom-left bin5, bottom-right bin6.\n"
        "- right_bins image (D): top-left bin3, top-right bin4, bottom-left bin2, bottom-right bin1.\n\n"
        "Slot numbering inside each bin is EXACTLY as reference image B (3x3, row-major):\n"
        "1 2 3\n"
        "4 5 6\n"
        "7 8 9\n"
        "A slot is written as bin<index>_<cell> (e.g., bin3_6).\n\n"
        "Detect ONLY parts on bins (ignore trays/IDs). Part types are EXACTLY:\n"
        "- battery, pump, sensor, regulator (use reference image A to classify by shape).\n"
        "Colors are: red, blue, green, orange, purple.\n\n"
        "Naming rules:\n"
        "- Part name format: {color}_{type} (e.g., blue_battery).\n"
        "- DO NOT add numeric suffixes. If multiple identical parts exist, repeat the same name multiple times.\n\n"
    )

    if orders_info:
        prompt += (
            f"ORDERS INFORMATION:\n{orders_info}\n\n"
            "Include the following in the PDDL output:\n"
            "1. Define all objects (parts, orders, agvs, etc.).\n"
            "2. In :init, include (order_needs_part ...), (order_uses_tray ...), (agv_at ...), (home ...), (agv_reach ...), (slot_of ...) based on the order info.\n"
            "3. In :goal, include (submitted <order_id>) for ALL incomplete orders listed in the ORDERS INFORMATION.\n"
            "   - Format: (:goal (and (submitted <order_id1>) (submitted <order_id2>) ...))\n\n"
        )

    prompt += (
        "Output STRICTLY one PDDL block and NOTHING ELSE:\n"
        "(define (problem Trial_new)\n"
        "  (:domain ariac_kitting)\n"
        "  (:objects ...)\n"
        "  (:init\n"
        "    (part_on <color_type> <bin_slot>)\n"
        "    (order_needs_part ...)\n"
        "    ...\n"
        "  )\n"
        "  (:goal (and\n"
        "    (submitted <order_id1>)\n"
        "    (submitted <order_id2>)\n"
        "    ...\n"
        "  ))\n"
        "  (:metric minimize (total-cost))\n"
        ")\n\n"
        "Rules for slot choice:\n"
        "- If a part touches multiple cells, choose the cell containing the part's center.\n"
        "- Use both trial images (C,D) to cover bins 1-8 without double counting.\n"
        "- Ensure ALL objects used in :init and :goal are defined in :objects.\n"
    )
    return prompt


def get_pddl_from_images(left_image: Path, right_image: Path, orders_info: str = "") -> Optional[str]:
    """
    Call the VLM with the provided images and return the parsed PDDL string.
    """
    api_key = load_api_key()
    client = openai.OpenAI(api_key=api_key, base_url=BASE_URL)

    # 参考图（A/B）
    if not PART_TYPES_REF_IMAGE.exists():
        raise FileNotFoundError(f"缺少参考图: {PART_TYPES_REF_IMAGE}")
    if not SLOT_MAP_REF_IMAGE.exists():
        raise FileNotFoundError(f"缺少参考图: {SLOT_MAP_REF_IMAGE}")

    ref_part_types_b64 = encode_image_to_base64(PART_TYPES_REF_IMAGE)
    ref_slot_map_b64 = encode_image_to_base64(SLOT_MAP_REF_IMAGE)

    # trial 图（C/D）
    if not left_image.exists():
        raise FileNotFoundError(f"Missing left image: {left_image}")
    if not right_image.exists():
        raise FileNotFoundError(f"Missing right image: {right_image}")

    left_b64 = encode_image_to_base64(left_image)
    right_b64 = encode_image_to_base64(right_image)
    
    prompt = build_prompt(orders_info)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                # A) Part Types reference
                {"type": "image_url", "image_url": {"url": ref_part_types_b64}},
                # B) Slot map reference
                {"type": "image_url", "image_url": {"url": ref_slot_map_b64}},
                # C) left_bins
                {"type": "image_url", "image_url": {"url": left_b64}},
                # D) right_bins
                {"type": "image_url", "image_url": {"url": right_b64}},
            ],
        }
    ]

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=4096,
        )
        content = resp.choices[0].message.content
        return parse_pddl_from_text(content)
    except Exception as e:
        print(f"Error calling VLM API: {e}")
        return None


# -------------------------------
# New Inspect Functions
# -------------------------------

def build_inspect_prompt(bin_id: str, current_state_hint: str = "") -> str:
    prompt = (
        f"# Task: Inspect {bin_id} and identify parts.\n"
        f"Problem: Inspect_Bin\n\n"
        "You will receive SEVEN items in this order:\n"
        "1) Text Prompt (this text)\n"
        "2) Reference Image - Part Types: uncolored 3D CAD models with labels.\n"
        "3) Reference Image - Slot Map: 3x3 grid showing slot numbers 1..9.\n"
        "4) Example Image - left_bins (Teaching Reference).\n"
        "5) Example Image - right_bins (Teaching Reference).\n"
        "6) Example PDDL Text - The Ground Truth for the Example Images above.\n"
        "7) Trial Image - Camera View containing the target bin.\n\n"
        "INSTRUCTION:\n"
        "1. Study the Example Images (4 & 5) and the Example PDDL (6) to learn how parts look and map to slots.\n"
        f"2. Then, focus ONLY on {bin_id} in the Trial Image (7).\n"
        "Slot numbering inside the bin is EXACTLY as reference image B (3x3, row-major):\n"
        "1 2 3\n"
        "4 5 6\n"
        "7 8 9\n"
        f"Slots are named {bin_id}_<cell> (e.g., {bin_id}_5).\n\n"
        "Detect ONLY parts inside this bin. Part types are:\n"
        "- battery, pump, sensor, regulator.\n"
        "Colors are: red, blue, green, orange, purple.\n\n"
        "Naming rules:\n"
        "- Part name format: {color}_{type} (e.g., blue_battery).\n"
        "- DO NOT add numeric suffixes.\n\n"
    )

    if current_state_hint:
        prompt += (
            "CURRENT SYSTEM BELIEF (Hint):\n"
            "The system currently believes the bin contains the following (PDDL format):\n"
            f"{current_state_hint}\n"
            "Please use this as a reference. If the visual evidence confirms this, output the same.\n"
            "If the visual evidence contradicts this (e.g., empty slot, wrong color), CORRECT it.\n\n"
        )

    prompt += (
        "Output STRICTLY one PDDL block:\n"
        "(define (problem Inspect_res)\n"
        "  (:domain ariac_kitting)\n"
        "  (:objects )\n"
        "  (:init\n"
        "    (part_on <color_type> <bin_slot>)\n"
        "  )\n"
        "  (:goal (and))\n"
        ")\n"
        "If the bin is empty, return an empty :init block.\n"
    )
    return prompt

def get_inspect_pddl(bin_id: str, image_path: Path, current_state_hint: str = "") -> Optional[str]:
    """
    Call VLM for a single bin inspection with few-shot examples and hint.
    """
    api_key = load_api_key()
    client = openai.OpenAI(api_key=api_key, base_url=BASE_URL)

    # Load references
    if not PART_TYPES_REF_IMAGE.exists():
        raise FileNotFoundError(f"Missing ref: {PART_TYPES_REF_IMAGE}")
    if not SLOT_MAP_REF_IMAGE.exists():
        raise FileNotFoundError(f"Missing ref: {SLOT_MAP_REF_IMAGE}")
    
    # Load examples
    if not LEFT_BINS_EXAMPLE_PATH.exists():
        raise FileNotFoundError(f"Missing example: {LEFT_BINS_EXAMPLE_PATH}")
    if not RIGHT_BINS_EXAMPLE_PATH.exists():
        raise FileNotFoundError(f"Missing example: {RIGHT_BINS_EXAMPLE_PATH}")

    ref_part_types_b64 = encode_image_to_base64(PART_TYPES_REF_IMAGE)
    ref_slot_map_b64 = encode_image_to_base64(SLOT_MAP_REF_IMAGE)
    example_left_b64 = encode_image_to_base64(LEFT_BINS_EXAMPLE_PATH)
    example_right_b64 = encode_image_to_base64(RIGHT_BINS_EXAMPLE_PATH)

    if not image_path.exists():
        raise FileNotFoundError(f"Missing inspect image: {image_path}")
    
    img_b64 = encode_image_to_base64(image_path)
    prompt = build_inspect_prompt(bin_id, current_state_hint)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                # 2) Reference Part Types
                {"type": "image_url", "image_url": {"url": ref_part_types_b64}},
                # 3) Reference Slot Map
                {"type": "image_url", "image_url": {"url": ref_slot_map_b64}},
                # 4) Example Left
                {"type": "image_url", "image_url": {"url": example_left_b64}},
                # 5) Example Right
                {"type": "image_url", "image_url": {"url": example_right_b64}},
                # 6) Example Text
                {"type": "text", "text": f"Example Ground Truth PDDL for the example images above:\n{EXAMPLE_PDDL_TEXT}"},
                # 7) Target Image
                {"type": "image_url", "image_url": {"url": img_b64}},
            ],
        }
    ]

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=1024,
        )
        content = resp.choices[0].message.content
        return parse_pddl_from_text(content)
    except Exception as e:
        print(f"Error calling VLM API for inspect: {e}")
        return None


# -------------------------------
# Replanning Functions
# -------------------------------

def build_replanning_prompt(order_description: str, current_state_hint: str) -> str:
    prompt = (
        f"# Task: Replan for Dynamic Order/Event\n"
        f"Problem: Replanning\n\n"
        "You will receive:\n"
        "1. Two current images of the bins (Left and Right).\n"
        "2. The Current Order Description (containing all incomplete orders).\n"
        "3. A hint of the current system state (what we thought was true).\n\n"
        "INSTRUCTION:\n"
        "1. UPDATE the environment state based on the images. If a part was picked (missing compared to hint), remove it. If a part is moved, update it.\n"
        "2. GENERATE the (:init ...) block reflecting the current REAL state of parts on bins, AND the order details (needs_part, uses_tray, etc.).\n"
        "3. GENERATE the (:goal ...) block based on the Order Description.\n"
        "   - The goal should specify (submitted <order_id>) for EACH incomplete order in the description.\n"
        "   - Ensure that the order needs and tray assignments are correctly defined in :init.\n\n"
        f"ORDER DESCRIPTION:\n{order_description}\n\n"
        f"CURRENT STATE HINT (PDDL):\n{current_state_hint}\n\n"
        "Output STRICTLY one PDDL block:\n"
        "(define (problem Replanning)\n"
        "  (:domain ariac_kitting)\n"
        "  (:objects ...)\n"
        "  (:init\n"
        "    (part_on ...)\n"
        "    (order_needs_part ...)\n"
        "    (order_uses_tray ...)\n"
        "    (agv_at ...)\n"
        "    (home ...)\n"
        "    ...\n"
        "  )\n"
        "  (:goal (and\n"
        "    (submitted <order_id_1>)\n"
        "    (submitted <order_id_2>)\n"
        "    ...\n"
        "  ))\n"
        "  (:metric minimize (total-cost))\n"
        ")\n"
        "IMPORTANT: Ensure all objects used in init/goal are defined in :objects or constants.\n"
    )
    return prompt

def get_replanning_pddl(order_description: str, current_state_hint: str, left_image: Path, right_image: Path) -> Optional[str]:
    """
    Call VLM to regenerate PDDL based on current images and order.
    """
    api_key = load_api_key()
    client = openai.OpenAI(api_key=api_key, base_url=BASE_URL)

    # Load references
    if not PART_TYPES_REF_IMAGE.exists():
        raise FileNotFoundError(f"Missing ref: {PART_TYPES_REF_IMAGE}")
    if not SLOT_MAP_REF_IMAGE.exists():
        raise FileNotFoundError(f"Missing ref: {SLOT_MAP_REF_IMAGE}")

    ref_part_types_b64 = encode_image_to_base64(PART_TYPES_REF_IMAGE)
    ref_slot_map_b64 = encode_image_to_base64(SLOT_MAP_REF_IMAGE)
    
    # Trial images
    if not left_image.exists() or not right_image.exists():
        raise FileNotFoundError("Missing trial images for replanning.")

    left_b64 = encode_image_to_base64(left_image)
    right_b64 = encode_image_to_base64(right_image)

    prompt = build_replanning_prompt(order_description, current_state_hint)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": ref_part_types_b64}},
                {"type": "image_url", "image_url": {"url": ref_slot_map_b64}},
                {"type": "image_url", "image_url": {"url": left_b64}},
                {"type": "image_url", "image_url": {"url": right_b64}},
            ],
        }
    ]

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=4096,
        )
        content = resp.choices[0].message.content
        return parse_pddl_from_text(content)
    except Exception as e:
        print(f"Error calling VLM API for replanning: {e}")
        return None


# -------------------------------
# Drop Recovery (LLM only, no images)
# -------------------------------


def _base_part_type(part_name: str) -> str:
    # e.g. red_battery_1 -> red_battery
    return re.sub(r"[_-]?\d+$", "", part_name.strip())


def _parse_objects_parts(pddl_text: str) -> List[str]:
    try:
        s, e = _extract_block("objects", pddl_text)
        block = pddl_text[s:e]
    except ValueError:
        return []
    parts: List[str] = []
    for ln in block.splitlines():
        if "- part" not in ln:
            continue
        left = ln.split("- part")[0]
        toks = [t.strip() for t in left.split() if t.strip()]
        parts.extend(toks)
    return parts


def _parse_init_facts(pddl_text: str) -> List[str]:
    try:
        s, e = _extract_block("init", pddl_text)
        block = pddl_text[s:e]
    except ValueError:
        return []
    # Return raw '(...)' facts in init
    return [m.group(0) for m in re.finditer(r"\([^\(\)]*\)", block)]


def _parse_order_needs(pddl_text: str) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for fact in _parse_init_facts(pddl_text):
        if not fact.startswith("(order_needs_part"):
            continue
        parts = fact[1:-1].split()
        if len(parts) >= 4:
            out.append((parts[1], parts[2], parts[3]))
    return out


def _parse_part_on(pddl_text: str) -> Dict[str, str]:
    loc: Dict[str, str] = {}
    for fact in _parse_init_facts(pddl_text):
        if not fact.startswith("(part_on"):
            continue
        parts = fact[1:-1].split()
        if len(parts) >= 3:
            loc[parts[1]] = parts[2]
    return loc


def build_drop_recovery_prompt(
    *,
    latest_pddl_text: str,
    dropped_part: str,
    dropped_order: str,
    dropped_slot: str,
) -> str:
    """
    Strict prompt: only swap the dropped part in a single (order_needs_part ...) fact to an existing unused same-type part.
    """
    base_type = _base_part_type(dropped_part)
    all_parts = _parse_objects_parts(latest_pddl_text)
    needs = _parse_order_needs(latest_pddl_text)
    used_parts = {p for _, p, _ in needs}
    part_locs = _parse_part_on(latest_pddl_text)

    candidates = [p for p in all_parts if (_base_part_type(p) == base_type and p not in used_parts)]
    candidates_with_loc = [(p, part_locs.get(p)) for p in candidates]
    candidates_on_bins = [(p, loc) for p, loc in candidates_with_loc if loc is not None]

    cand_lines = []
    for p, loc in sorted(candidates_on_bins):
        cand_lines.append(f"- {p} @ {loc}")
    if not cand_lines:
        for p, loc in sorted(candidates_with_loc):
            cand_lines.append(f"- {p} @ {loc or 'unknown'}")

    candidates_str = "\n".join(cand_lines) if cand_lines else "(none)"

    prompt = (
        "# Task: Repair PDDL after a dropped part\n"
        "You are given the CURRENT latest PDDL problem (already reflects the current world state).\n"
        "A part was DROPPED during execution and is no longer available.\n\n"
        "Your job is to produce a NEW PDDL problem by ONLY doing the minimal required edit:\n"
        f"- Find the init fact: (order_needs_part {dropped_order} {dropped_part} {dropped_slot})\n"
        f"- Replace {dropped_part} with ONE replacement part of the SAME type (same color+type, different id allowed), chosen from existing objects.\n"
        "- The replacement part MUST be an existing part object in (:objects) and MUST NOT already appear in any (order_needs_part ...).\n"
        "- Prefer a replacement part that currently appears in (part_on <part> <bin_slot>) (i.e., still on a bin).\n"
        "- Do NOT invent new objects.\n"
        "- Do NOT change anything else in :objects/:init/:goal/:metric.\n"
        "- Output STRICTLY one complete (define ...) PDDL block and NOTHING ELSE.\n\n"
        f"DROPPED_CONTEXT:\n- dropped_order: {dropped_order}\n- dropped_part: {dropped_part}\n- dropped_slot: {dropped_slot}\n- dropped_part_base_type: {base_type}\n\n"
        "CANDIDATE_REPLACEMENTS (unused, same base type):\n"
        f"{candidates_str}\n\n"
        "CURRENT_LATEST_PDDL:\n"
        f"{latest_pddl_text}\n"
    )
    return prompt


def build_drop_recovery_replacement_prompt(
    *,
    latest_pddl_text: str,
    dropped_part: str,
    dropped_order: str,
    dropped_slot: str,
) -> str:
    """
    Strict prompt: return ONLY the replacement part token (single word).
    """
    base_type = _base_part_type(dropped_part)
    all_parts = _parse_objects_parts(latest_pddl_text)
    needs = _parse_order_needs(latest_pddl_text)
    used_parts = {p for _, p, _ in needs}
    part_locs = _parse_part_on(latest_pddl_text)

    candidates = [p for p in all_parts if (_base_part_type(p) == base_type and p not in used_parts)]
    candidates_with_loc = [(p, part_locs.get(p)) for p in candidates]
    candidates_on_bins = [(p, loc) for p, loc in candidates_with_loc if loc is not None]

    cand_lines = []
    for p, loc in sorted(candidates_on_bins):
        cand_lines.append(f"- {p} @ {loc}")
    if not cand_lines:
        for p, loc in sorted(candidates_with_loc):
            cand_lines.append(f"- {p} @ {loc or 'unknown'}")

    candidates_str = "\n".join(cand_lines) if cand_lines else "(none)"

    prompt = (
        "# Task: Choose replacement part token after a dropped part\n"
        "Return ONLY ONE token: the replacement part name.\n\n"
        f"- The dropped part was: {dropped_part}\n"
        f"- Required slot: {dropped_slot} (order {dropped_order})\n"
        f"- Replacement must be SAME base type as {dropped_part} (base type: {base_type}).\n"
        "- Replacement must be an existing part in (:objects) and MUST NOT already appear in any (order_needs_part ...).\n"
        "- Prefer a replacement that appears in (part_on <part> <bin_slot>).\n"
        "- Do NOT output anything else (no punctuation, no explanations).\n\n"
        "CANDIDATE_REPLACEMENTS (unused, same base type):\n"
        f"{candidates_str}\n\n"
        "CURRENT_LATEST_PDDL:\n"
        f"{latest_pddl_text}\n"
    )
    return prompt


def _extract_first_token(text: str) -> Optional[str]:
    if not text:
        return None
    line = text.strip().splitlines()[0].strip()
    if not line:
        return None
    return line.split()[0]


def _validate_replacement_token(token: str, latest_pddl_text: str) -> bool:
    if not token:
        return False
    all_parts = set(_parse_objects_parts(latest_pddl_text))
    if token not in all_parts:
        return False
    needs = _parse_order_needs(latest_pddl_text)
    used_parts = {p for _, p, _ in needs}
    return token not in used_parts


def get_drop_recovery_replacement(
    *,
    latest_pddl_text: str,
    dropped_part: str,
    dropped_order: str,
    dropped_slot: str,
) -> Optional[str]:
    """
    Pure LLM (no images): return ONLY the replacement part token.
    """
    api_key = load_api_key()
    client = openai.OpenAI(api_key=api_key, base_url=BASE_URL)

    prompt = build_drop_recovery_replacement_prompt(
        latest_pddl_text=latest_pddl_text,
        dropped_part=dropped_part,
        dropped_order=dropped_order,
        dropped_slot=dropped_slot,
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=256,
        )
        content = resp.choices[0].message.content
        token = _extract_first_token(content or "")
        if token and _validate_replacement_token(token, latest_pddl_text):
            return token
        return None
    except Exception:
        # Keep quiet; caller will decide how to handle failure.
        return None


def build_drop_recovery_replacement_prompt(
    *,
    latest_pddl_text: str,
    dropped_part: str,
    dropped_order: str,
    dropped_slot: str,
) -> str:
    """
    Strict prompt: return ONLY the replacement part token (single token).
    """
    base_type = _base_part_type(dropped_part)
    all_parts = _parse_objects_parts(latest_pddl_text)
    needs = _parse_order_needs(latest_pddl_text)
    used_parts = {p for _, p, _ in needs}
    part_locs = _parse_part_on(latest_pddl_text)

    candidates = [p for p in all_parts if (_base_part_type(p) == base_type and p not in used_parts)]
    candidates_with_loc = [(p, part_locs.get(p)) for p in candidates]
    candidates_on_bins = [(p, loc) for p, loc in candidates_with_loc if loc is not None]

    cand_lines = []
    for p, loc in sorted(candidates_on_bins):
        cand_lines.append(f"- {p} @ {loc}")
    if not cand_lines:
        for p, loc in sorted(candidates_with_loc):
            cand_lines.append(f"- {p} @ {loc or 'unknown'}")

    candidates_str = "\n".join(cand_lines) if cand_lines else "(none)"

    prompt = (
        "# Task: Choose ONE replacement part for a dropped part\n"
        "You are given the CURRENT latest PDDL problem (already reflects the current world state).\n"
        "A part was DROPPED during execution and is no longer available.\n\n"
        "Your job is to output ONLY ONE replacement part token (single token), and NOTHING ELSE:\n"
        f"- Target: (order_needs_part {dropped_order} {dropped_part} {dropped_slot})\n"
        f"- Replace {dropped_part} with ONE part of the SAME base type (same color+type, different id allowed).\n"
        "- The replacement part MUST be an existing part object in (:objects) and MUST NOT already appear in any (order_needs_part ...).\n"
        "- Prefer a replacement part that currently appears in (part_on <part> <bin_slot>) (i.e., still on a bin).\n"
        "- Do NOT invent new objects.\n"
        "- Do NOT output any extra words, punctuation, or code fences.\n\n"
        f"DROPPED_CONTEXT:\n- dropped_order: {dropped_order}\n- dropped_part: {dropped_part}\n- dropped_slot: {dropped_slot}\n- dropped_part_base_type: {base_type}\n\n"
        "CANDIDATE_REPLACEMENTS (unused, same base type):\n"
        f"{candidates_str}\n\n"
        "CURRENT_LATEST_PDDL:\n"
        f"{latest_pddl_text}\n"
    )
    return prompt


def get_drop_recovery_replacement(
    *,
    latest_pddl_text: str,
    dropped_part: str,
    dropped_order: str,
    dropped_slot: str,
) -> Optional[str]:
    """
    Pure LLM (no images): return ONLY the replacement part token (single token).
    """
    api_key = load_api_key()
    client = openai.OpenAI(api_key=api_key, base_url=BASE_URL)

    prompt = build_drop_recovery_replacement_prompt(
        latest_pddl_text=latest_pddl_text,
        dropped_part=dropped_part,
        dropped_order=dropped_order,
        dropped_slot=dropped_slot,
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=64,
        )
        content = resp.choices[0].message.content or ""
        token = content.strip().split()[0] if content.strip() else ""
        return token or None
    except Exception:
        # Keep quiet; caller will decide how to log/fail.
        return None


def get_drop_recovery_pddl(
    *,
    latest_pddl_text: str,
    dropped_part: str,
    dropped_order: str,
    dropped_slot: str,
) -> Optional[str]:
    """
    Pure LLM (no images): Given the latest PDDL and a drop context, return a new PDDL that swaps the dropped part
    in (order_needs_part ...) to an existing unused same-type part.
    """
    api_key = load_api_key()
    client = openai.OpenAI(api_key=api_key, base_url=BASE_URL)

    prompt = build_drop_recovery_prompt(
        latest_pddl_text=latest_pddl_text,
        dropped_part=dropped_part,
        dropped_order=dropped_order,
        dropped_slot=dropped_slot,
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            max_tokens=4096,
        )
        content = resp.choices[0].message.content
        return parse_pddl_from_text(content)
    except Exception as e:
        # Keep quiet; caller will decide how to log/fallback.
        return None