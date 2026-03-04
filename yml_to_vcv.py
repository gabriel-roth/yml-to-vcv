#!/Users/gabrielroth/Dev/python-scripts/.venv/bin/python3
"""Convert a MetaModule YML patch to a VCV Rack 2 .vcv file.

Usage:
    yml-to-vcv <patch.yml> [output.vcv] [-r ROWS]

If no output path is given, the .vcv is written next to the .yml.
Modules are placed in the order they appear in the YML, wrapped across
ROWS rows (default 3). Row breaks are chosen to keep row widths roughly
equal; breaks between directly-connected adjacent modules are avoided.
"""

import sys
import json
import yaml
import argparse
from pathlib import Path

# Default module width in HP units used when the lookup table has no entry.
DEFAULT_HP = 8

# VCV Rack version string written to the output file.
VCV_VERSION = "2.5.2"

# Penalty added to the DP cost when a row break would split two adjacent
# modules that share a cable. Large enough to dominate width imbalance
# for typical patches.
CONNECTION_BREAK_PENALTY = 1_000_000

# Target HP per row when auto-detecting row count.
TARGET_ROW_HP = 100

# Load the width lookup table (Plugin:Slug -> HP) from the file next to this script.
# Resolve symlinks so it works when invoked via ~/bin/yml-to-vcv.
_WIDTHS_FILE = Path(__file__).resolve().parent / "module_widths.json"
try:
    MODULE_WIDTHS: dict[str, int] = json.loads(_WIDTHS_FILE.read_text())
except Exception as e:
    print(f"Warning: could not load {_WIDTHS_FILE}: {e}", file=sys.stderr)
    MODULE_WIDTHS = {}


def module_hp(slug: str) -> int:
    """Return the HP width for a 'Plugin:Model' slug, or DEFAULT_HP if unknown."""
    return MODULE_WIDTHS.get(slug, DEFAULT_HP)


def rgb565_to_hex(color565: int) -> str:
    """Convert an RGB565 integer to a VCV-style hex colour string."""
    r = ((color565 >> 11) & 0x1F) * 255 // 31
    g = ((color565 >> 5) & 0x3F) * 255 // 63
    b = (color565 & 0x1F) * 255 // 31
    return f"#{r:02x}{g:02x}{b:02x}"


def parse_slug(slug: str) -> tuple[str, str]:
    """Split 'Brand:Model' into (plugin, model). Falls back to (slug, slug)."""
    if ":" in slug:
        plugin, model = slug.split(":", 1)
        return plugin, model
    return slug, slug


def build_params(slot_idx: int, params_by_module: dict) -> list:
    """Return a sorted params array for a module slot."""
    params = []
    for pid, val in sorted((params_by_module.get(slot_idx) or {}).items()):
        params.append({"id": pid, "value": val})
    return params


def try_parse_json(text: str):
    """Return parsed JSON or None if it can't be decoded (e.g. binary blobs)."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def build_hub_mappings(pd: dict, hub_slot: int) -> dict:
    """
    Translate YML mapped_knobs into the Mappings/KnobSetNames format
    expected by the 4ms VCV Hub module.
    Returns a dict with "Mappings" (array of arrays) and "KnobSetNames".
    """
    NUM_SETS = 8
    knob_sets = pd.get("mapped_knobs") or []
    mappings = []
    knob_set_names = []

    for knob_set in knob_sets[:NUM_SETS]:
        entries = []
        for mk in knob_set.get("set") or []:
            entries.append(
                {
                    "DstModID": mk["module_id"],
                    "DstObjID": mk["param_id"],
                    "SrcModID": hub_slot,
                    "SrcObjID": mk["panel_knob_id"],
                    "RangeMin": mk.get("min", 0),
                    "RangeMax": mk.get("max", 1),
                    "CurveType": mk.get("curve_type", 0),
                    "AliasName": "",
                }
            )
        mappings.append(entries)
        knob_set_names.append(knob_set.get("name", "") or "")

    # Pad to NUM_SETS
    while len(mappings) < NUM_SETS:
        mappings.append([])
        knob_set_names.append("")

    return {"Mappings": mappings, "KnobSetNames": knob_set_names}


def auto_rows(slots: list) -> int:
    """Return a row count targeting TARGET_ROW_HP HP per row."""
    total_hp = sum(module_hp(slug) for _, slug in slots)
    return max(1, -(-total_hp // TARGET_ROW_HP))  # ceiling division


def compute_row_assignments(slots: list, int_cables: list, num_rows: int) -> list[int]:
    """
    Return a list of row indices (one per slot) determined by DP.

    The DP partitions the module sequence into `num_rows` contiguous segments,
    minimising squared deviation from the target row width while applying a
    large penalty whenever a break would fall between two adjacent modules
    that share a cable.
    """
    n = len(slots)
    num_rows = min(num_rows, n)  # can't have more rows than modules

    hps = [module_hp(slug) for _, slug in slots]

    # Prefix sums for O(1) range-width queries.
    prefix = [0] * (n + 1)
    for i, hp in enumerate(hps):
        prefix[i + 1] = prefix[i] + hp

    total_hp = prefix[n]
    target = total_hp / num_rows

    # Build set of adjacent order-pairs that share at least one cable.
    slot_to_order = {int(idx): i for i, (idx, _) in enumerate(slots)}
    connected_adjacent: set[tuple[int, int]] = set()
    for cable in int_cables or []:
        a = slot_to_order.get(cable["out"]["module_id"])
        for in_jack in cable.get("ins") or []:
            b = slot_to_order.get(in_jack["module_id"])
            if a is not None and b is not None and abs(a - b) == 1:
                lo, hi = min(a, b), max(a, b)
                connected_adjacent.add((lo, hi))

    def break_penalty(at: int) -> float:
        """Penalty for starting a new row at order index `at`."""
        return CONNECTION_BREAK_PENALTY if (at - 1, at) in connected_adjacent else 0.0

    def width_cost(start: int, end: int) -> float:
        """Squared deviation of a row's width from the target."""
        w = prefix[end] - prefix[start]
        return (w - target) ** 2

    INF = float("inf")

    # dp[j][i] = min cost to fit the first i modules into j rows.
    dp = [[INF] * (n + 1) for _ in range(num_rows + 1)]
    parent = [[-1] * (n + 1) for _ in range(num_rows + 1)]
    dp[0][0] = 0.0

    for j in range(1, num_rows + 1):
        for i in range(j, n + 1):          # need at least j modules for j rows
            for k in range(j - 1, i):      # row j covers slots[k:i]
                penalty = break_penalty(k) if k > 0 else 0.0
                cost = dp[j - 1][k] + width_cost(k, i) + penalty
                if cost < dp[j][i]:
                    dp[j][i] = cost
                    parent[j][i] = k

    # Backtrack to recover row-start indices.
    row_starts = []
    i = n
    for j in range(num_rows, 0, -1):
        k = parent[j][i]
        # k == 0 means "row starts at the beginning" — already covered by the
        # [0] prepended below, so only collect interior break points.
        if k > 0:
            row_starts.append(k)
        i = k
    row_starts.reverse()
    row_starts = [0] + row_starts  # row_starts[r] = first slot index on row r

    # Convert to per-slot row assignments.
    assignments = [0] * n
    for r, start in enumerate(row_starts):
        end = row_starts[r + 1] if r + 1 < len(row_starts) else n
        for i in range(start, end):
            assignments[i] = r

    return assignments


def convert(yml_path: str, vcv_path: str | None = None, num_rows: int | None = None) -> str:
    """
    Convert a MetaModule YML patch file to a VCV Rack .vcv file.
    Returns the path of the written file.
    """
    with open(yml_path, "r") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Could not parse YAML: {e}") from e

    if not isinstance(data, dict) or "PatchData" not in data:
        raise ValueError("Invalid patch file: missing top-level 'PatchData' key.")

    pd = data["PatchData"]
    patch_name = pd.get("patch_name", Path(yml_path).stem)

    raw_slugs = pd.get("module_slugs") or {}
    slots = sorted(raw_slugs.items(), key=lambda kv: int(kv[0]))

    if num_rows is None:
        num_rows = auto_rows(slots)

    params_by_module: dict[int, dict[int, float]] = {}
    for sk in pd.get("static_knobs") or []:
        mid, pid, val = sk["module_id"], sk["param_id"], sk["value"]
        params_by_module.setdefault(mid, {})[pid] = val

    state_by_module: dict[int, object] = {}
    for ms in pd.get("vcvModuleStates") or []:
        parsed = try_parse_json(ms.get("data", ""))
        if parsed is not None:
            state_by_module[ms["module_id"]] = parsed

    bypassed = set(pd.get("bypassed_modules") or [])
    int_cables = pd.get("int_cables") or []

    # --- Compute multi-row layout ---
    row_assignments = compute_row_assignments(slots, int_cables, num_rows)

    # x_pos resets at the start of each row.
    x_by_row: dict[int, int] = {}

    vcv_modules = []
    for order_idx, (raw_key, slug) in enumerate(slots):
        slot_idx = int(raw_key)
        plugin, model = parse_slug(slug)
        row = row_assignments[order_idx]

        x_pos = x_by_row.get(row, 0)
        x_by_row[row] = x_pos + module_hp(slug)

        module: dict = {
            "id": slot_idx,
            "plugin": plugin,
            "version": "2.0.0",
            "model": model,
            "params": build_params(slot_idx, params_by_module),
            "pos": [x_pos, row],
        }

        # Adjacency links: only within the same row.
        if order_idx > 0 and row_assignments[order_idx - 1] == row:
            module["leftModuleId"] = int(slots[order_idx - 1][0])
        if order_idx < len(slots) - 1 and row_assignments[order_idx + 1] == row:
            module["rightModuleId"] = int(slots[order_idx + 1][0])

        if slot_idx == 0:
            hub_data: dict = {"PatchName": patch_name}
            hub_data.update(build_hub_mappings(pd, slot_idx))
            module["data"] = hub_data
        elif slot_idx in state_by_module:
            module["data"] = state_by_module[slot_idx]

        if slot_idx in bypassed:
            module["bypass"] = True

        vcv_modules.append(module)

    # --- Build VCV cables ---
    vcv_cables = []
    cable_id = 1000
    for cable in int_cables:
        out_jack = cable["out"]
        raw_color = cable.get("color", 61865)
        hex_color = rgb565_to_hex(raw_color) if isinstance(raw_color, int) else raw_color
        for in_jack in cable.get("ins") or []:
            vcv_cables.append(
                {
                    "id": cable_id,
                    "outputModuleId": out_jack["module_id"],
                    "outputId": out_jack["jack_id"],
                    "inputModuleId": in_jack["module_id"],
                    "inputId": in_jack["jack_id"],
                    "color": hex_color,
                }
            )
            cable_id += 1

    vcv_patch = {
        "version": VCV_VERSION,
        "modules": vcv_modules,
        "cables": vcv_cables,
    }

    if vcv_path is None:
        vcv_path = str(Path(yml_path).with_suffix(".vcv"))

    with open(vcv_path, "w") as f:
        json.dump(vcv_patch, f, indent=2)

    return vcv_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a MetaModule YML patch to a VCV Rack 2 .vcv file."
    )
    parser.add_argument("yml", help="Input .yml patch file")
    parser.add_argument("vcv", nargs="?", help="Output .vcv file (default: next to yml)")
    parser.add_argument(
        "-r", "--rows",
        type=int,
        default=None,
        metavar="N",
        help="Number of rows to lay modules across (default: auto)",
    )
    args = parser.parse_args()

    try:
        out = convert(args.yml, args.vcv, num_rows=args.rows)
    except (ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
