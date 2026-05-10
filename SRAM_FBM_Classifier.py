# sram_pattern_classifier.py
# Python 3.7+
#
# Usage:
#   1. Set INPUT_PATH below.
#   2. Run: python SRAM_FBM_Classifier.py
#
# Input required columns:
#   lot_id, wafer_id, test_time, device#, diex, diey, macro,
#   testname, condition, address_dec, fail_bit, physical_x, physical_y

import argparse
import os
from collections import defaultdict, deque

import pandas as pd


INPUT_PATH = r"E:\Programming\Project\SRAM_Classifier\AS00706_55HV_277_normal_failbit.csv"
RESULT_DIR_NAME = "result"


DEFAULT_GROUP_COLS = [
    "lot_id",
    "wafer_id",
    "test_time",
    "device#",
    "diex",
    "diey",
    "macro",
    "testname",
    "condition",
]


class SRAMPatternClassifier:
    def __init__(
        self,
        macro_cols=512,
        macro_rows=512,
        macro_block_ratio=0.30,
        swr_ratio=0.40,
        sbc_ratio=0.40,
        partial_min_len=20,
        cross_min_arm_len=20,
        dotted_min_points=6,
        dotted_min_span=20,
        block_min_size=25,
        block_aspect_low=0.8,
        block_aspect_high=1.2,
        block_fill_ratio=0.60,
        random_density_threshold=200,
        random_density_min_span_ratio=0.30,
        random_density_max_gap=100,
        connectivity=8,
    ):
        self.macro_cols = macro_cols
        self.macro_rows = macro_rows

        self.macro_block_ratio = macro_block_ratio
        self.swr_ratio = swr_ratio
        self.sbc_ratio = sbc_ratio
        self.partial_min_len = partial_min_len
        self.cross_min_arm_len = cross_min_arm_len
        self.dotted_min_points = dotted_min_points
        self.dotted_min_span = dotted_min_span

        self.block_min_size = block_min_size
        self.block_aspect_low = block_aspect_low
        self.block_aspect_high = block_aspect_high
        self.block_fill_ratio = block_fill_ratio

        self.random_density_threshold = random_density_threshold
        self.random_density_min_span_ratio = random_density_min_span_ratio
        self.random_density_max_gap = random_density_max_gap
        self.connectivity = connectivity

        self.pattern_defs = {
            "MBLK" : ( "MACRO_BLOCK"    , 1  ) ,
            "SWR"  : ( "SWR"            , 3  ) ,
            "SBC"  : ( "SBC"            , 3  ) ,
            "PWR"  : ( "PWR"            , 4  ) ,
            "PBC"  : ( "PBC"            , 4  ) ,
            "MWR"  : ( "MWR"            , 5  ) ,
            "MBC"  : ( "MBC"            , 5  ) ,
            "CRS"  : ( "CROSS"          , 6  ) ,
            "BLK"  : ( "BLOCK"          , 7  ) ,
            "QB"   : ( "QUADRA_BIT"     , 8  ) ,
            "RCLU" : ( "RANDOM_CLUSTER" , 9  ) ,
            "DWR"  : ( "DOTTED_ROW"     , 9  ) ,
            "DWC"  : ( "DOTTED_COLUMN"  , 9  ) ,
            "SB"   : ( "SB"             , 10 ) ,
            "DBR"  : ( "DBR"            , 10 ) ,
            "DBC"  : ( "DBC"            , 10 ) ,
            "TB"   : ( "TB"             , 10 ) ,
            "RDEN" : ( "RANDOM_DENSITY" , 11 ) ,
            "UNCL" : ( "UNCLASSIFIED"   , 99 ) ,
        }

    # -----------------------------
    # Public API
    # -----------------------------
    def classify_dataframe(self, df, group_cols=None, show_progress=True):
        if group_cols is None:
            group_cols = [c for c in DEFAULT_GROUP_COLS if c in df.columns]

        self._validate_columns(df)

        all_pattern_rows = []
        all_labeled_rows = []
        grouped = df.groupby(group_cols, dropna=False)
        total_groups = grouped.ngroups
        processed_groups = 0
        last_progress_percent = 0

        if show_progress:
            print("Progress: 0/%d macro (0%%)" % total_groups, end="", flush=True)

        for group_key, g in grouped:
            processed_groups += 1
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            group_info = dict(zip(group_cols, group_key))

            pattern_rows, point_label_map = self.classify_one_macro(g, group_info)

            all_pattern_rows.extend(pattern_rows)

            labeled_g = g.copy()
            labels = []
            codes = []
            pattern_ids = []

            for _, row in labeled_g.iterrows():
                p = (int(row["physical_x"]), int(row["physical_y"]))
                item = point_label_map.get(p)

                if item is None:
                    labels.append("UNCLASSIFIED")
                    codes.append("UNCL")
                    pattern_ids.append("")
                else:
                    labels.append(item["name"])
                    codes.append(item["code"])
                    pattern_ids.append(item["pattern_id"])

            labeled_g["pattern_name"] = labels
            labeled_g["pattern_code"] = codes
            labeled_g["pattern_id"] = pattern_ids

            all_labeled_rows.append(labeled_g)

            if show_progress:
                progress_percent = int(processed_groups * 100 / total_groups)
                if progress_percent != last_progress_percent:
                    print(
                        "\rProgress: %d/%d macro (%d%%)" % (
                            processed_groups,
                            total_groups,
                            progress_percent,
                        ),
                        end="",
                        flush=True,
                    )
                    last_progress_percent = progress_percent

        if show_progress:
            print()

        summary_df = pd.DataFrame(all_pattern_rows)

        if all_labeled_rows:
            labeled_df = pd.concat(all_labeled_rows, ignore_index=True)
        else:
            labeled_df = pd.DataFrame()

        return summary_df, labeled_df

    def classify_one_macro(self, g, group_info):
        points = []
        for _, row in g.iterrows():
            points.append((int(row["physical_x"]), int(row["physical_y"])))

        points = sorted(set(points))
        remaining = set(points)

        pattern_rows = []
        point_label_map = {}
        pattern_index = 1

        def add_pattern(code, pts, extra=None, consume=True):
            nonlocal pattern_index, remaining, pattern_rows, point_label_map

            pts = sorted(set(pts))
            if not pts and code != "RDEN":
                return None

            name, priority = self.pattern_defs[code]
            pattern_id = self._make_pattern_id(pattern_index)
            pattern_index += 1

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]

            row = {}
            row.update(group_info)
            row.update({
                "pattern_id": pattern_id,
                "priority": priority,
                "pattern_name": name,
                "pattern_code": code,
                "fail_count": len(pts),
                "xmin": min(xs) if xs else None,
                "xmax": max(xs) if xs else None,
                "ymin": min(ys) if ys else None,
                "ymax": max(ys) if ys else None,
            })

            if extra:
                row.update(extra)

            pattern_rows.append(row)

            if consume:
                for p in pts:
                    if p in remaining:
                        remaining.remove(p)

                    if p not in point_label_map:
                        point_label_map[p] = {
                            "pattern_id": pattern_id,
                            "name": name,
                            "code": code,
                        }

            return pattern_id

        # 1. MACRO_BLOCK
        comps = self._connected_components(remaining)
        for comp in comps:
            if len(comp) / float(self.macro_cols * self.macro_rows) >= self.macro_block_ratio:
                add_pattern(
                    "MBLK",
                    comp,
                    extra={
                        "rule": "connected_component_fail_count / macro_total_bits >= %.3f" % self.macro_block_ratio
                    },
                    consume=True,
                )

        # 2-5. LINE and CROSS
        line_patterns = self._detect_line_patterns(remaining)

        row_patterns = [p for p in line_patterns if p["kind"] in ("SWR", "PWR", "MWR")]
        col_patterns = [p for p in line_patterns if p["kind"] in ("SBC", "PBC", "MBC")]

        used_line_pattern_ids = set()

        cross_patterns = self._detect_cross_patterns(row_patterns, col_patterns)

        for cp in cross_patterns:
            add_pattern(
                "CRS",
                cp["points"],
                extra={
                    "row_pattern": cp["row_kind"],
                    "col_pattern": cp["col_kind"],
                    "row_arm_span": cp["row_arm_span"],
                    "col_arm_span": cp["col_arm_span"],
                    "row_left_arm": cp["row_left_arm"],
                    "row_right_arm": cp["row_right_arm"],
                    "col_down_arm": cp["col_down_arm"],
                    "col_up_arm": cp["col_up_arm"],
                    "cross_count": cp["cross_count"],
                    "rule": "row/column patterns intersect with two-sided arms around the crossing",
                },
                consume=True,
            )
            used_line_pattern_ids.add(cp["row_id"])
            used_line_pattern_ids.add(cp["col_id"])

        line_patterns = sorted(line_patterns, key=self._line_pattern_strength, reverse=True)

        for lp in line_patterns:
            if lp["id"] in used_line_pattern_ids:
                continue

            line_points = self._remaining_line_points(lp, remaining)
            if not line_points:
                continue

            add_pattern(
                lp["kind"],
                line_points,
                extra=self._line_extra_with_overlap(lp, line_points),
                consume=True,
            )

        # 6. BLOCK
        comps = self._connected_components(remaining)
        for comp in comps:
            if self._is_block(comp):
                add_pattern(
                    "BLK",
                    comp,
                    extra=self._block_extra(comp),
                    consume=True,
                )

        # 7. DOTTED LINE
        dotted_patterns = self._detect_dotted_line_patterns(remaining)
        for dp in dotted_patterns:
            if any(p not in remaining for p in dp["points"]):
                continue

            add_pattern(
                dp["kind"],
                dp["points"],
                extra=dp.get("extra"),
                consume=True,
            )

        # 8. QUADRA_BIT
        comps = self._connected_components(remaining)
        for comp in comps:
            if self._is_quadra_bit(comp):
                add_pattern("QB", comp, consume=True)

        # 9. RANDOM_CLUSTER
        comps = self._connected_components(remaining)
        for comp in comps:
            if len(comp) >= 3 and not self._is_l_shape(comp):
                add_pattern(
                    "RCLU",
                    comp,
                    extra={"rule": "component size >= 3 and not row / column / quadra / block / TB"},
                    consume=True,
                )

        # 10. SB / DBR / DBC / TB
        comps = self._connected_components(remaining)
        for comp in comps:
            if self._is_sb(comp):
                add_pattern("SB", comp, consume=True)
            elif self._is_dbr(comp):
                add_pattern("DBR", comp, consume=True)
            elif self._is_dbc(comp):
                add_pattern("DBC", comp, consume=True)
            elif self._is_l_shape(comp):
                add_pattern("TB", comp, consume=True)
            else:
                add_pattern("UNCL", comp, consume=True)

        # 11. RANDOM_DENSITY
        self._apply_random_density(pattern_rows, point_label_map)

        return pattern_rows, point_label_map

    # -----------------------------
    # Validation
    # -----------------------------
    def _validate_columns(self, df):
        required = ["physical_x", "physical_y"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("Missing required columns: %s" % missing)

    # -----------------------------
    # Pattern ID
    # -----------------------------
    def _make_pattern_id(self, index):
        return index

    def _apply_random_density(self, pattern_rows, point_label_map):
        random_like_codes = set(["SB", "DBR", "DBC", "QB", "TB", "RCLU", "DWR", "DWC"])
        random_like_rows = [r for r in pattern_rows if r["pattern_code"] in random_like_codes]
        random_density_groups = self._random_density_groups(random_like_rows)
        qualified_groups = []

        for group in random_density_groups:
            random_like_count = len(group)
            random_like_fail_count = sum(r["fail_count"] for r in group)

            if (
                random_like_count <= self.random_density_threshold
                and random_like_fail_count <= self.random_density_threshold
            ):
                continue

            spans = self._random_density_group_spans(group)
            if spans is None:
                continue

            random_like_x_span, random_like_y_span, min_x_span, min_y_span = spans

            if random_like_x_span <= min_x_span or random_like_y_span <= min_y_span:
                continue

            qualified_groups.append({
                "rows": group,
                "count": random_like_count,
                "fail_count": random_like_fail_count,
                "x_span": random_like_x_span,
                "y_span": random_like_y_span,
                "min_x_span": min_x_span,
                "min_y_span": min_y_span,
            })

        if not qualified_groups:
            return

        name, priority = self.pattern_defs["RDEN"]
        covered_pattern_id_map = {}

        for group in qualified_groups:
            rden_pattern_id = min(row["pattern_id"] for row in group["rows"])
            for row in group["rows"]:
                original_pattern_id = row["pattern_id"]
                covered_pattern_id_map[original_pattern_id] = rden_pattern_id
                row["original_pattern_id"] = original_pattern_id
                row["original_pattern_name"] = row["pattern_name"]
                row["original_pattern_code"] = row["pattern_code"]
                row["pattern_id"] = rden_pattern_id
                row["pattern_name"] = name
                row["pattern_code"] = "RDEN"
                row["priority"] = priority
                row["random_like_component_count"] = group["count"]
                row["random_like_fail_count"] = group["fail_count"]
                row["random_like_x_span"] = group["x_span"]
                row["random_like_y_span"] = group["y_span"]
                row["random_density_min_x_span"] = group["min_x_span"]
                row["random_density_min_y_span"] = group["min_y_span"]
                row["random_density_max_gap"] = self.random_density_max_gap
                row["threshold"] = self.random_density_threshold
                row["rule"] = "nearby random-like group exceeds count/fail threshold and x/y span threshold"

        for item in point_label_map.values():
            if item["pattern_id"] in covered_pattern_id_map:
                item["pattern_id"] = covered_pattern_id_map[item["pattern_id"]]
                item["name"] = name
                item["code"] = "RDEN"

    def _random_density_groups(self, rows):
        rows = [r for r in rows if self._row_has_bbox(r)]
        if not rows:
            return []

        groups = []
        visited = set()

        for i, row in enumerate(rows):
            if i in visited:
                continue

            visited.add(i)
            q = deque([i])
            group = []

            while q:
                cur = q.popleft()
                group.append(rows[cur])

                for j, other in enumerate(rows):
                    if j in visited:
                        continue

                    if self._bbox_gap(rows[cur], other) <= self.random_density_max_gap:
                        visited.add(j)
                        q.append(j)

            groups.append(group)

        return groups

    def _row_has_bbox(self, row):
        return (
            row.get("xmin") is not None
            and row.get("xmax") is not None
            and row.get("ymin") is not None
            and row.get("ymax") is not None
        )

    def _bbox_gap(self, a, b):
        x_gap = max(0, max(a["xmin"], b["xmin"]) - min(a["xmax"], b["xmax"]))
        y_gap = max(0, max(a["ymin"], b["ymin"]) - min(a["ymax"], b["ymax"]))
        return max(x_gap, y_gap)

    def _random_density_group_spans(self, rows):
        xs = []
        ys = []
        for row in rows:
            if row["xmin"] is not None and row["xmax"] is not None:
                xs.extend([row["xmin"], row["xmax"]])
            if row["ymin"] is not None and row["ymax"] is not None:
                ys.extend([row["ymin"], row["ymax"]])

        if not xs or not ys:
            return

        random_like_x_span = max(xs) - min(xs) + 1
        random_like_y_span = max(ys) - min(ys) + 1
        min_x_span = self.macro_cols * self.random_density_min_span_ratio
        min_y_span = self.macro_rows * self.random_density_min_span_ratio

        return random_like_x_span, random_like_y_span, min_x_span, min_y_span

    # -----------------------------
    # Connected component
    # -----------------------------
    def _neighbors(self, p):
        x, y = p

        if self.connectivity == 4:
            offsets = [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
            ]
        else:
            offsets = [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]

        for dx, dy in offsets:
            yield (x + dx, y + dy)

    def _connected_components(self, points):
        points = set(points)
        visited = set()
        comps = []

        for p in list(points):
            if p in visited:
                continue

            q = deque([p])
            visited.add(p)
            comp = []

            while q:
                cur = q.popleft()
                comp.append(cur)

                for nb in self._neighbors(cur):
                    if nb in points and nb not in visited:
                        visited.add(nb)
                        q.append(nb)

            comps.append(comp)

        return comps

    # -----------------------------
    # LINE / CROSS
    # -----------------------------
    def _detect_line_patterns(self, points):
        points = set(points)

        row_map = defaultdict(list)
        col_map = defaultdict(list)

        for x, y in points:
            row_map[y].append(x)
            col_map[x].append(y)

        base_rows = []
        base_cols = []

        # SWR / PWR
        for y, xs in row_map.items():
            xs = sorted(set(xs))
            coverage = len(xs) / float(self.macro_cols)

            if coverage >= self.swr_ratio:
                row_points = [(x, y) for x in xs]
                base_rows.append({
                    "kind": "SWR",
                    "line": y,
                    "points": row_points,
                    "range_min": min(xs),
                    "range_max": max(xs),
                    "extra": {
                        "axis": "WL",
                        "line": y,
                        "coverage": coverage,
                        "rule": "row fail count / total BL >= %.3f" % self.swr_ratio,
                    },
                })
            else:
                segments = self._continuous_segments(xs)
                long_segments = [seg for seg in segments if len(seg) > self.partial_min_len]
                if long_segments:
                    selected = []
                    for seg in long_segments:
                        selected.extend(seg)
                    selected = sorted(set(selected))

                    row_points = [(x, y) for x in selected]
                    base_rows.append({
                        "kind": "PWR",
                        "line": y,
                        "points": row_points,
                        "range_min": min(selected),
                        "range_max": max(selected),
                        "extra": {
                            "axis": "WL",
                            "line": y,
                            "coverage": coverage,
                            "max_segment_len": max(len(seg) for seg in long_segments),
                            "rule": "continuous segment length > %d and coverage < %.3f"
                                    % (self.partial_min_len, self.swr_ratio),
                        },
                    })

        # SBC / PBC
        for x, ys in col_map.items():
            ys = sorted(set(ys))
            coverage = len(ys) / float(self.macro_rows)

            if coverage >= self.sbc_ratio:
                col_points = [(x, y) for y in ys]
                base_cols.append({
                    "kind": "SBC",
                    "line": x,
                    "points": col_points,
                    "range_min": min(ys),
                    "range_max": max(ys),
                    "extra": {
                        "axis": "BL",
                        "line": x,
                        "coverage": coverage,
                        "rule": "column fail count / total WL >= %.3f" % self.sbc_ratio,
                    },
                })
            else:
                segments = self._continuous_segments(ys)
                long_segments = [seg for seg in segments if len(seg) > self.partial_min_len]
                if long_segments:
                    selected = []
                    for seg in long_segments:
                        selected.extend(seg)
                    selected = sorted(set(selected))

                    col_points = [(x, y) for y in selected]
                    base_cols.append({
                        "kind": "PBC",
                        "line": x,
                        "points": col_points,
                        "range_min": min(selected),
                        "range_max": max(selected),
                        "extra": {
                            "axis": "BL",
                            "line": x,
                            "coverage": coverage,
                            "max_segment_len": max(len(seg) for seg in long_segments),
                            "rule": "continuous segment length > %d and coverage < %.3f"
                                    % (self.partial_min_len, self.sbc_ratio),
                        },
                    })

        patterns = []
        pid = 1

        # MWR
        row_groups = self._group_adjacent_lines(base_rows)
        used_base_row_ids = set()

        for group in row_groups:
            if len(group) >= 2:
                pts = []
                lines = []
                kinds = []

                for item in group:
                    pts.extend(item["points"])
                    lines.append(item["line"])
                    kinds.append(item["kind"])
                    used_base_row_ids.add(id(item))

                patterns.append({
                    "id": "line_%04d" % pid,
                    "kind": "MWR",
                    "points": sorted(set(pts)),
                    "line_min": min(lines),
                    "line_max": max(lines),
                    "range_min": min(min(i["range_min"] for i in group), max(i["range_min"] for i in group)),
                    "range_max": max(i["range_max"] for i in group),
                    "extra": {
                        "axis": "WL",
                        "line_start": min(lines),
                        "line_end": max(lines),
                        "line_count": len(lines),
                        "sub_kinds": ",".join(sorted(set(kinds))),
                        "rule": ">=2 adjacent WL satisfy SWR or PWR",
                    },
                })
                pid += 1

        # MBC
        col_groups = self._group_adjacent_lines(base_cols)
        used_base_col_ids = set()

        for group in col_groups:
            if len(group) >= 2:
                pts = []
                lines = []
                kinds = []

                for item in group:
                    pts.extend(item["points"])
                    lines.append(item["line"])
                    kinds.append(item["kind"])
                    used_base_col_ids.add(id(item))

                patterns.append({
                    "id": "line_%04d" % pid,
                    "kind": "MBC",
                    "points": sorted(set(pts)),
                    "line_min": min(lines),
                    "line_max": max(lines),
                    "range_min": min(i["range_min"] for i in group),
                    "range_max": max(i["range_max"] for i in group),
                    "extra": {
                        "axis": "BL",
                        "line_start": min(lines),
                        "line_end": max(lines),
                        "line_count": len(lines),
                        "sub_kinds": ",".join(sorted(set(kinds))),
                        "rule": ">=2 adjacent BL satisfy SBC or PBC",
                    },
                })
                pid += 1

        # single row lines not included in MWR
        for item in base_rows:
            if id(item) in used_base_row_ids:
                continue

            patterns.append({
                "id": "line_%04d" % pid,
                "kind": item["kind"],
                "points": item["points"],
                "line_min": item["line"],
                "line_max": item["line"],
                "range_min": item["range_min"],
                "range_max": item["range_max"],
                "extra": item["extra"],
            })
            pid += 1

        # single column lines not included in MBC
        for item in base_cols:
            if id(item) in used_base_col_ids:
                continue

            patterns.append({
                "id": "line_%04d" % pid,
                "kind": item["kind"],
                "points": item["points"],
                "line_min": item["line"],
                "line_max": item["line"],
                "range_min": item["range_min"],
                "range_max": item["range_max"],
                "extra": item["extra"],
            })
            pid += 1

        return patterns

    def _continuous_segments(self, values):
        values = sorted(set(values))
        if not values:
            return []

        segments = []
        cur = [values[0]]

        for v in values[1:]:
            if v == cur[-1] + 1:
                cur.append(v)
            else:
                segments.append(cur)
                cur = [v]

        segments.append(cur)
        return segments

    def _group_adjacent_lines(self, line_items):
        line_items = sorted(line_items, key=lambda d: d["line"])

        groups = []
        cur = []

        for item in line_items:
            if not cur:
                cur = [item]
                continue

            if item["line"] == cur[-1]["line"] + 1:
                cur.append(item)
            else:
                groups.append(cur)
                cur = [item]

        if cur:
            groups.append(cur)

        return groups

    def _line_pattern_strength(self, pattern):
        range_span = pattern["range_max"] - pattern["range_min"] + 1
        line_span = pattern["line_max"] - pattern["line_min"] + 1

        return (range_span, len(pattern["points"]), line_span)

    def _remaining_line_points(self, pattern, remaining):
        points = sorted(p for p in pattern["points"] if p in remaining)
        if not points:
            return []

        kind = pattern["kind"]

        if kind in ("SWR", "PWR", "MWR"):
            line_map = defaultdict(list)
            for x, y in points:
                line_map[y].append(x)
            if any(self._has_long_segment(xs) for xs in line_map.values()):
                return points
            if kind == "SWR" and len(points) / float(self.macro_cols) >= self.swr_ratio:
                return points
            return []

        if kind in ("SBC", "PBC", "MBC"):
            line_map = defaultdict(list)
            for x, y in points:
                line_map[x].append(y)
            if any(self._has_long_segment(ys) for ys in line_map.values()):
                return points
            if kind == "SBC" and len(points) / float(self.macro_rows) >= self.sbc_ratio:
                return points
            return []

        return points

    def _has_long_segment(self, values):
        return any(len(seg) > self.partial_min_len for seg in self._continuous_segments(values))

    def _line_extra_with_overlap(self, pattern, points):
        extra = {}
        if pattern.get("extra"):
            extra.update(pattern["extra"])

        original_count = len(pattern["points"])
        remaining_count = len(points)
        overlap_count = original_count - remaining_count

        if overlap_count > 0:
            extra["original_line_fail_count"] = original_count
            extra["overlap_consumed_count"] = overlap_count
            extra["rule"] = extra.get("rule", "") + "; kept remaining line points after overlap with higher-priority line"

        return extra

    def _detect_dotted_line_patterns(self, points):
        points = set(points)
        if not points:
            return []

        row_map = defaultdict(list)
        col_map = defaultdict(list)
        row_component_count = defaultdict(int)
        col_component_count = defaultdict(int)

        for x, y in points:
            row_map[y].append(x)
            col_map[x].append(y)

        for comp in self._connected_components(points):
            for y in set(p[1] for p in comp):
                row_component_count[y] += 1
            for x in set(p[0] for p in comp):
                col_component_count[x] += 1

        row_items = []
        for y, xs in row_map.items():
            item = self._dotted_line_item(
                line=y,
                values=xs,
                axis="WL",
                component_count=row_component_count[y],
                kind="DWR",
            )
            if item is not None:
                item["points"] = [(x, y) for x in sorted(set(xs))]
                row_items.append(item)

        col_items = []
        for x, ys in col_map.items():
            item = self._dotted_line_item(
                line=x,
                values=ys,
                axis="BL",
                component_count=col_component_count[x],
                kind="DWC",
            )
            if item is not None:
                item["points"] = [(x, y) for y in sorted(set(ys))]
                col_items.append(item)

        patterns = []
        for group in self._group_adjacent_lines(row_items):
            pts = []
            lines = []
            for item in group:
                pts.extend(item["points"])
                lines.append(item["line"])

            patterns.append({
                "kind": "DWR",
                "points": sorted(set(pts)),
                "extra": {
                    "axis": "WL",
                    "line_start": min(lines),
                    "line_end": max(lines),
                    "line_count": len(lines),
                    "rule": "same or adjacent WL has sparse clusters over a long span",
                },
            })

        for group in self._group_adjacent_lines(col_items):
            pts = []
            lines = []
            for item in group:
                pts.extend(item["points"])
                lines.append(item["line"])

            patterns.append({
                "kind": "DWC",
                "points": sorted(set(pts)),
                "extra": {
                    "axis": "BL",
                    "line_start": min(lines),
                    "line_end": max(lines),
                    "line_count": len(lines),
                    "rule": "same or adjacent BL has sparse clusters over a long span",
                },
            })

        return sorted(patterns, key=lambda d: len(d["points"]), reverse=True)

    def _dotted_line_item(self, line, values, axis, component_count, kind):
        values = sorted(set(values))
        if len(values) < self.dotted_min_points:
            return None

        span = values[-1] - values[0] + 1
        if span < self.dotted_min_span:
            return None

        if component_count < 3:
            return None

        max_segment_len = max(len(seg) for seg in self._continuous_segments(values))
        if max_segment_len > self.partial_min_len:
            return None

        return {
            "kind": kind,
            "line": line,
            "range_min": values[0],
            "range_max": values[-1],
            "extra": {
                "axis": axis,
                "line": line,
                "point_count": len(values),
                "component_count": component_count,
                "span": span,
                "max_segment_len": max_segment_len,
                "rule": "sparse non-continuous points/clusters form a visual dotted line",
            },
        }

    def _detect_cross_patterns(self, row_patterns, col_patterns):
        cross = []

        for rp in row_patterns:
            rset = set(rp["points"])

            for cp in col_patterns:
                cset = set(cp["points"])
                inter = rset.intersection(cset)

                if not inter:
                    continue

                cross_info = self._valid_crossing(rp, cp, rset, cset, inter)
                if cross_info is None:
                    continue

                pts = sorted(rset.union(cset))
                cross.append({
                    "row_id": rp["id"],
                    "col_id": cp["id"],
                    "row_kind": rp["kind"],
                    "col_kind": cp["kind"],
                    "points": pts,
                    "cross_points": cross_info["cross_points"],
                    "row_arm_span": cross_info["row_arm_span"],
                    "col_arm_span": cross_info["col_arm_span"],
                    "row_left_arm": cross_info["row_left_arm"],
                    "row_right_arm": cross_info["row_right_arm"],
                    "col_down_arm": cross_info["col_down_arm"],
                    "col_up_arm": cross_info["col_up_arm"],
                })

        return self._merge_cross_patterns(cross)

    def _valid_crossing(self, row_pattern, col_pattern, row_points, col_points, inter):
        valid_cross_points = []
        best = None

        for x, y in sorted(inter):
            row_xs = [px for px, py in row_points if py == y]
            col_ys = [py for px, py in col_points if px == x]

            if not row_xs or not col_ys:
                continue

            row_left = x - min(row_xs)
            row_right = max(row_xs) - x
            col_down = y - min(col_ys)
            col_up = max(col_ys) - y

            if min(row_left, row_right, col_down, col_up) < self.cross_min_arm_len:
                continue

            item = {
                "point": (x, y),
                "row_left_arm": row_left,
                "row_right_arm": row_right,
                "col_down_arm": col_down,
                "col_up_arm": col_up,
                "row_arm_span": row_left + row_right + 1,
                "col_arm_span": col_down + col_up + 1,
            }
            valid_cross_points.append((x, y))

            if best is None:
                best = item
            else:
                best_score = min(
                    best["row_left_arm"],
                    best["row_right_arm"],
                    best["col_down_arm"],
                    best["col_up_arm"],
                )
                item_score = min(row_left, row_right, col_down, col_up)
                if item_score > best_score:
                    best = item

        if best is None:
            return None

        best["cross_points"] = valid_cross_points
        return best

    def _merge_cross_patterns(self, cross_patterns):
        if not cross_patterns:
            return []

        merged = []
        used = [False] * len(cross_patterns)

        for i, pattern in enumerate(cross_patterns):
            if used[i]:
                continue

            used[i] = True
            group = [pattern]
            changed = True

            while changed:
                changed = False
                group_points = set()
                group_row_ids = set()
                group_col_ids = set()

                for item in group:
                    group_points.update(item["points"])
                    group_row_ids.add(item["row_id"])
                    group_col_ids.add(item["col_id"])

                for j, other in enumerate(cross_patterns):
                    if used[j]:
                        continue

                    shares_line = (
                        other["row_id"] in group_row_ids
                        or other["col_id"] in group_col_ids
                    )
                    shares_points = bool(group_points.intersection(other["points"]))

                    if shares_line or shares_points:
                        used[j] = True
                        group.append(other)
                        changed = True

            merged.append(self._merge_cross_group(group))

        return merged

    def _merge_cross_group(self, group):
        base = group[0].copy()
        points = set()
        cross_points = []
        row_kinds = set()
        col_kinds = set()

        for item in group:
            points.update(item["points"])
            cross_points.extend(item["cross_points"])
            row_kinds.add(item["row_kind"])
            col_kinds.add(item["col_kind"])

        base["points"] = sorted(points)
        base["cross_points"] = sorted(set(cross_points))
        base["row_kind"] = ",".join(sorted(row_kinds))
        base["col_kind"] = ",".join(sorted(col_kinds))
        base["cross_count"] = len(base["cross_points"])
        return base

    # -----------------------------
    # BLOCK / QB / Small shape
    # -----------------------------
    def _bbox(self, comp):
        xs = [p[0] for p in comp]
        ys = [p[1] for p in comp]
        return min(xs), max(xs), min(ys), max(ys)

    def _is_block(self, comp):
        size = len(comp)

        if size < self.block_min_size:
            return False

        xmin, xmax, ymin, ymax = self._bbox(comp)
        col_span = xmax - xmin + 1
        row_span = ymax - ymin + 1

        if col_span <= 0 or row_span <= 0:
            return False

        aspect = row_span / float(col_span)

        if not (self.block_aspect_low < aspect < self.block_aspect_high):
            return False

        fill_ratio = size / float(row_span * col_span)

        if fill_ratio <= self.block_fill_ratio:
            return False

        return True

    def _block_extra(self, comp):
        xmin, xmax, ymin, ymax = self._bbox(comp)
        col_span = xmax - xmin + 1
        row_span = ymax - ymin + 1
        fill_ratio = len(comp) / float(row_span * col_span)
        aspect = row_span / float(col_span)

        return {
            "row_span": row_span,
            "col_span": col_span,
            "aspect": aspect,
            "fill_ratio": fill_ratio,
            "rule": "size >= %d, %.2f < row_span/col_span < %.2f, fill_ratio > %.2f"
                    % (
                        self.block_min_size,
                        self.block_aspect_low,
                        self.block_aspect_high,
                        self.block_fill_ratio,
                    ),
        }

    def _is_quadra_bit(self, comp):
        if len(comp) != 4:
            return False

        xmin, xmax, ymin, ymax = self._bbox(comp)

        if xmax - xmin + 1 != 2:
            return False

        if ymax - ymin + 1 != 2:
            return False

        expected = set([
            (xmin, ymin),
            (xmin, ymax),
            (xmax, ymin),
            (xmax, ymax),
        ])

        return set(comp) == expected

    def _is_sb(self, comp):
        return len(comp) == 1

    def _is_dbr(self, comp):
        if len(comp) != 2:
            return False

        pts = list(comp)
        x1, y1 = pts[0]
        x2, y2 = pts[1]

        return y1 == y2 and abs(x1 - x2) == 1

    def _is_dbc(self, comp):
        if len(comp) != 2:
            return False

        pts = list(comp)
        x1, y1 = pts[0]
        x2, y2 = pts[1]

        return x1 == x2 and abs(y1 - y2) == 1

    def _is_l_shape(self, comp):
        if len(comp) != 3:
            return False

        xmin, xmax, ymin, ymax = self._bbox(comp)

        if xmax - xmin + 1 != 2:
            return False

        if ymax - ymin + 1 != 2:
            return False

        return True


def read_input(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path)

    raise ValueError("Unsupported file type: %s. Please use a csv file." % ext)


def default_outdir(input_path):
    input_dir = os.path.dirname(os.path.abspath(input_path))
    return os.path.join(input_dir, RESULT_DIR_NAME)


def main():
    parser = argparse.ArgumentParser(                                 )
    parser.add_argument             ("input"                          , nargs="?"   , default=INPUT_PATH     , help="input csv file")
    parser.add_argument             ("--outdir"                       , default=None, help="output directory")

    parser.add_argument             ("--macro-cols"                   , type=int    , default=512            )
    parser.add_argument             ("--macro-rows"                   , type=int    , default=512           )

    parser.add_argument             ("--macro-block-ratio"            , type=float  , default=0.30           )
    parser.add_argument             ("--swr-ratio"                    , type=float  , default=0.40           )
    parser.add_argument             ("--sbc-ratio"                    , type=float  , default=0.40           )
    parser.add_argument             ("--partial-min-len"              , type=int    , default=20             )
    parser.add_argument             ("--cross-min-arm-len"            , type=int    , default=20             )
    parser.add_argument             ("--dotted-min-points"            , type=int    , default=6              )
    parser.add_argument             ("--dotted-min-span"              , type=int    , default=20             )

    parser.add_argument             ("--block-min-size"               , type=int    , default=25             )
    parser.add_argument             ("--block-fill-ratio"             , type=float  , default=0.60           )

    parser.add_argument             ("--random-density-threshold"     , type=int    , default=200            )
    parser.add_argument             ("--random-density-min-span-ratio", type=float  , default=0.30           )
    parser.add_argument             ("--random-density-max-gap"       , type=int    , default=100            )

    args = parser.parse_args()

    input_path = args.input
    outdir = args.outdir or default_outdir(input_path)

    df = read_input(input_path)

    clf = SRAMPatternClassifier(
        macro_cols=args.macro_cols,
        macro_rows=args.macro_rows,
        macro_block_ratio=args.macro_block_ratio,
        swr_ratio=args.swr_ratio,
        sbc_ratio=args.sbc_ratio,
        partial_min_len=args.partial_min_len,
        cross_min_arm_len=args.cross_min_arm_len,
        dotted_min_points=args.dotted_min_points,
        dotted_min_span=args.dotted_min_span,
        block_min_size=args.block_min_size,
        block_fill_ratio=args.block_fill_ratio,
        random_density_threshold=args.random_density_threshold,
        random_density_min_span_ratio=args.random_density_min_span_ratio,
        random_density_max_gap=args.random_density_max_gap,
    )

    summary_df, labeled_df = clf.classify_dataframe(df)

    os.makedirs(outdir, exist_ok=True)

    summary_path = os.path.join(outdir, "pattern_summary.csv")
    labeled_path = os.path.join(outdir, "failbit_labeled.csv")

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    labeled_df.to_csv(labeled_path, index=False, encoding="utf-8-sig")

    print("Done.")
    print("Input:", input_path)
    print("Pattern summary:", summary_path)
    print("Failbit labeled:", labeled_path)


if __name__ == "__main__":
    main()
