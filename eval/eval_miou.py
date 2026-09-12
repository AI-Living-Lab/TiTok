#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_miou.py — multi-segment temporal-grounding 통합 평가기 (3-method).

입력: inference 결과(test_results_rank0.json; 단일 파일 또는 하위 재귀 탐색할 폴더)
출력: 각 결과 파일 옆에 세 개의 summary JSON (구조 동일, method/IoU 계산만 다름)
  - pairwise_miou_summary.json : MUSEG 방식 (GT 세그먼트별 best-match pred IoU)
  - union_miou_summary.json    : 기존 방식   (GT 세그먼트 vs 겹치는 pred 합집합 IoU)
  - sample_miou_summary.json   : R-AVST all_iou 방식 (전체 GT merge vs 전체 pred merge)

집계 단위: 전부 '샘플 단위' (샘플마다 값 산출 후 샘플 평균).

pred 파싱:
  - 기본       : SALMONN 시간 토큰  "From <t..> to <t..>. ..."
  - --natural  : 관대 파싱 (시간토큰 + HH:MM:SS / M:SS / X.X-Y.Y / from X to Y seconds /
                 second{X} to second{Y} / first N seconds 등 다양한 자연어 출력)
  - CoT 출력(<answer>...</answer> 포함)이면 <answer> 안의 내용만 파싱 (자동 감지).

GT 소스 해석 순서:
  1) --test_json 명시 시 그 파일
  2) results 항목에 gt_segments 가 박혀 있으면 그대로 사용
  3) results 옆 리프 폴더명으로 <test_dir>/<리프>.json (또는 <리프>/_full.json)
  매칭: (basename(id/video), gt_label) 키 → 실패 시 동일 길이일 때 positional fallback.

사용:
  python3 eval_miou.py <OUT_DIR 또는 test_results_rank0.json> --test_json <GT.json>
  python3 eval_miou.py <OUT_DIR> --test_json <GT.json> --natural
"""
import argparse
import datetime as _dt
import json
import os
import re

THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
METHODS = ["pairwise", "union", "sample"]
OUT_NAMES = {
    "pairwise": "pairwise_miou_summary.json",
    "union": "union_miou_summary.json",
    "sample": "sample_miou_summary.json",
}
MAX_TIME_DEFAULT = 9999.9


# ============================== pred 파싱 ==============================
def extract_answer_scope(text):
    """CoT: <answer>...</answer> 가 있으면 그 안의 내용만, 없으면 전체 텍스트."""
    spans = re.findall(r"<answer>(.*?)</answer>", text or "", re.DOTALL | re.IGNORECASE)
    if spans:
        return " ".join(spans)
    return text or ""


def _fix(s, e, max_time):
    """비정상 구간(e<=s)은 보수적으로 최소 0.1s 만 부여(점수 이득 최소화)."""
    if e <= s:
        e = min(s + 0.1, max_time)
    return [min(s, max_time), min(e, max_time)]


# ---- 시간 토큰 구간 ----
# "From <t..> to <t..>" 뿐 아니라 "<t..> - <t..>"(dash) 등 구분자 변형도 매칭.
# (v0 프롬프트엔 answer-format 지시가 없어 모델이 'in <t..> - <t..> seconds' 식으로 냄)
_TOKTIME = r"(?:<t\d>)+(?:<tdot>(?:<t\d>)+)?"
_TOK_SEG = re.compile(rf"({_TOKTIME})\s*(?:to|-|–|—|~)\s*({_TOKTIME})", re.IGNORECASE)


def _decode_tok(token_str, max_time):
    if "<tdot>" in token_str:
        a, _, b = token_str.partition("<tdot>")
        ip = re.findall(r"<t(\d)>", a)
        dp = re.findall(r"<t(\d)>", b)
    else:
        ip = re.findall(r"<t(\d)>", token_str)
        dp = []
    if not ip:
        return None
    return min(int("".join(ip)) + (int(dp[0]) / 10.0 if dp else 0.0), max_time)


def parse_tokens(text, max_time):
    out = []
    for sa, sb in _TOK_SEG.findall(text):
        s, e = _decode_tok(sa, max_time), _decode_tok(sb, max_time)
        if s is None or e is None:
            continue
        out.append(_fix(s, e, max_time))
    return out


# ---- 자연어 관대 파싱 ----
_HMS = r"(?:\d+:)?\d{1,2}:\d{2}"


def _hms_to_sec(tok):
    s = 0
    for p in (int(x) for x in tok.split(":")):
        s = s * 60 + p
    return float(s)


def parse_natural(text, max_time):
    """다양한 자연어 시간 출력 → [[s,e]]. consumed 마스킹으로 중복 매칭 방지.

    우선순위(구체적 → 일반): 시간토큰 → HH:MM:SS → M:SS → second{} → from X to Y(sec) →
    X-Y seconds → first N seconds.
    """
    segs = list(parse_tokens(text, max_time))  # 토큰이 섞여 있어도 회수
    consumed = [False] * len(text)

    def grab(pattern, conv):
        for m in re.finditer(pattern, text, re.IGNORECASE):
            a, b = m.span()
            if any(consumed[a:b]):
                continue
            r = conv(m)
            if r is not None:
                segs.append(r)
                for i in range(a, b):
                    consumed[i] = True

    # HH:MM:SS - HH:MM:SS  (3-필드 우선)
    grab(r"(\d{1,2}:\d{2}:\d{2})\s*(?:to|-|–|—|~)\s*(\d{1,2}:\d{2}:\d{2})",
         lambda m: _fix(_hms_to_sec(m.group(1)), _hms_to_sec(m.group(2)), max_time))
    # M:SS (to|-|...) M:SS
    grab(rf"({_HMS})\s*(?:to|-|–|—|~)\s*({_HMS})",
         lambda m: _fix(_hms_to_sec(m.group(1)), _hms_to_sec(m.group(2)), max_time))
    # second{X} to/- second{Y}
    grab(r"second\{?\s*([\d.]+)\s*\}?\s*(?:to|-)\s*second\{?\s*([\d.]+)\s*\}?",
         lambda m: _fix(float(m.group(1)), float(m.group(2)), max_time))
    # from X to Y (seconds)
    grab(r"from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(?:seconds|secs|sec)?\b",
         lambda m: _fix(float(m.group(1)), float(m.group(2)), max_time))
    # X-Y seconds
    grab(r"\b(\d+(?:\.\d+)?)\s*(?:to|-|–|—)\s*(\d+(?:\.\d+)?)\s*(?:seconds|secs|sec)\b",
         lambda m: _fix(float(m.group(1)), float(m.group(2)), max_time))
    # X.X-Y.Y  (소수 초 구간, 단위어 없는 prose 안 포함)
    grab(r"(\d+(?:\.\d+)?)\s*(?:to|-|–|—)\s*(\d+(?:\.\d+)?)",
         lambda m: _fix(float(m.group(1)), float(m.group(2)), max_time))
    # first N seconds -> [0, N]
    grab(r"first\s+(\d+(?:\.\d+)?)\s*(?:seconds|secs|sec)\b",
         lambda m: _fix(0.0, float(m.group(1)), max_time))

    segs.sort()
    return segs


def parse_pred(raw, natural, max_time):
    scope = extract_answer_scope(raw)
    return parse_natural(scope, max_time) if natural else parse_tokens(scope, max_time)


# ============================== 구간 연산 ==============================
def tiou(a, b):
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def merge_intervals(intervals):
    if not intervals:
        return []
    s = sorted([list(x) for x in intervals])
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def total_len(intervals):
    return sum(max(0.0, b - a) for a, b in intervals)


def intersect_lists(A, B):
    out, i, j = [], 0, 0
    while i < len(A) and j < len(B):
        s = max(A[i][0], B[j][0])
        e = min(A[i][1], B[j][1])
        if e > s:
            out.append([s, e])
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return out


def union_iou(seg, others):
    """seg 1개 vs 그와 겹치는 others 들의 합집합 IoU."""
    ov = [o for o in others if min(o[1], seg[1]) > max(o[0], seg[0])]
    if not ov:
        return 0.0
    U = merge_intervals(ov)
    G = [list(seg)]
    inter = total_len(intersect_lists(G, U))
    uni = total_len(merge_intervals(G + U))
    return inter / uni if uni > 0 else 0.0


def best_iou(seg, others):
    return max((tiou(seg, o) for o in others), default=0.0)


def sample_iou_parts(gt, pred):
    """all_iou(R-AVST): 전체 GT merge vs 전체 pred merge → (iou, recall_t, precision_t)."""
    G = merge_intervals(gt)
    P = merge_intervals(pred)
    inter = total_len(intersect_lists(G, P))
    gt_len, pred_len = total_len(G), total_len(P)
    uni = total_len(merge_intervals(G + P))
    iou = inter / uni if uni > 0 else 0.0
    rec = inter / gt_len if gt_len > 0 else 0.0
    prc = inter / pred_len if pred_len > 0 else 0.0
    return iou, rec, prc


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _clip01(x):
    return max(0.0, min(1.0, x))


# ============================== method별 집계 ==============================
def _gt_ious(gt, pred, method):
    if method == "pairwise":
        return [best_iou(g, pred) for g in gt]
    return [union_iou(g, pred) for g in gt]  # union


def _pred_ious(gt, pred, method):
    if method == "pairwise":
        return [best_iou(p, gt) for p in pred]
    return [union_iou(p, gt) for p in pred]  # union


def _frac_ge(arr, th):
    return _mean([1.0 if x >= th else 0.0 for x in arr]) if arr else 0.0


def compute_method_block(samples, method):
    """method별 mIoU / Recall / Precision / F1.

    집계 단위 (기존 eval_all_miou_multiseg 의 해당 block 과 일치):
      - pairwise (Best_IoU, GT세그먼트 단위) : 전체 GT 세그먼트를 한 풀로 평균
      - union    (Union_IoU, GT세그먼트 단위): 〃
      - sample   (All_IoU, 샘플 단위)        : 샘플당 all_iou 1개를 샘플 평균
    Recall = 해당 단위에서 (method-IoU ≥ θ) 비율 → 기존 R@1 과 동일.
    Precision = pred 측 단위에서 (IoU ≥ θ) 비율, F1 = 2PR/(P+R).
    """
    if method == "sample":
        # All_IoU 는 대칭 집합 IoU(=inter/union) → 샘플당 1값. P=R=F1=(all_iou≥θ 샘플비율).
        ious = [sample_iou_parts(gt, pred)[0] for gt, pred in samples if gt]
        miou = 100.0 * _mean(ious)
        rec = {str(t): round(100.0 * _frac_ge(ious, t), 4) for t in THRESHOLDS}
        prc = dict(rec)
        f1 = dict(rec)
    else:
        gt_iou_all, pred_iou_all = [], []   # 전체 GT/pred 세그먼트 풀
        for gt, pred in samples:
            if not gt:
                continue
            gt_iou_all.extend(_gt_ious(gt, pred, method))   # GT 세그먼트별 IoU
            pred_iou_all.extend(_pred_ious(gt, pred, method))  # pred 세그먼트별 IoU
        miou = 100.0 * _mean(gt_iou_all)
        rec, prc, f1 = {}, {}, {}
        for t in THRESHOLDS:
            r = _frac_ge(gt_iou_all, t)
            p = _frac_ge(pred_iou_all, t)
            rec[str(t)] = round(100.0 * r, 4)
            prc[str(t)] = round(100.0 * p, 4)
            f1[str(t)] = round(100.0 * (2 * r * p / (r + p) if (r + p) > 0 else 0.0), 4)

    return {
        "method": method,
        "mIoU_%": round(miou, 4),
        "Recall_avg_%": round(_mean(list(rec.values())), 4),
        "Precision_avg_%": round(_mean(list(prc.values())), 4),
        "F1_avg_%": round(_mean(list(f1.values())), 4),
        "Recall": rec,
        "Precision": prc,
        "F1": f1,
    }


def compute_count_metrics(samples):
    """CountF1 및 구성요소(CR*, FMR) — 샘플별 (N_gt, N_pred) 개수만 사용(위치 무관).

    서브셋(정답 개수 기준):
      - 멀티  S  = {i : N_gt >= 2}
      - 싱글  S1 = {i : N_gt == 1}
      - N_gt == 0 샘플은 양 서브셋 모두에서 제외.

    1) CR_multi = mean_{i∈S} min(N_gt,N_pred)/max(N_gt,N_pred)
       chance_floor b = mean_{i∈S} 1/N_gt   ("무조건 1개 찍기"의 기대 점수)
       CR* = (CR_multi - b)/(1 - b), [0,1] 로 클립(찍기보다 못하면 0).
    2) FMR = mean_{i∈S1} 1[N_pred >= 2]      (단일을 쪼갠 비율; 낮을수록 좋음)
    3) SingleAcc = 1 - FMR
    4) CountF1 = 조화평균(CR*, SingleAcc) = 2·CR*·SingleAcc/(CR*+SingleAcc),
       분모 0(둘 다 0)이면 CountF1 = 0.

    서브셋이 비면 해당 지표는 None → CountF1 도 None(표에선 '-').
    """
    multi_ratios, multi_floor, single_split = [], [], []
    for gt, pred in samples:
        ng, npd = len(gt), len(pred)
        if ng >= 2:
            hi = max(ng, npd)
            multi_ratios.append(min(ng, npd) / hi if hi > 0 else 0.0)
            multi_floor.append(1.0 / ng)
        elif ng == 1:
            single_split.append(1.0 if npd >= 2 else 0.0)

    out = {"n_multi": len(multi_ratios), "n_single": len(single_split),
           "CR_multi": None, "chance_floor": None, "CR_star": None,
           "FMR": None, "SingleAcc": None, "CountF1": None}

    if multi_ratios:
        cr = _mean(multi_ratios)
        b = _mean(multi_floor)
        out["CR_multi"] = round(cr, 6)
        out["chance_floor"] = round(b, 6)
        out["CR_star"] = round(_clip01((cr - b) / (1 - b)) if (1 - b) > 0 else 0.0, 6)
    if single_split:
        fmr = _mean(single_split)
        out["FMR"] = round(fmr, 6)
        out["SingleAcc"] = round(1.0 - fmr, 6)

    cs, sa = out["CR_star"], out["SingleAcc"]
    if cs is not None and sa is not None:
        denom = cs + sa
        out["CountF1"] = round(2 * cs * sa / denom, 6) if denom > 0 else 0.0
    return out


def compute_shared_block(samples):
    """method 무관 공통: FP/FN(best-match overlap 기준, threshold별), 세그먼트 분포, 파싱."""
    total_gt = total_pred = 0
    n_fn = {t: 0 for t in THRESHOLDS}
    n_fp = {t: 0 for t in THRESHOLDS}
    gt_counts, pred_counts = [], []
    parse_ok = parse_fail = 0

    for gt, pred in samples:
        gt_counts.append(len(gt))
        pred_counts.append(len(pred))
        parse_ok += 1 if pred else 0
        parse_fail += 0 if pred else 1
        total_gt += len(gt)
        total_pred += len(pred)
        for g in gt:
            bi = best_iou(g, pred)
            for t in THRESHOLDS:
                if bi < t:
                    n_fn[t] += 1
        for p in pred:
            bi = best_iou(p, gt)
            for t in THRESHOLDS:
                if bi < t:
                    n_fp[t] += 1

    n = len(samples)

    def buckets(counts):
        b = {"0": 0, "1": 0, "2": 0, "3": 0, "4+": 0}
        for c in counts:
            b["4+" if c >= 4 else str(c)] += 1
        return {k: round(100.0 * v / max(n, 1), 4) for k, v in b.items()}

    fp_rate = {str(t): round(100.0 * n_fp[t] / max(total_pred, 1), 4) for t in THRESHOLDS}
    fn_rate = {str(t): round(100.0 * n_fn[t] / max(total_gt, 1), 4) for t in THRESHOLDS}

    return {
        "n_samples": n,
        "count_metrics": compute_count_metrics(samples),
        "FP_rate_avg_%": round(_mean(list(fp_rate.values())), 4),
        "FN_rate_avg_%": round(_mean(list(fn_rate.values())), 4),
        "FP_rate_%": fp_rate,
        "FN_rate_%": fn_rate,
        "n_fp_segments": {str(t): n_fp[t] for t in THRESHOLDS},
        "n_fn_segments": {str(t): n_fn[t] for t in THRESHOLDS},
        "gt_segments": {
            "total": total_gt,
            "mean_per_sample": round(_mean(gt_counts), 4),
            "count_buckets_pct": buckets(gt_counts),
        },
        "pred_segments": {
            "total": total_pred,
            "mean_per_sample": round(_mean(pred_counts), 4),
            "count_buckets_pct": buckets(pred_counts),
        },
        "parse_ok": parse_ok,
        "parse_fail": parse_fail,
        "parse_fail_rate_%": round(100.0 * parse_fail / max(n, 1), 4),
    }


def build_summaries(samples, meta):
    """method별 summary dict 3개 반환 ({method: summary})."""
    shared = compute_shared_block(samples)
    out = {}
    for method in METHODS:
        block = compute_method_block(samples, method)
        summary = {
            "label": meta.get("label", ""),
            "testset": meta.get("testset", ""),
            "method": method,
            "n_samples": shared["n_samples"],
        }
        summary.update({k: v for k, v in block.items() if k != "method"})
        summary["thresholds"] = THRESHOLDS
        summary.update({k: v for k, v in shared.items() if k != "n_samples"})
        summary.update({k: v for k, v in meta.items()
                        if k not in ("label", "testset")})
        out[method] = summary
    return out


# ============================== GT 소스 해석 ==============================
def _basename(p):
    return os.path.basename(str(p)) if p else ""


def _coerce_segments(val):
    if not val:
        return []
    if isinstance(val, str):
        out = []
        for m in re.finditer(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)", val):
            out.append([float(m.group(1)), float(m.group(2))])
        return out
    out = []
    for s in val:
        try:
            out.append([float(s[0]), float(s[1])])
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _default_test_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    # <WS>/master/Team4/eval -> <WS>/data/test
    return os.path.normpath(os.path.join(here, "..", "..", "..", "data", "test"))


def resolve_gt_json_path(results_file, test_dir):
    leaf = os.path.basename(os.path.dirname(os.path.abspath(results_file)))
    flat = os.path.join(test_dir, f"{leaf}.json")
    if os.path.isfile(flat):
        return flat
    nested = os.path.join(test_dir, leaf, "_full.json")
    return nested if os.path.isfile(nested) else None


def _gt_from_ref(results, max_time):
    """GT 를 각 항목 'ref'(SALMONN 시간토큰) 에서 파싱."""
    gt_list = [parse_tokens(r.get("ref", "") or "", max_time) for r in results]
    return gt_list, sum(1 for g in gt_list if g), sum(1 for g in gt_list if not g)


def _gt_from_test_json(results, gt_path, source_tag, allow_positional=False):
    """test_json 의 gt_segments 매칭.

    매칭 순서: (video/id, gt_label) 키 → video 단독(모호하지 않을 때) →
    (allow_positional 일 때만) positional. 묵시적 positional 은 순서 어긋난 결과에서
    엉뚱한 GT 와 비교하는 사고를 내므로 기본 비활성.
    """
    gt_data = _load_json(gt_path)
    key_map, ambiguous = {}, set()          # (video, gt_label) -> segs
    vid_map, vid_amb = {}, set()            # video -> segs (단독 fallback)
    for item in gt_data:
        vid = _basename(item.get("video") or item.get("id") or item.get("audio"))
        segs = _coerce_segments(item.get("gt_segments"))
        key = (vid, item.get("gt_label", ""))
        if key in key_map and key_map[key] != segs:
            ambiguous.add(key)
        key_map[key] = segs
        if vid in vid_map and vid_map[vid] != segs:
            vid_amb.add(vid)               # 같은 video 에 다른 segs → 단독 매칭 불가
        vid_map[vid] = segs

    gt_list, matched, unmatched, by_pos = [], 0, 0, 0
    for i, r in enumerate(results):
        vid = _basename(r.get("id") or r.get("video") or r.get("audio"))
        gl = r.get("gt_label", "")
        key = (vid, gl)
        if gl and key in key_map and key not in ambiguous:
            gt_list.append(key_map[key]); matched += 1
        elif vid and vid in vid_map and vid not in vid_amb:
            gt_list.append(vid_map[vid]); matched += 1
        elif allow_positional and len(gt_data) == len(results):
            gt_list.append(_coerce_segments(gt_data[i].get("gt_segments")))
            matched += 1; by_pos += 1
        else:
            gt_list.append([]); unmatched += 1
    if by_pos:
        print(f"[WARN] {os.path.basename(gt_path)}: {by_pos} 샘플을 positional 로 매칭 "
              f"(키/ video 매칭 실패). 순서가 어긋나면 GT 오정렬 위험.")
    return gt_list, source_tag, matched, unmatched


def build_gt_for_results(results, results_file, args):
    """GT 소스 자동감지 우선순위:
    --gt_ref(강제) > embedded(gt_segments) > ref(자동) > --test_json/리프명 test_json.

    ref/embedded 가 test_json 보다 우선인 이유: 둘 다 '각 샘플 자신의' GT 라 순서/매칭
    문제가 없다. test_json 은 (video,gt_label) 매칭에 의존하며, gt_label 없는 결과에서는
    취약하다(과거 positional fallback 으로 오정렬 사고가 있었음).
    """
    n = len(results)
    if args.gt_ref:
        gl, m, u = _gt_from_ref(results, args.max_time)
        return gl, "ref(forced)", m, u
    if all(r.get("gt_segments") is not None for r in results):
        gl = [_coerce_segments(r.get("gt_segments")) for r in results]
        return gl, "embedded", sum(1 for g in gl if g), sum(1 for g in gl if not g)
    if all(r.get("ref") for r in results):
        gl, m, u = _gt_from_ref(results, args.max_time)
        # ref 가 실제 SALMONN 토큰 GT 로 파싱될 때만 사용. HH:MM:SS 등 자연어 ref 는
        # parse_tokens 로 0개 → test_json(gt_segments) 로 폴백.
        if m >= 0.5 * n:
            return gl, "ref(auto)", m, u
    gt_path = args.test_json or resolve_gt_json_path(results_file, args.test_dir)
    if not gt_path or not os.path.isfile(gt_path):
        raise FileNotFoundError(
            f"GT 를 찾을 수 없음 (results={results_file}). embedded gt_segments/ref 없음, "
            f"리프명 매칭 실패. --test_json 또는 --gt_ref 로 지정하세요. (test_dir={args.test_dir})"
        )
    gl, src, m, u = _gt_from_test_json(results, gt_path, gt_path, args.allow_positional)
    if u > 0:
        print(f"[WARN] test_json 매칭 실패 {u}/{n} (gt=[] 로 제외). "
              f"이 결과가 ref/gt_segments 를 가지면 그쪽이 더 정확합니다.")
    return gl, src, m, u


# ============================== 한 폴더 평가 ==============================
def write_and_report(results_file, summaries, args):
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(results_file))
    os.makedirs(out_dir, exist_ok=True)
    for method in METHODS:
        path = os.path.join(out_dir, OUT_NAMES[method])
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(summaries[method], f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    if not args.quiet:
        s0 = summaries["pairwise"]
        SEP = "=" * 60
        print(f"\n{SEP}")
        print(f"  {results_file}")
        print(f"  GT: {s0.get('gt_source')} (matched={s0.get('gt_matched')}, "
              f"unmatched={s0.get('gt_unmatched')})  natural={s0.get('natural')}")
        print(f"  Samples: {s0['n_samples']}  GT segs: {s0['gt_segments']['total']}  "
              f"Pred segs: {s0['pred_segments']['total']}  "
              f"parse_ok/fail: {s0['parse_ok']}/{s0['parse_fail']}  "
              f"parse-fail rate: {s0['parse_fail_rate_%']:.2f}%")
        print(f"  {'method':9s} | mIoU%  | R@.1/.3/.5/.7  | F1_avg | FP/FN_avg%")
        for m in METHODS:
            s = summaries[m]
            r = s["Recall"]
            print(f"  {m:9s} | {s['mIoU_%']:5.2f} | "
                  f"{r['0.1']:.1f}/{r['0.3']:.1f}/{r['0.5']:.1f}/{r['0.7']:.1f} | "
                  f"{s['F1_avg_%']:5.2f} | {s['FP_rate_avg_%']:.1f}/{s['FN_rate_avg_%']:.1f}")
        print(f"  [SAVED] {', '.join(OUT_NAMES[m] for m in METHODS)}  -> {out_dir}")
        print(SEP)

    if args.progress_log:
        snap = {"timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
                "results_file": results_file,
                **{m: summaries[m] for m in METHODS}}
        with open(args.progress_log, "a") as f:
            f.write(json.dumps(snap, ensure_ascii=False) + "\n")


def _scale_percent(segs, dur, max_time):
    """avicuna 류: pred 구간이 영상 길이 대비 %(0~100) → 초로 환산."""
    if not dur:
        return []
    return [_fix(s / 100.0 * dur, e / 100.0 * dur, max_time) for s, e in segs]


def evaluate_one(results_file, args):
    results = _load_json(results_file)
    gt_list, gt_source, matched, unmatched = build_gt_for_results(results, results_file, args)
    samples = []
    for r, gt in zip(results, gt_list):
        pred = parse_pred(r.get("pred", ""), args.natural, args.max_time)
        if args.pred_percent:
            pred = _scale_percent(pred, r.get(args.duration_key), args.max_time)
        samples.append((gt, pred))
    testset = args.testset or os.path.basename(os.path.dirname(os.path.abspath(results_file)))
    meta = {"label": args.label, "testset": testset,
            "gt_source": gt_source, "gt_matched": matched,
            "gt_unmatched": unmatched, "natural": bool(args.natural)}
    summaries = build_summaries(samples, meta)
    write_and_report(results_file, summaries, args)
    return summaries


def find_results_files(root):
    if os.path.isfile(root):
        return [root]
    found = []
    for dirpath, _dirs, files in os.walk(root):
        if "test_results_rank0.json" in files:
            found.append(os.path.join(dirpath, "test_results_rank0.json"))
    return sorted(found)


def main():
    ap = argparse.ArgumentParser(
        description="multi-segment 통합 mIoU 평가 (pairwise/union/sample 3-method)")
    ap.add_argument("results", help="test_results_rank0.json 또는 하위 재귀 탐색할 폴더")
    ap.add_argument("--test_json", default=None, help="GT json (생략 시 embedded/리프명 자동)")
    ap.add_argument("--test_dir", default=None, help="GT json 탐색 디렉토리 (기본 <WS>/data/test)")
    ap.add_argument("--natural", action="store_true",
                    help="관대 파싱(HH:MM:SS/M:SS/소수초/second{} 등). 기본 off=시간토큰만")
    ap.add_argument("--label", default="", help="summary 에 기록할 모델 라벨 (예: CKPT_MODEL_ID)")
    ap.add_argument("--testset", default="", help="summary 에 기록할 testset 태그")
    ap.add_argument("--gt_ref", action="store_true",
                    help="GT 를 각 항목 'ref'(SALMONN 시간토큰)에서 파싱 (강제). 미지정 시 자동감지")
    ap.add_argument("--pred_percent", action="store_true",
                    help="pred 구간을 영상길이 대비 %%로 보고 duration 곱해 초 환산 (avicuna)")
    ap.add_argument("--allow_positional", action="store_true",
                    help="test_json 매칭 실패 시 순서대로(positional) GT 끼움 허용 (위험; 기본 off)")
    ap.add_argument("--duration_key", default="duration",
                    help="--pred_percent 환산용 duration 필드명")
    ap.add_argument("--max_time", type=float, default=MAX_TIME_DEFAULT)
    ap.add_argument("--out_dir", default=None, help="단일 결과 파일일 때만 의미. 생략 시 결과 옆")
    ap.add_argument("--progress_log", default=None, help="JSONL 스냅샷 append")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.test_dir is None:
        args.test_dir = _default_test_dir()

    files = find_results_files(args.results)
    if not files:
        raise SystemExit(f"[에러] test_results_rank0.json 을 찾지 못함: {args.results}")
    # 폴더 재귀 모드에서는 out_dir 를 각 결과 옆으로 강제(혼선 방지)
    if len(files) > 1:
        args.out_dir = None

    if not args.quiet:
        print(f"[INFO] {len(files)} 개 결과 파일  (natural={args.natural})")

    for rf in files:
        try:
            evaluate_one(rf, args)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {rf}: {e}")


if __name__ == "__main__":
    main()
