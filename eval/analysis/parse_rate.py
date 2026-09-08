#!/usr/bin/env python3
"""baseline 모델별 output parsing 성공률.

parse 정의 = 채점기(Team4/eval/eval_miou.py)와 동일:
  extract_answer_scope(<answer> 있으면 그 안) -> parse_natural(관대 파싱)
  세그먼트 1개 이상 회수 => parse 성공 (실제 채점에서 이 샘플이 IoU 0 이 아니게 되는 조건).
컬럼:
  parse율    : 관대 파싱으로 세그가 1개 이상 나온 비율 (= 채점에 실제로 쓰인 성공률)
  포맷준수   : 프롬프트가 지시한 answer format 을 그대로 지킨 비율(엄격)
  구제       : 포맷은 어겼지만 관대 파싱으로 살린 샘플 수
  빈출력     : pred 가 빈 문자열
  degen      : 파싱은 됐지만 end<=start 라 0.1s 로 보정된 세그 수
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Team4/eval (스크립트 위치 기준)
WS = os.environ.get("WORKSPACE", os.path.dirname(os.path.dirname(HERE)))  # workspace
sys.path.insert(0, HERE)
import eval_miou as EM

BASE = os.path.join(WS, "outputs", "base")
MAX_T = EM.MAX_TIME_DEFAULT

# --- end<=start 세그를 세기 위해 _fix 를 계측 버전으로 교체 ---
_orig_fix = EM._fix
_degen = [0]


def _fix_counting(s, e, max_time):
    if e <= s:
        _degen[0] += 1
    return _orig_fix(s, e, max_time)


EM._fix = _fix_counting

# 모델별 '지시한 answer format' 정규식 (eval/COMPARISON_MODELS_HANDOFF.md 표 기준)
STRICT = {
    "Qwen2.5-Omni": re.compile(r"from\s+\d+(?:\.\d+)?\s+to\s+\d+(?:\.\d+)?\s*(?:seconds|secs|sec)", re.I),
    "MUSEG": re.compile(r"\d+\.\d{2}\s*-\s*\d+\.\d{2}"),
    "ARC-Hunyuan-Video-7B": re.compile(r"\d{1,2}:\d{2}:\d{2}\s*-\s*\d{1,2}:\d{2}:\d{2}"),
    "ChronusOmni": re.compile(r"second\{?\s*[\d.]+\s*\}?\s*(?:to|-)\s*second\{?\s*[\d.]+\s*\}?", re.I),
    "AVicuna": re.compile(r"from\s+\d+(?:\.\d+)?\s+to\s+\d+(?:\.\d+)?", re.I),
}
RAW_KEY = {"AVicuna": "raw_pred"}  # avicuna 는 pred 가 %→초 환산본, 원문은 raw_pred
MODELS = ["Qwen2.5-Omni", "MUSEG", "ARC-Hunyuan-Video-7B", "ChronusOmni", "AVicuna"]


def analyze(model, path):
    rows = json.load(open(path))
    raw_key, strict_re = RAW_KEY.get(model), STRICT[model]
    n = len(rows)
    empty = ok = fail_nonempty = strict_ok = rescued = 0
    nsegs = 0
    fails = []
    _degen[0] = 0
    for r in rows:
        pred = str(r.get("pred", "") or "")
        fmt_src = str(r.get(raw_key, "") or "") if raw_key else pred
        strict = bool(strict_re.search(fmt_src or pred))
        strict_ok += strict
        if not pred.strip():
            empty += 1
            fails.append(("<EMPTY>", r.get("id", "")))
            continue
        segs = EM.parse_natural(EM.extract_answer_scope(pred), MAX_T)
        if segs:
            ok += 1
            nsegs += len(segs)
            rescued += (not strict)
        else:
            fail_nonempty += 1
            fails.append((pred[:150].replace("\n", " "), r.get("id", "")))
    return dict(model=model, n=n, empty=empty, ok=ok, fail_nonempty=fail_nonempty,
                strict_ok=strict_ok, rescued=rescued, degen=_degen[0],
                segs_per_ok=(nsegs / ok if ok else 0.0), fails=fails)


def report(title, items):
    print(f"\n### {title}")
    hdr = (f"{'model':<24}{'n':>6}{'parseOK':>9}{'parse율':>9}{'포맷준수':>10}"
           f"{'구제':>6}{'빈출력':>7}{'파싱실패':>9}{'degen':>7}{'segs/샘플':>10}")
    print(hdr); print("-" * len(hdr))
    for a in items:
        print(f"{a['model']:<24}{a['n']:>6}{a['ok']:>9}{100*a['ok']/a['n']:>8.2f}%"
              f"{100*a['strict_ok']/a['n']:>9.2f}%{a['rescued']:>6}{a['empty']:>7}"
              f"{a['fail_nonempty']:>9}{a['degen']:>7}{a['segs_per_ok']:>10.2f}")
    for a in items:
        if a["fails"]:
            print(f"\n[{a['model']}] parse 실패 {len(a['fails'])}건 예시:")
            for t, i in a["fails"][:4]:
                print(f"  - ({i}) {t!r}")


unav = [analyze(m, f"{BASE}/{m}/unav100_multiseg/eval/test_results_rank0.json") for m in MODELS]
report("UnAV-100 multiseg (공통 프롬프트, n=3455)", unav)

th = [(m, f"{BASE}/{m}/thumos_tail/test_results_rank0.json") for m in MODELS]
th = [analyze(m, p) for m, p in th if os.path.exists(p)]
if th:
    report("THUMOS tail (참고)", th)

# MUSEG 빈 pred 는 추론 래퍼의 <answer> 추출 실패 → 원문(raw)에는 무엇이 있는지 확인
import glob
raw_rows = []
for f in sorted(glob.glob(f"{BASE}/MUSEG/unav100_multiseg/results/chunk_*.json")):
    raw_rows += json.load(open(f))
bad = [r for r in raw_rows if not str(r.get("output", "")).strip()]
raw_recover = sum(1 for r in bad if EM.parse_natural(EM.extract_answer_scope(str(r.get("raw", ""))), MAX_T))
print(f"\n[MUSEG] 빈 pred {len(bad)}건 = 모델이 <answer> 를 못 낸 케이스(think 루프/토큰 초과). "
      f"raw 텍스트에서 시간표현이라도 회수되는 건: {raw_recover}건")

# --- parse 실패가 mIoU 를 얼마나 깎는지 (실패 샘플은 IoU 0 으로 들어감) ---
print(f"\n### parse 실패의 mIoU 영향 (UnAV-100)")
print(f"{'model':<24}{'sample mIoU(all)':>18}{'mIoU(parseOK만)':>18}{'상승폭':>9}")
for m in MODELS:
    rows = json.load(open(f"{BASE}/{m}/unav100_multiseg/eval/test_results_rank0.json"))
    ious = []
    for r in rows:
        gt = EM._coerce_segments(r.get("gt_segments"))
        pred = EM.parse_natural(EM.extract_answer_scope(str(r.get("pred", "") or "")), MAX_T)
        iou, _, _ = EM.sample_iou_parts(gt, pred)
        ious.append((iou, bool(pred)))
    allm = 100 * sum(x for x, _ in ious) / len(ious)
    okl = [x for x, o in ious if o]
    okm = 100 * sum(okl) / len(okl)
    print(f"{m:<24}{allm:>17.2f}%{okm:>17.2f}%{okm-allm:>8.2f}")
