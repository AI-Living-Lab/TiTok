#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_charades_all.py — Charades-STA 전체(3,720) 결과를 통일 기준으로 재채점.

채점 규칙 (ChronusOmni 공식 inference/cal_iou.py 와 동일하게 맞춤):
  - R@1: 쿼리당 예측 1개. 멀티세그 출력이면 '첫 시작 ~ 마지막 끝' 으로 병합
  - IoU = 교집합 / (max(end) - min(start))
  - 파싱 실패 = IoU 0
  - mIoU = 샘플 IoU 평균
GT 는 결과 파일의 embedded gt_segments 또는 ref(시간토큰)에서 읽고,
공식 charades_sta_test.txt 와 (video,start,end) 다중집합으로 대조해 검증한다.
"""
import json, os, re, glob, collections, statistics

WS = os.environ.get("WORKSPACE",
     os.path.dirname(os.path.dirname(os.path.dirname(
         os.path.dirname(os.path.abspath(__file__))))))  # workspace
OFFICIAL = f"{WS}/datasets/charades_sta/annotations/charades_sta_test.txt"
THR = (0.1, 0.3, 0.5, 0.7, 0.9)


def calc_iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    uni = max(a[1], b[1]) - min(a[0], b[0])
    return inter / uni if uni > 0 else 0.0


# 구분자는 eval_miou.py 의 _TOK_SEG 와 동일하게 to / - / – / — / ~ 를 모두 허용.
# ("The event happens in <t..> - <t..> seconds." 형태 출력이 실제로 존재)
_TOK = re.compile(r"((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)\s*(?:to|-|–|—|~)\s*((?:<t\d>)+(?:<tdot>(?:<t\d>)+)?)",
                  re.IGNORECASE)


def _tok2sec(t):
    head, _, tail = t.partition("<tdot>")
    ip = re.findall(r"<t(\d)>", head)
    dp = re.findall(r"<t(\d)>", tail)
    return int("".join(ip)) + (int(dp[0]) / 10.0 if dp else 0.0) if ip else None


def span_tokens(txt):
    """TiTok: 'From <t..> to <t..>. ...' → (첫 시작, 마지막 끝)"""
    m = _TOK.findall(txt or "")
    return (_tok2sec(m[0][0]), _tok2sec(m[-1][1])) if m else None


def span_numeric(txt):
    """ChronusOmni: 'second{a}-second{b}' → 공식과 동일하게 첫/마지막 숫자"""
    n = re.findall(r"(\d+\.?\d*)", txt or "")
    return (float(n[0]), float(n[-1])) if n else None


def official_multiset():
    c = collections.Counter()
    for line in open(OFFICIAL):
        line = line.strip()
        if not line:
            continue
        head, _ = line.split("##", 1)
        v, s, e = head.split()
        c[(v.split("/")[-1], round(float(s), 1), round(float(e), 1))] += 1
    return c


def score(path):
    """결과 파일 하나 채점 → (metrics, n, gt_mismatch) / 미완성이면 None"""
    d = json.load(open(path))
    if len(d) != 3720:
        return None, len(d), None
    tokens = "pred" in d[0] and "<t" in (d[0].get("ref") or d[0].get("pred") or "")
    span = span_tokens if tokens else span_numeric
    ious, got = [], collections.Counter()
    for x in d:
        if x.get("gt_segments"):
            g = tuple(x["gt_segments"][0])
        else:
            g = span_tokens(x.get("ref", ""))
        if g is None:
            return None, len(d), None
        vid = os.path.basename(x.get("video", "")).replace(".mp4", "") or x.get("id", "").split("_")[0]
        got[(vid, round(g[0], 1), round(g[1], 1))] += 1
        p = span(x.get("pred", ""))
        ious.append(calc_iou(p, g) if p else 0.0)
    mism = sum((got - official_multiset()).values())
    n = len(ious)
    m = {f"R@{t}": 100.0 * sum(i >= t for i in ious) / n for t in THR}
    m["mIoU"] = 100.0 * statistics.mean(ious)
    return m, n, mism


def charades_status(run_id):
    """Charades 학습 여부 — 이름 추측이 아니라 실제 실행 스크립트/설정으로 확인한 결과.

    근거(Team4 git 히스토리, `_tools/GDPO/`):
      run_mu2_unpu.sh                    RUN=..._mlp_headoff_unpu        dataset=unpu_v2.json
      run_unpucha_batch4.sh              RUN=..._mlp_headoff_unpucha_b4  dataset=unpucha_v2.json
      config_sep2fp_lr.yaml (헤더 주석)   run_name=..._rMsep2fp_lr_clip_mu2  dataset=unav100_v2.json
      config_sep3.yaml / run_rMsep3_*.sh RUN=...rMsep3_unpucha_batch4*   dataset=unpucha_v2.json
      run_rMsep3_natural.sh              RUN=..._chronus_..._natural     dataset=unpucha_chronus.json
    unpucha_v2.json = UnAV 4,063 + PU-VALOR 2,884 + Charades 3,411
    unpu_v2.json    = UnAV 7,474 + PU-VALOR 2,884 (Charades 0)
    unav100_v2.json = UnAV 단독
    """
    VERIFIED = {
        "sft_7b_unav_v8_rl_rMsep2fp_lr_clip_mu2": ("unav100_v2.json", False),
        "sft_7b_unav_v8_rl_rMsep2fp_lr_clip_mu2_mlp_headoff_unpu": ("unpu_v2.json", False),
        "sft_7b_unav_v8_rl_rMsep2fp_lr_clip_mu2_mlp_headoff_unpucha_batch4": ("unpucha_v2.json", True),
        "sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling": ("unpucha_v2.json", True),
        "sft_7b_unpucha_v8_rl_rMsep3_unpucha_batch4_noscaling": ("unpucha_v2.json", True),
        "sft_7b_unpucha_v8_rl_rMsep3_unpucha_batch4_noscaling_ttifix": ("unpucha_v2.json", True),
        "sft_7b_unav_v8_chronus_rl_rMsep3_unpucha_batch4_natural": ("unpucha_chronus.json", True),
    }
    if "ChronusOmni" in run_id:
        return "공개ckpt(grounding FT 없음)"
    if run_id in VERIFIED:
        ds, trained = VERIFIED[run_id]
        return ("학습함 " if trained else "미학습 ") + ds
    return "미확인(스크립트 못 찾음)"


def _short(rel):
    """gdpo/<run>/<ckpt>/<tag>/<testset>/test_results_rank0.json → 'run @ckpt (tag,testset)'"""
    p = rel.split("/")
    if len(p) >= 6:
        run, ck, tag, ts = p[1], p[2].replace("checkpoint-", ""), p[3], p[4]
        run = run.replace("sft_7b_", "").replace("_rl_rMsep3_unpucha_batch4", "")
        run = run.replace("_rl_rMsep2fp_lr_clip_mu2", "~mu2")
        ts = ts.replace("charades_rlp_", "")
        return f"{run} @{ck} ({tag},{ts})"[:62]
    return rel[:62]


def main():
    targets = []
    for p in glob.glob(f"{WS}/outputs/**/test_results_rank0.json", recursive=True):
        if "charades" not in p.lower():
            continue
        rel = p[len(WS) + 9:]
        targets.append((rel, p))
    rows, partial = [], []
    for rel, p in sorted(targets):
        m, n, mism = score(p)
        parts = rel.split("/")
        run = parts[1] if len(parts) > 1 else parts[0]
        if m is None:
            partial.append((rel, n))
            continue
        rows.append((rel, run, m, n, mism))

    print("=" * 118)
    print("Charades-STA 공식 test set (3,720 쿼리 / 1,334 영상) — R@1, IoU=교집합/(max end - min start)")
    print("=" * 118)
    hdr = f"{'모델 / 체크포인트 / 설정':<62}{'R@0.3':>7}{'R@0.5':>7}{'R@0.7':>7}{'mIoU':>7}  {'Charades 학습':<22}"
    print(hdr)
    print("-" * 118)
    for rel, run, m, n, mism in sorted(rows, key=lambda r: -r[2]["mIoU"]):
        flag = "" if mism <= 1 else f"  [GT불일치 {mism}]"
        print(f"{_short(rel):<62}{m['R@0.3']:>7.2f}{m['R@0.5']:>7.2f}{m['R@0.7']:>7.2f}"
              f"{m['mIoU']:>7.2f}  {charades_status(run):<22}{flag}")
    if partial:
        print("-" * 118)
        print("미완성(3,720 미만) — 위 표에서 제외:")
        for rel, n in sorted(partial):
            print(f"  {rel:<62} n={n}")
    print("=" * 118)
    print("주) CountF1 은 Charades-STA 전 샘플이 단일 세그먼트(N_gt=1)라 정의되지 않음.")
    print("    F1@k 는 Precision≈Recall 이라 R@1 과 사실상 동일값 — 별도 보고 불필요.")


if __name__ == "__main__":
    main()
