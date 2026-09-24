"""Validate trained models on REAL drone patches, per FLIGHT (deployment OOD).

The training sim val set uses SIMULATED drone patches; the real question is
whether a model scores the TRUE map location above false peaks for REAL
orthoprojection patches — and crucially on a HELD-OUT FLIGHT it never trained
on (finetune.val_flights in train_config.yml). This script:

  1. Loads all truth pairings from real_pairs.npz (train_similarity.py
     --build-real output: every debug patch k*_4_patch.png with its flight
     tag, GPS-truth map pixel and coverage).
  2. For EACH FLIGHT (patches read from that flight's run_dir/debug):
       truth score  = max over 180 rotations (2 deg) of score(patch, map@truth)
       false score  = max over rotations AND over near (176-660 m ring,
                      the observed false-peak distances) + far random spots
     with both the NEW (Huizhou fine-tuned) and OLD (Sweden) checkpoints.
  3. Prints win rates / margins per flight, marking the held-out flight as
     the deployment gate (true med >~0.5, win ~100%), saves side-by-side
     debug images, and (--profile) runs scale-sweep / embedding-gap
     failure diagnosis.

Run:  ~/anaconda3/envs/sivl/bin/python eval_real_patches.py [--frames 40]
"""

import argparse
import glob
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, '/home/one/GNSS-denied-Localization/sivl')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_similarity import (Scorer, extract_all, sim_drone)  # noqa: E402
from utils.utils import cropRectangularSectionFromImage  # noqa: E402

ROOT = '/home/one/GNSS-denied-Localization'
RUN = os.path.join(ROOT, 'sample_images/mcl_test/bag0001_debug')
DEBUG = os.path.join(RUN, 'debug')
MAP_PATH = os.path.join(ROOT, 'z17_5120.png')
CKPT_NEW = os.path.join(
    ROOT, 'checkpoints_huizhou/huizhou_ft_1_epoch_040_of_40.pt')
CKPT_OLD = os.path.join(ROOT, 'sivl/checkpoints/training_4_epoch_00200_of_200.pt')
OUT_DIR = os.path.join(ROOT, 'eval_real_patches_out')

GSD = 1.10            # z17_5120.png m/px (same as mcl_node default)
PATCH_PX = 96
N_ROT = 180           # rotation sweep every 2 deg — the model is extremely
                      # rotation-sensitive (10 deg steps miss peaks: observed
                      # score 0.33 -> 1.00 when refined to 2 deg)
N_NEAR, N_FAR = 10, 10
NEAR_MIN_M, NEAR_MAX_M = 176, 660


def load_real_pairs():
    """All real pairs from real_pairs.npz (built by train_similarity.py
    --build-real): every debug patch (k = camera-frame index) with its
    flight tag, GPS-truth map pixel and coverage.

    Returns dict(flight, k, px, py, cov) as arrays; k repeats across flights
    so pairs MUST be keyed by (flight, k) — see flight_rows().
    """
    d = np.load(os.path.join(ROOT, 'huizhou_map/patch_bank/real_pairs.npz'))
    flight = (d['flight'].astype(str) if 'flight' in d.files
              else np.full(len(d['k']), 'flight0'))
    return dict(flight=flight, k=d['k'].astype(int), px=d['px'],
                py=d['py'], cov=d['cov'].astype(float))


def flight_rows(rp, flight):
    """k -> row index within real_pairs for one flight (keyed (flight,k))."""
    idx = np.where(rp['flight'] == flight)[0]
    return {int(rp['k'][i]): i for i in idx}


def flight_run_dir(flight):
    """Debug run_dir (contains debug/k*_4_patch.png) for a flight name."""
    from train_similarity import cfg
    for fl in cfg.data.path.flights:
        if fl.name == flight:
            return fl.run_dir
    return None


def held_out_flights():
    """flights in finetune.val_flights (cross-flight validation set)."""
    from train_similarity import cfg
    return set(getattr(cfg.finetune, 'val_flights', []) or [])


def profile_fails(args):
    """Diagnose why the Huizhou model scores truth ~0 on ~78% of real frames.

    (The old Sweden model scores truth >0.5 on 100% of frames, which proves
    the truth position / coverage scale / rotation sweep are geometrically
    right — the miss is an appearance-domain problem.) Three probes:

    a) scale sweep at truth (0.60-1.50) — would a different coverage (VIO
       altitude error) recover the peak? (deployment: particles share one
       coverage per frame, so a systematic scale miss kills the update)
    b) image stats of the drone patch (mean/std/Laplacian/coverage) —
       which appearance regime passes vs fails
    c) branch-embedding distance: real patch vs truth map patch, compared
       to sim_drone(map) vs map (the training anchor distribution)
    """
    rp = load_real_pairs()
    # DEBUG is one flight's run dir; restrict truth rows to that flight
    # (k repeats across flights, so the (flight,k) key is required)
    run_dirs = {f: flight_run_dir(f) for f in sorted(set(rp['flight']))}
    dbg_flight = next((f for f, rd in run_dirs.items()
                       if rd and os.path.join(rd, 'debug') == DEBUG), None)
    row_of_k = flight_rows(rp, dbg_flight) if dbg_flight else {
        int(k): i for i, k in enumerate(rp['k'])}
    truth_px = np.stack([rp['px'], rp['py']], 1)
    cov_all = rp['cov']
    files = sorted(glob.glob(os.path.join(DEBUG, 'k*_4_patch.png')))
    ks = [int(os.path.basename(f)[1:6]) for f in files
          if int(os.path.basename(f)[1:6]) in row_of_k]
    pick = ks if len(ks) <= args.frames else [
        ks[i] for i in np.linspace(0, len(ks) - 1, args.frames).astype(int)]
    print(f'profiling {len(pick)} of {len(ks)} debug frames')
    map_rgb = cv2.cvtColor(cv2.imread(MAP_PATH, cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
    angles = np.arange(N_ROT) * (360.0 / N_ROT)
    scales = np.round(np.arange(0.60, 1.501, 0.05), 2)
    sc = Scorer(args.ckpt or CKPT_NEW)

    rows = []
    for k in pick:
        row = row_of_k[k]
        patch = cv2.cvtColor(
            cv2.imread(os.path.join(DEBUG, f'k{k:05d}_4_patch.png')),
            cv2.COLOR_BGR2RGB)
        g = patch.mean(2)
        lap = cv2.Laplacian(g, cv2.CV_64F).var()
        px, py = truth_px[row]
        src_dim = cov_all[row] / GSD

        tp = extract_all(map_rgb, px, py, src_dim, angles)
        ok = tp.reshape(len(tp), -1).mean(1) >= 5.0
        st = np.where(ok, sc.score_many(patch, tp), 0.0)
        s10 = float(st.max())
        best_mp = tp[int(np.argmax(st))]

        # a) scale sweep only when scale 1.0 missed
        best_s, best_sc = 1.0, s10
        if s10 < 0.5:
            for s in scales:
                if s == 1.0:
                    continue
                pp = np.stack([
                    cropRectangularSectionFromImage(
                        map_rgb, px, py, math.radians(a), src_dim, PATCH_PX,
                        s) for a in angles])
                ss = sc.score_many(patch, pp)
                ss = np.where(pp.reshape(len(pp), -1).mean(1) >= 5.0, ss, 0.0)
                if ss.max() > best_sc:
                    best_s, best_sc = float(s), float(ss.max())

        # c) embedding distances + training-style anchor scores
        sims = np.stack([sim_drone(best_mp, np.random.default_rng(i))
                         for i in range(8)])
        s_sim = float(np.median(
            [sc.score_many(sims[i], best_mp[None])[0] for i in range(8)]))
        e_map = sc.embed(best_mp[None])[0]
        e_real = sc.embed(patch[None])[0]
        e_sims = sc.embed(sims)
        d_real = float(np.linalg.norm(e_real - e_map))
        d_sim = float(np.linalg.norm(e_sims - e_map, axis=1).mean())

        rows.append(dict(k=k, s10=s10, best_s=best_s, best_sc=best_sc,
                         mean=g.mean(), std=g.std(), lap=lap,
                         cov=cov_all[row], d_real=d_real, d_sim=d_sim,
                         s_sim=s_sim))
        print(f'  k={k:5d} s@1.0={s10:.3f} best(s={best_s:.2f})={best_sc:.3f} '
              f'std={g.std():5.1f} lap={lap:6.0f} cov={cov_all[row]:5.1f}m '
              f'd_real={d_real:6.1f} d_sim={d_sim:6.1f} s_sim={s_sim:.3f}',
              flush=True)

    # ---- group report ----
    def grp(sel, name):
        if not sel:
            print(f'{name:16s} n=0')
            return
        m = lambda f: np.mean([r[f] for r in sel])
        print(f'{name:16s} n={len(sel):2d}  std={m("std"):5.1f} '
              f'lap={m("lap"):6.0f} cov={m("cov"):5.1f}m '
              f'd_real={m("d_real"):6.1f} d_sim={m("d_sim"):6.1f} '
              f's_sim={m("s_sim"):.3f}')

    print('\n===== profile: new(Huizhou) on real patches =====')
    pas = [r for r in rows if r['s10'] > 0.5]
    fail = [r for r in rows if r['s10'] <= 0.5]
    rec = [r for r in fail if r['best_sc'] > 0.5]
    grp(pas, 'pass@1.0')
    grp(fail, 'miss@1.0')
    grp(rec, 'miss,recov@s')
    grp([r for r in fail if r['best_sc'] <= 0.5], 'miss,notrecov')
    if rec:
        print('  recovering scales: '
              + ', '.join(f"{r['best_s']:.2f}" for r in rec))
    keys = list(rows[0].keys())
    np.savez(os.path.join(OUT_DIR, 'profile.npz'),
             **{key: np.array([r[key] for r in rows]) for key in keys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=40,
                    help='number of debug frames to evaluate (spread evenly)')
    ap.add_argument('--save-sb', type=int, default=8,
                    help='side-by-side debug images to save')
    ap.add_argument('--profile', action='store_true',
                    help='diagnose new-model misses: scale sweep at truth, '
                         'patch stats, branch-embedding domain gap')
    ap.add_argument('--ckpt', default='',
                    help='override the new-model checkpoint path')
    args = ap.parse_args()

    if args.profile:
        profile_fails(args)
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    rp = load_real_pairs()
    held = held_out_flights()
    map_rgb = cv2.cvtColor(cv2.imread(MAP_PATH, cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
    angles = np.arange(N_ROT) * (360.0 / N_ROT)
    scorers = {'new(Huizhou)': Scorer(args.ckpt or CKPT_NEW),
               'old(Sweden)': Scorer(CKPT_OLD)}

    def ok_mask(pp):
        return pp.reshape(len(pp), -1).mean(1) >= 5.0

    def eval_flight(flight):
        """Score truth vs false peaks for one flight; returns per-model
        true/near/far lists."""
        run_dir = flight_run_dir(flight)
        dbg = os.path.join(run_dir, 'debug') if run_dir else ''
        row_of_k = flight_rows(rp, flight)
        files = sorted(glob.glob(os.path.join(dbg, 'k*_4_patch.png')))
        ks = [int(os.path.basename(f)[1:6]) for f in files
              if int(os.path.basename(f)[1:6]) in row_of_k]
        if not ks:
            print(f'\n===== {flight}: no debug patches found in {dbg} — skip '
                  f'(need an MCL run with save_debug into that run_dir) =====')
            return None
        pick = ks if len(ks) <= args.frames else [
            ks[i] for i in np.linspace(0, len(ks) - 1, args.frames).astype(int)]
        tag = 'HELD-OUT VALIDATION' if flight in held else 'train flight'
        print(f'\n===== {flight} [{tag}]: evaluating {len(pick)} of {len(ks)} '
              f'debug frames =====')
        rng = np.random.default_rng(7)
        results = {name: {'true': [], 'near': [], 'far': []}
                   for name in scorers}
        for fi, k in enumerate(pick):
            row = row_of_k[k]
            patch = cv2.cvtColor(
                cv2.imread(os.path.join(dbg, f'k{k:05d}_4_patch.png')),
                cv2.COLOR_BGR2RGB)
            src_dim = rp['cov'][row] / GSD
            px, py = rp['px'][row], rp['py'][row]
            tp = extract_all(map_rgb, px, py, src_dim, angles)
            near_pts = px + np.cos(rng.uniform(0, 2 * np.pi, N_NEAR)) * \
                rng.uniform(NEAR_MIN_M, NEAR_MAX_M, N_NEAR) / GSD, \
                py + np.sin(rng.uniform(0, 2 * np.pi, N_NEAR)) * \
                rng.uniform(NEAR_MIN_M, NEAR_MAX_M, N_NEAR) / GSD
            np_pts = list(zip(near_pts[0], near_pts[1]))
            far_pts = [(rng.uniform(300, 5120 - 300),
                        rng.uniform(300, 5120 - 300))
                       for _ in range(N_FAR)]
            fp = [extract_all(map_rgb, qx, qy, src_dim, angles)
                  for qx, qy in np_pts + far_pts]
            fp_all = np.concatenate(fp)
            for name, sc in scorers.items():
                st = np.where(ok_mask(tp), sc.score_many(patch, tp), 0.0)
                sf = np.where(ok_mask(fp_all),
                              sc.score_many(patch, fp_all), 0.0)
                results[name]['true'].append(st.max())
                results[name]['near'].append(sf[:N_NEAR * N_ROT].max())
                results[name]['far'].append(sf[N_NEAR * N_ROT:].max())
            if fi < args.save_sb:
                best = int(np.argmax(st))
                side = np.concatenate(
                    [patch, np.full((96, 8, 3), 255, np.uint8), tp[best]],
                    axis=1)
                cv2.imwrite(os.path.join(
                    OUT_DIR, f'{flight[-6:]}_k{k:05d}_sb.png'),
                    cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
            print(f'  k={k:5d} true(new)={results["new(Huizhou)"]["true"][-1]:.3f}'
                  f' near={results["new(Huizhou)"]["near"][-1]:.3f}'
                  f' far={results["new(Huizhou)"]["far"][-1]:.3f}')

        hdr = (f'{"model":14s} {"true med":>9s} {"true p10":>9s} '
               f'{"near med":>9s} {"far med":>9s} {"win%":>6s} '
               f'{"win|t>.5":>8s} {"margin":>7s}')
        print(hdr)
        for name, r in results.items():
            tr = np.array(r['true'])
            nf = np.maximum(np.array(r['near']), np.array(r['far']))
            win = 100 * (tr > nf).mean()
            good = tr > 0.5
            winc = 100 * (tr[good] > nf[good]).mean() if good.any() \
                else float('nan')
            print(f'{name:14s} {np.median(tr):9.3f} '
                  f'{np.percentile(tr, 10):9.3f} '
                  f'{np.median(r["near"]):9.3f} {np.median(r["far"]):9.3f} '
                  f'{win:5.0f}% {winc:7.0f}% {np.median(tr - nf):+7.3f} '
                  f'({good.mean() * 100:.0f}% true>.5)')
        return results

    all_results = {}
    for flight in sorted(set(rp['flight'])):
        all_results[flight] = eval_flight(flight)

    # ---- gate summary ----
    print('\n===== CROSS-FLIGHT GATE (truth med / win%) =====')
    for flight, res in all_results.items():
        if res is None:
            continue
        tag = 'HELD-OUT VALIDATION' if flight in held else 'train flight  '
        gates = []
        for name, r in res.items():
            tr = np.array(r['true'])
            nf = np.maximum(np.array(r['near']), np.array(r['far']))
            gates.append(f'{name}: true med {np.median(tr):.3f}, '
                         f'win {100 * (tr > nf).mean():.0f}%')
        print(f'  {flight} [{tag}]  ' + '  |  '.join(gates))
    print('  deploy only checkpoints whose HELD-OUT flight has true med >~0.5'
          ' and win ~100%.')

    # ---- 5. the original false-peak failure case (legacy 184004 files) ----
    dp_path = os.path.join(ROOT, 'k00290_4_patch.png')
    mp_path = os.path.join(ROOT, 'k00290_5_mapatch.png')
    if os.path.exists(dp_path) and os.path.exists(mp_path):
        print('\n===== known false-peak pair (k00290) =====')
        dp = cv2.cvtColor(cv2.imread(dp_path), cv2.COLOR_BGR2RGB)
        mp = cv2.cvtColor(cv2.imread(mp_path), cv2.COLOR_BGR2RGB)
        for name, sc in scorers.items():
            s = sc.score_many(dp, mp[np.newaxis])[0]
            print(f'  {name}: score(false mapatch) = {s:.3f}')


if __name__ == '__main__':
    main()
