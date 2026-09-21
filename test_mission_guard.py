#!/usr/bin/env python3
"""Pure-logic tests for mission_guard.py (step 5). No ROS, plain python3.

Each scenario drives the guard with explicit timestamps and asserts the full
(state, reason) transition sequence, so the safety behaviour is pinned:
bounded paths, takeover tolerance during the climb, hold budget, margin
return + timeout, blind low-altitude abort, land timeout, recovery.

Run:
    python3 test_mission_guard.py
"""

import sys

from mission_guard import (DONE, HOLD, LAND, MISSION, RETURN, STATE_NAMES,
                           TAKEOFF, WAIT, Guard)

TRACKING = {'state': 'tracking', 'vio': 'ok'}


class Sim:
    """Tiny driver around Guard: advance time, feed facts, record changes."""

    def __init__(self, **guard_kw):
        self.g = Guard(**guard_kw)
        self.t = 0.0
        self.trans = [(self.g.state, self.g.reason)]

    def step(self, dt, **kw):
        self.t = round(self.t + dt, 6)
        facts = dict(ready=True, health=TRACKING, ev_age=0.1, pos=(0.0, 0.0),
                     alt=50.0, mission_done=False, landed=False)
        facts.update(kw)
        st = self.g.update(self.t, facts['ready'], facts['health'],
                           facts['ev_age'], facts['pos'], facts['alt'],
                           facts['mission_done'], facts['landed'])
        if st != self.trans[-1][0]:
            self.trans.append((st, self.g.reason))
        return st

    def hold_for(self, secs, step_s=5.0, **kw):
        while secs > 0.0:
            dt = min(step_s, secs)
            self.step(dt, **kw)
            secs -= dt


def expect(name, sim, want):
    ok = sim.trans == want
    print(('PASS ' if ok else 'FAIL ') + name)
    if not ok:
        print('   want: %s' % fmt(want))
        print('   got : %s' % fmt(sim.trans))
    return ok


def fmt(trans):
    return ' -> '.join('%s(%s)' % (STATE_NAMES[s], r) for s, r in trans)


# --------------------------------------------------------------- scenarios
def happy_path():
    """wait -> takeoff -> mission -> land -> done, each on its trigger."""
    s = Sim()
    s.step(0.1, ready=False, alt=0.0)
    s.step(0.1, alt=0.0)
    s.step(0.2, alt=30.0)
    s.step(0.2, alt=47.0)          # cruise_alt - ALT_EPS_M
    s.step(0.1, mission_done=True)
    s.step(0.1, landed=True)
    return expect('happy path wait->takeoff->mission->land->done', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (MISSION, 'altitude_reached'), (LAND, 'mission_complete'),
        (DONE, 'landed')])


def wait_never_flies_early():
    """Not armed+OFFBOARD: the guard stays in wait, degradations ignored."""
    s = Sim()
    for _ in range(5):
        s.step(1.0, ready=False, alt=0.0, ev_age=1.0,
               health={'state': 'no_frames', 'vio': 'drop'})
    return expect('wait holds until ready (degradations ignored)', s,
                  [(WAIT, 'wait_ev')])


def scan_during_climb():
    """init_wait / no_frames are tolerated while climbing (MCL scans)."""
    s = Sim()
    s.step(0.1, alt=0.0)
    for st in ('init_wait', 'no_frames'):
        s.hold_for(30.0, step_s=10.0, alt=20.0,
                   health={'state': st, 'vio': 'ok'})
    s.step(0.1, alt=50.0, health={'state': 'no_frames', 'vio': 'ok'})
    return expect('climb: init_wait/no_frames tolerated until altitude', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (MISSION, 'altitude_reached')])


def takeoff_timeout():
    """Never reaching cruise altitude is bounded by takeoff_max_s."""
    s = Sim(takeoff_max_s=30.0)
    s.step(0.1, alt=20.0)
    s.step(30.0, alt=20.0)         # dt = 30.0, not > 30
    s.step(0.3, alt=20.0)
    return expect('takeoff timeout -> LAND', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (LAND, 'takeoff_timeout')])


def blind_low_alt_abort():
    """E V stale below 10 m in takeoff: no velocity source -> LAND."""
    s = Sim()
    s.step(0.1, alt=5.0, ev_age=1.0)       # -> TAKEOFF
    s.step(0.1, alt=5.0, ev_age=1.0)       # TAKEOFF decision tick
    return expect('blind low-alt (ev stale, alt<10) -> LAND', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (LAND, 'blind_low_alt')])


def ev_stale_high_alt_hold():
    """E V stale above 10 m during climb -> HOLD, recovery -> MISSION."""
    s = Sim()
    s.step(0.1, alt=20.0)
    s.step(0.1, alt=20.0, ev_age=1.0)      # condition appears
    s.step(0.1, alt=20.0, ev_age=1.0)      # grace 0 -> degraded
    s.step(0.1, alt=20.0, ev_age=0.1)      # recovered
    return expect('ev stale (alt>=10) -> HOLD -> recovered MISSION', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (HOLD, 'degraded:ev_stale'), (MISSION, 'recovered')])


def hold_budget_single():
    """A single long hold is capped by hold_budget_s."""
    s = Sim(hold_budget_s=60.0)
    s.step(0.1)
    s.step(0.1, alt=50.0)
    s.step(0.1, ev_age=1.0)
    s.step(0.1, ev_age=1.0)                # -> HOLD
    s.hold_for(70.0, ev_age=1.0)
    return expect('hold budget exceeded -> LAND', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (MISSION, 'altitude_reached'), (HOLD, 'degraded:ev_stale'),
        (LAND, 'hold_budget')])


def hold_budget_repeated():
    """Repeated short losses accumulate: 3 x 45 s > 120 s -> LAND."""
    s = Sim(hold_budget_s=120.0)
    s.step(0.1)
    s.step(0.1, alt=50.0)
    for _ in range(3):
        s.step(0.1, ev_age=1.0)
        s.step(0.1, ev_age=1.0)            # -> HOLD
        s.hold_for(45.0, ev_age=1.0)
        s.step(0.1, ev_age=0.1)            # recovered (no-op if LANDed)
    return expect('repeated holds accumulate to the budget', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (MISSION, 'altitude_reached'),
        (HOLD, 'degraded:ev_stale'), (MISSION, 'recovered'),
        (HOLD, 'degraded:ev_stale'), (MISSION, 'recovered'),
        (HOLD, 'degraded:ev_stale'), (LAND, 'hold_budget')])


def mission_timeout():
    """The mission clock is bounded by mission_max_s."""
    s = Sim(mission_max_s=50.0)
    s.step(0.1)
    s.step(0.1, alt=50.0)
    s.step(50.0)
    return expect('mission timeout -> LAND', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (MISSION, 'altitude_reached'), (LAND, 'mission_timeout')])


def margin_return():
    """Sustained honest breach of the margin -> RETURN; honest inside -> MISSION."""
    s = Sim(margin_m=1000.0, map_half_m=2500.0)
    s.step(0.1)
    s.step(0.1, alt=50.0)
    s.step(0.1, pos=(1400.0, 0.0))         # inside the 1500 m limit
    s.step(0.1, pos=(1600.0, 0.0))         # breach starts (t=0.3)
    s.step(3.0, pos=(1600.0, 0.0))         # 3.0 not > 3
    s.step(0.5, pos=(1600.0, 0.0))         # -> RETURN
    s.step(0.1, pos=(1600.0, -1600.0))     # still outside
    s.step(0.1, pos=(100.0, 0.0))          # honest inside
    return expect('map margin -> RETURN -> MISSION', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (MISSION, 'altitude_reached'), (RETURN, 'map_margin'),
        (MISSION, 'back_in_margin')])


def margin_gated_is_not_inside():
    """A gated frame (pos=None) must NOT read as 'back inside the margin'."""
    s = Sim(margin_m=1000.0, map_half_m=2500.0)
    s.step(0.1)
    s.step(0.1, alt=50.0)
    s.step(0.1, pos=(1600.0, 0.0))         # breach starts
    s.step(3.2, pos=(1600.0, 0.0))         # -> RETURN
    s.step(0.5, pos=None)                  # gated: still outside
    s.step(0.5, pos=None)
    s.step(0.1, pos=(900.0, 0.0))          # honest inside -> MISSION
    return expect('gated frames do not reset the margin breach', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (MISSION, 'altitude_reached'), (RETURN, 'map_margin'),
        (MISSION, 'back_in_margin')])


def margin_brief_breach_ignored():
    """A single-frame breach shorter than margin_persist_s is ignored."""
    s = Sim(margin_m=1000.0, map_half_m=2500.0)
    s.step(0.1)
    s.step(0.1, alt=50.0)
    s.step(0.2, pos=(1600.0, 0.0))         # brief breach
    s.step(0.1, pos=(900.0, 0.0))          # honest inside resets it
    s.step(2.0, pos=(900.0, 0.0))
    return expect('brief margin breach ignored', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (MISSION, 'altitude_reached')])


def offboard_lost_lands():
    """Losing armed+OFFBOARD for > offboard_lost_s -> LAND (external takeover)."""
    s = Sim()
    s.step(0.1)
    s.step(0.1, alt=50.0)                  # -> MISSION
    s.step(2.0, ready=False)               # brief dropout: tolerated
    s.step(0.1, ready=True)                # PX4 re-engaged
    s.step(1.0, ready=False)               # loss starts here
    s.step(3.2, ready=False)               # 3.2 > 3 -> LAND
    return expect('OFFBOARD loss -> LAND, brief dropout tolerated', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (MISSION, 'altitude_reached'), (LAND, 'offboard_lost')])


def offboard_lost_during_takeoff():
    """The same during the climb."""
    s = Sim()
    s.step(0.1, alt=10.0)                  # -> TAKEOFF (still climbing)
    s.step(3.5, alt=10.0, ready=False)     # loss starts at t=3.6
    s.step(3.2, alt=10.0, ready=False)     # -> LAND
    return expect('OFFBOARD loss during takeoff -> LAND', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (LAND, 'offboard_lost')])


def hold_mission_done_lands():
    """Stream end while holding -> LAND (no waiting out the hold budget)."""
    s = Sim(hold_budget_s=600.0)
    s.step(0.1)
    s.step(0.1, alt=50.0)
    s.step(0.1, ev_age=1.0)
    s.step(0.1, ev_age=1.0)                # -> HOLD (ev stale)
    s.step(0.1, ev_age=6.0, mission_done=True)
    return expect('hold + mission_done -> LAND', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (MISSION, 'altitude_reached'), (HOLD, 'degraded:ev_stale'),
        (LAND, 'mission_complete')])


def margin_return_timeout():
    """A return that never gets back inside is bounded by return_max_s."""
    s = Sim(return_max_s=20.0, margin_m=1000.0, map_half_m=2500.0)
    s.step(0.1)
    s.step(0.1, alt=50.0)
    s.step(0.1, pos=(1600.0, 0.0))         # breach starts (t=0.3)
    s.step(3.2, pos=(1600.0, 0.0))         # -> RETURN (t=3.5)
    s.step(20.0, pos=(1600.0, 0.0))        # dt = 20.0, not > 20
    s.step(0.3, pos=(1600.0, 0.0))
    return expect('return timeout -> LAND', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (MISSION, 'altitude_reached'), (RETURN, 'map_margin'),
        (LAND, 'return_timeout')])


def margin_return_degraded():
    """Degraded while returning -> HOLD; recovery goes back to RETURN."""
    s = Sim(margin_m=1000.0, map_half_m=2500.0)
    s.step(0.1)
    s.step(0.1, alt=50.0)
    s.step(0.1, pos=(1600.0, 0.0))
    s.step(3.2, pos=(1600.0, 0.0))                 # -> RETURN
    s.step(0.1, pos=(1600.0, 0.0), ev_age=1.0)
    s.step(0.1, pos=(1600.0, 0.0), ev_age=1.0)     # -> HOLD
    s.step(0.1, pos=(1600.0, 0.0), ev_age=0.1)     # recovered, still outside
    return expect('return + degraded -> HOLD -> RETURN', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (MISSION, 'altitude_reached'), (RETURN, 'map_margin'),
        (HOLD, 'degraded:ev_stale'), (RETURN, 'recovered')])


def land_timeout():
    """LAND without a land-detector trip still ends in DONE."""
    s = Sim(land_max_s=10.0)
    s.step(0.1)
    s.step(0.1, alt=50.0)
    s.step(0.1, mission_done=True)         # -> LAND
    s.step(10.0)                           # dt = 10.0, not > 10
    s.step(0.3)
    return expect('land timeout -> DONE', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (MISSION, 'altitude_reached'), (LAND, 'mission_complete'),
        (DONE, 'land_timeout')])


def vio_drop_grace():
    """A VIO dropout is tolerated for vio_grace_s (from first appearance)."""
    s = Sim(vio_grace_s=3.0)
    s.step(0.1)
    s.step(0.1, alt=50.0)
    vio = {'state': 'tracking', 'vio': 'drop'}
    s.step(0.1, health=vio)                # condition appears
    s.step(3.0, health=vio)                # 3.0 not > 3
    s.step(0.5, health=vio)                # 3.5 > 3
    return expect('vio drop grace 3 s -> HOLD', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (MISSION, 'altitude_reached'), (HOLD, 'degraded:vio_drop')])


def motion_only_grace():
    """motion_only is tolerated for 25 s (from first appearance), then HOLD."""
    s = Sim()
    s.step(0.1)
    s.step(0.1, alt=50.0)
    mo = {'state': 'motion_only', 'vio': 'ok'}
    s.step(0.1, health=mo)                 # condition appears
    s.step(25.0, health=mo)                # 25.0 not > 25
    s.step(0.5, health=mo)
    return expect('motion_only grace 25 s -> HOLD', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (MISSION, 'altitude_reached'), (HOLD, 'degraded:motion_only')])


def lost_is_hard():
    """lost has grace 0: degraded after one tick (no health topic case)."""
    s = Sim()
    s.step(0.1)
    s.step(0.1, alt=50.0)
    s.step(0.1, health=None)                       # no health: EV fresh only
    s.step(0.1, health={'state': 'lost', 'vio': 'ok'})
    s.step(0.1, health={'state': 'lost', 'vio': 'ok'})   # grace 0 -> HOLD
    return expect('lost -> HOLD after one tick', s, [
        (WAIT, 'wait_ev'), (TAKEOFF, 'armed_offboard'),
        (MISSION, 'altitude_reached'), (HOLD, 'degraded:lost')])


def margin_helper():
    """outside_margin() boundaries and None handling."""
    g = Guard(margin_m=1000.0, map_half_m=2500.0)
    cases = [((1499.9, 0.0), False), ((1500.1, 0.0), True),
             ((0.0, 1500.1), True), ((0.0, -1499.9), False),
             (None, False)]
    ok = all(g.outside_margin(p) is v for p, v in cases)
    print(('PASS ' if ok else 'FAIL ') + 'outside_margin boundaries')
    return ok


ALL = [happy_path, wait_never_flies_early, scan_during_climb,
       takeoff_timeout, blind_low_alt_abort, ev_stale_high_alt_hold,
       hold_budget_single, hold_budget_repeated, mission_timeout,
       margin_return, margin_gated_is_not_inside, margin_brief_breach_ignored,
       margin_return_timeout, margin_return_degraded,
       offboard_lost_lands, offboard_lost_during_takeoff,
       hold_mission_done_lands,
       land_timeout, vio_drop_grace, motion_only_grace, lost_is_hard,
       margin_helper]


def main():
    results = [scenario() for scenario in ALL]
    print('\n%d/%d scenarios passed' % (sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())