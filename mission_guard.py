#!/usr/bin/env python3
"""Guard state machine for the companion-side mission manager (step 5).

Pure logic (NO ROS imports) so the safety behaviour is unit-testable with a
plain `python3 test_mission_guard.py`. The `mission_manager` node feeds it
/mcl/health + EV freshness + position/altitude every tick and executes the
state it returns.

States and transitions — EVERY path is bounded and ends in MISSION or LAND:

    WAIT ── ready (armed + OFFBOARD) ──> TAKEOFF ── alt>=cruise_alt ─> MISSION
                                             │          MISSION ── outside margin ─┐
               init_wait / no_frames / low   │             ▲                     ▼
               coverage tolerated for        │             │ inside        RETURN
               takeoff_max_s (MCL scans      ▼             │ (bounded      │
               during the climb)          HOLD ── recovered┘  return_max_s) │
                                             │ hold_budget_s                 │
                                             ▼                               │
                                any timeout / blind low alt / mission done ──► LAND
                                                                              │ landed
                                                                              ▼
                                                                             DONE
    LAND is bounded too: land_max_s without a land-detector trip -> DONE
    (the manager forces DISARM on DONE).

Degraded = /mcl/health state != 'tracking' beyond its per-state grace, an
/mcl/odom silence > ev_stale_s (grace 0), or a VIO dropout (vio_drop, its own
short grace). HOLD accumulates time across the WHOLE mission (hold_total vs
hold_budget_s), so repeated short losses cannot loop forever — the bounded
end state is always LAND.

Two robustness rules added after the step-7 rehearsal (2026-09-18):
  - the map-edge margin is evaluated on HONEST fixes only and must persist for
    margin_persist_s: a gated frame (pos=None) does NOT count as "back inside"
    (that produced 0.2 s RETURN<->MISSION flapping), and a single-frame breach
    is ignored;
  - losing OFFBOARD/armed (PX4 failsafe took over, e.g. RTL/land) for longer
    than offboard_lost_s in TAKEOFF/MISSION/HOLD/RETURN -> LAND, so an
    external takeover cannot leave the manager commanding a vehicle that is
    no longer listening.
"""

# --------------------------------------------------------------- guard cfg
EV_STALE_S = 0.5          # /mcl/odom silence -> EV invalid (NaN EV stream)
STREAM_END_S = 5.0        # feedforward: this much EV silence = stream ended
ALT_EPS_M = 3.0           # takeoff altitude tolerance [m]
WP_RADIUS_M = 8.0         # waypoint arrival radius [m]

# health states the guard tolerates, with the grace [s] before they count as
# degraded ('no_frames'/'lost' are hard conditions: grace 0)
DEGRADE_GRACE = {'no_frames': 0.0, 'lost': 0.0, 'init_wait': 30.0,
                 'motion_only': 25.0, 'recovering': 25.0}
VIO_GRACE_S = 3.0         # /mcl/health vio == 'drop' -> degraded after this
OFFBOARD_LOST_S = 3.0     # not armed+OFFBOARD this long -> external takeover
MARGIN_PERSIST_S = 3.0    # honest margin breach must persist this long

# guard states
WAIT, TAKEOFF, MISSION, HOLD, RETURN, LAND, DONE = range(7)
STATE_NAMES = {WAIT: 'wait', TAKEOFF: 'takeoff', MISSION: 'mission',
               HOLD: 'hold', RETURN: 'return', LAND: 'land', DONE: 'done'}


class Guard:
    """Health-driven mission guard (pure logic).

    update() is called every manager tick with the current facts and returns
    the (possibly new) state. All limits are injectable for tests.
    """

    def __init__(self, cruise_alt=50.0, takeoff_max_s=120.0,
                 hold_budget_s=180.0, return_max_s=60.0, mission_max_s=900.0,
                 land_max_s=120.0, margin_m=1000.0, map_half_m=2500.0,
                 degrade_grace=None, vio_grace_s=VIO_GRACE_S,
                 ev_stale_s=EV_STALE_S, offboard_lost_s=OFFBOARD_LOST_S,
                 margin_persist_s=MARGIN_PERSIST_S):
        self.cruise_alt = float(cruise_alt)
        self.takeoff_max_s = float(takeoff_max_s)
        self.hold_budget_s = float(hold_budget_s)
        self.return_max_s = float(return_max_s)
        self.mission_max_s = float(mission_max_s)
        self.land_max_s = float(land_max_s)
        self.margin_m = float(margin_m)
        self.map_half_m = float(map_half_m)
        self.degrade_grace = dict(DEGRADE_GRACE if degrade_grace is None
                                  else degrade_grace)
        self.vio_grace_s = float(vio_grace_s)
        self.ev_stale_s = float(ev_stale_s)
        self.offboard_lost_s = float(offboard_lost_s)
        self.margin_persist_s = float(margin_persist_s)

        self.state = WAIT
        self.reason = 'wait_ev'
        self.degrade = None          # current degradation condition
        self.degrade_since = None    # when that condition first appeared
        self.hold_total = 0.0        # cumulative HOLD time [s]
        self.t_start = None          # mission clock
        self.t_enter = None          # state-entry clock
        self.last_t = None
        self.offboard_since = None   # not ready since (None = ready)
        self.margin_since = None     # last honest breach started (None = in)
        self.pos_out = False         # last honest fix was outside

    # ------------------------------------------------------------- helpers
    def outside_margin(self, pos):
        """True when pos (map ENU [m]) violates the map-edge margin."""
        if pos is None:
            return False
        lim = self.map_half_m - self.margin_m
        return abs(pos[0]) > lim or abs(pos[1]) > lim

    def _evaluate(self, now, health, ev_age):
        """(degraded, condition) from EV freshness / VIO / health state."""
        cond = grace = None
        if ev_age is None or ev_age > self.ev_stale_s:
            cond, grace = 'ev_stale', 0.0
        elif health is None:
            pass                          # no health topic: EV freshness only
        elif health.get('vio') in ('drop', 'bad'):
            # mcl_node classifies every processed frame's VIO as one of
            # {'none', 'ok', 'resume', 'bad'} — 'bad' = missing/catastrophic
            # (diverged SchurVINS, z to -11 km observed on bag 190020).
            # 'drop' is kept for back-compat with the original step-5 tests.
            cond, grace = 'vio_drop', self.vio_grace_s
        else:
            st = str(health.get('state', 'no_frames'))
            if st != 'tracking':
                cond = st
                grace = self.degrade_grace.get(st, 0.0)
                if self.state == TAKEOFF and st in ('init_wait', 'no_frames'):
                    grace = self.takeoff_max_s   # MCL scans during the climb
        if cond is None:
            self.degrade = None
            self.degrade_since = None
            return False, None
        if cond != self.degrade:
            self.degrade = cond
            self.degrade_since = now
        return (now - self.degrade_since) > grace, cond

    # -------------------------------------------------------------- update
    def update(self, now, ready, health, ev_age, pos, alt, mission_done,
               landed):
        """Advance the guard; returns the new state.

        ready         armed + OFFBOARD (manager)
        health        /mcl/health JSON dict (None -> not seen yet)
        ev_age        seconds since the last /mcl/odom (None -> never)
        pos           honest EV map ENU (x east, y north) or None
        alt           altitude [m] above the local frame origin (None -> ?)
        mission_done  mission finished (waypoints exhausted / EV stream end)
        landed        PX4 land detector
        """
        if self.t_start is None:
            self.t_start = self.t_enter = self.last_t = now
        if self.last_t is not None and self.state == HOLD:
            self.hold_total += now - self.last_t
        self.last_t = now
        dt = now - self.t_enter
        degraded, cond = self._evaluate(now, health, ev_age)

        # ready = armed + OFFBOARD (manager). Losing it (PX4 failsafe, RTL,
        # land, disarm) must not go unnoticed while we are commanding.
        if ready:
            self.offboard_since = None
        elif self.offboard_since is None:
            self.offboard_since = now
        offboard_lost = (self.offboard_since is not None
                         and now - self.offboard_since > self.offboard_lost_s)

        # margin: honest fixes only (gated frames are NOT "back inside"), and
        # a breach must persist before it counts
        if pos is not None:
            if self.outside_margin(pos):
                if self.margin_since is None:
                    self.margin_since = now
            else:
                self.margin_since = None
        outside = (self.margin_since is not None
                   and now - self.margin_since > self.margin_persist_s)

        def go(state, reason):
            if state != self.state:
                self.state = state
                self.reason = reason
                self.t_enter = now

        st = self.state
        if st == WAIT:
            if ready:
                go(TAKEOFF, 'armed_offboard')
        elif st == TAKEOFF:
            if alt is not None and alt >= self.cruise_alt - ALT_EPS_M:
                go(MISSION, 'altitude_reached')
            elif offboard_lost:
                go(LAND, 'offboard_lost')
            elif dt > self.takeoff_max_s:
                go(LAND, 'takeoff_timeout')
            elif cond == 'ev_stale' and (alt or 0.0) < 10.0:
                go(LAND, 'blind_low_alt')      # no EV = no velocity source
            elif degraded and cond != 'init_wait':
                go(HOLD, 'degraded:' + str(cond))
        elif st == MISSION:
            if offboard_lost:
                go(LAND, 'offboard_lost')
            elif outside:
                go(RETURN, 'map_margin')
            elif degraded:
                go(HOLD, 'degraded:' + str(cond))
            elif mission_done:
                go(LAND, 'mission_complete')
            elif now - self.t_start > self.mission_max_s:
                go(LAND, 'mission_timeout')
        elif st == HOLD:
            if offboard_lost:
                go(LAND, 'offboard_lost')
            elif mission_done:
                go(LAND, 'mission_complete')   # stream ended while holding
            elif not degraded:
                go(MISSION if not outside else RETURN, 'recovered')
            elif self.hold_total > self.hold_budget_s:
                go(LAND, 'hold_budget')
        elif st == RETURN:
            if offboard_lost:
                go(LAND, 'offboard_lost')
            elif not outside:
                go(MISSION, 'back_in_margin')
            elif degraded:
                go(HOLD, 'degraded:' + str(cond))
            elif dt > self.return_max_s:
                go(LAND, 'return_timeout')
        elif st == LAND:
            if landed:
                go(DONE, 'landed')
            elif dt > self.land_max_s:
                go(DONE, 'land_timeout')      # manager forces DISARM
        return self.state

    # --------------------------------------------------------------- misc
    @property
    def state_name(self):
        return STATE_NAMES[self.state]