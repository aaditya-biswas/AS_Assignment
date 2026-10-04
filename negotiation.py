"""Negotiation as POP operations (``SPEC.md`` section 9.2).

``modify.py`` owns the escalation ladder; this module owns the *protocol* the
ladder speaks when a candidate plan wants a link that somebody else has
reserved::

    negotiate(i, depth, chain):
        path, C = st_astar_soft(...)      # C = clashed agents
        if C empty: apply to a PEG copy; return success
        for j in C within comm_radius, j not in chain:
            PROPOSE_ORDER(i->j, visit, link, 'before'|'after')
            j: 1. acyclic and delay(j) <= 2*delay_threshold -> ACCEPT(delay)
               2. else j reroutes its own leg (others frozen) -> ACCEPT(delay)
               3. else if depth < max_depth: sub-negotiation -> ACCEPT/REJECT
               4. else REJECT(reason)
        if all ACCEPT: COMMIT; check acyclic -> success
        else: ABORT; add the rejected clash as a hard constraint; retry <= 3

The figure maps onto this code base like this:

* ``st_astar_soft`` is :func:`planner.st_astar` in soft mode (a clash costs
  ``lambda_soft``).  The proposal is the whole remaining tour, computed by
  ``modify._plan_from`` with ``soft_lambda``/``clash_sink``; the sink collects
  the set ``C`` of clashed agents.
* **ACCEPT by waiting is free.**  The agents that are replanned *after* the
  initiator are re-derived against the initiator's new reservations, so they
  simply wait or reroute -- no further message is needed.  That is the
  "``j`` only waits" branch of SPEC 9.2 and it is why most negotiations alter
  few agents.
* **REJECT** is the honest verdict for a holder that cannot help: a physically
  held agent (accident/breakdown/emergency hold, dead agent), an agent whose
  plan is frozen (outside the replan set, or already committed earlier in the
  pass) or a chain that would close a cycle.  The rejected clash then stays a
  *hard* constraint for the initiator, which is what the hard ST-A*
  regrounding of the ladder does -- the retry of the pseudocode.
* **R3 widening.**  A holder outside ``comm_radius`` but within
  ``comm_radius + 3*(max_depth - depth)`` is pulled into the replan set, so it
  can yield after all (SPEC 9.1 "R2 with ``comm_radius += 3``").
* A grant is only taken when it *pays*: the initiator's saving must beat
  ``delay_threshold`` **and** the ``beta_alter`` price of every agent the grant
  newly alters (SPEC 9.1 ``cost = added_delay_total + beta_alter*altered``).
  The granted proposal is additionally verified clash-free against every
  non-yielding holder (:func:`first_clash`), so it can never introduce a
  collision.

Every message is recorded, counted by type (:meth:`Negotiation.by_type`) and
lands in ``RepairRecord.messages_by_type`` (SPEC 9.5), in the simulator trace,
and in the message arrows of the visualisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (Callable, Dict, Iterable, List, Optional, Sequence, Set,
                    Tuple)

from planner import PlanStep
from reservation import ReservationTable
from world import Cell

PROPOSE_ORDER = "PROPOSE_ORDER"
PROPOSE_REROUTE = "PROPOSE_REROUTE"
ACCEPT = "ACCEPT"
REJECT = "REJECT"
COMMIT = "COMMIT"
ABORT = "ABORT"

#: Message types of ``SPEC.md`` 9.2, in protocol order.
MESSAGE_TYPES = (PROPOSE_ORDER, PROPOSE_REROUTE, ACCEPT, REJECT, COMMIT,
                 ABORT)

#: Rung ``R3`` widens ``comm_radius`` by this much per sub-negotiation.
WIDEN_STEP = 3



@dataclass(frozen=True)
class Message:
    """One protocol message (kept for the trace and the message arrows)."""

    kind: str
    sender: int
    receiver: int
    t: int
    detail: str = ""


@dataclass
class Verdict:
    """A holder's answer to one proposal."""

    holder: int
    granted: bool
    how: str = "ordering"            # ordering | reroute | held | frozen | ...
    reason: str = ""
    #: agents that must also be replanned for the grant to hold (rung R3)
    extra: Set[int] = field(default_factory=set)


@dataclass
class Negotiation:
    """Message log + counters for one repair pass (``SPEC.md`` 9.2/9.5)."""

    t: int = 0
    messages: List[Message] = field(default_factory=list)
    rounds: int = 0                  # proposals sent
    committed: int = 0
    aborted: int = 0

    # ---- sending ------------------------------------------------------
    def send(self, kind: str, sender: int, receiver: int,
             detail: str = "") -> Message:
        msg = Message(kind, sender, receiver, self.t, detail)
        self.messages.append(msg)
        return msg

    def propose_order(self, initiator: int, holder: int,
                      detail: str = "") -> Message:
        self.rounds += 1
        return self.send(PROPOSE_ORDER, initiator, holder, detail)

    def propose_reroute(self, initiator: int, holder: int,
                        detail: str = "") -> Message:
        self.rounds += 1
        return self.send(PROPOSE_REROUTE, initiator, holder, detail)

    def accept(self, holder: int, initiator: int,
               detail: str = "") -> Message:
        return self.send(ACCEPT, holder, initiator, detail)

    def reject(self, holder: int, initiator: int,
               detail: str = "") -> Message:
        return self.send(REJECT, holder, initiator, detail)

    def commit(self, initiator: int, holders: Iterable[int],
               detail: str = "") -> List[Message]:
        holders = list(holders)
        self.committed += 1
        return [self.send(COMMIT, initiator, j, detail) for j in holders]

    def abort(self, initiator: int, holders: Iterable[int],
              detail: str = "") -> List[Message]:
        holders = list(holders)
        self.aborted += 1
        return [self.send(ABORT, initiator, j, detail) for j in holders]

    # ---- reading ------------------------------------------------------
    def by_type(self) -> Dict[str, int]:
        """Counts per message type (always contains every type)."""
        out: Dict[str, int] = {k: 0 for k in MESSAGE_TYPES}
        for m in self.messages:
            out[m.kind] = out.get(m.kind, 0) + 1
        return out

    def arrows(self) -> List[Tuple[int, int]]:
        """``(sender, receiver)`` of every directed message (for the viz)."""
        return [(m.sender, m.receiver) for m in self.messages
                if m.kind in (PROPOSE_ORDER, PROPOSE_REROUTE, ACCEPT, REJECT)]

    def details(self, kind: str) -> List[str]:
        return [m.detail for m in self.messages if m.kind == kind]

    def __len__(self) -> int:
        return len(self.messages)


# ----------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------
def first_clash(res: ReservationTable, aid: int, path: Sequence[PlanStep],
                ignore: Iterable[int] = ()) -> Optional[Tuple[Cell, int, int]]:
    """First ``(cell, t, holder)`` of ``path`` booked by a non-ignored agent.

    ``ignore`` are the agents that granted the proposal and are therefore
    re-derived; everybody else keeps its reservations, so a clash with them
    would be a real collision.
    """
    skip = set(ignore) | {aid}
    for a, b in zip(path, path[1:]):
        for other in sorted(res.clashes(aid, a.cell, b.cell, a.t)):
            if other not in skip:
                return b.cell, b.t, other
    return None


def displaced_ticks(res: ReservationTable, holder: int,
                    path: Sequence[PlanStep]) -> int:
    """How many booked cell-times of ``holder`` the proposal would take over.

    A conservative stand-in for ``delay(j)`` (SPEC 9.2): every cell-time ``j``
    has already booked that the initiator now wants is one tick ``j`` has to
    give up.
    """
    return sum(1 for s in path if res.vertex.get((s.cell, s.t)) == holder)


def first_blocked_tick(res: ReservationTable, aid: int, holder: int,
                       path: Sequence[PlanStep]) -> Optional[int]:
    """Earliest tick at which ``path`` is blocked *by this holder*."""
    for a, b in zip(path, path[1:]):
        if holder in res.clashes(aid, a.cell, b.cell, a.t):
            return b.t
    return None


def paths_clash(a: Sequence[PlanStep], b: Sequence[PlanStep]
                ) -> Optional[Tuple[Cell, int]]:
    """First ``(cell, t)`` where two paths disagree (vertex or swap clash).

    A granted proposal must not only avoid the reservations that stayed frozen
    (:func:`first_clash`) -- it must also stay consistent with the *new* plan
    of every holder that rerouted its own leg for it.
    """
    cells_a = {(s.cell, s.t) for s in a}
    cells_b = {(s.cell, s.t) for s in b}
    both = sorted(cells_a & cells_b)
    if both:
        return both[0]
    pos_a = {s.t: s.cell for s in a}
    pos_b = {s.t: s.cell for s in b}
    for t in sorted(set(pos_a) & set(pos_b)):
        if t + 1 in pos_a and t + 1 in pos_b and \
                pos_a[t] == pos_b[t + 1] and pos_a[t + 1] == pos_b[t]:
            return pos_a[t], t
    return None


def grant_is_worthwhile(*, hard_completion: int, prop_completion: int,
                        n_new_altered: int, beta_alter: int,
                        delay_threshold: int) -> bool:
    """SPEC 9.1 ``cost = added_delay_total + beta_alter*altered`` for a grant.

    The granted proposal has to save the initiator more than
    ``delay_threshold`` ticks (9.4: "a leg is rerouted only if waiting costs
    more than ``delay_threshold``") **and** more than the ``beta_alter`` price
    of every agent the grant newly alters.
    """
    saved = hard_completion - prop_completion
    return saved > delay_threshold and saved > beta_alter * n_new_altered


# ----------------------------------------------------------------------
# the protocol
# ----------------------------------------------------------------------
def resolve_clash(holder: int, initiator: int, *, distance: float,
                  replan_set: Set[int], order_index: Dict[int, int],
                  comm_radius: int, chain: Sequence[int] = (),
                  depth: int = 0, max_depth: int = 3, displaced: int = 0,
                  delay_threshold: int = 8,
                  fixed: Iterable[int] = (),
                  hold_end: Optional[Dict[int, int]] = None,
                  contested: Optional[int] = None,
                  reroute_cost: Optional[Callable[[int],
                                                  Optional[int]]] = None,
                  log: Optional[Negotiation] = None,
                  widen: bool = True) -> Verdict:
    """Ask ``holder`` to make room for ``initiator`` (``SPEC.md`` 9.2).

    Returns the holder's :class:`Verdict`.  ``order_index`` is the priority
    rank inside the current pass.  The branches of the pseudocode are:

    1. the holder is re-derived *after* the initiator in this pass, so it
       simply waits -- ACCEPT (it may never cost the initiator a wait, which is
       why the caller only takes such a grant when it saves time anyway);
    2. the holder is already committed *earlier* in the pass: it repairs
       itself, rerouting its own leg with the other agents frozen.  The caller
       prices that reroute through ``reroute_cost`` (``holder -> added delay``,
       ``None`` when the holder cannot reroute) and it is accepted only if the
       delay stays within ``2*delay_threshold``;
    3. rung ``R3``: a holder outside ``comm_radius`` but within
       ``comm_radius + 3*(max_depth - depth)`` is pulled into the replan set;
    4. otherwise REJECT -- the clash stays *hard* for the initiator, which is
       exactly the retry of the ladder's hard ST-A* regrounding.

    ``fixed`` are the agents that can never move (permanent breakdown, dead
    agent) and ``hold_end``/``contested`` make the accident hold time-aware: an
    agent that is held *now* can still yield later, so it rejects only when the
    proposal wants its cell before its hold ends.
    """
    if holder == initiator:
        return Verdict(holder, True, "self", "the initiator holds the link")
    if holder in chain:
        if log is not None:
            log.propose_order(initiator, holder, "cycle")
            log.reject(holder, initiator, "ordering would close a cycle")
        return Verdict(holder, False, "cycle", "ordering would close a cycle")
    until = (hold_end or {}).get(holder)
    if holder in set(fixed) or (until is not None and contested is not None
                                and contested < until):
        if log is not None:
            log.propose_order(initiator, holder, "held")
            log.reject(holder, initiator, "the holder cannot move in time")
        why = ("the holder is physically held in place" if until is None
               else f"the holder is held until t={until}")
        return Verdict(holder, False, "held", why)

    n = len(order_index)
    later = order_index.get(holder, n + 1) > order_index.get(initiator, n)
    within = distance <= comm_radius

    if holder in replan_set and later:
        if displaced > 2 * delay_threshold:
            if log is not None:
                log.propose_order(initiator, holder, f"delay={displaced}")
                log.reject(holder, initiator,
                           f"delay {displaced} > 2*{delay_threshold}")
            return Verdict(holder, False, "delay",
                           f"delay {displaced} exceeds 2*{delay_threshold}")
        detail = f"delay={displaced}"
        if log is not None:
            log.propose_order(initiator, holder, detail)
            log.accept(holder, initiator, detail)
        return Verdict(holder, True, "ordering" if within else "reroute")

    if holder in replan_set and reroute_cost is not None:
        delay = reroute_cost(holder)
        if delay is None:
            if log is not None:
                log.propose_reroute(initiator, holder, "no detour")
                log.reject(holder, initiator, "the holder cannot reroute")
            return Verdict(holder, False, "blocked",
                           "the holder cannot reroute around the proposal")
        if delay > 2 * delay_threshold:
            if log is not None:
                log.propose_reroute(initiator, holder, f"delay={delay}")
                log.reject(holder, initiator,
                           f"delay {delay} > 2*{delay_threshold}")
            return Verdict(holder, False, "delay",
                           f"reroute would delay it by {delay}")
        if log is not None:
            log.propose_reroute(initiator, holder, f"delay={delay}")
            log.accept(holder, initiator, f"delay={delay}")
        return Verdict(holder, True, "reroute", "holder reroutes its own leg")

    if not within and widen and depth < max_depth:
        reach = comm_radius + WIDEN_STEP * (max_depth - depth)
        if distance <= reach:
            if log is not None:
                log.propose_reroute(
                    initiator, holder,
                    f"radius {comm_radius}->{comm_radius + WIDEN_STEP}")
            return Verdict(holder, True, "reroute",
                           "holder pulled into the replan set (R3)",
                           extra={holder})

    if log is not None:
        log.propose_order(initiator, holder, "frozen")
        log.reject(holder, initiator, "the holder's plan is frozen")
    return Verdict(holder, False, "frozen" if within else "radius",
                   "the holder is not replanned in this pass")


def negotiate(initiator: int, clash: Iterable[int], *,
              distance: Callable[[int], float], replan_set: Set[int],
              order_index: Dict[int, int], comm_radius: int,
              res: Optional[ReservationTable] = None,
              path: Sequence[PlanStep] = (),
              chain: Sequence[int] = (), depth: int = 0, max_depth: int = 3,
              delay_threshold: int = 8, fixed: Iterable[int] = (),
              hold_end: Optional[Dict[int, int]] = None,
              reroute_cost: Optional[Callable[[int], Optional[int]]] = None,
              log: Optional[Negotiation] = None
              ) -> Tuple[List[Verdict], Set[int]]:
    """Run the ``for j in C`` loop of SPEC 9.2 for one initiator.

    Returns ``(verdicts, extra)`` where ``extra`` are the agents that must also
    be replanned for the granted verdicts to hold (rung R3).  ``distance`` maps
    a holder to its distance from the initiator; ``reroute_cost`` prices the
    "``j`` reroutes its own leg" branch (see :func:`resolve_clash`).
    """
    verdicts: List[Verdict] = []
    extra: Set[int] = set()
    for holder in sorted(set(clash) - {initiator}):
        displaced = displaced_ticks(res, holder, path) if res else 0
        contested = (first_blocked_tick(res, initiator, holder, path)
                     if res else None)
        v = resolve_clash(
            holder, initiator, distance=distance(holder),
            replan_set=replan_set, order_index=order_index,
            comm_radius=comm_radius, chain=chain, depth=depth,
            max_depth=max_depth, displaced=displaced,
            delay_threshold=delay_threshold, fixed=fixed, hold_end=hold_end,
            contested=contested, reroute_cost=reroute_cost, log=log)
        verdicts.append(v)
        extra |= set(v.extra)
    return verdicts, extra


def granted_holders(verdicts: Sequence[Verdict]) -> List[int]:
    """The holders that accepted (for ``COMMIT`` / :func:`first_clash`)."""
    return [v.holder for v in verdicts if v.granted]



