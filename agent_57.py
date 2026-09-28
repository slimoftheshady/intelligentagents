import time
#import timeout_decorator
import math
import random
import signal
from collections import defaultdict, deque
from agent_baselines import Agent
from game import copy_game

'''
WINDOWS COMPATIBILITY NOTE:
    The timeout_decorator package may not work correctly on Windows. For local
    development on Windows, you may comment out the import and all four
    @timeout_decorator.timeout(1) lines in this file. If you do so, measure the
    running time of __init__, new_game, update_game, and get_actions yourself
    (for example, with time.perf_counter). This local workaround does not relax
    the one-second limit: it is a hard constraint and will be enforced
    independently during marking.
'''

if hasattr(signal, 'SIGALRM'):
    import timeout_decorator # type: ignore
    _timeout = timeout_decorator.timeout
else:
    def _timeout(seconds):
        def deco(f):
            return f
        return deco

INF = 10 ** 6


def loc_adjacent(agent, t, prov, target):
    return prov in agent.stage[t].get(target, ())


def base(loc):
    return loc.split('/')[0].upper()

class StudentAgent(Agent):
    '''
    Implement your agent here. 

    Please read the abstract Agent class from agent_baselines.py first.
    
    You can add/override attributes and methods as needed.

    v0 architecture:
        new_game    -> precompute adjacency + BFS distance tables
        update_game -> opponent model (static detection from observed orders)
        get_actions -> movement: greedy plan allocation (capture / defend) with
                        coordinated supports, then advance free units, then a
                        fix-up pass that removes self-bounces.
                        retreat / adjustment: scored choices.
    Flags let each technique be switched off for ablation experiments.
    '''
    @_timeout(1)
    def __init__(self, agent_name='UCB1Agent', support_alloc=True,threat_aware=False, opp_model=True, use_ucb=False, budget=0.5, n_candidates=12, ucb_c=0.1, n_samples=8):
        super().__init__(agent_name)
        self.USE_UCB = use_ucb          # basic technique: 1-ply UCB1 lookahead
        self.BUDGET = budget            # seconds of search per movement phase
        self.K = n_candidates
        self.C = ucb_c
        self.M = n_samples
        self.USE_SUPPORT_ALLOC = support_alloc
        self.USE_THREAT = threat_aware
        self.USE_OPP_MODEL = opp_model

    @_timeout(1)
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name

        m = game.map
        self.scs = sorted(set(s.upper() for s in m.scs))
        self.homes = {p: set(h.upper() for h in hs) for p, hs in m.homes.items()}

        ltype = {k.upper(): v for k, v in m.loc_type.items()}
        allowed = {'A': ('LAND', 'COAST'), 'F': ('WATER', 'COAST')}
        self.nbrs = {'A': {}, 'F': {}}
        for u in ('A', 'F'):
            nodes = [l for l, t in ltype.items() if t in allowed[u]]
            for i in nodes:
                self.nbrs[u][i] = [j for j in nodes if m.abuts(u, i, '-', j)]
        self.nbr_base = {u: {i: set(base(j) for j in js) for i, js in self.nbrs[u].items()}
                            for u in ('A', 'F')}
        
        # dist[u][node][province] = moves needed for a unit of type u at node
        self.dist = {'A': {}, 'F': {}}
        for u in ('A', 'F'):
            for src in self.nbrs[u]:
                d = {src: 0}
                q = deque([src])
                while q:
                    x = q.popleft()
                    for y in self.nbrs[u][x]:
                        if y not in d:
                            d[y] = d[x] + 1
                            q.append(y)
                best = {}
                for node, k in d.items():
                    b = base(node)
                    if k < best.get(b, INF):
                        best[b] = k
                self.dist[u][src] = best

        # stage[u][t]: provinces from which a type-u unit can move into t
        self.stage = {u: defaultdict(set) for u in ('A', 'F')}
        for u in ('A', 'F'):
            for node, bs in self.nbr_base[u].items():
                for t in bs:
                    self.stage[u][t].add(base(node))

        # opponent model state
        self.seen_moves = defaultdict(int)     # movement phases observed
        self.active_moves = defaultdict(int)   # phases where it moved/supported
        self.deadline = None

    # This is only for updating the game engine and other states if any. Do not implement heavy stratergy here.
    @_timeout(1)
    def update_game(self, all_power_orders):
        if self.game.phase_type == 'M':
            for p, orders in all_power_orders.items():
                if p == self.power_name or not self.game.powers[p].units:
                    continue
                self.seen_moves[p] += 1
                if any((' - ' in o) or (' S ' in o) or (' C ' in o) for o in orders):
                    self.active_moves[p] += 1

        # do not make changes to the following codes
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

    def is_active(self, p):
        '''Could power p move against us? Static = never issued a non-hold order.'''
        if not self.USE_OPP_MODEL:
            return True
        if self.seen_moves[p] == 0:
            return True   # unknown -> assume dangerous
        return self.active_moves[p] > 0

    @_timeout(1)
    def get_actions(self):

        self._t0 = time.perf_counter()
        self.deadline = self._t0 + 0.5
        pt = self.game.phase_type
        try:
            if pt == 'M':
                return self._movement()
            if pt == 'R':
                return self._retreats()
            if pt == 'A':
                return self._adjustments()
        except Exception:
            self.errors = getattr(self, 'errors', 0) + 1
            return []   # never crash: holding is better than a lost game
        return []

        # '''
        # Return a list of orders. Each order is a string, with specific format. For the format, read the game rule and game engine documentation.
        
        # Expected format:
        # A LON H                  # Army at LON holds
        # F IRI - MAO              # Fleet at IRI moves to MAO (and attack)
        # A WAL S F LON            # Army at WAL supports Fleet at LON (and hold)
        # F NTH S A EDI - YOR      # Fleet at NTH supports Army at EDI to move to YOR
        # F NWG C A NWY - EDI      # Fleet at NWG convoys Army at NWY to EDI
        # A NWY - EDI VIA          # Army at NWY moves to EDI via convoy
        # A WAL R LON              # Army at WAL retreats to LON
        # A LON D                  # Disband Army at LON
        # A LON B                  # Build Army at LON
        # F EDI B                  # Build Fleet at EDI

        # Note: If an invalid order is sent to the engine, it will be accepted but with a result of 'void' (no effect).
        # Note: For a 'support' action, two orders are needed, one for the supporter and one for the supportee. (Same for 'convoy')
        # Note: For each unit, if no order is given, it will 'hold' by default.

        # Useful Functions:
        
        # # This is a dict of all the possible orders for each unit at each location (for all powers).
        # possible_orders = self.game.get_all_possible_orders()

        # # This is a list of all orderable locations for the power you control.
        # orderable_locations = self.game.get_orderable_locations(self.power_name)
    
        # # Combining these two, you can have the full action space for the power you control.

        # # You can re-use the build_map_graphs function in the GreedyAgent to build the connection graph of the map if needed.
        
        # '''
    # ----------------------------------------------------------- state utils
    def _snapshot(self):
        g = self.game
        self.units = {}      # power -> [(type, full_loc)]
        self.occ = {}        # province -> (power, type, full_loc)
        self.owner = {}      # sc -> power
        for p, pw in g.powers.items():
            lst = []
            for u in pw.units:
                t, loc = u.lstrip('*').split()[:2]
                lst.append((t, loc.upper()))
                self.occ[base(loc)] = (p, t, loc.upper())
            self.units[p] = lst
            for c in pw.centers:
                self.owner[c.upper()] = p

    def adj_count(self, p, prov, exclude_loc=None):
        '''Number of p's units that could move into / support into prov.'''
        n = 0
        for t, loc in self.units.get(p, []):
            if loc == exclude_loc or base(loc) == prov:
                continue
            if prov in self.nbr_base[t].get(loc, ()):
                n += 1
        return n

    def threat(self, prov):
        '''Strongest single-power attack that could hit prov next phase.'''
        if not self.USE_THREAT:
            return 0
        best = 0
        for p in self.units:
            if p == self.power_name or not self.is_active(p):
                continue
            best = max(best, self.adj_count(p, prov))
        return best

    def unit_dist(self, t, loc, prov):
        return self.dist[t].get(loc, {}).get(prov, INF)

    # --------------------------------------------------------------- movement
    def _index_orders(self, locs, po):
        idx = {}
        for loc in locs:
            e = {'hold': None, 'move': {}, 'via': {}, 'sup_move': {},
                    'sup_hold': {}, 'convoy': {}}
            for o in po.get(loc, []):
                w = o.split()
                if len(w) == 3 and w[2] == 'H':
                    e['hold'] = o
                elif len(w) == 4 and w[2] == '-':
                    e['move'].setdefault(base(w[3]), []).append((o, w[3]))
                elif len(w) == 5 and w[2] == '-' and w[4] == 'VIA':
                    e['via'][base(w[3])] = o
                elif w[2] == 'S' and len(w) == 5:
                    e['sup_hold'][base(w[4])] = o
                elif w[2] == 'S' and len(w) == 7:
                    e['sup_move'][(base(w[4]), base(w[6]))] = o
                elif w[2] == 'C' and len(w) == 7:
                    e['convoy'][(base(w[4]), base(w[6]))] = o
            idx[loc] = e
        return idx

    def _value(self, prov, fall):
        own = self.owner.get(prov)
        v = 1.0 if own is None else 1.2
        return v * (1.5 if fall else 1.0)

    def _movement(self):
        g = self.game
        me = self.power_name
        self._snapshot()
        fall = g.get_current_phase().startswith('F')
        po = g.get_all_possible_orders()
        my_locs = [l for l in g.get_orderable_locations(me) if po.get(l)]
        idx = self._index_orders(my_locs, po)
        unit_of = {base(loc): (t, loc) for t, loc in self.units[me]}
        my_scs = set(c for c, p in self.owner.items() if p == me)
        targets = [s for s in self.scs if s not in my_scs]

        orders = {}
        free = set(my_locs)
        planned = set()   # provinces we already committed to capture/defend

        # ---- phase 1: greedy plan allocation --------------------------------
        while free and time.perf_counter() < self.deadline:
            best = None
            for prov in self.scs:
                if prov in planned:
                    continue
                plan = self._plan_for(prov, free, idx, unit_of, fall, my_scs)
                if plan and (best is None or plan[0] > best[0]):
                    best = plan
            if best is None or best[0] <= 0.05:
                break
            score, prov, new_orders = best
            for loc, o in new_orders.items():
                orders[loc] = o
                free.discard(loc)
            planned.add(prov)

        # ---- phase 2: advance remaining units ------------------------------
        claimed = set()
        for loc, o in orders.items():
            w = o.split()
            if len(w) >= 4 and w[2] == '-':
                claimed.add(base(w[3]))
        staying = set(l for l, o in orders.items() if not self._is_move(o))
        staying |= set(l for l in free)   # not yet decided: assume staying

        # armies that cannot reach any target by land try a convoy first
        for loc in sorted(free, key=lambda l: (unit_of[l][0] != 'A', l)):
            if loc not in free:
                continue
            t, full = unit_of[loc]
            if t == 'A' and min((self.unit_dist('A', full, s) for s in targets), default=INF) >= INF:
                self._try_convoy(loc, full, idx, unit_of, free, orders, claimed, targets)

        # need[t]: attackers required next to target t; have[t]: already staged
        need, have = {}, defaultdict(int)
        for s in targets:
            o = self.occ.get(s)
            if o is None:
                need[s] = 1 + self.threat(s)
            elif o[0] == me:
                need[s] = 0
            else:
                need[s] = 2 + (self.adj_count(o[0], s, o[2]) if self.is_active(o[0]) and self.USE_THREAT else 0)
        taken = set(base(l) for l in staying) | claimed   # squares our units will occupy

        mine_at = set(base(l) for _, l in self.units[me])
        enemy_at = set(p for p, v in self.occ.items() if v[0] != me)
        sd = self._stage_distances(targets, need, enemy_at, mine_at)

        def stage_dist(t, full, s, own_loc):
            if s in self.nbr_base[t].get(full, ()):
                return 0          # already next to the target
            return sd[t].get(s, {}).get(full, INF)

        def pos_score(t, full, own_loc):
            best, arg = 0.0, None
            for s in targets:
                if need[s] == 0:
                    continue
                d = stage_dist(t, full, s, own_loc)
                if d >= INF:
                    continue
                v = self._value(s, fall) / (1 + d)
                if have[s] >= need[s]:
                    v *= 0.3            # already enough attackers: go elsewhere
                if s in planned:
                    v *= 0.3
                if v > best:
                    best, arg = v, s
            return best, arg

        order_units = sorted(free, key=lambda l: (-pos_score(unit_of[l][0], unit_of[l][1], l)[0], l))
        for loc in order_units:
            if loc not in free:
                continue
            t, full = unit_of[loc]
            hold_sc, hold_t = pos_score(t, full, loc)
            choice = (hold_sc, None, loc, hold_t)
            for dest, lst in sorted(idx[loc]['move'].items()):
                if dest in claimed or dest in staying - {loc}:
                    continue
                o_occ = self.occ.get(dest)
                if o_occ and o_occ[0] != me:
                    continue   # lone unit into an enemy unit only bounces
                for o, dfull in lst:
                    sc, st = pos_score(t, dfull, dest)
                    sc -= 0.02 * self.threat(dest)
                    if sc > choice[0] + 1e-9:
                        choice = (sc, o, dest, st)
            _, o, dest, st = choice
            if o is None:
                o = self._useful_support(loc, idx, orders) or idx[loc]['hold']
            else:
                claimed.add(dest)
                taken.add(dest)
                staying.discard(loc)
                taken.discard(loc)
            if st is not None and loc_adjacent(self, t, dest, st):
                have[st] += 1
            orders[loc] = o
            free.discard(loc)

        orders = self._fixup(orders, idx)
        if self.USE_UCB:
            orders = self._ucb_refine(orders, idx, po)
        return list(orders.values())

    def _stage_distances(self, targets, need, enemy_at, mine_at):
        """sd[u][t][node]: cost for a type-u unit at node to reach a free square
        from which it can move into t. Enemy-occupied provinces are impassable,
        our own units cost 2 to pass through (they may or may not move)."""
        import heapq
        sd = {'A': {}, 'F': {}}
        for u in ('A', 'F'):
            for t in targets:
                if need.get(t, 0) == 0:
                    continue
                dist = {}
                heap = []
                for n, bs in self.nbr_base[u].items():
                    if t in bs and base(n) not in enemy_at and base(n) not in mine_at:
                        dist[n] = 0
                        heap.append((0, n))
                heapq.heapify(heap)
                while heap:
                    d, x = heapq.heappop(heap)
                    if d > dist.get(x, INF):
                        continue
                    for y in self.nbrs[u][x]:
                        by = base(y)
                        if by in enemy_at:
                            continue
                        nd = d + (2 if base(x) in mine_at else 1)
                        if nd < dist.get(y, INF):
                            dist[y] = nd
                            heapq.heappush(heap, (nd, y))
                sd[u][t] = dist
        return sd


    # ============================================================ UCB1 search
    # Basic technique: one-ply Monte Carlo lookahead. Candidate order sets are
    # the heuristic plan plus single-unit perturbations; each "pull" samples
    # opponent orders, simulates one phase with the real engine, and scores the
    # result with a heuristic evaluation. UCB1 decides which candidate to pull.

    def _phase_seed(self):
        return sum(ord(c) * (i + 1) for i, c in enumerate(self.game.get_current_phase() + self.power_name))

    def _alternatives(self, loc, idx, orders):
        e = idx[loc]
        alts = []
        if e['hold']:
            alts.append(e['hold'])
        for dest, lst in sorted(e['move'].items()):
            alts.extend(o for o, _ in lst)
        for mv, o in sorted(orders.items()):
            if mv == loc:
                continue
            w = o.split()
            if self._is_move(o):
                so = e['sup_move'].get((mv, base(w[3])))
            else:
                so = e['sup_hold'].get(mv)
            if so:
                alts.append(so)
        cur = orders.get(loc)
        return [a for a in alts if a != cur]

    def _make_candidates(self, base_orders, idx, rng):
        cands = [dict(base_orders)]
        seen = {frozenset(base_orders.items())}
        locs = sorted(base_orders)
        tries = 0
        while len(cands) < self.K and tries < 10 * self.K and locs:
            tries += 1
            loc = rng.choice(locs)
            alts = self._alternatives(loc, idx, base_orders)
            if not alts:
                continue
            new = dict(base_orders)
            new[loc] = rng.choice(alts)
            new = self._fixup(new, idx)
            key = frozenset(new.items())
            if key in seen:
                continue
            seen.add(key)
            cands.append(new)
        return cands

    def _predict_greedy(self, p, locs, po):
        """Opponent model: each unit heads for the nearest centre p doesn't own."""
        owned = set(c for c, q in self.owner.items() if q == p)
        tg = [s for s in self.scs if s not in owned]
        out = []
        for loc in locs:
            opts = po.get(loc, [])
            if not opts:
                continue
            if base(loc) in tg:
                h = next((o for o in opts if o.endswith(' H')), None)
                if h:
                    out.append(h)
                continue
            best, bd = None, INF
            for o in opts:
                w = o.split()
                if len(w) == 4 and w[2] == '-':
                    d = min((self.unit_dist(w[0], w[3], s) for s in tg), default=INF)
                    if d < bd:
                        best, bd = o, d
            if best:
                out.append(best)
        return out

    def _sample_opponents(self, po, rng):
        g = self.game
        samples = []
        opp_locs = {p: g.get_orderable_locations(p) for p in g.powers if p != self.power_name}
        greedy = {p: self._predict_greedy(p, locs, po) for p, locs in opp_locs.items()}
        for _ in range(self.M):
            s = {}
            for p, locs in opp_locs.items():
                if not self.is_active(p):
                    s[p] = []
                elif rng.random() < 0.5:
                    s[p] = greedy[p]
                else:
                    s[p] = [rng.choice(po[l]) for l in locs if po.get(l)]
            samples.append(s)
        return samples

    def _evaluate(self, g2, fall):
        """Leaf evaluation of the simulated next state, normalised to [0, 1]."""
        me = self.power_name
        occ = {}
        mine = dislodged = 0
        for p, pw in g2.powers.items():
            for u in pw.units:
                if u.startswith('*'):
                    if p == me:
                        dislodged += 1
                    continue
                occ[base(u.split()[1])] = p
                if p == me:
                    mine += 1
        dislodged += len(g2.powers[me].retreats) if hasattr(g2.powers[me], 'retreats') else 0
        my_sc = set(c.upper() for c in g2.get_centers(me))
        lost = sum(1 for c in my_sc if occ.get(c, me) != me)
        gained = sum(1 for c in self.scs if c not in my_sc and occ.get(c) == me)
        w = 1.0 if fall else 0.6       # Spring occupation is only provisional
        sc_eff = len(my_sc) + w * (gained - lost)
        E = sc_eff + 0.2 * mine - 0.4 * dislodged
        return max(0.0, min(1.0, E / 30.0))

    def _simulate(self, cand, sample, fall):
        g2 = copy_game(self.game)
        for p, o in sample.items():
            g2.set_orders(p, o)
        g2.set_orders(self.power_name, list(cand.values()))
        g2.process()
        return self._evaluate(g2, fall)

    def _ucb_refine(self, base_orders, idx, po):
        start = time.perf_counter()
        stop = self._t0 + self.BUDGET
        rng = random.Random(self._phase_seed())
        fall = self.game.get_current_phase().startswith('F')
        cands = self._make_candidates(base_orders, idx, rng)
        if len(cands) < 2:
            return base_orders
        samples = self._sample_opponents(po, rng)
        K = len(cands)
        n = [0] * K
        s = [0.0] * K
        N = 0
        pull_t = 0.0
        while time.perf_counter() + 1.5 * pull_t < stop:
            if N < K:
                k = N                         # pull every candidate once
            else:
                logN = math.log(N)
                k = max(range(K), key=lambda i: s[i] / n[i] + self.C * math.sqrt(logN / n[i]))
            t = time.perf_counter()
            # common random numbers: candidates face the same opponent samples
            r = self._simulate(cands[k], samples[n[k] % self.M], fall)
            pull_t = max(pull_t, time.perf_counter() - t)
            n[k] += 1
            s[k] += r
            N += 1
        self.last_pulls = N
        tried = [i for i in range(K) if n[i] > 0]
        if 0 not in tried:
            return base_orders
        best = max(tried, key=lambda i: (s[i] / n[i], i == 0))
        # only abandon the heuristic plan if it is actually beaten
        if best != 0 and s[best] / n[best] <= s[0] / n[0]:
            best = 0
        return cands[best]

    def _plan_for(self, prov, free, idx, unit_of, fall, my_scs):
        '''Return (score, prov, {loc: order}) for the best way to capture or
        defend prov using free units, or None.'''
        me = self.power_name
        occ = self.occ.get(prov)
        support_ok = self.USE_SUPPORT_ALLOC

        # --- my unit is sitting on prov ---
        if occ and occ[0] == me:
            loc = prov
            if loc not in free:
                return None
            T = self.threat(prov)
            capturing = prov not in my_scs
            if not capturing and T == 0:
                return None  # safe own centre: unit is free to leave
            val = self._value(prov, fall) if capturing else \
                (1.0 + (0.3 if prov in self.homes.get(me, ()) else 0)) * (1.5 if fall else 1.0)
            sups = []
            if support_ok and T > 1:
                for s in sorted(free):
                    if s != loc and prov in idx[s]['sup_hold']:
                        sups.append(s)
                        if len(sups) >= T - 1:
                            break
            p = 1.0 if 1 + len(sups) >= T else 0.3
            new = {loc: idx[loc]['hold']}
            for s in sups:
                new[s] = idx[s]['sup_hold'][prov]
            return (val * p - 0.05 * len(new), prov, new)

        # --- target / threatened own centre not occupied by me ---
        if prov in my_scs:
            if occ is None:
                T = self.threat(prov)
                if T == 0:
                    return None
                need_strength = T          # a bounce keeps the centre ours
                val = (1.0 + (0.3 if prov in self.homes.get(me, ()) else 0)) * (1.5 if fall else 1.0)
            else:
                # enemy sitting in our centre: treat like a capture
                q = occ[0]
                H = 1 + (self.adj_count(q, prov, occ[2]) if self.is_active(q) and self.USE_THREAT else 0)
                need_strength = H + 1
                val = 1.3 * (1.5 if fall else 1.0)
        else:
            val = self._value(prov, fall)
            if occ is None:
                C = self.threat(prov)
                need_strength = C + 1
            else:
                q = occ[0]
                H = 1 + (self.adj_count(q, prov, occ[2]) if self.is_active(q) and self.USE_THREAT else 0)
                need_strength = H + 1

        movers = [l for l in sorted(free) if prov in idx[l]['move']]
        if not movers:
            return None
        best = None
        for mv in movers:
            sups = []
            if support_ok:
                for s in sorted(free):
                    if s != mv and (mv, prov) in idx[s]['sup_move']:
                        sups.append(s)
                        if 1 + len(sups) >= need_strength:
                            break
            strength = 1 + len(sups)
            if strength >= need_strength:
                p = 1.0
            elif strength == need_strength - 1 and occ is None:
                p = 0.35   # likely bounce with a contesting unit
            else:
                continue
            # pick the order variant (coast) closest to future targets
            o = idx[mv]['move'][prov][0][0]
            new = {mv: o}
            for s in sups:
                new[s] = idx[s]['sup_move'][(mv, prov)]
            sc = val * p - 0.05 * len(new)
            if best is None or sc > best[0]:
                best = (sc, prov, new)
        return best

    def _try_convoy(self, loc, full, idx, unit_of, free, orders, claimed, targets):
        best = None
        for dest, via_o in idx[loc]['via'].items():
            if dest in claimed:
                continue
            for s in sorted(free):
                if s == loc or unit_of[s][0] != 'F':
                    continue
                c = idx[s]['convoy'].get((loc, dest))
                if not c:
                    continue
                snode = unit_of[s][1]
                # single-fleet convoy only: fleet adjacent to both ends
                if loc not in self.nbr_base['F'].get(snode, ()) or dest not in self.nbr_base['F'].get(snode, ()):
                    continue
                d = 0 if dest in targets else min((self.unit_dist('A', dest, t) for t in targets), default=INF)
                if best is None or d < best[0]:
                    best = (d, via_o, s, c, dest)
        if best and best[0] < INF:
            _, via_o, s, c, dest = best
            orders[loc] = via_o
            orders[s] = c
            free.discard(loc)
            free.discard(s)
            claimed.add(dest)

    def _useful_support(self, loc, idx, orders):
        '''Idle unit: add support to one of our moves (extra margin).'''
        for mv, o in orders.items():
            w = o.split()
            if len(w) == 4 and w[2] == '-':
                so = idx[loc]['sup_move'].get((mv, base(w[3])))
                if so:
                    return so
        return None

    @staticmethod
    def _is_move(o):
        w = o.split()
        return len(w) >= 4 and w[2] == '-'

    def _fixup(self, orders, idx):
        '''Remove self-inflicted failures: moving into our own staying unit,
        two moves to one province, head-to-head swaps, orphan supports/convoys.'''
        changed = True
        guard = 0
        while changed and guard < 20:
            changed = False
            guard += 1
            moves = {l: base(o.split()[3]) for l, o in orders.items() if self._is_move(o)}
            staying = set(l for l in idx if l not in moves)
            seen = {}
            for l, d in sorted(moves.items()):
                bad = d in staying
                bad = bad or (d in seen)
                bad = bad or (moves.get(d) == l and not orders[l].endswith('VIA'))
                if bad:
                    orders[l] = idx[l]['hold']
                    changed = True
                    break
                seen[d] = l
            if changed:
                continue
            for l, o in list(orders.items()):
                w = o.split()
                if w[2] == 'S' and len(w) == 7:
                    if moves.get(base(w[4])) != base(w[6]):
                        orders[l] = idx[l]['hold']
                        changed = True
                elif w[2] == 'S' and len(w) == 5:
                    if base(w[4]) in moves:
                        orders[l] = idx[l]['hold']
                        changed = True
                elif w[2] == 'C':
                    if moves.get(base(w[4])) != base(w[6]):
                        orders[l] = idx[l]['hold']
                        changed = True
        return {l: o for l, o in orders.items() if o}

    # ---------------------------------------------------------------- retreat
    def _retreats(self):
        self._snapshot()
        me = self.power_name
        po = self.game.get_all_possible_orders()
        my_scs = set(c for c, p in self.owner.items() if p == me)
        targets = [s for s in self.scs if s not in my_scs]
        out = []
        for loc in self.game.get_orderable_locations(me):
            opts = po.get(loc, [])
            best, best_s = None, -INF
            for o in opts:
                w = o.split()
                if len(w) == 4 and w[2] == 'R':
                    t, dest = w[0], w[3]
                    d = min((self.unit_dist(t, dest, s) for s in targets), default=INF)
                    sc = (2 if base(dest) in targets else 1 if base(dest) in my_scs else 0) - 0.1 * min(d, 20)
                    if sc > best_s:
                        best, best_s = o, sc
            if best is None:
                best = next((o for o in opts if o.endswith(' D')), None)
            if best:
                out.append(best)
        return out

    # ------------------------------------------------------------- adjustment
    def _adjustments(self):
        self._snapshot()
        me = self.power_name
        po = self.game.get_all_possible_orders()
        centres = self.game.get_centers(me)
        n = len(centres) - len(self.units[me])
        my_scs = set(c.upper() for c in centres)
        targets = [s for s in self.scs if s not in my_scs]
        locs = self.game.get_orderable_locations(me)

        def near(t, node):
            return min((self.unit_dist(t, node, s) for s in targets), default=INF)

        out = []
        if n > 0:
            cands = []
            for loc in locs:
                for o in po.get(loc, []):
                    w = o.split()
                    if len(w) == 3 and w[2] == 'B':
                        cands.append((near(w[0], w[1]), 0 if w[0] == 'A' else 1, loc, o))
            cands.sort()
            used = set()
            for d, _, loc, o in cands:
                if len(out) >= n:
                    break
                if loc in used:
                    continue
                used.add(loc)
                out.append(o)
        elif n < 0:
            ranked = []
            for t, loc in self.units[me]:
                d = near(t, loc)
                on_target = base(loc) in targets
                ranked.append((on_target, -d, loc, t))
            ranked.sort()
            for on_target, negd, loc, t in ranked[:-n]:
                o = f'{t} {loc} D'
                if o in po.get(base(loc), []) or o in po.get(loc, []):
                    out.append(o)
                else:
                    alt = next((x for x in po.get(base(loc), []) if x.endswith(' D')), None)
                    if alt:
                        out.append(alt)
        return out
