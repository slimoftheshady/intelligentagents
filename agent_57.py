import random
import signal
import networkx as nx
from agent_baselines import Agent

if hasattr(signal, 'SIGALRM'):
    import timeout_decorator
    _timeout = timeout_decorator.timeout
else:
    def _timeout(seconds):
        def decorator(func):
            return func
        return decorator

DEBUG = False

def _dbg(*args):
    if DEBUG:
        print('[StudentAgent]', *args)


class StudentAgent(Agent):
    '''Scenario 1 agent: opponents are all Static Agents (always Hold).'''

    @_timeout(1)
    def __init__(self, agent_name='Scenario1Bot'):
        super().__init__(agent_name)
        self.map_graph_army = None
        self.map_graph_navy = None
        self._adjacent_turns = {}

    @_timeout(1)
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self.build_map_graphs()
        self._adjacent_turns = {}

    def build_map_graphs(self):
        self.map_graph_army = nx.Graph()
        self.map_graph_navy = nx.Graph()
        locations = list(self.game.map.loc_type.keys())
        for i in locations:
            if self.game.map.loc_type[i] in ['LAND', 'COAST']:
                self.map_graph_army.add_node(i.upper())
            if self.game.map.loc_type[i] in ['WATER', 'COAST']:
                self.map_graph_navy.add_node(i.upper())
        locations = [i.upper() for i in locations]
        for i in locations:
            for j in locations:
                if self.game.map.abuts('A', i, '-', j):
                    self.map_graph_army.add_edge(i, j)
                if self.game.map.abuts('F', i, '-', j):
                    self.map_graph_navy.add_edge(i, j)

    @_timeout(1)
    def update_game(self, all_power_orders):
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

    @_timeout(1)
    def get_actions(self):
        phase_type = self.game.phase_type
        if phase_type == 'M':
            return self._movement_orders()
        elif phase_type == 'R':
            return self._retreat_orders()
        elif phase_type == 'A':
            return self._adjustment_orders()
        return []

    # ---- case-insensitive order matching ----
    def _hold_order(self, loc, possible):
        lu = loc.upper()
        for o in possible:
            p = o.split(' ')
            if len(p) == 3 and p[1].upper() == lu and p[2].upper() == 'H':
                return o
        return None

    def _find_move(self, loc, dest, possible):
        lu, du = loc.upper(), dest.upper()
        for o in possible:
            p = o.split(' ')
            if (len(p) >= 4 and p[2] == '-' and
                    p[1].upper() == lu and p[3].upper() == du):
                return o
        return None

    def _find_support(self, loc, sup_orig, sup_dest, possible):
        lu, ou, du = loc.upper(), sup_orig.upper(), sup_dest.upper()
        for o in possible:
            p = o.split(' ')
            if len(p) >= 7 and p[2] == 'S':
                if (p[1].upper() == lu and p[4].upper() == ou
                        and p[5] == '-' and p[6].upper() == du):
                    return o
        return None

    def _any_move(self, loc, possible):
        lu = loc.upper()
        for o in possible:
            p = o.split(' ')
            if len(p) >= 4 and p[2] == '-' and p[1].upper() == lu:
                return o
        return None

    def _find_convoy(self, loc, dest, possible):
        """Try to find a legal convoy order for the army at loc to dest."""
        lu, du = loc.upper(), dest.upper()
        for o in possible:
            p = o.split(' ')
            if (len(p) >= 5 and p[0] == 'A' and p[1].upper() == lu
                    and p[2] == '-' and p[3].upper() == du
                    and p[4].upper() == 'VIA'):
                return o
        return None

    # ------------------------------------------------------------------
    # England opening (hard-coded for the first two phases)
    # ------------------------------------------------------------------
    def _england_opening(self, all_possible_orders):
        plan = {}
        current = self.game.get_current_phase()
        if 'S1901M' in current:
            plan['LON'] = 'F LON - ENG'
            plan['EDI'] = 'F EDI - NTH'
            plan['LVP'] = 'A LVP - YOR'
        elif 'F1901M' in current:
            plan['ENG'] = 'F ENG C A YOR - BEL'
            plan['NTH'] = 'F NTH - NWY'
            plan['YOR'] = 'A YOR - BEL'
        return plan

    # ------------------------------------------------------------------
    # Movement phase
    # ------------------------------------------------------------------
    def _movement_orders(self):
        all_possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(self.power_name)
        my_centers = set(c.upper() for c in self.game.get_centers(self.power_name))
        all_scs = [sc.upper() for sc in self.game.map.scs]
        targets = [sc for sc in all_scs if sc not in my_centers]
        targets_set = set(targets)

        # England opening override (only for the first two moves)
        if self.power_name == 'ENGLAND':
            plan = self._england_opening(all_possible_orders)
            if plan:
                out = []
                matched_all = True
                for loc in orderable_locations:
                    matched = False
                    for key, order in plan.items():
                        parts = order.split(' ')
                        if len(parts) >= 2 and parts[1].upper() == loc.upper():
                            if order in all_possible_orders.get(loc, []):
                                out.append(order)
                                matched = True
                                break
                    if not matched:
                        matched_all = False
                        break
                if matched_all and len(out) == len(orderable_locations):
                    return out
                # else fall through to normal logic

        # Enemy-occupied centres
        occupied_by = {}
        for p in self.game.powers.keys():
            try:
                units = self.game.get_units(power_name=p)
            except Exception:
                try:
                    units = self.game.get_units(p)
                except Exception:
                    units = []
            for u in units:
                parts = u.split(' ')
                if len(parts) >= 2:
                    occupied_by[parts[1].upper()] = p

        # Per-unit data
        unit_options = {}
        unit_paths = {}
        unit_kind = {}
        for loc in orderable_locations:
            possible = all_possible_orders.get(loc, [])
            if not possible:
                continue
            unit_options[loc] = possible
            if any(o.startswith('A') for o in possible):
                graph = self.map_graph_army
                unit_kind[loc] = 'A'
            elif any(o.startswith('F') for o in possible):
                graph = self.map_graph_navy
                unit_kind[loc] = 'F'
            else:
                unit_paths[loc] = {}
                continue
            lu = loc.upper()
            if lu in graph:
                try:
                    unit_paths[loc] = nx.shortest_path(graph, source=lu)
                except Exception:
                    unit_paths[loc] = {}
            else:
                unit_paths[loc] = {}

        def dist(loc, t):
            p = unit_paths.get(loc, {})
            return len(p[t]) if t in p else float('inf')

        orders = {}
        used = set()

        # Step 1: hold on targets we're standing on
        for loc in unit_options:
            if loc.upper() in targets_set:
                h = self._hold_order(loc, unit_options[loc])
                if h is not None:
                    orders[loc] = h
                    used.add(loc)

        # Step 2: claim UNOCCUPIED targets
        unoccupied_targets = [t for t in targets
                              if t not in occupied_by or occupied_by[t] == self.power_name]

        def nearest_free_to(t):
            best = (float('inf'), None)
            for loc in unit_options:
                if loc in used:
                    continue
                d = dist(loc, t)
                if d < best[0]:
                    best = (d, loc)
            return best

        unoccupied_targets.sort(key=lambda t: nearest_free_to(t)[0])

        for t in unoccupied_targets:
            d, loc = nearest_free_to(t)
            if loc is None or d == float('inf'):
                continue
            p = unit_paths[loc].get(t, [])
            if len(p) <= 1:
                h = self._hold_order(loc, unit_options[loc])
                orders[loc] = h if h is not None else unit_options[loc][0]
                used.add(loc)
                continue
            step = p[1]

            mv = self._find_move(loc, step, unit_options[loc])
            # Only England uses convoys; continental land paths exist.
            if mv is None and self.power_name == 'ENGLAND' and unit_kind.get(loc) == 'A':
                cv = self._find_convoy(loc, t, unit_options[loc])
                if cv is not None:
                    orders[loc] = cv
                    used.add(loc)
                    continue
            if mv is None:
                mv = self._any_move(loc, unit_options[loc])
            if mv is not None:
                orders[loc] = mv
                used.add(loc)

        # Step 3: dislodge OCCUPIED targets
        occupied_targets = [t for t in targets
                            if t in occupied_by and occupied_by[t] != self.power_name]
        occupied_targets.sort(key=lambda t: nearest_free_to(t)[0])

        new_adjacent = {}

        for t in occupied_targets:
            candidates = []
            for loc in unit_options:
                if loc in used:
                    continue
                d = dist(loc, t)
                if d != float('inf'):
                    candidates.append((d, loc))
            candidates.sort()
            if not candidates:
                continue

            adjacent = [c for c in candidates if c[0] == 2]
            approaching = [c for c in candidates if c[0] > 2]

            if len(adjacent) >= 2:
                mover = adjacent[0][1]
                supporter = adjacent[1][1]
                sup = self._find_support(supporter, mover, t, unit_options[supporter])
                mv = self._find_move(mover, t, unit_options[mover])
                if sup is not None and mv is not None:
                    orders[mover] = mv
                    orders[supporter] = sup
                    used.add(mover)
                    used.add(supporter)
                    continue

            if len(adjacent) == 1 and len(approaching) >= 1:
                adj_unit = adjacent[0][1]
                stuck = self._adjacent_turns.get(adj_unit, 0)
                if stuck >= 2:
                    if not self._send_to_nearest_unoccupied(
                            adj_unit, unoccupied_targets, unit_options,
                            unit_paths, orders, used):
                        h = self._hold_order(adj_unit, unit_options[adj_unit])
                        if h is not None:
                            orders[adj_unit] = h
                            used.add(adj_unit)
                else:
                    h = self._hold_order(adj_unit, unit_options[adj_unit])
                    if h is not None:
                        orders[adj_unit] = h
                        used.add(adj_unit)
                    new_adjacent[adj_unit] = stuck + 1
                appr_unit = approaching[0][1]
                p = unit_paths[appr_unit].get(t, [])
                if len(p) > 1:
                    mv = self._find_move(appr_unit, p[1], unit_options[appr_unit]) or self._any_move(appr_unit, unit_options[appr_unit])
                    if mv is not None:
                        orders[appr_unit] = mv
                        used.add(appr_unit)
                continue

            if len(adjacent) == 1 and not approaching:
                adj_unit = adjacent[0][1]
                stuck = self._adjacent_turns.get(adj_unit, 0)
                if stuck >= 2:
                    if not self._send_to_nearest_unoccupied(
                            adj_unit, unoccupied_targets, unit_options,
                            unit_paths, orders, used):
                        h = self._hold_order(adj_unit, unit_options[adj_unit])
                        if h is not None:
                            orders[adj_unit] = h
                            used.add(adj_unit)
                else:
                    h = self._hold_order(adj_unit, unit_options[adj_unit])
                    if h is not None:
                        orders[adj_unit] = h
                        used.add(adj_unit)
                    new_adjacent[adj_unit] = stuck + 1
                continue

            if len(approaching) >= 2:
                for _, loc in approaching[:2]:
                    p = unit_paths[loc].get(t, [])
                    if len(p) > 1:
                        mv = self._find_move(loc, p[1], unit_options[loc]) or self._any_move(loc, unit_options[loc])
                        if mv is not None:
                            orders[loc] = mv
                            used.add(loc)
                continue

            if len(approaching) == 1:
                loc = approaching[0][1]
                p = unit_paths[loc].get(t, [])
                if len(p) > 1:
                    mv = self._find_move(loc, p[1], unit_options[loc]) or self._any_move(loc, unit_options[loc])
                    if mv is not None:
                        orders[loc] = mv
                        used.add(loc)

        # Step 4: remaining
        for loc in unit_options:
            if loc in used:
                continue
            best_d, best_t = float('inf'), None
            for t in targets:
                d = dist(loc, t)
                if d < best_d:
                    best_d, best_t = d, t
            if best_t is None or best_d == float('inf') or best_d <= 1:
                h = self._hold_order(loc, unit_options[loc])
                orders[loc] = h if h is not None else unit_options[loc][0]
            else:
                p = unit_paths[loc].get(best_t, [])
                if len(p) > 1:
                    mv = self._find_move(loc, p[1], unit_options[loc])
                    if mv is None and self.power_name == 'ENGLAND' and unit_kind.get(loc) == 'A':
                        cv = self._find_convoy(loc, best_t, unit_options[loc])
                        if cv is not None:
                            orders[loc] = cv
                            used.add(loc)
                            continue
                    mv = mv or self._any_move(loc, unit_options[loc])
                    orders[loc] = mv if mv is not None else (self._hold_order(loc, unit_options[loc]) or unit_options[loc][0])
                else:
                    h = self._hold_order(loc, unit_options[loc])
                    orders[loc] = h if h is not None else unit_options[loc][0]
            used.add(loc)

        self._adjacent_turns = new_adjacent
        return [orders[loc] for loc in orderable_locations if loc in orders]

    def _send_to_nearest_unoccupied(self, loc, unoccupied_targets, unit_options,
                                     unit_paths, orders, used):
        best_d, best_t = float('inf'), None
        loc_paths = unit_paths.get(loc, {})
        for t in unoccupied_targets:
            claimed = False
            for other_loc, o in orders.items():
                parts = o.split(' ')
                if len(parts) >= 4 and parts[2] == '-':
                    if parts[3].upper() == t:
                        claimed = True
                        break
            if claimed:
                continue
            if t not in loc_paths:
                continue
            d = len(loc_paths[t])
            if d < best_d:
                best_d, best_t = d, t
        if best_t is None or best_d == float('inf') or best_d <= 1:
            return False
        p = loc_paths.get(best_t, [])
        if len(p) <= 1:
            return False
        mv = self._find_move(loc, p[1], unit_options[loc]) or self._any_move(loc, unit_options[loc])
        if mv is not None:
            orders[loc] = mv
            used.add(loc)
            return True
        return False

    # ------------------------------------------------------------------
    def _retreat_orders(self):
        all_possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(self.power_name)
        out = []
        for loc in orderable_locations:
            possible = all_possible_orders.get(loc, [])
            if not possible:
                continue
            r = [o for o in possible if ' R ' in o]
            d = [o for o in possible if o.endswith(' D')]
            if r:
                out.append(random.choice(r))
            elif d:
                out.append(d[0])
            else:
                out.append(possible[0])
        return out

    # ------------------------------------------------------------------
    def _adjustment_orders(self):
        all_possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(self.power_name)
        out = []
        prefer_fleet = (self.power_name == 'ENGLAND')
        for loc in orderable_locations:
            possible = all_possible_orders.get(loc, [])
            if not possible:
                continue
            b = [o for o in possible if o.endswith(' B')]
            if b:
                fleet_builds = [o for o in b if o.startswith('F')]
                army_builds = [o for o in b if o.startswith('A')]
                if prefer_fleet and fleet_builds:
                    out.append(fleet_builds[0])
                elif army_builds:
                    out.append(army_builds[0])
                else:
                    out.append(b[0])
            else:
                d = [o for o in possible if o.endswith(' D')]
                out.append(d[0] if d else possible[0])
        return out