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


class StudentAgent(Agent):
    '''
    Scenario 1 agent: opponents are all Static Agents (always Hold).

    Since Static Agents never move and never support, the map is essentially
    a race to capture all 34 supply centres. Two mechanisms are needed:

    1. Capture neutral (unowned) centres by moving onto them.
    2. Dislodge Static units sitting on their home centres using SUPPORTED
       ATTACKS - a supported attack (strength 2) beats an unsupported
       defence (strength 1). A bare move into an occupied centre bounces.

    Strategy per phase:
      - Movement:
          * Build a target list: every supply centre we don't own.
          * Decide per unit what action helps most:
              (a) If standing on a target centre -> HOLD (capture).
              (b) Else, move toward the nearest target that no other unit
                  of ours is already assigned to "move into" this turn,
                  UNLESS that target is occupied by an enemy unit and we
                  already have one mover assigned -> then SUPPORT the mover.
          * Concretely: for each target, pick the 1 or 2 nearest of our
            units. If the target is occupied, use one as mover and one as
            supporter. If empty, just one mover.
      - Adjustment: build at every home centre where legal (prefer armies).
      - Retreat: safety net (should never trigger).
    '''

    @_timeout(1)
    def __init__(self, agent_name='Scenario1Bot'):
        super().__init__(agent_name)
        self.map_graph_army = None
        self.map_graph_navy = None

    @_timeout(1)
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self.build_map_graphs()

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
        else:
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _hold_order(self, loc, possible):
        if not possible:
            return None
        unit_type = possible[0][0]
        hold = f'{unit_type} {loc} H'
        if hold in possible:
            return hold
        for o in possible:
            parts = o.split(' ')
            if len(parts) == 3 and parts[0] == unit_type and parts[1] == loc and parts[2] == 'H':
                return o
        return None

    def _find_move(self, loc, dest, possible):
        """Return a valid move order from loc to dest, if one exists in possible."""
        if not possible:
            return None
        unit_type = possible[0][0]
        candidate = f'{unit_type} {loc} - {dest}'
        if candidate in possible:
            return candidate
        for o in possible:
            parts = o.split(' ')
            if (len(parts) >= 4 and parts[0] == unit_type
                    and parts[1] == loc and parts[2] == '-'
                    and parts[3].upper() == dest.upper()):
                return o
        return None

    def _find_support(self, loc, supported_unit_loc, supported_unit_dest, possible):
        """
        Return a valid support order for the unit at loc supporting the move
        of a friendly unit from supported_unit_loc to supported_unit_dest.
        """
        if not possible:
            return None
        unit_type = possible[0][0]
        supported_type = 'A'  # opponent units on SCs are armies or fleets; try both
        # Try army-supported and fleet-supported forms
        for st in ('A', 'F'):
            candidate = f'{unit_type} {loc} S {st} {supported_unit_loc} - {supported_unit_dest}'
            if candidate in possible:
                return candidate
        # Fuzzy match: unit_type, loc, 'S', then 'X ORIGIN - DEST'
        target_origin = supported_unit_loc.upper()
        target_dest = supported_unit_dest.upper()
        for o in possible:
            parts = o.split(' ')
            if len(parts) >= 7 and parts[0] == unit_type and parts[1] == loc and parts[2] == 'S':
                if parts[4].upper() == target_origin and parts[5] == '-' and parts[6].upper() == target_dest:
                    return o
        return None

    def _find_support_hold(self, loc, supported_unit_loc, possible):
        """Support a friendly unit that is HOLDING at supported_unit_loc."""
        if not possible:
            return None
        unit_type = possible[0][0]
        for st in ('A', 'F'):
            candidate = f'{unit_type} {loc} S {st} {supported_unit_loc} H'
            if candidate in possible:
                return candidate
        target_origin = supported_unit_loc.upper()
        for o in possible:
            parts = o.split(' ')
            if len(parts) >= 6 and parts[0] == unit_type and parts[1] == loc and parts[2] == 'S':
                if parts[4].upper() == target_origin and parts[5] == 'H':
                    return o
        return None

    # ------------------------------------------------------------------
    # Movement phase
    # ------------------------------------------------------------------
    def _movement_orders(self):
        all_possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(self.power_name)
        my_centers = set(c.upper() for c in self.game.get_centers(self.power_name))

        # All supply centres not currently owned by us
        all_scs = [sc.upper() for sc in self.game.map.scs]
        targets = [sc for sc in all_scs if sc not in my_centers]
        target_set = set(targets)

        # Current unit locations of ALL powers (to detect occupied centres)
        occupied_by = {}  # loc.upper() -> power_name
        for p in self.game.powers.keys():
            for u in self.game.get_units(power_name=p):
                # u looks like 'A PAR' or 'F LON'
                parts = u.split(' ')
                if len(parts) >= 2:
                    occupied_by[parts[1].upper()] = p

        # Build per-unit information
        unit_data = {}  # loc -> dict(type, graph, possible, army_opts, navy_opts)
        for loc in orderable_locations:
            possible = all_possible_orders.get(loc, [])
            if not possible:
                continue
            army_opts = [o for o in possible if o.startswith('A')]
            navy_opts = [o for o in possible if o.startswith('F')]
            if army_opts:
                graph = self.map_graph_army
                utype = 'A'
            elif navy_opts:
                graph = self.map_graph_navy
                utype = 'F'
            else:
                continue
            unit_data[loc] = {'type': utype, 'graph': graph, 'possible': possible}

        # Cache shortest paths per unit (from its origin)
        paths_cache = {}
        for loc, data in unit_data.items():
            graph = data['graph']
            lu = loc.upper()
            if lu not in graph:
                paths_cache[loc] = {}
                continue
            try:
                paths_cache[loc] = nx.shortest_path(graph, source=lu)
            except Exception:
                paths_cache[loc] = {}

        # ------------------------------------------------------------------
        # Decide assignments
        # ------------------------------------------------------------------
        # order_map: loc -> order string
        order_map = {}
        # Track which unit is assigned to move into which target
        mover_of_target = {}   # target -> loc (mover unit)
        supporter_of_target = {}  # target -> loc (supporter unit)
        assigned = set()  # locations of units that already have an order

        # Step 1: any unit standing on a target centre -> HOLD (capture)
        for loc in unit_data:
            if loc.upper() in target_set:
                hold = self._hold_order(loc, unit_data[loc]['possible'])
                if hold is not None:
                    order_map[loc] = hold
                    assigned.add(loc)
                    mover_of_target[loc.upper()] = loc

        # Step 2: assign movers/supporters for each remaining target.
        # Sort targets by distance from the nearest available unit so the
        # closest opportunities are claimed first.
        def nearest_available_dist(target):
            best = (float('inf'), None)
            for loc, data in unit_data.items():
                if loc in assigned:
                    continue
                paths = paths_cache.get(loc, {})
                if target in paths:
                    d = len(paths[target])
                    if d < best[0]:
                        best = (d, loc)
            return best

        remaining_targets = [t for t in targets if t not in mover_of_target]

        # Iteratively assign: pick target with globally shortest available unit
        while True:
            best = (float('inf'), None, None)  # (dist, target, unit_loc)
            for t in remaining_targets:
                if t in mover_of_target:
                    continue
                d, loc = nearest_available_dist(t)
                if loc is None:
                    continue
                if d < best[0]:
                    best = (d, t, loc)
            d, target, mover_loc = best
            if mover_loc is None:
                break

            # We have a mover for `target`
            mover_of_target[target] = mover_loc
            assigned.add(mover_loc)

            # Is the target occupied by an enemy (Static) unit?
            occupied_enemy = (target in occupied_by and
                              occupied_by[target] != self.power_name)

            if occupied_enemy:
                # Find a second unit adjacent to `target` that is free and
                # can support. Prefer a unit whose shortest path to target
                # is length 2 (i.e., can reach target in one step by staying
                # adjacent and supporting), but any unit adjacent to target
                # with a valid support order works.
                supporter_loc = None
                # We need a unit that can give "S X mover_loc - target",
                # i.e. a unit already adjacent to target.
                for loc, data in unit_data.items():
                    if loc in assigned:
                        continue
                    if loc == mover_loc:
                        continue
                    # Must be adjacent to target
                    if target not in paths_cache.get(loc, {}):
                        continue
                    if len(paths_cache[loc][target]) != 2:
                        continue
                    # Try to construct a support order
                    sup = self._find_support(loc, mover_loc, target, data['possible'])
                    if sup is not None:
                        supporter_loc = loc
                        order_map[loc] = sup
                        assigned.add(loc)
                        supporter_of_target[target] = loc
                        break

                # If no adjacent supporter is available, we need the mover to
                # arrive adjacent this turn. That's fine - we'll support next
                # turn. For now the mover just moves toward target and we
                # defer the actual capture.
                if supporter_loc is None:
                    pass  # mover moves toward target as usual

            # Remove this target from consideration
            remaining_targets = [t for t in remaining_targets if t != target]

        # Step 3: every still-unassigned unit moves toward the nearest
        # unassigned target (its final destination may be one that already
        # has a mover - that's fine, we just need the unit to advance).
        for loc, data in unit_data.items():
            if loc in assigned:
                continue
            possible = data['possible']
            paths = paths_cache.get(loc, {})

            # Prefer any target with no mover yet
            best_target = None
            best_dist = float('inf')
            for t in remaining_targets:
                if t in paths and len(paths[t]) < best_dist:
                    best_dist = len(paths[t])
                    best_target = t

            # If no unclaimed target is reachable, advance toward the
            # nearest target overall (to set up future support/moves)
            if best_target is None:
                for t in targets:
                    if t in paths and len(paths[t]) < best_dist:
                        best_dist = len(paths[t])
                        best_target = t

            if best_target is None or best_dist <= 1:
                hold = self._hold_order(loc, possible)
                if hold is not None:
                    order_map[loc] = hold
                    assigned.add(loc)
                    continue
                order_map[loc] = random.choice(possible)
                assigned.add(loc)
                continue

            step = paths[best_target][1]
            move = self._find_move(loc, step, possible)
            if move is not None:
                order_map[loc] = move
            else:
                hold = self._hold_order(loc, possible)
                order_map[loc] = hold if hold is not None else random.choice(possible)
            assigned.add(loc)

        # Step 4: convert to list
        power_orders = []
        for loc in orderable_locations:
            if loc in order_map:
                power_orders.append(order_map[loc])
            else:
                possible = all_possible_orders.get(loc, [])
                if possible:
                    hold = self._hold_order(loc, possible)
                    power_orders.append(hold if hold is not None else random.choice(possible))

        return power_orders

    # ------------------------------------------------------------------
    # Retreat phase (safety net)
    # ------------------------------------------------------------------
    def _retreat_orders(self):
        all_possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(self.power_name)
        power_orders = []
        for loc in orderable_locations:
            possible = all_possible_orders.get(loc, [])
            if not possible:
                continue
            retreat_opts = [o for o in possible if ' R ' in o]
            disband_opts = [o for o in possible if o.endswith(' D')]
            if retreat_opts:
                power_orders.append(random.choice(retreat_opts))
            elif disband_opts:
                power_orders.append(disband_opts[0])
            else:
                power_orders.append(random.choice(possible))
        return power_orders

    # ------------------------------------------------------------------
    # Adjustment phase
    # ------------------------------------------------------------------
    def _adjustment_orders(self):
        all_possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(self.power_name)
        power_orders = []

        for loc in orderable_locations:
            possible = all_possible_orders.get(loc, [])
            if not possible:
                continue

            build_opts = [o for o in possible if o.endswith(' B')]
            if build_opts:
                army_builds = [o for o in build_opts if o.startswith('A')]
                if army_builds:
                    power_orders.append(army_builds[0])
                else:
                    power_orders.append(build_opts[0])
            else:
                disband_opts = [o for o in possible if o.endswith(' D')]
                if disband_opts:
                    power_orders.append(disband_opts[0])
                else:
                    power_orders.append(random.choice(possible))

        return power_orders