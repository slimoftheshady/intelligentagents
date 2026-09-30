import signal
import networkx as nx
from collections import defaultdict
from agent_baselines import Agent

if hasattr(signal, 'SIGALRM'):
    import timeout_decorator
    _timeout = timeout_decorator.timeout
else:
    def _timeout(seconds):
        def decorator(function): return function
        return decorator

DEBUG = False

def _dbg(*args):
    if DEBUG: print('[StudentAgent]', *args)


class StudentAgent(Agent):
    """
    Scenario 1/2 Student Agent.
    Stage 1: existing strategy with deterministic decisions.
    """

    @_timeout(1)
    def __init__(self, agent_name='Scenario1Bot'):
        super().__init__(agent_name)
        self.game = None
        self.power_name = None
        self.map_graph_army = None
        self.map_graph_navy = None
        self.army_paths = {}
        self.navy_paths = {}
        self._adjacent_turns = {}

    @_timeout(1)
    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self.build_map_graphs()
        self._adjacent_turns = {}

    @_timeout(1)
    def update_game(self, all_power_orders):
        for power_name in sorted(all_power_orders):
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

    @_timeout(1)
    def get_actions(self):
        if self.game.phase_type == 'M': return self._movement_orders()
        if self.game.phase_type == 'R': return self._retreat_orders()
        if self.game.phase_type == 'A': return self._adjustment_orders()
        return []

    def build_map_graphs(self):
        self.map_graph_army = nx.Graph()
        self.map_graph_navy = nx.Graph()
        locations = sorted(self.game.map.loc_type.keys(), key=lambda x: x.upper())

        for raw_loc in locations:
            loc, loc_type = raw_loc.upper(), self.game.map.loc_type[raw_loc]
            if loc_type in ('LAND', 'COAST'): self.map_graph_army.add_node(loc)
            if loc_type in ('WATER', 'COAST'): self.map_graph_navy.add_node(loc)

        for raw_a in locations:
            a = raw_a.upper()
            for raw_b in locations:
                b = raw_b.upper()
                if self.game.map.abuts('A', a, '-', b): self.map_graph_army.add_edge(a, b)
                if self.game.map.abuts('F', a, '-', b): self.map_graph_navy.add_edge(a, b)

        self.army_paths = dict(nx.all_pairs_shortest_path(self.map_graph_army))
        self.navy_paths = dict(nx.all_pairs_shortest_path(self.map_graph_navy))

    @staticmethod
    def _province(loc): return loc.split('/')[0].upper()

    @staticmethod
    def _exact(loc): return loc.upper()

    def _hold_order(self, loc, possible):
        source = self._exact(loc)
        for order in sorted(possible):
            parts = order.split()
            if len(parts) != 3 or parts[1].upper() != source or parts[2].upper() != 'H': continue
            return order

        province = self._province(loc)
        for order in sorted(possible):
            parts = order.split()
            if len(parts) == 3 and self._province(parts[1]) == province and parts[2].upper() == 'H': return order
        return None

    def _find_move(self, loc, destination, possible):
        source, destination = self._exact(loc), self._exact(destination)

        for order in sorted(possible):
            parts = order.split()
            if len(parts) != 4 or parts[2] != '-' or parts[1].upper() != source or parts[3].upper() != destination: continue
            return order

        source_province, destination_province = self._province(loc), self._province(destination)
        for order in sorted(possible):
            parts = order.split()
            if len(parts) == 4 and parts[2] == '-' and self._province(parts[1]) == source_province and self._province(parts[3]) == destination_province:
                return order
        return None

    def _find_support(self, loc, supported_origin, supported_destination, possible):
        source = self._province(loc)
        origin = self._province(supported_origin)
        destination = self._province(supported_destination)

        for order in sorted(possible):
            parts = order.split()
            if len(parts) < 7 or parts[2].upper() != 'S' or self._province(parts[1]) != source or self._province(parts[4]) != origin or parts[5] != '-' or self._province(parts[6]) != destination:
                continue
            return order
        return None

    def _any_move(self, loc, possible):
        source = self._province(loc)
        for order in sorted(possible):
            parts = order.split()
            if len(parts) == 4 and parts[2] == '-' and self._province(parts[1]) == source: return order
        return None

    def _find_convoy(self, loc, destination, possible):
        source, target = self._province(loc), self._province(destination)
        for order in sorted(possible):
            parts = order.split()
            if len(parts) < 5 or parts[0].upper() != 'A' or self._province(parts[1]) != source or parts[2] != '-' or self._province(parts[3]) != target or parts[4].upper() != 'VIA':
                continue
            return order
        return None

    def _unit_type(self, possible):
        for order in sorted(possible):
            if order.startswith('A '): return 'A'
        for order in sorted(possible):
            if order.startswith('F '): return 'F'
        return None

    def _unit_locations(self):
        return sorted(self.game.get_orderable_locations(self.power_name))

    def _unit_paths(self, loc, unit_type):
        source = self._exact(loc)
        if unit_type == 'A': return self.army_paths.get(source, {})
        if unit_type == 'F': return self.navy_paths.get(source, {})
        return {}

    def _distance(self, loc, destination, unit_type):
        paths = self._unit_paths(loc, unit_type)
        destination = self._exact(destination)
        path = paths.get(destination)
        if path is not None: return len(path) - 1

        target_province = self._province(destination)
        best = float('inf')
        for node in sorted(paths):
            path = paths[node]
            if self._province(node) == target_province: best = min(best, len(path) - 1)
        return best

    def _my_centres(self):
        return {self._province(x) for x in self.game.get_centers(self.power_name)}

    def _all_supply_centres(self):
        return {self._province(x) for x in self.game.map.scs}

    def _enemy_occupied(self):
        occupied = {}
        for power_name in sorted(self.game.powers.keys()):
            if power_name == self.power_name: continue
            try:
                units = self.game.get_units(power_name=power_name)
            except Exception:
                try: units = self.game.get_units(power_name)
                except Exception: units = []

            for unit in sorted(units):
                parts = unit.split()
                if len(parts) < 2: continue
                occupied[self._province(parts[1])] = power_name
        return occupied

    def _known_opening(self):
        phase = self.game.get_current_phase()
        plans = {
            'ENGLAND': {
                'S1901M': {'LON': 'F LON - ENG', 'EDI': 'F EDI - NTH', 'LVP': 'A LVP - YOR'},
                'F1901M': {'ENG': 'F ENG C A YOR - BEL', 'NTH': 'F NTH - NWY', 'YOR': 'A YOR - BEL'}
            },
            'FRANCE': {
                'S1901M': {'PAR': 'A PAR - BUR', 'MAR': 'A MAR - SPA', 'BRE': 'F BRE - MAO'},
                'F1901M': {'BUR': 'A BUR - BEL', 'SPA': 'A SPA - POR'}
            },
            'ITALY': {
                'S1901M': {'VEN': 'A VEN - TYR', 'ROM': 'A ROM - TUS', 'NAP': 'F NAP - ION'},
                'F1901M': {'TYR': 'A TYR - VIE', 'TUS': 'A TUS - PIE', 'ION': 'F ION - TUN'}
            }
        }
        return plans.get(self.power_name, {}).get(phase, {})

    def _try_opening(self, possible_orders, orderable_locations):
        plan = self._known_opening()
        if not plan: return None
        result = []

        for loc in sorted(orderable_locations):
            source = self._province(loc)
            if source not in plan: return None
            desired = plan[source]
            possible = possible_orders.get(loc, [])
            if desired not in possible: return None
            result.append(desired)

        if len(result) != len(orderable_locations): return None
        return result

    def _reachable_units(self, target, unit_options, unit_types):
        result = []
        for loc in sorted(unit_options):
            distance = self._distance(loc, target, unit_types[loc])
            if distance != float('inf'): result.append((distance, loc))
        result.sort()
        return result

    def _target_score(self, target, unit_options, unit_types, enemy_occupied, my_centres):
        target = self._province(target)
        if target in my_centres: return -100000

        reachable = self._reachable_units(target, unit_options, unit_types)
        if not reachable: return float('-inf')

        nearest = reachable[0][0]
        adjacent = sum(1 for distance, _ in reachable if distance == 1)
        within_two = sum(1 for distance, _ in reachable if distance <= 2)
        occupied = target in enemy_occupied
        score = 0.0

        if nearest == 1: score += 100
        elif nearest == 2: score += 70
        elif nearest == 3: score += 40
        elif nearest == 4: score += 20
        else: score += max(0, 10 - nearest)

        score += adjacent * 35
        score += within_two * 10

        if occupied:
            score += 25
            if adjacent >= 2: score += 60
        else:
            score += 30
        return score

    def _movement_orders(self):
        possible_orders = self.game.get_all_possible_orders()
        orderable_locations = sorted(self.game.get_orderable_locations(self.power_name))
        my_centres = self._my_centres()
        all_scs = self._all_supply_centres()
        targets = sorted(x for x in all_scs if x not in my_centres)
        enemy_occupied = self._enemy_occupied()

        opening = self._try_opening(possible_orders, orderable_locations)
        if opening is not None: return opening

        unit_options, unit_types = {}, {}
        for loc in orderable_locations:
            possible = sorted(possible_orders.get(loc, []))
            if not possible: continue
            unit_type = self._unit_type(possible)
            if unit_type is None: continue
            unit_options[loc], unit_types[loc] = possible, unit_type

        target_scores = {target: self._target_score(target, unit_options, unit_types, enemy_occupied, my_centres) for target in targets}
        orders, used = {}, set()

        for loc in sorted(unit_options):
            province = self._province(loc)
            if province not in targets: continue
            hold = self._hold_order(loc, unit_options[loc])
            if hold is not None:
                orders[loc] = hold
                used.add(loc)

        occupied_targets = [target for target in targets if target in enemy_occupied]
        occupied_targets.sort(key=lambda target: (-target_scores.get(target, float('-inf')), target))
        new_adjacent = {}

        for target in occupied_targets:
            adjacent = []
            for loc in sorted(unit_options):
                if loc in used: continue
                distance = self._distance(loc, target, unit_types[loc])
                if distance == 1: adjacent.append(loc)

            if len(adjacent) >= 2:
                adjacent.sort()
                mover, supporter = adjacent[0], adjacent[1]
                move = self._find_move(mover, target, unit_options[mover])
                support = self._find_support(supporter, mover, target, unit_options[supporter])
                if move is not None and support is not None:
                    orders[mover], orders[supporter] = move, support
                    used.add(mover)
                    used.add(supporter)
                    continue

            if len(adjacent) == 1:
                loc = adjacent[0]
                stuck = self._adjacent_turns.get(loc, 0)
                if stuck < 2:
                    hold = self._hold_order(loc, unit_options[loc])
                    if hold is not None:
                        orders[loc] = hold
                        used.add(loc)
                    new_adjacent[loc] = stuck + 1
                else:
                    self._move_toward_best_target(loc, targets, unit_options, unit_types, target_scores, orders, used)

        unoccupied_targets = [target for target in targets if target not in enemy_occupied]
        unoccupied_targets.sort(key=lambda target: (-target_scores.get(target, float('-inf')), target))
        claimed = set()

        for target in unoccupied_targets:
            candidates = []
            for loc in sorted(unit_options):
                if loc in used: continue
                distance = self._distance(loc, target, unit_types[loc])
                if distance != float('inf'): candidates.append((distance, loc))
            candidates.sort()
            if not candidates: continue
            _, loc = candidates[0]
            if self._move_toward_target(loc, target, unit_options, unit_types, orders, used): claimed.add(target)

        for loc in sorted(unit_options):
            if loc in used: continue

            best_target, best_score = None, float('-inf')
            for target in sorted(targets):
                distance = self._distance(loc, target, unit_types[loc])
                if distance == float('inf'): continue
                score = target_scores.get(target, float('-inf')) - distance * 8
                if target in enemy_occupied: score += 15

                if score > best_score or (score == best_score and (best_target is None or target < best_target)):
                    best_score, best_target = score, target

            if best_target is not None and self._move_toward_target(loc, best_target, unit_options, unit_types, orders, used):
                continue

            hold = self._hold_order(loc, unit_options[loc])
            orders[loc] = hold if hold is not None else sorted(unit_options[loc])[0]
            used.add(loc)

        self._free_home_centres(orders, used, unit_options, unit_types, targets, my_centres)
        self._remove_self_bounces(orders, orderable_locations, unit_options)
        self._adjacent_turns = new_adjacent

        result = []
        for loc in orderable_locations:
            if loc in orders:
                result.append(orders[loc])
                continue
            possible = unit_options.get(loc, possible_orders.get(loc, []))
            if not possible: continue
            hold = self._hold_order(loc, possible)
            result.append(hold if hold is not None else sorted(possible)[0])
        return result

    def _move_toward_target(self, loc, target, unit_options, unit_types, orders, used):
        if loc in used: return False
        unit_type = unit_types[loc]
        distance = self._distance(loc, target, unit_type)
        if distance <= 0 or distance == float('inf'): return False

        paths = self._unit_paths(loc, unit_type)
        target_exact = self._exact(target)
        path = paths.get(target_exact)

        if path is None:
            target_province = self._province(target)
            candidates = [candidate_path for node in sorted(paths) for candidate_path in [paths[node]] if self._province(node) == target_province]
            if not candidates: return False
            candidates.sort(key=lambda p: (len(p), tuple(p)))
            path = candidates[0]

        if len(path) < 2: return False
        next_location = path[1]
        move = self._find_move(loc, next_location, unit_options[loc])

        if move is None and self.power_name == 'ENGLAND' and unit_type == 'A':
            move = self._find_convoy(loc, target, unit_options[loc])

        if move is None: move = self._any_move(loc, unit_options[loc])
        if move is None: return False

        orders[loc] = move
        used.add(loc)
        return True

    def _move_toward_best_target(self, loc, targets, unit_options, unit_types, target_scores, orders, used):
        best_target, best_score = None, float('-inf')

        for target in sorted(targets):
            distance = self._distance(loc, target, unit_types[loc])
            if distance == float('inf'): continue
            score = target_scores.get(target, float('-inf')) - distance * 5

            if score > best_score or (score == best_score and (best_target is None or target < best_target)):
                best_score, best_target = score, target

        if best_target is None: return False
        return self._move_toward_target(loc, best_target, unit_options, unit_types, orders, used)

    def _free_home_centres(self, orders, used, unit_options, unit_types, targets, my_centres):
        home_centres = {self._province(x) for x in self.game.map.homes.get(self.power_name, [])}
        unit_count = len(unit_options)
        surplus = len(my_centres) - unit_count
        if surplus <= 0: return

        for loc in sorted(unit_options):
            if surplus <= 0: break
            if self._province(loc) not in home_centres: continue

            order = orders.get(loc)
            if order is None or not order.endswith(' H'): continue

            best_target, best_distance = None, float('inf')
            for target in sorted(targets):
                distance = self._distance(loc, target, unit_types[loc])
                if distance < best_distance or (distance == best_distance and (best_target is None or target < best_target)):
                    best_distance, best_target = distance, target

            if best_target is None or best_distance == float('inf'): continue
            if self._move_toward_target(loc, best_target, unit_options, unit_types, orders, used):
                surplus -= 1

    def _remove_self_bounces(self, orders, orderable_locations, unit_options):
        own_locations = {self._province(loc): loc for loc in sorted(orderable_locations)}
        changed = True

        while changed:
            changed = False

            for loc in sorted(list(orders)):
                order = orders[loc]
                parts = order.split()
                if len(parts) != 4 or parts[2] != '-': continue

                destination = self._province(parts[3])
                other_loc = own_locations.get(destination)
                if other_loc is None or other_loc == loc: continue

                other_order = orders.get(other_loc)
                if other_order is not None and other_order.endswith(' H'):
                    hold = self._hold_order(loc, unit_options[loc])
                    if hold is not None:
                        orders[loc] = hold
                        changed = True

            destination_movers = defaultdict(list)
            for loc in sorted(orders):
                parts = orders[loc].split()
                if len(parts) == 4 and parts[2] == '-':
                    destination_movers[self._province(parts[3])].append(loc)

            for destination in sorted(destination_movers):
                movers = sorted(destination_movers[destination])
                if len(movers) <= 1: continue

                for loc in movers[1:]:
                    hold = self._hold_order(loc, unit_options[loc])
                    if hold is not None:
                        orders[loc] = hold
                        changed = True

    def _retreat_orders(self):
        possible_orders = self.game.get_all_possible_orders()
        orderable_locations = sorted(self.game.get_orderable_locations(self.power_name))
        my_centres = self._my_centres()
        targets = sorted(x for x in self._all_supply_centres() if x not in my_centres)
        result = []

        for loc in orderable_locations:
            possible = sorted(possible_orders.get(loc, []))
            if not possible: continue

            retreats = sorted(x for x in possible if ' R ' in x)
            disbands = sorted(x for x in possible if x.endswith(' D'))

            if retreats:
                best_order, best_distance = None, float('inf')
                unit_type = 'F' if retreats[0].startswith('F') else 'A'

                for order in retreats:
                    parts = order.split()
                    if len(parts) < 4: continue
                    destination = parts[3]

                    for target in targets:
                        distance = self._distance(destination, target, unit_type)
                        if distance < best_distance or (distance == best_distance and (best_order is None or order < best_order)):
                            best_distance, best_order = distance, order

                result.append(best_order if best_order is not None else retreats[0])
            elif disbands:
                result.append(disbands[0])
            else:
                result.append(possible[0])

        return result

    def _adjustment_orders(self):
        possible_orders = self.game.get_all_possible_orders()
        orderable_locations = sorted(self.game.get_orderable_locations(self.power_name))
        my_centres = self._my_centres()
        targets = sorted(x for x in self._all_supply_centres() if x not in my_centres)
        result = []

        for loc in orderable_locations:
            possible = sorted(possible_orders.get(loc, []))
            if not possible: continue

            builds = sorted(x for x in possible if x.endswith(' B'))
            if builds:
                result.append(self._choose_build(builds, targets))
                continue

            disbands = sorted(x for x in possible if x.endswith(' D'))
            if disbands:
                result.append(self._choose_disband(disbands, targets))
            else:
                result.append(possible[0])

        return result

    def _choose_build(self, builds, targets):
        best_order, best_score = sorted(builds)[0], float('-inf')

        for order in sorted(builds):
            parts = order.split()
            if len(parts) < 2: continue

            unit_type, location = parts[0], parts[1]
            score, distances = 0.0, []

            for target in sorted(targets):
                distance = self._distance(location, target, unit_type)
                if distance != float('inf'): distances.append(distance)

            if distances:
                distances.sort()
                score += 100.0 / (1 + distances[0])
                score += len(distances) * 5
                score += sum(15 for d in distances if d <= 3)

            if score > best_score or (score == best_score and order < best_order):
                best_score, best_order = score, order

        return best_order

    def _choose_disband(self, disbands, targets):
        disbands = sorted(disbands)
        if len(disbands) == 1: return disbands[0]

        worst_order, worst_score = disbands[0], float('inf')

        for order in disbands:
            parts = order.split()
            if len(parts) < 2: continue

            unit_type, location = parts[0], parts[1]
            reachable = []

            for target in sorted(targets):
                distance = self._distance(location, target, unit_type)
                if distance != float('inf'): reachable.append(distance)

            if not reachable:
                score = -1000
            else:
                reachable.sort()
                score = 100.0 / (1 + reachable[0]) + len(reachable) * 5

            if score < worst_score or (score == worst_score and order < worst_order):
                worst_score, worst_order = score, order

        return worst_order