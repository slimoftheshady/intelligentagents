import random
import signal
import networkx as nx
from agent_baselines import Agent

if hasattr(signal, 'SIGALRM'):
    import timeout_decorator
    _timeout = timeout_decorator.timeout
else:
    # Windows has no SIGALRM, so timeout_decorator can't work here.
    # This is a LOCAL DEV WORKAROUND ONLY: the 1-second limit is still a hard
    # constraint and will be enforced independently during marking (on Linux).
    def _timeout(seconds):
        def decorator(func):
            return func
        return decorator

class StudentAgent(Agent):
    '''
    Scenario 1 agent: opponents are all Static Agents (always hold, never attack).
    Since there is no real opposition, the optimal strategy is simply to march
    every unit toward the nearest unclaimed supply centre each movement phase,
    build new units whenever possible, and never need to worry about retreats
    or defence (Static Agents never attack, so combat should not occur).
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
        '''Reused from GreedyAgent: builds adjacency graphs for army/navy movement.'''
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
        # do not make changes to the following codes
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

    # ---------------- Movement phase ----------------

    def _movement_orders(self):
        all_possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(self.power_name)
        my_centers = self.game.get_centers(self.power_name)

        # any supply centre on the map I don't already own is a target
        unclaimed = [sc for sc in self.game.map.scs if sc not in my_centers]

        power_orders = []
        for loc in orderable_locations:
            possible = all_possible_orders.get(loc, [])
            if not possible:
                continue

            army_opts = [o for o in possible if o.startswith('A')]
            navy_opts = [o for o in possible if o.startswith('F')]

            if army_opts:
                power_orders.append(self._best_move(loc, unclaimed, self.map_graph_army, army_opts))
            elif navy_opts:
                power_orders.append(self._best_move(loc, unclaimed, self.map_graph_navy, navy_opts))
            else:
                power_orders.append(random.choice(possible))

        return power_orders

    def _best_move(self, loc, targets, graph, options):
        # already standing on an unclaimed centre -> hold so it gets captured
        if loc in targets:
            hold = f'{options[0][0]} {loc} H'
            return hold if hold in options else options[0]

        if loc not in graph:
            return random.choice(options)

        try:
            paths = nx.shortest_path(graph, source=loc)
        except Exception:
            return random.choice(options)

        best_target, best_dist = None, float('inf')
        for t in targets:
            if t in paths and len(paths[t]) < best_dist:
                best_dist = len(paths[t])
                best_target = t

        if best_target is None:
            return random.choice(options)

        step = paths[best_target][1]
        unit_type = options[0][0]
        move = f'{unit_type} {loc} - {step}'
        return move if move in options else random.choice(options)

    # ---------------- Retreat phase ----------------

    def _retreat_orders(self):
        # Static Agents never attack, so retreats should rarely if ever be
        # needed. Kept simple as a safety net.
        all_possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(self.power_name)
        power_orders = []
        for loc in orderable_locations:
            possible = all_possible_orders.get(loc, [])
            if possible:
                power_orders.append(random.choice(possible))
        return power_orders

    # ---------------- Build/adjustment phase ----------------

    def _adjustment_orders(self):
        all_possible_orders = self.game.get_all_possible_orders()
        orderable_locations = self.game.get_orderable_locations(self.power_name)
        power_orders = []
        for loc in orderable_locations:
            possible = all_possible_orders.get(loc, [])
            if possible:
                build_opts = [o for o in possible if o.endswith(' B')]
                power_orders.append(build_opts[0] if build_opts else random.choice(possible))
        return power_orders