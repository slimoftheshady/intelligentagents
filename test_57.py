'''
CITS3011 Intelligent Agent Project - Group 57 experiments.

Extends the provided test.py with:
  * seeded games, so different agent configurations meet identical opponents
    from identical starting conditions (paired comparison)
  * Scenario 3, using a copy of our own agent as a stand-in for the unknown
    Hidden Agent
  * technique ablations (each technique switched on and off) reported as the
    mean paired difference with its standard error
  * worst-case get_actions timing, to check the 1 second limit

Ablations are skipped automatically if the agent does not expose the
corresponding constructor flag, so this file runs against any version of
agent_57.py.

Run:  python3 test_57.py          (10 repeats = 70 games per arm)
      python3 test_57.py 30       (30 repeats = 210 games per arm)
'''

import sys
import time
import random
import inspect
import numpy as np
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from game import run_one_game
from agent_baselines import StaticAgent, RandomAgent, GreedyAgent, AttitudeAgent
from agent_57 import StudentAgent

ALL_POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']

# Opponent pools. Scenarios 2 and 3 follow the project description: the Random
# Agent is less likely to appear than the Attitude and Greedy agents.
POOL_1 = [StaticAgent]
POOL_2 = [RandomAgent, AttitudeAgent, AttitudeAgent, GreedyAgent, GreedyAgent]

AGENT_FLAGS = set(inspect.signature(StudentAgent.__init__).parameters)


def scoring(centres):
    '''Unchanged from the provided test.py.'''
    scores = {k: min(v, 18) for k, v in centres.items()}
    wins = {}
    for k, v in scores.items():
        if v == 18:
            wins[k] = 'WIN'
        elif v == 0:
            wins[k] = 'DEFEAT'
        else:
            wins[k] = 'SURVIVE'
    return scores, wins


class TimedAgent(StudentAgent):
    '''Records the worst-case get_actions time across a run.'''
    worst = 0.0

    def get_actions(self):
        t = time.perf_counter()
        actions = super().get_actions()
        TimedAgent.worst = max(TimedAgent.worst, time.perf_counter() - t)
        return actions


def build_agents(scenario, player_power, player_agent):
    '''Build one game's agents. Called after random.seed(), so the opponent
    draw is reproducible for a given (scenario, repeat, power).'''
    agents = {}
    others = [p for p in ALL_POWERS if p != player_power]
    hidden = random.choice(others) if scenario == 3 else None
    pool = POOL_1 if scenario == 1 else POOL_2
    for p in ALL_POWERS:
        if p == player_power:
            agents[p] = player_agent()
        elif p == hidden:
            # stand-in for the unknown Hidden Agent: a copy of our own agent
            agents[p] = StudentAgent()
        else:
            agents[p] = random.choice(pool)()
    return agents


def experiment(scenario, player_agent=None, repeat_nums=10, label='', quiet=False):
    '''Play repeat_nums games as each power. Returns {(repeat, power): score}
    so two configurations run on the same seeds are comparable game by game.'''
    player_agent = player_agent or TimedAgent
    TimedAgent.worst = 0.0
    per_game = {}
    all_scores, all_wins = defaultdict(list), defaultdict(list)

    with tqdm(total=repeat_nums * len(ALL_POWERS), desc=f'S{scenario} {label}') as pbar:
        for r in range(repeat_nums):
            for i, power in enumerate(ALL_POWERS):
                random.seed(r * 1000 + i)        # identical game across arms
                np.random.seed(r * 1000 + i)
                agents = build_agents(scenario, power, player_agent)
                results, _ = run_one_game(agents)
                scores, wins = scoring(results)
                per_game[(r, power)] = scores[power]
                all_scores[power].append(scores[power])
                all_scores['ALL'].append(scores[power])
                all_wins[power].append(wins[power])
                all_wins['ALL'].append(wins[power])
                pbar.update(1)

    if not quiet:
        report(scenario, label, all_scores, all_wins)
    return per_game


def report(scenario, label, all_scores, all_wins):
    print(f'----- Scenario {scenario} {label}: per-power and overall performance -----')
    for p in ALL_POWERS + ['ALL']:
        v, w = all_scores[p], all_wins[p]
        win = round(sum(i == 'WIN' for i in w) / len(w) * 100, 2)
        sur = round(sum(i == 'SURVIVE' for i in w) / len(w) * 100, 2)
        dfe = round(sum(i == 'DEFEAT' for i in w) / len(w) * 100, 2)
        print(f'{p}: SCs - {round(np.mean(v), 2)}±{round(np.std(v), 2)}, '
              f'Wins - {win}%, Survives - {sur}%, Defeats - {dfe}%')
    print(f'worst-case get_actions: {TimedAgent.worst * 1000:.0f} ms (limit 1000 ms)')
    print('-------------------------------------------------------------------------')


def compare(name, scenario, on_kwargs, off_kwargs, repeat_nums):
    '''Paired ablation: identical games, one technique switched on and off.'''
    needed = set(on_kwargs) | set(off_kwargs)
    missing = needed - AGENT_FLAGS
    if missing:
        print(f'\n--- Ablation: {name} SKIPPED: agent has no flag(s) {sorted(missing)} ---')
        return None
    on = experiment(scenario, partial(TimedAgent, **on_kwargs), repeat_nums,
                    label=f'{name}=on', quiet=True)
    off = experiment(scenario, partial(TimedAgent, **off_kwargs), repeat_nums,
                     label=f'{name}=off', quiet=True)
    keys = sorted(set(on) & set(off))
    a = np.array([on[k] for k in keys], dtype=float)
    b = np.array([off[k] for k in keys], dtype=float)
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan')
    print(f'\n--- Ablation: {name} (Scenario {scenario}, {len(keys)} paired games) ---')
    print(f'  ON  : SCs {a.mean():5.2f}  wins {(a >= 18).mean() * 100:5.1f}%  '
          f'defeats {(a == 0).mean() * 100:5.1f}%')
    print(f'  OFF : SCs {b.mean():5.2f}  wins {(b >= 18).mean() * 100:5.1f}%  '
          f'defeats {(b == 0).mean() * 100:5.1f}%')
    print(f'  paired difference (on - off): {d.mean():+.2f} ± {se:.2f} SCs (mean ± SE)')
    print(f'  better in {(d > 0).sum()} games, worse in {(d < 0).sum()}, '
          f'equal in {(d == 0).sum()}')
    return d


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    # 1. headline performance of the submitted agent 
    for s in (1, 2, 3):
        experiment(s, TimedAgent, repeat_nums=reps, label='final agent')

    # 2. technique ablations, Scenario 2 
    # New technique: coordinated support allocation
    compare('support allocation', 2, {}, {'support_alloc': False}, reps)

    # New technique: threat-aware evaluation (all-or-nothing version)
    compare('threat awareness', 2, {'threat_aware': True}, {}, reps)

    # Opponent model: only affects play while threat awareness is on, becaus is_active() is used solely inside the threat and hold-strength estimates
    compare('opponent model', 2, {'threat_aware': True},
            {'threat_aware': True, 'opp_model': False}, reps)

    # Basic technique: one-ply UCB1 Monte Carlo lookahead. Roughly 10x slower per game, so fewer repeats by default.
    compare('UCB1 lookahead (basic technique)', 2,
            {'use_ucb': True}, {}, max(3, reps // 3))

    # New technique: staging-square positioning. Runs only if the agent exposes a 'staging' flag; otherwise reported as skipped.
    compare('staging positioning', 2, {}, {'staging': False}, reps)

    # 3. Scenario 1 ablations: the techniques matter even against opponents that never move 
    compare('support allocation', 1, {}, {'support_alloc': False}, max(2, reps // 5))
    compare('staging positioning', 1, {}, {'staging': False}, max(2, reps // 5))