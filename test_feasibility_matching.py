"""本轮修改的回归检查；不写仿真缓存。运行 python -B -m unittest -v test_feasibility_matching。"""
import importlib.util
import math
import unittest
from unittest.mock import patch

from config import build_config
from proposed_algo import initialize_simulation, proposed_jbhco_mb
from baselines import run_computing_locally, run_jbhco_pg
import solvers as s


def single(**overrides):
    params = dict(num_satellites=1, num_ground_cells=1, num_beams_per_satellite=1,
                  task_data_size_range_Mbit=(10, 10))
    params.update(overrides)
    cfg = build_config(**params)
    net, channel = initialize_simulation(cfg)
    plan = s.design_beam_hopping_matching(cfg, net, channel)
    return cfg, net, channel, plan


class FeasibilityTests(unittest.TestCase):
    def test_zero_compute_is_invalid_offload_not_local(self):
        c, n, h, p = single()
        r = s.evaluate_latency(c, n, h, [[1]], [[0]], p)
        self.assertTrue(math.isinf(r['max_lambda']))
        self.assertEqual(r['local_users'], [])
        self.assertEqual(r['invalid_offloaded_users'], [0])
        self.assertFalse(r['constraint_C4_ok'])
        with self.assertRaises(s.FeasibleSolutionNotFoundError):
            s.validate_solution(c, n, h, [[1]], [[0]], p)

    def test_zero_capacity_is_invalid_for_both_resource_backends(self):
        c, n, h, p = single()
        with patch.object(h, 'average_capacity_table', return_value={}):
            r = s.evaluate_latency(c, n, h, [[1]], [[5e9]], p)
        self.assertTrue(math.isinf(r['max_lambda']))
        with self.assertRaises(s.InfeasibleSubproblemError):
            s.solve_resource_allocation(c, n, [[1]], [[float('inf')]])

    def test_p2_and_p3_reject_deadline_violation(self):
        c, n, h, p = single(max_latency_s=0.001)
        with self.assertRaises(s.InfeasibleSubproblemError):
            s.solve_offloading(c, n, [[0]])
        psi = s.build_psi_matrix(c, n, h.average_capacity_table(p))
        with self.assertRaises(s.InfeasibleSubproblemError):
            s.solve_resource_allocation(c, n, [[1]], psi)
        with self.assertRaises(s.FeasibleSolutionNotFoundError):
            s.solve_joint_offloading_and_allocation(c, n, h, p)

    def test_local_infeasible_does_not_imply_joint_infeasible(self):
        c, n, h, p = single(max_latency_s=0.1, sat_compute_capacity_GHz=100)
        with self.assertRaises(s.FeasibleSolutionNotFoundError):
            run_computing_locally(c, n, h)
        r = s.solve_joint_offloading_and_allocation(c, n, h, p)
        self.assertLessEqual(r['max_lambda'], 1)
        self.assertEqual(r['access_matrix'], [[1]])

    def test_greedy_baseline_rejects_infeasible_output(self):
        c, n, h, p = single(max_latency_s=0.001)
        with self.assertRaises(s.FeasibleSolutionNotFoundError):
            run_jbhco_pg(c, n, h)

    def test_fast_p3_does_not_silently_round_fixed_input(self):
        c, n, h, p = single()
        psi = s.build_psi_matrix(c, n, h.average_capacity_table(p))
        with self.assertRaises(ValueError):
            s.solve_resource_allocation(c, n, [[0.5]], psi)

    def test_final_validator_checks_budget_nonnegative_binary_and_shape(self):
        c, n, h, p = single()
        for a, f in [([[1]], [[6e9]]), ([[1]], [[-1]]), ([[0.5]], [[1e9]])]:
            with self.subTest(a=a, f=f), self.assertRaises(ValueError):
                s.validate_solution(c, n, h, a, f, p)
        bad = [[[1]] for _ in range(c.num_slots_per_BHP)]
        bad[-1] = [[0.5, 0.5]]
        with self.assertRaises(ValueError):
            s.validate_beam_hopping_plan(c, n, bad)

    def test_infeasible_swap_is_rejected_and_history_retained(self):
        c, n, h, p = single()
        initial = s.solve_joint_offloading_and_allocation(c, n, h, p)
        with patch('proposed_algo.propose_swap', return_value=p), patch(
            'proposed_algo.solve_joint_offloading_and_allocation',
            side_effect=[initial, s.FeasibleSolutionNotFoundError('test candidate')]
        ):
            result = proposed_jbhco_mb(c, n, h, max_swaps=1)
        self.assertEqual(result['accepted_swaps'], 0)
        self.assertEqual(result['history'], [initial['max_lambda']] * 2)

    @unittest.skipUnless(importlib.util.find_spec('cvxpy'), '当前环境未安装 CVXPY')
    def test_cvxpy_preserves_c4_and_valid_solutions(self):
        c, n, h, p = single(solver_backend='cvxpy')
        psi = s.build_psi_matrix(c, n, h.average_capacity_table(p))
        self.assertLessEqual(s.solve_resource_allocation(c, n, [[1]], psi)['epsilon'], 1 + 1e-7)
        c, n, h, p = single(solver_backend='cvxpy', max_latency_s=0.001)
        with self.assertRaises((s.InfeasibleSubproblemError, RuntimeError)):
            s.solve_offloading(c, n, [[0]])
        psi = s.build_psi_matrix(c, n, h.average_capacity_table(p))
        with self.assertRaises((s.InfeasibleSubproblemError, RuntimeError)):
            s.solve_resource_allocation(c, n, [[1]], psi)


class MatchingTests(unittest.TestCase):
    def test_many_to_many_recomputes_preferences_without_old_fill(self):
        c = build_config(num_satellites=1, num_ground_cells=5,
                         num_beams_per_satellite=3, num_slots_per_BHP=3)
        n, h = initialize_simulation(c)
        with patch.object(s, '_fill_slot_to_beam_capacity', side_effect=AssertionError('old fill')), \
                patch.object(h, 'compute_sinr_for_user', wraps=h.compute_sinr_for_user) as spy:
            p = s.design_beam_hopping_matching(c, n, h)
        s.validate_beam_hopping_plan(c, n, p)
        self.assertTrue(any(sum(t[0][i] for t in p) > 1 for i in range(5)))
        self.assertTrue(any(call.args[2] for call in spy.call_args_list))
        self.assertEqual(p, s.design_beam_hopping_matching(c, n, h))

    def test_paper_network_sizes_and_redundant_illumination(self):
        scenarios = [(v, 25) for v in range(2, 8)] + [(5, v) for v in [21, 23, 27, 29, 31]]
        for ns, ni in scenarios:
            with self.subTest(ns=ns, ni=ni):
                c = build_config(num_satellites=ns, num_ground_cells=ni)
                n, h = initialize_simulation(c)
                p = s.design_beam_hopping_matching(c, n, h)
                s.validate_beam_hopping_plan(c, n, p)
                if ns * c.num_beams_per_satellite <= ni:
                    self.assertTrue(all(sum(t[sat][i] for sat in range(ns)) <= 1
                                        for t in p for i in range(ni)))

    def test_impossible_beam_or_coverage_counts_fail(self):
        for params in [dict(num_ground_cells=2, num_beams_per_satellite=3),
                       dict(num_ground_cells=3, num_beams_per_satellite=1)]:
            c = build_config(num_satellites=1, num_slots_per_BHP=1, **params)
            n, h = initialize_simulation(c)
            with self.assertRaises(ValueError):
                s.design_beam_hopping_matching(c, n, h)


if __name__ == '__main__':
    unittest.main()
