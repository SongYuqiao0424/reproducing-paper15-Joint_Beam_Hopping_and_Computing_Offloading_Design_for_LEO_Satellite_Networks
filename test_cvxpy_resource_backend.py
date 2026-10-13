"""实际资源分配的后端回归测试。

运行：python -B -m unittest -v test_cvxpy_resource_backend
调用路径测试使用受控返回值验证 CVXPY 结果确实被保存；数值测试需要安装 CVXPY。
"""
import importlib.util
import unittest
from unittest.mock import patch

from config import build_config
from proposed_algo import initialize_simulation, proposed_jbhco_mb
import solvers


def build_case(backend="cvxpy"):
    """构造一个可直接核算时延的小场景。"""
    config = build_config(
        solver_backend=backend, num_satellites=2, num_ground_cells=2,
        num_beams_per_satellite=1, task_data_size_range_Mbit=(10, 10),
        max_bcd_iterations=2, max_repair_iterations=2,
    )
    network, channel_model = initialize_simulation(config)
    plan = solvers.design_beam_hopping_matching(config, network, channel_model)
    capacity_table = channel_model.average_capacity_table(plan)  # 固定计划对应的平均容量。
    psi_unit = solvers.build_psi_matrix(config, network, capacity_table)
    return config, network, channel_model, plan, psi_unit


def controlled_resource_result(config, network, association, psi_unit):
    """为每条选中的链路返回 1 GC/s，故意区别于二分分配，便于识别覆盖错误。"""
    allocation = [[0.0] * config.num_users for _ in range(config.num_satellites)]
    for sat_id in range(config.num_satellites):
        for user_index in range(config.num_users):
            if association[sat_id][user_index] == 1:
                allocation[sat_id][user_index] = 1e9
    return {"allocation": allocation, "epsilon": 0.9, "status": "controlled"}


class ResourceBackendRoutingTests(unittest.TestCase):
    def test_cvxpy_allocation_is_preserved_and_objective_is_recomputed(self):
        config, network, channel_model, plan, psi_unit = build_case()
        with patch.object(solvers, '_solve_resource_allocation_cvxpy',
                          side_effect=controlled_resource_result) as solver_call, \
                patch.object(solvers, '_minimax_allocation_on_satellite',
                             side_effect=AssertionError('实际分配不应调用二分')):
            state = solvers.OffloadingState(config, network, psi_unit, [0, 1])
        self.assertEqual(solver_call.call_count, 1)
        self.assertEqual(state.allocation_matrix(), [[1e9, 0.0], [0.0, 1e9]])
        result = solvers.evaluate_latency(config, network, channel_model,
                                          state.access_matrix(), state.allocation_matrix(), plan)
        self.assertAlmostEqual(state.objective(), result['max_lambda'])
        self.assertNotAlmostEqual(state.objective(), 0.9)  # 目标应来自实际分配，不是辅助上界。

    def test_local_search_also_uses_cvxpy_and_trial_does_not_mutate_state(self):
        config, network, channel_model, plan, psi_unit = build_case()
        with patch.object(solvers, '_solve_resource_allocation_cvxpy',
                          side_effect=controlled_resource_result) as solver_call, \
                patch.object(solvers, '_minimax_allocation_on_satellite',
                             side_effect=AssertionError('局部实际分配不应调用二分')):
            state = solvers.OffloadingState(config, network, psi_unit, [0, None])
            objective = state.evaluate_move(1, 1)  # 只评估用户 1 改接卫星 1。
            self.assertEqual(state.assignment, [0, None])
            state.apply_move(1, 1)
        self.assertEqual(solver_call.call_count, 3)
        self.assertEqual(state.assignment, [0, 1])
        self.assertEqual(state.allocation_matrix(), [[1e9, 0.0], [0.0, 1e9]])
        self.assertAlmostEqual(state.objective(), objective)

    def test_infeasible_move_keeps_current_state(self):
        config, network, channel_model, plan, psi_unit = build_case()
        state = solvers.OffloadingState(config, network, psi_unit, [None, None])
        with patch.object(solvers, '_solve_resource_allocation_cvxpy',
                          side_effect=solvers.InfeasibleSubproblemError('固定 A 不可行')):
            self.assertEqual(state.evaluate_move(0, 0), float('inf'))
            with self.assertRaises(solvers.InfeasibleSubproblemError):
                state.apply_move(0, 0)
        self.assertEqual(state.assignment, [None, None])
        self.assertEqual(state.allocation_matrix(), [[0.0, 0.0], [0.0, 0.0]])

    def test_infeasible_bcd_candidate_is_rejected_without_fast_fallback(self):
        config, network, channel_model, plan, psi_unit = build_case()
        forecast = [[5e9, 5e9], [5e9, 5e9]]  # 隔离预测步骤，只检查候选实际分配路径。
        offloading_result = {"association_relaxed": [[1, 1], [0, 0]]}
        with patch.object(solvers.OffloadingState, 'forecast_matrix', return_value=forecast), \
                patch.object(solvers, '_solve_offloading_cvxpy', return_value=offloading_result), \
                patch.object(solvers, '_solve_resource_allocation_cvxpy',
                             side_effect=solvers.InfeasibleSubproblemError('拥塞')) as resource_call, \
                patch.object(solvers, '_minimax_allocation_on_satellite',
                             side_effect=AssertionError('不应回退到二分')):
            result = solvers.solve_joint_offloading_and_allocation(config, network, channel_model, plan)
        self.assertGreater(resource_call.call_count, 0)
        self.assertEqual(result['assignment'], [None, None])
        self.assertTrue(result['latency_result']['constraint_C4_ok'])

    def test_fast_backend_does_not_call_cvxpy(self):
        config, network, channel_model, plan, psi_unit = build_case(backend="fast")
        with patch.object(solvers, '_solve_resource_allocation_cvxpy',
                          side_effect=AssertionError('fast 不应调用 CVXPY')):
            state = solvers.OffloadingState(config, network, psi_unit, [0, 1])
            state.evaluate_move(0, None)
        self.assertAlmostEqual(state.allocation_matrix()[0][0], 5e9, delta=1e-3)


@unittest.skipUnless(importlib.util.find_spec('cvxpy'), '未安装 CVXPY，跳过真实数值求解')
class CvxpyNumericalTests(unittest.TestCase):
    def test_fixed_assignment_matches_fast_objective_and_satisfies_constraints(self):
        config, network, channel_model, plan, psi_unit = build_case()
        with patch.object(solvers, '_minimax_allocation_on_satellite',
                          side_effect=AssertionError('CVXPY 实际分配不能使用二分')):
            state = solvers.OffloadingState(config, network, psi_unit, [0, 1])
        result = solvers.validate_solution(config, network, channel_model,
                                           state.access_matrix(), state.allocation_matrix(), plan)
        fast_config = build_config(solver_backend="fast", num_satellites=2, num_ground_cells=2,
                                   num_beams_per_satellite=1, task_data_size_range_Mbit=(10, 10))
        fast_state = solvers.OffloadingState(fast_config, network, psi_unit, [0, 1])
        self.assertAlmostEqual(result['max_lambda'], fast_state.objective(), delta=1e-5)

    def test_full_algorithm_calls_real_cvxpy_resource_solver(self):
        config, network, channel_model, plan, psi_unit = build_case()
        with patch.object(solvers, '_solve_resource_allocation_cvxpy',
                          wraps=solvers._solve_resource_allocation_cvxpy) as resource_call:
            result = proposed_jbhco_mb(config, network, channel_model, max_swaps=2)
        self.assertGreater(resource_call.call_count, 0)
        self.assertTrue(result['constraint_C4_ok'])
        solvers.validate_solution(config, network, channel_model, result['access_matrix'],
                                   result['allocation'], result['beam_hopping_plan'])


if __name__ == '__main__':
    unittest.main()
