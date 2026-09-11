"""Deterministic timer/payload tests; no model, CUDA, or scientific execution."""

import ast
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.causal_comparison.measurement import (
    CommunicationLedger,
    MethodTimer,
    timed_call,
    validate_timing_output_path,
    write_timing_json,
)


ROOT = Path(__file__).resolve().parents[1]


def clock(values):
    readings = iter(values)
    return lambda: next(readings)


class MeasurementTests(unittest.TestCase):
    def test_data_preparation_is_a_separate_measured_stage(self):
        timer = MethodTimer("fixture", wall_clock=clock([0, 3]), cpu_clock=clock([0, 2]))
        with timer.measure("data_preparation"):
            pass
        report = timer.to_dict()
        self.assertEqual(report["stages"]["data_preparation"]["wall_seconds"], 3)
        self.assertFalse(report["stages"]["train"]["measured"])
        self.assertEqual(report["measured_stage_sum"]["cpu_seconds"], 2)

    def test_disjoint_stages_components_and_independent_methods(self):
        timer = MethodTimer("first", wall_clock=clock([0, 1, 3, 5, 7, 11]),
                            cpu_clock=clock([10, 11, 12, 13, 14, 16]))
        with timer.measure("train"):
            with timer.measure("train", component="sgd"):
                pass
        with timer.measure("evaluation"):
            pass
        report = timer.to_dict()
        self.assertEqual(report["stages"]["train"]["wall_seconds"], 5)
        self.assertEqual(report["stages"]["train"]["cpu_seconds"], 3)
        self.assertEqual(report["stages"]["train"]["components"]["sgd"]["wall_seconds"], 2)
        self.assertEqual(report["measured_stage_sum"], {"wall_seconds": 9, "cpu_seconds": 5})
        self.assertFalse(report["stages"]["tuning"]["measured"])
        self.assertFalse(MethodTimer("second").to_dict()["stages"]["train"]["measured"])
        report["stages"]["train"]["wall_seconds"] = 100
        self.assertEqual(timer.to_dict()["stages"]["train"]["wall_seconds"], 5)

    def test_repeated_calls_accumulate_and_exception_remains_failure(self):
        timer = MethodTimer("test", wall_clock=clock([0, 2, 3, 6]), cpu_clock=clock([0, 1, 1, 2]))
        self.assertEqual(timed_call(timer, "train", lambda x: x + 1, 3), 4)
        with self.assertRaisesRegex(RuntimeError, "fit failed"):
            with timer.measure("train"):
                raise RuntimeError("fit failed")
        stage = timer.to_dict()["stages"]["train"]
        self.assertEqual((stage["wall_seconds"], stage["calls"], stage["failed_calls"]), (5, 2, 1))
        self.assertEqual(timed_call(None, "train", lambda: "untimed"), "untimed")

    def test_clocks_must_produce_finite_nonnegative_durations(self):
        for values in ([2, 1], [0, float("nan")], [0, float("inf")]):
            with self.subTest(values=values):
                timer = MethodTimer("bad", wall_clock=clock(values), cpu_clock=clock([0, 1]))
                with self.assertRaises(ValueError):
                    with timer.measure("train"):
                        pass
                self.assertFalse(timer.to_dict()["stages"]["train"]["measured"])
        timer = MethodTimer("bad-cpu", wall_clock=clock([0, 1]), cpu_clock=clock([2, 1]))
        with self.assertRaises(ValueError):
            with timer.measure("train"):
                pass

    def test_stages_cannot_overlap_and_components_need_parent(self):
        timer = MethodTimer("test")
        with self.assertRaises(ValueError):
            with timer.measure("train", component="orphan"):
                pass
        with timer.measure("train"):
            with self.assertRaises(ValueError):
                timer.to_dict()
            with self.assertRaises(ValueError):
                with timer.measure("tuning"):
                    pass
            with self.assertRaises(ValueError):
                with timer.measure("evaluation", component="wrong-parent"):
                    pass
        with self.assertRaises(ValueError):
            with timer.measure("unknown"):
                pass

    def test_explicit_synchronization_brackets_clock_observations(self):
        events = []

        def sync():
            events.append("sync")

        def wall():
            events.append("wall")
            return events.count("wall")

        def cpu():
            events.append("cpu")
            return events.count("cpu")

        timer = MethodTimer("device", wall_clock=wall, cpu_clock=cpu, synchronize=sync)
        with timer.measure("aggregation"):
            events.append("body")
        self.assertEqual(events, ["sync", "wall", "cpu", "body", "sync", "wall", "cpu"])
        self.assertTrue(timer.to_dict()["clock"]["explicit_synchronization"])

    def test_logical_payload_totals_and_round_deduplication(self):
        ledger = CommunicationLedger()
        self.assertFalse(ledger.to_dict()["accounted"])
        ledger.add("upload", 120, messages=3, round_id="fit-1")
        ledger.add("download", 80, messages=2, round_id="fit-1")
        ledger.add("upload", 16, round_id="fit-2")
        report = ledger.to_dict()
        self.assertEqual((report["upload_bytes"], report["download_bytes"]), (136, 80))
        self.assertEqual(report["messages"], 6)
        self.assertEqual(report["identified_logical_rounds"], 2)
        self.assertTrue(report["round_accounting_complete"])
        self.assertFalse(report["actual_network_measured"])
        ledger.add("upload", 0)
        self.assertFalse(ledger.to_dict()["round_accounting_complete"])
        for size in (-1, 1.5, float("inf"), True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                ledger.add("upload", size)
        with self.assertRaises(ValueError):
            ledger.add("upload", 8, messages=0)

    def test_sidecar_refuses_existing_symlink_nonfinite_and_foreign_paths(self):
        scratch = Path("/scr/del6500/FedFisher/comparison-measurement-tests")
        validate_timing_output_path(scratch / "preflight.json")
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(scratch)) as temporary:
            root = Path(temporary)
            target = root / "timing.json"
            write_timing_json(target, {"wall_seconds": 1})
            original = target.read_text()
            with self.assertRaises(FileExistsError):
                write_timing_json(target, {"wall_seconds": 2})
            self.assertEqual(target.read_text(), original)
            link = root / "link.json"
            link.symlink_to(target)
            with self.assertRaises(ValueError):
                write_timing_json(link, {})
            parent_link = root / "linked-parent"
            parent_link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                validate_timing_output_path(parent_link / "new.json")
            invalid = root / "nonfinite.json"
            with self.assertRaises(ValueError):
                write_timing_json(invalid, {"wall_seconds": float("nan")})
            self.assertFalse(invalid.exists())
        # Rejected by lexical scope checks without reading the foreign path.
        with self.assertRaises(ValueError):
            validate_timing_output_path("/outside-fedfisher/timing.json")

    def test_legacy_main_timing_flag_is_opt_in_without_importing_main(self):
        tree = ast.parse((ROOT / "main.py").read_text())
        flags = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "add_argument" and node.args
                 and isinstance(node.args[0], ast.Constant)
                 and node.args[0].value == "--record_timing"]
        self.assertEqual(len(flags), 1)
        self.assertEqual({kw.arg: ast.literal_eval(kw.value) for kw in flags[0].keywords}["action"], "store_true")
        self.assertNotIn("default", {kw.arg for kw in flags[0].keywords})
        synchronizers = [node for node in ast.walk(tree) if isinstance(node, ast.keyword)
                         and node.arg == "synchronize"]
        self.assertEqual(len(synchronizers), 1)
        self.assertEqual(ast.unparse(synchronizers[0].value), "torch.cuda.synchronize")

    def test_local_training_timing_preserves_operation_order_with_fakes(self):
        # Compile just the method against harmless fakes. Never import torch,
        # run real SGD/Fisher code, or initialize a CUDA context.
        tree = ast.parse((ROOT / "train_model.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        function = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                        and node.name == "train_and_compute_fisher")
        self.assertEqual(function.args.args[-1].arg, "timing")
        self.assertIsNone(ast.literal_eval(function.args.defaults[-1]))
        module = ast.Module(body=[function], type_ignores=[])
        calls = []
        net = mock.Mock()
        net.train.side_effect = lambda: calls.append("train")
        net.parameters.return_value = []
        fisher = mock.Mock()
        fisher.get_diag.side_effect = lambda: calls.append("get_diag") or "diagonal"
        torch = mock.Mock()
        torch.optim.SGD.side_effect = lambda *args, **kwargs: calls.append("SGD")

        def fim(**kwargs):
            calls.append(kwargs["representation"])
            return fisher

        namespace = {"nullcontext": nullcontext, "torch": torch, "FIM": fim,
                     "PMatKFAC": "KFAC", "PMatDiag": "Diag",
                     "parameters_to_vector": lambda params: calls.append("vector") or "vector"}
        exec(compile(module, str(ROOT / "train_model.py"), "exec"), namespace)
        local = mock.Mock()
        local.args = {"eta": 0.01, "local_epochs": 0}
        function = namespace["train_and_compute_fisher"]
        plain = function(local, net, 2)
        plain_calls = list(calls)
        calls.clear()
        timer = MethodTimer("fake")
        with timer.measure("train"):
            timed = function(local, net, 2, timing=timer)
        self.assertEqual(plain_calls, ["train", "SGD", "KFAC", "Diag", "get_diag", "vector"])
        self.assertEqual(calls, plain_calls)
        self.assertEqual(plain, timed)
        self.assertEqual(set(timer.to_dict()["stages"]["train"]["components"]),
                         {"sgd", "fisher_diag", "fisher_kfac"})


if __name__ == "__main__":
    unittest.main()
