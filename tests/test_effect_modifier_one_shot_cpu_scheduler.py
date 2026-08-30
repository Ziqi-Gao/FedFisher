import subprocess
import unittest
from pathlib import Path

from effect_modifier_experiment.manage_one_shot_lasso_cpu_run import (
    FORMAL_SEED_END,
    FORMAL_SEED_START,
    MODELS,
    expected_evaluation_rounds,
    formal_run_id,
    formal_runner_args,
)
from effect_modifier_experiment.run_one_shot_lasso_effect_modifier import (
    method_config,
    run_config,
)


PROJECT_ROOT = Path("/home/del6500/projects/FedFisher")
SCRIPTS = PROJECT_ROOT / "scripts"
SHELLS = (
    SCRIPTS / "effect_modifier_one_shot_cpu_lock.sh",
    SCRIPTS / "run_effect_modifier_one_shot_lasso_cpu_worker.sh",
    SCRIPTS / "launch_effect_modifier_one_shot_lasso_cpu.sh",
    SCRIPTS / "status_effect_modifier_one_shot_lasso_cpu.sh",
)


class OneShotCpuSchedulerTests(unittest.TestCase):
    def test_formal_configuration_is_full_and_one_thread(self):
        logistic = formal_runner_args(
            "one_shot_local_logistic_lasso_fedavg", "control"
        )
        mlp = formal_runner_args("one_shot_mlp_group_lasso_fedavg", "control")
        for args in (logistic, mlp):
            self.assertEqual(args.seed_start, FORMAL_SEED_START)
            self.assertEqual(args.seed_end, FORMAL_SEED_END)
            self.assertEqual((args.num_sites, args.site_min_samples, args.site_max_samples), (10, 500, 1000))
            self.assertEqual((args.test_size, args.covariate_dim), (10000, 100))
            self.assertEqual(args.intervention_repeats, 5)
            self.assertEqual(args.threads, 1)
            self.assertEqual(run_config(args)["splits"], ["iid", "noniid"])
            self.assertEqual(method_config(args)["training_communication_rounds"], 1)
        self.assertEqual((logistic.cv_folds, logistic.lambda_count), (5, 30))
        self.assertEqual(logistic.optimizer_polish_max_iter, 50)
        self.assertEqual(method_config(logistic)["optimizer_polish_max_iter"], 50)
        self.assertEqual((mlp.local_epochs, mlp.batch_size), (30, 64))
        self.assertEqual(mlp.lambda_ratios, (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0))

    def test_formal_run_ids_models_and_evaluation_rounds(self):
        self.assertEqual(
            formal_run_id("one_shot_local_logistic_lasso_fedavg", "control"),
            "one_shot_local_logistic_lasso_fedavg_control_stationarity_polish_300seeds",
        )
        self.assertEqual(MODELS["one_shot_local_logistic_lasso_fedavg"], ("linear",))
        self.assertEqual(
            MODELS["one_shot_mlp_group_lasso_fedavg"],
            ("SyntheticMLP", "SyntheticMLPDeep"),
        )
        self.assertEqual(expected_evaluation_rounds("control"), 0)
        self.assertEqual(expected_evaluation_rounds("site_gaussian_shift"), 0)
        self.assertEqual(expected_evaluation_rounds("local_permutation"), 1)
        self.assertEqual(expected_evaluation_rounds("site_gaussian_local_permutation"), 1)

    def test_scheduler_shells_parse_and_forbid_gpu_or_foreign_control(self):
        subprocess.run(
            ["bash", "-n", *map(str, SHELLS)],
            cwd=str(PROJECT_ROOT),
            check=True,
        )
        combined = "\n".join(path.read_text(encoding="utf-8") for path in SHELLS)
        for forbidden in (
            "CUDA_VISIBLE_DEVICES",
            "nvidia-smi",
            "screen -wipe",
            "kill ",
            "renice",
            "effect_modifier_experiment/locks",
            "effect_modifier_experiment/cpu_locks",
        ):
            self.assertNotIn(forbidden, combined)
        launcher = SHELLS[2].read_text(encoding="utf-8")
        for session in ("ff_osp_ctl300", "ff_osp_gau300", "ff_osp_loc300", "ff_osp_cmb300"):
            self.assertIn(session, launcher)
            self.assertLess(len(session), 32)
        self.assertIn("total_configured_cpu_threads=4", launcher)

    def test_real_two_process_variant_lock_contention_and_release(self):
        helper = SCRIPTS / "effect_modifier_one_shot_cpu_lock.sh"
        holder_script = (
            'source "$1"; '
            'one_shot_acquire_variant_lock control "unittest_holder"; '
            'trap \'one_shot_release_cpu_lock "unittest_holder"\' EXIT; '
            'echo holder_ready; read -r release_signal'
        )
        holder = subprocess.Popen(
            ["bash", "-c", holder_script, "bash", str(helper)],
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            lines = []
            while True:
                line = holder.stdout.readline()
                self.assertNotEqual(line, "")
                lines.append(line)
                if "holder_ready" in line:
                    break
            contender = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; one_shot_acquire_variant_lock control "unittest_contender"',
                    "bash",
                    str(helper),
                ],
                cwd=str(PROJECT_ROOT),
                text=True,
                capture_output=True,
            )
            self.assertEqual(contender.returncode, 73, contender.stdout + contender.stderr)
            self.assertIn("one_shot_cpu_lock_status=busy", contender.stderr)
            holder.stdin.write("release\n")
            holder.stdin.flush()
            holder_stdout, holder_stderr = holder.communicate(timeout=10)
            self.assertEqual(holder.returncode, 0, "".join(lines) + holder_stdout + holder_stderr)

            reacquire = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; one_shot_acquire_variant_lock control "unittest_reacquire"; '
                    'one_shot_release_cpu_lock "unittest_reacquire"',
                    "bash",
                    str(helper),
                ],
                cwd=str(PROJECT_ROOT),
                text=True,
                capture_output=True,
            )
            self.assertEqual(reacquire.returncode, 0, reacquire.stdout + reacquire.stderr)
            self.assertIn("one_shot_cpu_lock_status=released", reacquire.stdout)
        finally:
            if holder.poll() is None:
                holder.stdin.write("release\n")
                holder.stdin.flush()
                holder.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
