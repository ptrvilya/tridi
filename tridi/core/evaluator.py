from logging import getLogger
from pathlib import Path

import numpy as np

from config.config import ProjectConfig
from tridi.utils.metrics import generation
from tridi.utils.metrics import reconstruction

logger = getLogger(__name__)


class Evaluator:
    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg

    def evaluate(self):
        base_samples_folder = (Path(self.cfg.run.path) / "artifacts" /
                          f"step_{self.cfg.resume.step}_samples")
        logger.info(f"Experiment: {self.cfg.run.name} step: {self.cfg.resume.step}")
        # Generation
        if self.cfg.eval.use_gen_metrics:
            logger.info("Evaluating generation")
            for dataset in self.cfg.run.datasets:
                logger.info(f"\ton {dataset}")
                samples_folder = base_samples_folder / dataset
                # 1-NNA, COV, MMD
                for sample_target in self.cfg.eval.sampling_target:
                    logger.info(f"\t  sampling target: {sample_target}")
                    metrics = {
                        "1-NNA": [], "COV": [], "MMD": [], "SD": []
                    }

                    samples_files = list(samples_folder.glob(f"{sample_target}/samples_rep_*.hdf5"))
                    for samples_file in samples_files:
                        metrics["1-NNA"].append(generation.nearest_neighbor_accuracy(
                            self.cfg, samples_file, dataset,"test",
                            sample_target,
                        ))
                        metrics["COV"].append(generation.coverage(
                            self.cfg, samples_file, dataset,"test",
                            sample_target,
                        ))
                        metrics["MMD"].append(generation.minimum_matching_distance(
                            self.cfg, samples_file, dataset, "test",
                            sample_target,
                        ))
                        metrics["SD"].append(generation.sample_distance(
                            self.cfg, samples_file, dataset, "train",
                            sample_target,
                        ))
                    for k, v in metrics.items():
                        logger.info(f"\t\t{k:<6s} - {sample_target}: {np.mean(v):.4f} ± {np.std(v):.4f}")

        # Reconstruction
        if self.cfg.eval.use_rec_metrics:
            logger.info("Evaluating reconstruction")
            for dataset in self.cfg.run.datasets:
                logger.info(f"\ton {dataset}")
                samples_folder = base_samples_folder / dataset

                # Reconstruction
                metrics = {
                    "MPJPE": [], "MPJPE_PA": [], "SBJ_CONTACT_MESHES":[], "SBJ_CONTACT_DIFFUSED": [],
                    "OBJ_V2V": [], "OBJ_CENTER": [], "OBJ_CONTACT_MESHES": [], "OBJ_CONTACT_DIFFUSED": []

                }
                for sample_target in self.cfg.eval.sampling_target:
                    # subject
                    if 'sbj' in sample_target:
                        samples_files = list(samples_folder.glob(f"{sample_target}/samples_rep_*.hdf5"))
                        for samples_file in samples_files:
                            mpjpe, mpjpe_pa, sbj_contact_meshes, sbj_contact_diffused = \
                                reconstruction.get_sbj_metrics(
                                    self.cfg, samples_file, dataset
                                )
                            metrics["MPJPE"].append(mpjpe)
                            metrics["MPJPE_PA"].append(mpjpe_pa)
                            metrics["SBJ_CONTACT_MESHES"].append(sbj_contact_meshes)
                            metrics["SBJ_CONTACT_DIFFUSED"].append(sbj_contact_diffused)

                    # object
                    if 'obj' in sample_target:
                        samples_files = list(samples_folder.glob(f"{sample_target}/samples_rep_*.hdf5"))
                        for samples_file in samples_files:
                            obj_v2v, obj_center_dist, obj_contact_meshes, obj_contact_diffused = \
                                reconstruction.get_obj_metrics(
                                    self.cfg, samples_file, dataset
                                )
                            metrics["OBJ_V2V"].append(obj_v2v)
                            metrics["OBJ_CENTER"].append(obj_center_dist)
                            metrics["OBJ_CONTACT_MESHES"].append(obj_contact_meshes)
                            metrics["OBJ_CONTACT_DIFFUSED"].append(obj_contact_diffused)

                for k, v in metrics.items():
                    if len(v) > 0:
                        if k in ["SBJ_CONTACT_MESHES", "SBJ_CONTACT_DIFFUSED", "OBJ_CONTACT_MESHES", "OBJ_CONTACT_DIFFUSED"]:
                            logger.info(f"\t\t{k:<12s}: {100 * np.mean(np.max(np.stack(v, 1), axis=1)):.4f}")
                        else:
                            logger.info(f"\t\t{k:<12s}: {np.mean(np.min(np.stack(v, 1), axis=1)):.4f}")
