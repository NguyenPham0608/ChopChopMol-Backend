import asyncio
from typing import Any

import numpy as np


def _get_torch_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _detect_mace_device() -> tuple[str, str]:
    device = _get_torch_device()
    mace_device = "cpu" if device == "mps" else device
    mace_dtype = "float64" if mace_device != "mps" else "float32"
    return mace_device, mace_dtype

MACE_MODELS = {
    "mace-mp-0a": {"url": "medium", "name": "MACE-MP-0a"},
    "mace-mp-0b3": {
        "url": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model",
        "name": "MACE-MP-0b3",
    },
    "mace-mpa-0": {
        "url": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
        "name": "MACE-MPA-0",
    },
}

MODEL_MAP = {"small": "small", "medium": "medium", "large": "large", "mace-mpa-0": "medium"}


class MACEService:
    """Manages lazy-loaded MACE calculators and runs computations in threads."""

    def __init__(self, device: str = ""):
        self._calculators: dict[str, Any] = {}
        mace_device, mace_dtype = _detect_mace_device()
        self.device = device or mace_device
        self.dtype = mace_dtype

    def _get_calculator(self, model_id: str):
        if model_id not in self._calculators:
            from mace.calculators import mace_mp

            model_url = MACE_MODELS.get(model_id, MACE_MODELS["mace-mp-0a"])["url"]
            self._calculators[model_id] = mace_mp(
                model=model_url, default_dtype="float32", device=self.device
            )
        return self._calculators[model_id]

    def _get_optimizer_calculator(self, model_name: str):
        from mace.calculators import mace_mp

        mace_model = MODEL_MAP.get(model_name, "medium")
        return mace_mp(model=mace_model, device=self.device, default_dtype=self.dtype)

    # ── Energy ────────────────────────────────────────────────────────────

    async def calculate_energy(
        self, atoms_data: list[dict], model_id: str, include_forces: bool
    ) -> dict:
        return await asyncio.to_thread(
            self._calculate_energy_sync, atoms_data, model_id, include_forces
        )

    def _calculate_energy_sync(
        self, atoms_data: list[dict], model_id: str, include_forces: bool
    ) -> dict:
        from ase import Atoms

        symbols = [a["element"] for a in atoms_data]
        positions = [[a["x"], a["y"], a["z"]] for a in atoms_data]
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.calc = self._get_calculator(model_id)

        energy = float(atoms.get_potential_energy())
        result = {
            "success": True,
            "energy_eV": energy,
            "energy_kcal": energy * 23.0609,
        }
        if include_forces:
            forces = atoms.get_forces().tolist()
            result["forces"] = forces
            result["max_force"] = float(np.max(np.linalg.norm(forces, axis=1)))
        return result

    # ── Batch Energy ──────────────────────────────────────────────────────

    async def calculate_energy_batch(
        self, frames_data: list[list[dict]], model_id: str, include_forces: bool
    ) -> dict:
        return await asyncio.to_thread(
            self._calculate_energy_batch_sync, frames_data, model_id, include_forces
        )

    def _calculate_energy_batch_sync(
        self, frames_data: list[list[dict]], model_id: str, include_forces: bool
    ) -> dict:
        from ase import Atoms

        calc = self._get_calculator(model_id)
        results = []

        first_frame = frames_data[0]
        symbols = [a["element"] for a in first_frame]
        positions = [[a["x"], a["y"], a["z"]] for a in first_frame]
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.calc = calc

        for i, atoms_data in enumerate(frames_data):
            positions = [[a["x"], a["y"], a["z"]] for a in atoms_data]
            atoms.set_positions(positions)

            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            max_force = float(np.max(np.linalg.norm(forces, axis=1)))

            frame_result = {
                "frame": i,
                "energy_eV": round(energy, 6),
                "energy_kcal": round(energy * 23.0609, 4),
                "max_force_eV_A": round(max_force, 6),
            }
            if include_forces:
                frame_result["forces"] = forces.tolist()
            results.append(frame_result)

        energies = [r["energy_eV"] for r in results]
        min_idx = int(np.argmin(energies))
        max_idx = int(np.argmax(energies))

        return {
            "success": True,
            "frameCount": len(results),
            "energies": results,
            "lowestEnergyFrame": min_idx,
            "highestEnergyFrame": max_idx,
            "energyRange_eV": round(max(energies) - min(energies), 6),
        }

    # ── Optimize ──────────────────────────────────────────────────────────

    async def optimize_geometry(
        self,
        atoms_data: list[dict],
        model_name: str,
        fmax: float,
        max_steps: int,
        include_forces: bool,
    ) -> dict:
        return await asyncio.to_thread(
            self._optimize_sync, atoms_data, model_name, fmax, max_steps, include_forces
        )

    def _optimize_sync(
        self,
        atoms_data: list[dict],
        model_name: str,
        fmax: float,
        max_steps: int,
        include_forces: bool,
    ) -> dict:
        from ase import Atoms
        from ase.optimize import BFGS

        symbols = [a["element"] for a in atoms_data]
        positions = np.array([[a["x"], a["y"], a["z"]] for a in atoms_data])

        pos_min = positions.min(axis=0) if len(positions) > 0 else np.zeros(3)
        pos_max = positions.max(axis=0) if len(positions) > 0 else np.full(3, 10.0)
        cell_size = np.maximum(pos_max - pos_min + 20.0, 30.0)

        atoms = Atoms(symbols=symbols, positions=positions, cell=cell_size, pbc=False)
        atoms.calc = self._get_optimizer_calculator(model_name)

        trajectory_frames = []

        def observer():
            pos = atoms.get_positions().copy()
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            max_f = float(np.sqrt((forces**2).sum(axis=1).max()))
            frame = {"positions": pos.tolist(), "energy_eV": energy, "max_force": max_f}
            if include_forces:
                frame["forces"] = forces.tolist()
            trajectory_frames.append(frame)

        opt = BFGS(atoms, logfile=None, trajectory=None, restart=None)
        opt.attach(observer, interval=1)
        opt.run(fmax=fmax, steps=max_steps)

        forces = atoms.get_forces()
        max_force = np.sqrt((forces**2).sum(axis=1).max())
        converged = bool(max_force < fmax)
        final_positions = atoms.get_positions().tolist()
        final_energy = float(atoms.get_potential_energy())

        result = {
            "success": True,
            "converged": converged,
            "steps": int(opt.nsteps),
            "energy_eV": final_energy,
            "energy_kcal": final_energy * 23.0609,
            "max_force": float(max_force),
            "positions": [
                {"index": i, "x": p[0], "y": p[1], "z": p[2]}
                for i, p in enumerate(final_positions)
            ],
            "trajectory": trajectory_frames,
        }
        if include_forces:
            result["forces"] = forces.tolist()
        return result

    # ── Molecular Dynamics ────────────────────────────────────────────────

    async def run_md(
        self,
        atoms_data: list[dict],
        model_name: str,
        temperature: float,
        timestep: float,
        friction: float,
        frames: int | None,
        steps: int,
        save_interval: int,
        include_forces: bool,
    ) -> dict:
        return await asyncio.to_thread(
            self._run_md_sync,
            atoms_data,
            model_name,
            temperature,
            timestep,
            friction,
            frames,
            steps,
            save_interval,
            include_forces,
        )

    def _run_md_sync(
        self,
        atoms_data: list[dict],
        model_name: str,
        temperature: float,
        timestep: float,
        friction: float,
        requested_frames: int | None,
        steps: int,
        save_interval: int,
        include_forces: bool,
    ) -> dict:
        from ase import Atoms, units
        from ase.md.langevin import Langevin
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        if requested_frames and requested_frames >= 2:
            n_steps = (requested_frames - 1) * save_interval
        elif requested_frames == 1:
            save_interval = 1
            n_steps = 0
        else:
            n_steps = steps

        symbols = [a["element"] for a in atoms_data]
        positions = np.array([[a["x"], a["y"], a["z"]] for a in atoms_data])

        pos_min = positions.min(axis=0) if len(positions) > 0 else np.zeros(3)
        pos_max = positions.max(axis=0) if len(positions) > 0 else np.full(3, 10.0)
        cell_size = np.maximum(pos_max - pos_min + 20.0, 30.0)

        atoms = Atoms(symbols=symbols, positions=positions, cell=cell_size, pbc=False)
        atoms.calc = self._get_optimizer_calculator(model_name)

        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

        trajectory_frames = []

        def observer():
            pos = atoms.get_positions().copy()
            energy = float(atoms.get_potential_energy())
            kinetic = float(atoms.get_kinetic_energy())
            temp = float(kinetic / (1.5 * len(atoms) * units.kB))
            frame = {
                "positions": pos.tolist(),
                "energy_eV": energy,
                "kinetic_eV": kinetic,
                "total_eV": energy + kinetic,
                "temperature_K": temp,
                "step": len(trajectory_frames) * save_interval,
            }
            if include_forces:
                forces = atoms.get_forces()
                frame["forces"] = forces.tolist()
                frame["max_force"] = float(np.max(np.linalg.norm(forces, axis=1)))
            trajectory_frames.append(frame)

        dyn = Langevin(
            atoms,
            timestep=timestep * units.fs,
            temperature_K=temperature,
            friction=friction / units.fs,
        )
        dyn.attach(observer, interval=save_interval)
        observer()  # Capture initial frame
        dyn.run(n_steps)

        final_positions = atoms.get_positions().tolist()
        final_energy = float(atoms.get_potential_energy())

        result = {
            "success": True,
            "steps": n_steps,
            "temperature_K": temperature,
            "timestep_fs": timestep,
            "energy_eV": final_energy,
            "energy_kcal": final_energy * 23.0609,
            "frameCount": len(trajectory_frames),
            "positions": [
                {"index": i, "x": p[0], "y": p[1], "z": p[2]}
                for i, p in enumerate(final_positions)
            ],
            "trajectory": trajectory_frames,
        }
        if include_forces:
            forces = atoms.get_forces()
            result["forces"] = forces.tolist()
            result["max_force"] = float(np.max(np.linalg.norm(forces, axis=1)))
        return result

    # ── Test ──────────────────────────────────────────────────────────────

    async def test(self) -> dict:
        try:
            await asyncio.to_thread(self._get_calculator, "mace-mp-0a")
            return {"success": True, "message": "MACE loaded"}
        except Exception as e:
            return {"error": str(e)}
