import asyncio
import io
import logging
import queue
import sys
import threading
import time
import warnings
from typing import Any, AsyncGenerator

import numpy as np

log = logging.getLogger(__name__)

# Suppress noisy but harmless warnings from e3nn/torch
warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")


class _SuppressMACEOutput:
    """Context manager to suppress MACE's print() and root logging.warning() noise.

    MACE uses raw print() for info messages (cuequivariance, float32 notices)
    and logging.warning() on the root logger for dtype mismatch.
    These are harmless and clutter the terminal.
    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = io.StringIO()
        self._root_level = logging.root.level
        logging.root.setLevel(logging.ERROR)
        return self

    def __exit__(self, *exc):
        sys.stdout = self._stdout
        logging.root.setLevel(self._root_level)


def _detect_device() -> tuple[str, str]:
    """Auto-detect best available device and matching dtype.

    Returns (device, dtype):
      - cuda  → float64  (full precision, fast on GPU)
      - mps   → float32  (Apple Silicon; no float64 support in Metal)
      - cpu   → float64  (fallback)
    """
    try:
        import torch

        if torch.cuda.is_available():
            log.info("CUDA available — using GPU: %s", torch.cuda.get_device_name(0))
            return "cuda", "float64"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            log.info("Apple Silicon MPS available — using GPU with float32")
            return "mps", "float32"
    except ImportError:
        pass
    log.info("No GPU detected — using CPU")
    return "cpu", "float64"

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


def _patch_mps_double(calc) -> None:
    """Monkey-patch ScaleShiftMACE models so .double() → .float() on MPS.

    MACE's forward pass calls `tensor.double()` for numerical precision in
    per-atom energy decomposition (models.py:580).  MPS does not support
    float64, so we swap .double() for .float() inside the forward method.
    Total energy and forces are unaffected — they never touch .double().
    """
    import types, torch
    from mace.modules.models import ScaleShiftMACE

    for model in calc.models:
        if not isinstance(model, ScaleShiftMACE):
            continue
        orig_forward = model.forward

        def _patched_forward(*args, _orig=orig_forward, **kwargs):
            # Temporarily make .double() return .float() for MPS tensors
            _real_double = torch.Tensor.double
            torch.Tensor.double = lambda self, *a, **kw: self.float()
            try:
                return _orig(*args, **kwargs)
            finally:
                torch.Tensor.double = _real_double

        model.forward = types.MethodType(lambda self, *a, _pf=_patched_forward, **kw: _pf(*a, **kw), model)
    log.info("Patched ScaleShiftMACE .double()→.float() for MPS compatibility")


class MACEService:
    """Manages lazy-loaded MACE calculators and runs computations in threads."""

    def __init__(self, device: str = "", compile_mode: str = ""):
        self._calculators: dict[str, Any] = {}
        self._lock = threading.Lock()  # Prevent concurrent MPS/GPU access
        detected_device, detected_dtype = _detect_device()
        self.device = device or detected_device
        self.dtype = "float32" if self.device == "mps" else detected_dtype
        self.compile_mode = compile_mode or None  # "" → None (disabled)
        log.info(
            "MACEService initialised — device=%s  dtype=%s  compile_mode=%s",
            self.device, self.dtype, self.compile_mode or "off",
        )

    def _load_calculator(self, *, model: str, dtype: str):
        """Load a MACE calculator, handling MPS float64→float32 conversion."""
        from mace.calculators import mace_mp

        with _SuppressMACEOutput():
            if self.device == "mps":
                # MPS cannot handle float64 tensors — load on CPU then transfer.
                # MACE's ScaleShiftMACE.forward() also hardcodes .double() for
                # per-atom energy; we monkey-patch that to .float() on MPS.
                calc = mace_mp(model=model, default_dtype="float32", device="cpu")
                import torch
                for m in calc.models:
                    m.float().to(torch.device("mps"))
                calc.device = "mps"
                _patch_mps_double(calc)
                return calc

            try:
                return mace_mp(
                    model=model,
                    default_dtype=dtype,
                    device=self.device,
                    compile_mode=self.compile_mode,
                )
            except Exception:
                if self.compile_mode:
                    log.warning("torch.compile failed — falling back without compilation")
                    return mace_mp(
                        model=model,
                        default_dtype=dtype,
                        device=self.device,
                    )
                raise

    def _get_calculator(self, model_id: str):
        if model_id not in self._calculators:
            model_url = MACE_MODELS.get(model_id, MACE_MODELS["mace-mp-0a"])["url"]
            self._calculators[model_id] = self._load_calculator(
                model=model_url, dtype="float32"
            )
            log.info("Loaded calculator %s on %s", model_id, self.device)
        return self._calculators[model_id]

    def _get_optimizer_calculator(self, model_name: str):
        cache_key = f"opt:{model_name}"
        if cache_key not in self._calculators:
            mace_model = MODEL_MAP.get(model_name, "medium")
            self._calculators[cache_key] = self._load_calculator(
                model=mace_model, dtype=self.dtype
            )
            log.info("Loaded optimizer calculator %s on %s", model_name, self.device)
        return self._calculators[cache_key]

    def warmup(self, model_id: str = "mace-mp-0a") -> None:
        """Preload calculators and JIT-compile GPU kernels with a tiny H2 calculation."""
        t0 = time.perf_counter()
        try:
            with self._lock:
                self._get_calculator(model_id)
                self._get_optimizer_calculator("medium")

                from ase import Atoms
                atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]],
                              cell=[10, 10, 10], pbc=False)
                atoms.calc = self._get_calculator(model_id)
                atoms.get_potential_energy()
                atoms.get_forces()

            elapsed = time.perf_counter() - t0
            log.info("MACE warmup complete in %.1fs (device=%s)", elapsed, self.device)
        except Exception:
            elapsed = time.perf_counter() - t0
            log.warning("MACE warmup failed after %.1fs — first request will be slow", elapsed, exc_info=True)

    def device_info(self) -> dict:
        """Return current device configuration for diagnostics."""
        import torch

        info: dict[str, Any] = {
            "device": self.device,
            "dtype": self.dtype,
            "compile_mode": self.compile_mode or "off",
            "torch_version": torch.__version__,
            "cached_models": list(self._calculators.keys()),
        }
        if self.device == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem
            info["gpu_memory_GB"] = round(mem / 1e9, 1)
        elif self.device == "mps":
            info["gpu_name"] = "Apple Silicon (MPS)"
        return info

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

        with self._lock:
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
        with self._lock:
            try:
                results = self._calculate_energy_batch_native(frames_data, model_id, include_forces)
                log.info("Batch energy: native path succeeded (%d frames)", len(frames_data))
            except Exception as e:
                log.warning("Native batch failed (%s), falling back to ASE loop", e)
                results = self._calculate_energy_batch_ase(frames_data, model_id, include_forces)

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

        with self._lock:
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

        with self._lock:
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

    # ── Streaming Optimize ─────────────────────────────────────────────────

    async def optimize_geometry_stream(
        self,
        atoms_data: list[dict],
        model_name: str,
        fmax: float,
        max_steps: int,
        include_forces: bool,
    ) -> AsyncGenerator[dict, None]:
        q: queue.Queue[dict] = queue.Queue()
        t = threading.Thread(
            target=self._optimize_stream_worker,
            args=(q, atoms_data, model_name, fmax, max_steps, include_forces),
            daemon=True,
        )
        t.start()
        while True:
            try:
                item = q.get(timeout=0.1)
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue
            yield item
            if item.get("type") in ("done", "error"):
                break

    def _optimize_stream_worker(self, q, atoms_data, model_name, fmax, max_steps, include_forces):
        try:
            from ase import Atoms
            from ase.optimize import BFGS

            with self._lock:
                symbols = [a["element"] for a in atoms_data]
                positions = np.array([[a["x"], a["y"], a["z"]] for a in atoms_data])

                pos_min = positions.min(axis=0) if len(positions) > 0 else np.zeros(3)
                pos_max = positions.max(axis=0) if len(positions) > 0 else np.full(3, 10.0)
                cell_size = np.maximum(pos_max - pos_min + 20.0, 30.0)

                atoms = Atoms(symbols=symbols, positions=positions, cell=cell_size, pbc=False)
                atoms.calc = self._get_optimizer_calculator(model_name)

                frame_index = [0]

                def observer():
                    pos = atoms.get_positions().copy()
                    energy = float(atoms.get_potential_energy())
                    forces = atoms.get_forces()
                    max_f = float(np.sqrt((forces**2).sum(axis=1).max()))
                    frame = {
                        "type": "frame",
                        "index": frame_index[0],
                        "positions": pos.tolist(),
                        "energy_eV": energy,
                        "max_force": max_f,
                    }
                    if include_forces:
                        frame["forces"] = forces.tolist()
                    q.put(frame)
                    frame_index[0] += 1

                opt = BFGS(atoms, logfile=None, trajectory=None, restart=None)
                opt.attach(observer, interval=1)
                opt.run(fmax=fmax, steps=max_steps)

                forces = atoms.get_forces()
                max_force = float(np.sqrt((forces**2).sum(axis=1).max()))
                q.put({
                    "type": "done",
                    "summary": {
                        "success": True,
                        "converged": bool(max_force < fmax),
                        "steps": int(opt.nsteps),
                        "energy_eV": float(atoms.get_potential_energy()),
                        "energy_kcal": float(atoms.get_potential_energy()) * 23.0609,
                        "max_force": max_force,
                    },
                })
        except Exception as e:
            q.put({"type": "error", "error": str(e)})

    # ── Streaming MD ─────────────────────────────────────────────────────

    async def run_md_stream(
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
    ) -> AsyncGenerator[dict, None]:
        q: queue.Queue[dict] = queue.Queue()
        t = threading.Thread(
            target=self._run_md_stream_worker,
            args=(q, atoms_data, model_name, temperature, timestep, friction,
                  frames, steps, save_interval, include_forces),
            daemon=True,
        )
        t.start()
        while True:
            try:
                item = q.get(timeout=0.1)
            except queue.Empty:
                await asyncio.sleep(0.02)
                continue
            yield item
            if item.get("type") in ("done", "error"):
                break

    def _run_md_stream_worker(
        self, q, atoms_data, model_name, temperature, timestep, friction,
        requested_frames, steps, save_interval, include_forces,
    ):
        try:
            from ase import Atoms, units
            from ase.md.langevin import Langevin
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

            with self._lock:
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

                frame_index = [0]

                def observer():
                    pos = atoms.get_positions().copy()
                    energy = float(atoms.get_potential_energy())
                    kinetic = float(atoms.get_kinetic_energy())
                    temp = float(kinetic / (1.5 * len(atoms) * units.kB))
                    frame = {
                        "type": "frame",
                        "index": frame_index[0],
                        "positions": pos.tolist(),
                        "energy_eV": energy,
                        "kinetic_eV": kinetic,
                        "total_eV": energy + kinetic,
                        "temperature_K": temp,
                        "step": frame_index[0] * save_interval,
                    }
                    if include_forces:
                        forces = atoms.get_forces()
                        frame["forces"] = forces.tolist()
                        frame["max_force"] = float(np.max(np.linalg.norm(forces, axis=1)))
                    q.put(frame)
                    frame_index[0] += 1

                dyn = Langevin(
                    atoms,
                    timestep=timestep * units.fs,
                    temperature_K=temperature,
                    friction=friction / units.fs,
                )
                dyn.attach(observer, interval=save_interval)
                observer()  # Capture initial frame
                dyn.run(n_steps)

                final_energy = float(atoms.get_potential_energy())
                q.put({
                    "type": "done",
                    "summary": {
                        "success": True,
                        "steps": n_steps,
                        "temperature_K": temperature,
                        "timestep_fs": timestep,
                        "energy_eV": final_energy,
                        "energy_kcal": final_energy * 23.0609,
                        "frameCount": frame_index[0],
                    },
                })
        except Exception as e:
            q.put({"type": "error", "error": str(e)})

    # ── Batched Energy (native) ──────────────────────────────────────────

    def _calculate_energy_batch_ase(
        self, frames_data: list[list[dict]], model_id: str, include_forces: bool
    ) -> list[dict]:
        """Original ASE-loop fallback for batch energy."""
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

        return results

    def _calculate_energy_batch_native(
        self, frames_data: list[list[dict]], model_id: str, include_forces: bool
    ) -> list[dict]:
        """Batched inference using MACE's native torch_geometric batching."""
        import torch
        from ase.data import atomic_numbers as ase_atomic_numbers
        from mace.tools import torch_geometric, to_numpy, utils as mace_utils

        calc = self._get_calculator(model_id)
        model = calc.models[0]
        r_max = float(calc.r_max)
        z_table = calc.z_table

        # Build AtomicData list
        data_list = []
        for frame_atoms in frames_data:
            symbols = [a["element"] for a in frame_atoms]
            positions = np.array([[a["x"], a["y"], a["z"]] for a in frame_atoms])

            config = mace_utils.Configuration(
                atomic_numbers=np.array([ase_atomic_numbers[s] for s in symbols]),
                positions=positions,
                pbc=np.array([False, False, False]),
            )
            atomic_data = mace_utils.AtomicData.from_config(
                config, z_table=z_table, cutoff=r_max
            )
            data_list.append(atomic_data)

        # Process in chunks
        n_atoms = len(frames_data[0])
        chunk_size = max(1, min(32, 2000 // max(n_atoms, 1)))
        results = []

        for chunk_start in range(0, len(data_list), chunk_size):
            chunk = data_list[chunk_start:chunk_start + chunk_size]
            batch = torch_geometric.Batch.from_data_list(chunk)
            batch = batch.to(self.device)

            if include_forces:
                batch.positions.requires_grad_(True)
                output = model(batch.to_dict(), training=False)
            else:
                with torch.no_grad():
                    output = model(batch.to_dict(), training=False)

            energies = to_numpy(output["energy"]).flatten()
            forces_all = to_numpy(output["forces"]) if include_forces else None
            batch_indices = to_numpy(batch.batch) if include_forces else None

            for i in range(len(chunk)):
                idx = chunk_start + i
                e = float(energies[i])
                frame_result = {
                    "frame": idx,
                    "energy_eV": round(e, 6),
                    "energy_kcal": round(e * 23.0609, 4),
                }
                if include_forces and forces_all is not None and batch_indices is not None:
                    mask = batch_indices == i
                    frame_forces = forces_all[mask]
                    max_f = float(np.max(np.linalg.norm(frame_forces, axis=1)))
                    frame_result["forces"] = frame_forces.tolist()
                    frame_result["max_force_eV_A"] = round(max_f, 6)
                else:
                    # Compute max_force via a quick single-frame ASE call? Skip for speed.
                    frame_result["max_force_eV_A"] = 0.0
                results.append(frame_result)

        return results

    # ── Test ──────────────────────────────────────────────────────────────

    async def test(self) -> dict:
        try:
            await asyncio.to_thread(self._get_calculator, "mace-mp-0a")
            return {"success": True, "message": "MACE loaded"}
        except Exception as e:
            return {"error": str(e)}
