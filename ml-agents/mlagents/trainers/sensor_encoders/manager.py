from typing import Dict, Any, List, Optional

import numpy as np
from mlagents.torch_utils import torch
from mlagents_envs.logging_util import get_logger

from mlagents.trainers.sensor_encoders.vae import VectorVAE
from mlagents.trainers.sensor_encoders.egnn import EGNNEncoder
from mlagents.trainers.sensor_encoders.buffers import RingBuffer
from mlagents.trainers.settings import (
    SensorEncodersSettings,
    EGNNSensorEncodersSettings,
)


logger = get_logger(__name__)

# Must match ObservationEncoder.ATTENTION_EMBEDDING_SIZE: per-entity embeddings
# from variable-length processors feed directly into the fixed-size RSA block.
RSA_EMBEDDING_SIZE = 128


def build_egnn_registry(
    behavior_name: str,
    settings: EGNNSensorEncodersSettings,
    observation_specs: List[Any],
) -> Dict[str, Any]:
    """
    Builds EGNN encoder modules for variable-length (BufferSensor) observations from
    the egnn_encoders block of NetworkSettings. Returns a registry mapping sensor
    name -> config consumable by ModelUtils.get_encoder_for_obs.

    Modules are created fresh on every call so that separately constructed policies
    (e.g. self-play ghost policies) never share parameters.
    """
    registry: Dict[str, Any] = {}
    for spec in observation_specs:
        name = getattr(spec, "name", None)
        if not name:
            continue
        shape = spec.shape
        if len(shape) != 2 or "variable" not in str(spec.dimension_property).lower():
            continue
        override = next((s for s in settings.sensors if s.name == name), None)
        if override is None and not (settings.auto and "egnn" in name.lower()):
            continue

        def pick(field: str) -> Any:
            if override is not None and getattr(override, field, None) is not None:
                return getattr(override, field)
            return getattr(settings.defaults, field)

        embedding_size = int(pick("embedding_size"))
        if embedding_size != RSA_EMBEDDING_SIZE:
            logger.warning(
                f"[EGNN] embedding_size={embedding_size} for sensor '{name}' is not supported: "
                f"the attention block expects {RSA_EMBEDDING_SIZE}. Using {RSA_EMBEDDING_SIZE}."
            )
            embedding_size = RSA_EMBEDDING_SIZE
        egnn = EGNNEncoder(
            input_dim=int(shape[1]),
            embedding_size=embedding_size,
            hidden_dim=int(pick("hidden_dim")),
            message_dim=int(pick("message_dim")),
            num_layers=int(pick("num_layers")),
            k_neighbors=int(pick("k_neighbors")),
            pos_dim=int(pick("pos_dim")),
            max_entities=int(shape[0]),
        )
        registry[name] = {
            "module": egnn,
            "latent_size": embedding_size,
            "normalize": False,
            # Trained on-policy as part of the actor network
            "stop_gradient": False,
            "egnn": True,
        }
        logger.info(
            f"[EGNN] Built encoder for sensor '{name}' on behavior '{behavior_name}': "
            f"max_entities={shape[0]}, entity_size={shape[1]}, pos_dim={int(pick('pos_dim'))}, "
            f"k_neighbors={int(pick('k_neighbors'))}, num_layers={int(pick('num_layers'))}, "
            f"hidden_dim={int(pick('hidden_dim'))}, message_dim={int(pick('message_dim'))}, "
            f"embedding_size={embedding_size}"
        )
    if not registry:
        logger.warning(
            f"[EGNN] egnn_encoders enabled for behavior '{behavior_name}' but no matching "
            f"variable-length sensors were found (auto={settings.auto}, "
            f"overrides={[s.name for s in settings.sensors]})."
        )
    return registry


class VAESensorManager:
    """
    Manages per-sensor VAE encoders, replay buffers, and training for vector observations.
    """

    def __init__(
        self,
        behavior_name: str,
        settings: SensorEncodersSettings,
        observation_specs: List[Any],
    ):
        self.behavior_name = behavior_name
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Registry keyed by sensor name
        self.encoders: Dict[str, VectorVAE] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.buffers: Dict[str, RingBuffer] = {}
        self.meta: Dict[str, Dict[str, Any]] = {}
        # EGNN registry (no off-policy training loop here; trained on-policy only or frozen)
        self.egnn_modules: Dict[str, EGNNEncoder] = {}

        # Preserve observation order of names to align incoming obs arrays
        self.spec_names: List[str] = [getattr(spec, "name", "") for spec in observation_specs]

        # Build encoders for vector specs
        built_count = 0
        included_names: List[str] = []
        for spec in observation_specs:
            dim_prop = spec.dimension_property
            shape = spec.shape
            name = spec.name
            # Only named sensors; skip legacy CollectObservations vector sensor (including stacked variants)
            if name is None or name == "":
                continue
            if name.startswith("Stacking("):
                inner_name = name[len("Stacking(") : -1] if name.endswith(")") else name
                if "VectorSensor" in inner_name:
                    continue
            else:
                if "VectorSensor" in name:
                    continue
            # Handle variable-length entity specs (BufferSensor): shape = [max_entities, entity_size]
            if len(shape) == 2 and "variable" in str(dim_prop).lower():
                # EGNN support now reads from egnn_encoders settings on NetworkSettings. If invoked here,
                # we only auto-include when legacy settings are in use and name implies EGNN.
                override = next((s for s in getattr(self.settings, "sensors", []) if s.name == name), None)
                if override is None and not (getattr(self.settings, "auto", False) and isinstance(name, str) and ("egnn" in name.lower())):
                    continue
                entity_size = int(shape[1])
                # Reasonable EGNN defaults (legacy): NOTE embedding size used for attention is 128
                egnn = EGNNEncoder(
                    input_dim=entity_size,
                    embedding_size=128,
                    hidden_dim=128,
                    message_dim=64,
                    num_layers=3,
                    k_neighbors=8,
                    pos_dim=3,
                    max_entities=int(shape[0]),
                ).to(self.device)
                self.egnn_modules[name] = egnn
                self.meta[name] = {
                    "latent_size": 128,
                    "normalize": False,
                    "stop_gradient": True,
                    "egnn": True,
                }
                built_count += 1
                included_names.append(name)
                continue
            # Vector encoders (VAE)
            if len(shape) != 1:
                continue
            if "vector" not in self.settings.apply_to_types:
                continue
            input_dim = int(shape[0])
            # Determine latent and MLP sizes
            latent = max(
                self.settings.min_latent_size,
                min(self.settings.max_latent_size, int(round(input_dim * self.settings.latent_proportion))),
            )
            override = next((s for s in self.settings.sensors if s.name == name), None)
            if override is not None and override.latent_size is not None:
                latent = int(override.latent_size)
            hidden_layers = self.settings.defaults.hidden_layers
            if override is not None and override.hidden_layers is not None:
                hidden_layers = int(override.hidden_layers)
            hidden_size = (
                input_dim if self.settings.defaults.hidden_size == "auto" else int(self.settings.defaults.hidden_size)
            )
            if override is not None and override.hidden_size is not None:
                hidden_size = int(override.hidden_size)

            vae = VectorVAE(input_dim, latent, hidden_size, hidden_layers).to(self.device)
            # Determine training mode
            offpolicy_default = bool(self.settings.defaults.offpolicy_reconstruction)
            offpolicy = offpolicy_default if (override is None or override.offpolicy_reconstruction is None) else bool(override.offpolicy_reconstruction)

            lr = self.settings.defaults.lr if override is None or override.lr is None else float(override.lr)
            opt = None
            if offpolicy:
                opt = torch.optim.Adam(vae.parameters(), lr=lr)
            buf_size = (
                self.settings.defaults.buffer_size
                if override is None or override.buffer_size is None
                else int(override.buffer_size)
            )
            self.encoders[name] = vae
            if opt is not None:
                self.optimizers[name] = opt
            # Maintain buffers for off-policy VAE updates only
            if offpolicy:
                self.buffers[name] = RingBuffer(buf_size, input_dim)
            self.meta[name] = {
                "latent_size": latent,
                "normalize": self.settings.defaults.normalize if (override is None or override.normalize is None) else bool(override.normalize),
                # For policy build: detach if offpolicy mode (no policy gradients)
                "stop_gradient": offpolicy,
                "batch_size": self.settings.defaults.batch_size if (override is None or override.batch_size is None) else int(override.batch_size),
                "beta": self.settings.defaults.beta if (override is None or override.beta is None) else float(override.beta),
                "offpolicy": offpolicy,
                "onpolicy_reconstruction_weight": (
                    self.settings.defaults.onpolicy_reconstruction_weight
                    if (override is None or override.onpolicy_reconstruction_weight is None)
                    else float(override.onpolicy_reconstruction_weight)
                ),
            }
            built_count += 1
            included_names.append(name)

        if built_count > 0:
            logger.info(
                f"[VAE] Initialized {built_count} vector sensor encoder(s) for behavior '{self.behavior_name}' on device {self.device}: {included_names}"
            )
        else:
            logger.warning(
                f"[VAE] No eligible vector sensors found for behavior '{self.behavior_name}'."
            )

    def registry_for_utils(self) -> Dict[str, Any]:
        """
        Returns a mapping usable by ModelUtils to build VAE processors:
        name -> {"module": encoder, "latent_size": int, "normalize": bool, "stop_gradient": bool}
        """
        reg: Dict[str, Any] = {}
        for name, vae in self.encoders.items():
            reg[name] = {
                "module": vae,
                "latent_size": self.meta[name]["latent_size"],
                "normalize": self.meta[name]["normalize"],
                "stop_gradient": self.meta[name]["stop_gradient"],
            }
        # Add EGNN modules under same registry interface
        for name, egnn in self.egnn_modules.items():
            reg[name] = {
                "module": egnn,
                "latent_size": self.meta[name]["latent_size"],
                "normalize": False,
                "stop_gradient": self.meta[name]["stop_gradient"],
                "egnn": True,
            }
        return reg

    def push_observations(self, obs_list: List[np.ndarray]) -> None:
        # obs_list aligns with observation_specs order; map by name only for included sensors
        for idx, obs in enumerate(obs_list):
            if idx >= len(self.spec_names):
                break
            name = self.spec_names[idx]
            # Skip unnamed or excluded sensors to maintain alignment
            if name is None or name == "":
                continue
            if name.startswith("Stacking("):
                inner_name = name[len("Stacking(") : -1] if name.endswith(")") else name
                if "VectorSensor" in inner_name:
                    continue
            else:
                if "VectorSensor" in name:
                    continue
            if name in self.buffers:
                self.buffers[name].push(np.asarray(obs).reshape(-1))
        # Lightweight debug: show total buffered per update interval via stats; avoid per-step logs

    def maybe_update(self) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        any_update = False
        for name, vae in self.encoders.items():
            meta = self.meta[name]
            # If policy gradients flow (stop_gradient=False), skip off-policy VAE updates here.
            if not meta.get("stop_gradient", True):
                continue
            batch_size = meta["batch_size"]
            beta = meta["beta"]
            buf = self.buffers[name]
            if not buf.can_sample(batch_size):
                continue
            batch = torch.as_tensor(buf.sample(batch_size), device=self.device)
            recon, mu, logvar = vae(batch)
            loss, recon_loss, kl = VectorVAE.loss_function(
                recon,
                batch,
                mu,
                logvar,
                beta,
                loss_type=self.settings.defaults.reconstruction_loss,
                huber_delta=self.settings.defaults.huber_delta,
            )
            opt = self.optimizers.get(name)
            if opt is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            stats[f"VAE/{name}/loss"] = float(loss.detach().cpu().item())
            stats[f"VAE/{name}/recon"] = float(recon_loss.detach().cpu().item())
            stats[f"VAE/{name}/kl"] = float(kl.detach().cpu().item())
            size, cap = buf.stats()
            stats[f"VAE/{name}/buffer_fill"] = float(size) / float(cap)
            any_update = True
        if any_update:
            logger.debug(f"[VAE] Updated encoders for behavior '{self.behavior_name}' with {len(stats)//4} sensor(s).")
        return stats


