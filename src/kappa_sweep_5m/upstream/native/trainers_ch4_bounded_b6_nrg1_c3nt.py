"""Chapter 4 trainers with durable optimizer-boundary resume.

No validation is run INSIDE a trainer. experiments_ch4 validates each completed
case/seed run before advancing to the next run. Model-selection checkpoints and
full recovery generations are deliberately separate.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import platform
import random
import sys
import time
import traceback
import numpy as np
from core_ch4_bounded_b6 import ExperimentEnvConfig, make_core
from full_v2_utils import BalancedMapOrder, CSVLog, write_json, read_json, array_hash, file_hash
from resume_ch4 import ResumeStore, StopController, TrainingPaused, execution_lock, capture_rng, restore_rng, runtime_signature, check_runtime, capture_log_offsets, restore_log_offsets, restore_checkpoint_index, sync_file, sync_directory, torch_load_local

def software_record():
    import importlib.metadata as im
    data = {'python': sys.version, 'platform': platform.platform(), 'cpu_count': os.cpu_count()}
    for package in ('numpy', 'torch', 'gymnasium', 'stable-baselines3', 'sb3-contrib', 'scipy', 'pyproj', 'shapely'):
        try:
            data[package] = im.version(package)
        except im.PackageNotFoundError:
            data[package] = None
    try:
        import torch
        data['cuda_available'] = torch.cuda.is_available()
        data['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        data['cuda_version'] = torch.version.cuda
    except ImportError:
        pass
    return data

class TrainingStream:
    """Balanced sampler + complete environment state, shared by all trainers."""

    def __init__(self, records, cfg, map_seed, seed, rank, run_dir=None):
        self.records = list(records)
        self.core = make_core(cfg)
        self.seed, self.rank = (seed, rank)
        self.order = BalancedMapOrder(records, map_seed + seed * 1009 + rank)
        self.cache = {}
        self.episode = self.total_steps = 0
        self.run_dir = Path(run_dir) if run_dir else None
        self.reopen_log()

    def reopen_log(self):
        self.log = CSVLog(self.run_dir / f'map_order_rank{self.rank}.csv') if self.run_dir else None

    def reset(self):
        idx = self.order.next_index()
        rec = self.records[idx]
        if rec.map_id not in self.cache:
            a = np.load(rec.path, allow_pickle=False)
            if array_hash(a) != rec.sha256:
                raise ValueError('Training map content changed')
            self.cache[rec.map_id] = a
        self.episode += 1
        obs, info = self.core.reset(self.cache[rec.map_id], seed=self.seed + self.rank * 100003 + self.episode, map_name=rec.map_id)
        if self.log:
            self.log.append(dict(rank=self.rank, episode=self.episode, sequence_position=self.order.position, map_id=rec.map_id, source=rec.source, pattern=rec.pattern, tier=rec.tier))
        return (obs, info)

    def step(self, action):
        self.total_steps += 1
        return self.core.step(action)

    def state_dict(self):
        return dict(core_type=type(self.core).__name__, config=asdict(self.core.config), core=deepcopy(vars(self.core)), order=self.order.state_dict(), episode=self.episode, total_steps=self.total_steps, seed=self.seed, rank=self.rank)

    def load_state_dict(self, state):
        if state['core_type'] != type(self.core).__name__ or state['config'] != asdict(self.core.config) or state['seed'] != self.seed or (state['rank'] != self.rank):
            raise ValueError('Training stream configuration changed on resume')
        self.order.load_state_dict(state['order'])
        vars(self.core).clear()
        vars(self.core).update(deepcopy(state['core']))
        self.episode = int(state['episode'])
        self.total_steps = int(state['total_steps'])
        self.core.assert_consistent()
        self.reopen_log()

def run_identity(spec, settings, records):
    content = {'spec': spec, 'settings': settings, 'maps': [(r.map_id, r.sha256) for r in records]}
    return hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()

def _check_settings(settings):
    for key in ('total_timesteps', 'num_envs', 'checkpoint_every_timesteps', 'episode_print_every'):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if int(settings.get('resume_every_timesteps', 50000)) < 1:
        raise ValueError('resume_every_timesteps must be positive')

def train_run(spec, settings, records, out):
    """Idempotent per-model entrypoint. A partial v2 run automatically resumes."""
    out = Path(out).resolve()
    with execution_lock(out.parent / '.run_locks' / (out.name + '.lock')):
        return _train_run_locked(spec, settings, records, out)

def _train_run_locked(spec, settings, records, out):
    _check_settings(settings)
    out.mkdir(parents=True, exist_ok=True)
    identity = run_identity(spec, settings, records)
    status_path = out / 'status.json'
    previous = None
    if status_path.exists():
        previous = read_json(status_path)
        if previous.get('identity') != identity:
            raise ValueError(f'Run identity mismatch: {out}. Budgets/configs are not changed during resume')
        if previous.get('state') == 'completed':
            for cp in read_json(out / 'checkpoints.json'):
                if not Path(cp['path']).exists() or file_hash(cp['path']) != cp['sha256']:
                    raise ValueError('Completed checkpoint hash mismatch')
            print(f"[TRAIN SKIP] completed {spec['run_id']}", flush=True)
            return previous
        if not (out / 'resume' / 'latest.json').exists():
            had_work = any(((out / name).exists() for name in ('training_episodes.csv', 'training_updates.csv', 'checkpoints.json', 'last_model.pt', 'last_model.zip')))
            had_maps = any(out.glob('map_order_rank*.csv'))
            if not had_work and previous.get('recovery_version') == 2:
                for f in out.glob('map_order_rank*.csv'):
                    archive = out / 'abandoned_initialization_logs'
                    archive.mkdir(exist_ok=True)
                    os.replace(f, archive / (f.stem + f".attempt{previous.get('attempts', 0)}.csv"))
                had_maps = False
            if had_work or had_maps:
                raise RuntimeError(f'Partial run has no full recovery bundle: {out}. V1 model-only files cannot restore replay/environment; use a new experiment rather than pretending to resume')
    elif any(out.iterdir()):
        raise ValueError(f'Nonempty run directory without status: {out}')
    import torch
    torch.set_num_threads(settings.get('torch_threads', 4))
    torch.manual_seed(spec['seed'])
    np.random.seed(spec['seed'])
    random.seed(spec['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec['seed'])
    write_json(out / 'run_spec.json', spec)
    write_json(out / 'training_settings.json', settings)
    if not (out / 'software.json').exists():
        write_json(out / 'software.json', software_record())
    attempt = int((previous or {}).get('attempts', 0)) + 1
    base_status = dict(identity=identity, attempts=attempt, recovery_version=2, validation_executed=False, test_executed=False)
    write_json(out / 'attempts' / f'attempt_{attempt:04d}.json', dict(attempt=attempt, started_unix_time=time.time(), software=software_record(), resume_available=(out / 'resume' / 'latest.json').exists()))
    write_json(status_path, dict(base_status, state='training'))
    try:
        if spec['algorithm'] in {'ppo', 'a2c'}:
            result = train_sb3(spec, settings, records, out)
        else:
            result = train_native(spec, settings, records, out)
        status = dict(base_status, state='completed', **result)
        write_json(status_path, status)
        if not settings.get('retain_completed_resume', False):
            try:
                ResumeStore(out, identity).clear_after_completion()
            except OSError as exc:
                print(f'[CLEANUP WARNING] completed recovery bundle retained: {exc}', flush=True)
        return status
    except BaseException as exc:
        if status_path.exists() and read_json(status_path).get('state') == 'completed':
            raise
        state = 'paused' if isinstance(exc, (TrainingPaused, KeyboardInterrupt)) else 'failed'
        write_json(status_path, dict(base_status, state=state, error=repr(exc), traceback=traceback.format_exc(), recovery='last committed optimizer boundary'))
        raise

def _register_checkpoint(out, checkpoints, path, step, final):
    sync_file(path)
    sync_directory(path.parent)
    row = dict(path=str(path.resolve()), num_timesteps=int(step), sha256=file_hash(path), final=bool(final))
    checkpoints[:] = [cp for cp in checkpoints if cp['path'] != row['path']]
    checkpoints.append(row)
    write_json(out / 'checkpoints.json', checkpoints)

def _episode_print(episode_count, info):
    print(f"[EP {episode_count}] {info['map_name']} success={info['is_success']} T={info['mission_time_sec']:.2f} Lw={info['priority_weighted_first_visit_latency_sec']} R={info['total_reward']:.3f}", flush=True)

def train_native(spec, settings, records, out):
    import torch
    from agents_ch4 import NativeAgent, Replay, continuous_to_discrete
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentEnvConfig(**spec['environment'])
    nenv, total = (settings['num_envs'], settings['total_timesteps'])
    acfg = dict(settings['off_policy'], grid_size=cfg.grid_size, gamma=settings['ppo']['gamma'], mapping=spec.get('mapping', 'preference4'))
    if cfg.paper_mode != 'none':
        acfg.update(channels=8 if cfg.paper_mode == 'sac' else 4, extra_dim=8 if cfg.paper_mode == 'ddqn' else 4 if cfg.paper_mode == 'sac' else 0)
    store = ResumeStore(out, run_identity(spec, settings, records), settings.get('resume_keep_generations', 2))
    bundle = store.load()
    state = torch_load_local(bundle[0] / 'state.pt') if bundle else None
    agent = NativeAgent.load(bundle[0] / 'model.pt', settings.get('device', 'auto')) if bundle else NativeAgent(spec['algorithm'], acfg, spec['seed'], settings.get('device', 'auto'))
    if agent.cfg != acfg:
        raise ValueError('Native agent configuration differs from recovery')
    runtime = runtime_signature(str(agent.device))
    if state:
        check_runtime(state['runtime'], runtime)
        restore_log_offsets(out, state['log_offsets'])
        restore_checkpoint_index(out, state['checkpoints'])
    print(f"{spec['run_id']} | native {spec['algorithm']} | device={agent.device}", flush=True)
    write_json(out / 'model_architecture.json', agent.architecture())
    streams = [TrainingStream(records, cfg, settings['map_sequence_seed'], spec['seed'], i, out) for i in range(nenv)]
    adim = agent.adim if spec['algorithm'] in {'td3', 'sac_continuous'} else 0
    capacity = acfg['buffer_size']
    estimated = capacity * (2 * acfg.get('channels', 6) * cfg.grid_size ** 2 + 128)
    if estimated > settings.get('maximum_replay_bytes', 4000000000):
        raise ValueError(f'Replay needs about {estimated} bytes; explicitly raise maximum_replay_bytes or reduce buffer_size')
    replay = Replay(capacity, cfg.grid_size, adim, channels=acfg.get('channels', 6), extra_dim=acfg.get('extra_dim') if cfg.paper_mode != 'none' else None)
    write_json(out / 'replay_configuration.json', dict(capacity=capacity, estimated_bytes=replay.estimated_bytes, stored_masks='current_and_next', saved_for_resume=True))
    cp_interval = settings['checkpoint_every_timesteps']
    resume_interval = int(settings.get('resume_every_timesteps', 50000))
    if state:
        if agent.num_timesteps != state['num_timesteps'] or len(state['streams']) != nenv:
            raise ValueError('Inconsistent native recovery step/stream count')
        for stream, saved in zip(streams, state['streams']):
            stream.load_state_dict(saved)
        obs = deepcopy(state['observations'])
        replay.load_recovery(bundle[0] / 'replay')
        checkpoints = deepcopy(state['checkpoints'])
        episode_count, vec_steps = (state['episode_count'], state['vec_steps'])
        next_cp, next_resume = (state['next_cp'], state['next_resume'])
        elapsed_before = state['elapsed_sec']
        resume_count = state.get('resume_count', 0) + 1
        restore_rng(state['rng'])
        print(f"[RESUME] {spec['run_id']}: step {agent.num_timesteps:,}/{total:,}; updates={agent.updates}, replay={replay.n}, original epsilon/LR budget preserved", flush=True)
    else:
        obs = [stream.reset()[0] for stream in streams]
        checkpoints = []
        episode_count = vec_steps = 0
        next_cp, next_resume = (cp_interval, resume_interval)
        elapsed_before = 0.0
        resume_count = 0
    ep_log = CSVLog(out / 'training_episodes.csv')
    upd_log = CSVLog(out / 'training_updates.csv')
    started = time.perf_counter()

    def save_model(final=False):
        if final and any((cp.get('final') and cp['num_timesteps'] == agent.num_timesteps for cp in checkpoints)):
            return
        path = out / ('last_model.pt' if final else f'checkpoints/step_{agent.num_timesteps}.pt')
        agent.save(path)
        _register_checkpoint(out, checkpoints, path, agent.num_timesteps, final)

    def save_resume(reason):
        payload = dict(format='native_training_state_v2', runtime=runtime, num_timesteps=agent.num_timesteps, streams=[stream.state_dict() for stream in streams], observations=deepcopy(obs), episode_count=episode_count, vec_steps=vec_steps, next_cp=next_cp, next_resume=next_resume, elapsed_sec=elapsed_before + time.perf_counter() - started, resume_count=resume_count, checkpoints=deepcopy(checkpoints), log_offsets=capture_log_offsets(out, nenv), rng=capture_rng())

        def writer(directory):
            agent.save(directory / 'model.pt')
            replay.save_recovery(directory / 'replay')
            torch.save(payload, directory / 'state.pt')
        store.commit(agent.num_timesteps, writer, reason=reason)
        print(f"[RECOVERY SAVED] {spec['run_id']} step={agent.num_timesteps:,} replay={replay.n}", flush=True)
    with StopController() as stop:
        if not state:
            save_resume('initial')
        stop.raise_if_requested(agent.num_timesteps)
        while agent.num_timesteps < total:
            for i, stream in enumerate(streams):
                if agent.num_timesteps >= total:
                    break
                mask = stream.core.action_masks()
                before = obs[i]
                epsilon = acfg['epsilon_final'] + (acfg['epsilon_initial'] - acfg['epsilon_final']) * max(0, 1 - agent.num_timesteps / max(1, acfg['epsilon_fraction'] * total))
                if agent.num_timesteps < acfg['learning_starts']:
                    if adim:
                        raw = agent.rng.uniform(-1, 1, adim).astype(np.float32)
                        if agent.mapping == 'preference4':
                            raw = np.where(mask, raw, -1.0).astype(np.float32)
                        action = continuous_to_discrete(raw, mask, agent.mapping, stream.core.heading)
                    else:
                        action = int(agent.rng.choice(np.flatnonzero(mask)))
                        raw = action
                else:
                    action, raw, _ = agent.act(before, mask, deterministic=False, epsilon=epsilon, noise=acfg['exploration_noise'])
                nxt, reward, done, trunc, info = stream.step(action)
                replay.add(before, mask, raw, reward, nxt, stream.core.action_masks(), done or trunc)
                agent.num_timesteps += 1
                if done or trunc:
                    stream.core.assert_consistent()
                    episode_count += 1
                    ep_log.append(dict(training_seed=spec['seed'], env_rank=i, episode_global=episode_count, timesteps=agent.num_timesteps, **info))
                    if episode_count % settings['episode_print_every'] == 0:
                        _episode_print(episode_count, info)
                    obs[i] = stream.reset()[0]
                else:
                    obs[i] = nxt
            vec_steps += 1
            if agent.num_timesteps >= acfg['learning_starts'] and replay.n >= acfg['batch_size'] and (vec_steps % acfg['train_freq_vector_steps'] == 0):
                for _ in range(acfg['gradient_steps']):
                    metrics = agent.update(replay.sample(acfg['batch_size'], agent.rng, agent.device))
                    upd_log.append(dict(timesteps=agent.num_timesteps, updates=agent.updates, learning_rate=acfg['learning_rate'], **metrics))
            if agent.num_timesteps >= next_cp:
                save_model()
                next_cp = (agent.num_timesteps // cp_interval + 1) * cp_interval
            if agent.num_timesteps >= next_resume or stop.requested:
                next_resume = (agent.num_timesteps // resume_interval + 1) * resume_interval
                save_resume('stop_requested' if stop.requested else 'periodic')
            stop.raise_if_requested(agent.num_timesteps)
        save_model(final=True)
        save_resume('training_complete')
        stop.raise_if_requested(agent.num_timesteps)
        if settings.get('save_replay_final', False):
            replay.save_recovery(out / 'replay_final')
    return dict(requested_timesteps=total, actual_timesteps=agent.num_timesteps, episodes=episode_count, gradient_updates=agent.updates, resume_count=resume_count, elapsed_sec=elapsed_before + time.perf_counter() - started)

def train_sb3(spec, settings, records, out):
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from sb3_contrib import MaskablePPO
        from stable_baselines3 import A2C
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import BaseCallback
        from nrg1_c3nt_features import CNN3CombinedExtractor as CombinedExtractor
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError('PPO/A2C requires requirements_chapter4.txt; this run was NOT trained') from exc
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentEnvConfig(**spec['environment'])
    store = ResumeStore(out, run_identity(spec, settings, records), settings.get('resume_keep_generations', 2))
    bundle = store.load()
    state = torch_load_local(bundle[0] / 'state.pt') if bundle else None
    if state:
        restore_log_offsets(out, state['log_offsets'])
        restore_checkpoint_index(out, state['checkpoints'])
    ep_log = CSVLog(out / 'training_episodes.csv')
    upd_log = CSVLog(out / 'training_updates.csv')
    nenv = settings['num_envs']
    total = settings['total_timesteps']

    class Adapter(gym.Env):
        metadata = {'render_modes': []}

        def __init__(self, rank):
            self.stream = TrainingStream(records, cfg, settings['map_sequence_seed'], spec['seed'], rank, out)
            if cfg.paper_mode != 'none':
                channels = 8 if cfg.paper_mode == 'sac' else 4
                extras = 8 if cfg.paper_mode == 'ddqn' else 4 if cfg.paper_mode == 'sac' else 0
                self.observation_space = spaces.Dict({'map': spaces.Box(0, 255, (channels, cfg.grid_size, cfg.grid_size), dtype=np.uint8), 'scalars': spaces.Box(0, 1, (extras,), dtype=np.float32)})
            else:
                self.observation_space = spaces.Dict({'map': spaces.Box(0, 255, (6, cfg.grid_size, cfg.grid_size), dtype=np.uint8), 'heading': spaces.Discrete(4), 'detour_budget': spaces.Box(0.0, 1.0, (1,), dtype=np.float32)})
            self.action_space = spaces.Discrete(4)
            self.observation_space = nrg_extend_space(self.observation_space)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.stream.reset()

        def step(self, action):
            return self.stream.step(int(action))

        def action_masks(self):
            return self.stream.core.action_masks()
    env = DummyVecEnv([lambda rank=i: Adapter(rank) for i in range(nenv)])
    checkpoints = deepcopy(state['checkpoints']) if state else []
    episode_count = state['episode_count'] if state else 0
    cp_interval = settings['checkpoint_every_timesteps']
    resume_interval = int(settings.get('resume_every_timesteps', 50000))
    next_cp = state['next_cp'] if state else cp_interval
    next_resume = state['next_resume'] if state else resume_interval
    resume_count = state.get('resume_count', 0) + 1 if state else 0
    elapsed_before = state['elapsed_sec'] if state else 0.0
    started = time.perf_counter()
    stop = StopController()
    Base = MaskablePPO if spec['algorithm'] == 'ppo' else A2C

    def save_model(model, final=False):
        if final and any((cp.get('final') and cp['num_timesteps'] == model.num_timesteps for cp in checkpoints)):
            return
        path = out / ('last_model.zip' if final else f'checkpoints/step_{model.num_timesteps}.zip')
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.stem + '.writing.zip')
        model.save(temp)
        sync_file(temp)
        os.replace(temp, path)
        _register_checkpoint(out, checkpoints, path, model.num_timesteps, final)

    def save_resume(model, reason):
        vector_names = ('buf_obs', 'buf_dones', 'buf_rews', 'buf_infos', 'reset_infos', '_seeds', '_options')
        adapters = []
        for adapter in env.envs:
            adapters.append(dict(stream=adapter.stream.state_dict(), gym_rng=deepcopy(getattr(adapter, '_np_random', None)), gym_rng_seed=getattr(adapter, '_np_random_seed', None), action_rng=deepcopy(getattr(adapter.action_space, '_np_random', None))))
        payload = dict(format='sb3_training_state_v2', runtime=runtime_signature(str(model.device)), num_timesteps=int(model.num_timesteps), streams=adapters, vector_state={k: deepcopy(getattr(env, k)) for k in vector_names if hasattr(env, k)}, last_obs=deepcopy(model._last_obs), last_episode_starts=deepcopy(model._last_episode_starts), episode_count=episode_count, next_cp=next_cp, next_resume=next_resume, resume_count=resume_count, elapsed_sec=elapsed_before + time.perf_counter() - started, checkpoints=deepcopy(checkpoints), log_offsets=capture_log_offsets(out, nenv), rng=capture_rng())

        def writer(directory):
            model.save(directory / 'model.zip')
            torch.save(payload, directory / 'state.pt')
        store.commit(int(model.num_timesteps), writer, reason=reason)
        print(f"[RECOVERY SAVED] {spec['run_id']} step={model.num_timesteps:,} (completed PPO/A2C update)", flush=True)

    class LogCallback(BaseCallback):

        def _on_training_start(self):
            cnn3_write_model_audit(self.model, out / 'network_observation_audit.json')
            nrg_audit_model(self.model)
            if state:
                restore_rng(state['rng'])
            else:
                save_resume(self.model, 'initial')
            stop.raise_if_requested(self.model.num_timesteps)

        def _on_step(self):
            nonlocal episode_count
            nrg_audit_delivery(self.locals, self.model)
            for i, (done, info) in enumerate(zip(self.locals['dones'], self.locals['infos'])):
                if done:
                    episode_count += 1
                    clean = {k: v for k, v in info.items() if k not in {'terminal_observation', 'TimeLimit.truncated'}}
                    ep_log.append(dict(training_seed=spec['seed'], env_rank=i, episode_global=episode_count, timesteps=int(self.num_timesteps), **clean))
                    if episode_count % settings['episode_print_every'] == 0:
                        _episode_print(episode_count, info)
            return True

        def _on_rollout_end(self):
            c3nt_audit_rollout_end(self.locals, self.model)

    class SaveAfterUpdate(Base):

        def train(self):
            nonlocal next_cp, next_resume
            super().train()
            vals = self.logger.name_to_value
            keys = ('train/approx_kl', 'train/clip_fraction', 'train/entropy_loss', 'train/explained_variance', 'train/value_loss', 'train/policy_gradient_loss', 'train/learning_rate')
            upd_log.append(dict(timesteps=int(self.num_timesteps), updates=int(self._n_updates), **{k.split('/')[-1]: float(vals[k]) if k in vals else None for k in keys}))
            if self.num_timesteps >= next_cp:
                save_model(self)
                next_cp = (self.num_timesteps // cp_interval + 1) * cp_interval
            if self.num_timesteps >= next_resume or stop.requested:
                next_resume = (self.num_timesteps // resume_interval + 1) * resume_interval
                save_resume(self, 'stop_requested' if stop.requested else 'periodic')
            stop.raise_if_requested(self.num_timesteps)
    p = settings['ppo']
    initial, floor = (p['learning_rate_initial'], p['learning_rate_floor'])
    common = dict(policy='MultiInputPolicy', env=env, learning_rate=lambda progress: max(floor, initial * progress), n_steps=p['n_steps'], gamma=p['gamma'], gae_lambda=p['gae_lambda'], ent_coef=p['ent_coef'], vf_coef=p['vf_coef'], max_grad_norm=p['max_grad_norm'], device=settings.get('device', 'auto'), seed=spec['seed'], verbose=1, tensorboard_log=str(out / 'tensorboard') if settings.get('tensorboard', False) else None, policy_kwargs=dict(features_extractor_class=CombinedExtractor, features_extractor_kwargs={'cnn_output_dim': 256}, share_features_extractor=True, net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=nn.ReLU))
    if spec['algorithm'] == 'ppo':
        common.update(batch_size=p['batch_size'], n_epochs=p['n_epochs'], clip_range=p['clip_range'], target_kl=p['target_kl'])
    try:
        with stop:
            if state:
                model = SaveAfterUpdate.load(bundle[0] / 'model.zip', env=env, device=settings.get('device', 'auto'), force_reset=False)
                check_runtime(state['runtime'], runtime_signature(str(model.device)))
                if int(model.num_timesteps) != state['num_timesteps'] or len(state['streams']) != nenv:
                    raise ValueError('Inconsistent SB3 recovery state')
                for adapter, saved in zip(env.envs, state['streams']):
                    adapter.stream.load_state_dict(saved['stream'])
                    adapter._np_random = deepcopy(saved['gym_rng'])
                    adapter._np_random_seed = saved['gym_rng_seed']
                    adapter.action_space._np_random = deepcopy(saved['action_rng'])
                for name, value in state['vector_state'].items():
                    setattr(env, name, deepcopy(value))
                model._last_obs = deepcopy(state['last_obs'])
                model._last_episode_starts = deepcopy(state['last_episode_starts'])
                for i, adapter in enumerate(env.envs):
                    current = adapter.stream.core.observation()
                    for key in current:
                        if not np.array_equal(current[key], model._last_obs[key][i]):
                            raise ValueError('Restored environment and policy observation do not match')
                restore_rng(state['rng'])
                print(f"[RESUME] {spec['run_id']}: step {model.num_timesteps:,}/{total:,}; optimizer, environment, sampler, RNG and ORIGINAL LR denominator restored", flush=True)
            else:
                model = SaveAfterUpdate(**common)
            write_json(out / 'model_architecture.json', dict(policy=str(model.policy), trainable_parameters=sum((v.numel() for v in model.policy.parameters() if v.requires_grad)), optimizer=str(model.policy.optimizer)))
            remaining = max(0, total - int(model.num_timesteps))
            if remaining:
                model.learn(total_timesteps=remaining, callback=LogCallback(), progress_bar=False, reset_num_timesteps=not bool(state), tb_log_name=f"{spec['algorithm']}_attempt_{resume_count}")
            save_model(model, final=True)
            save_resume(model, 'training_complete')
            stop.raise_if_requested(model.num_timesteps)
            return dict(requested_timesteps=total, actual_timesteps=int(model.num_timesteps), episodes=episode_count, ppo_epoch_counter=int(model._n_updates), resume_count=resume_count, elapsed_sec=elapsed_before + time.perf_counter() - started)
    finally:
        env.close()
from nrg1_c3nt_core import wrap_factory, extend_observation_space as nrg_extend_space
from nrg1_c3nt_core import audit_model as nrg_audit_model, audit_delivery as nrg_audit_delivery
_nrg_original_make_core = make_core
make_core = wrap_factory(_nrg_original_make_core)
from nrg1_c3nt_features import write_cnn3_model_audit as cnn3_write_model_audit
from nrg1_c3nt_core import audit_rollout_end as c3nt_audit_rollout_end
