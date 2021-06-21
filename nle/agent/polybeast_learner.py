# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Run with OMP_NUM_THREADS=1.
#

import collections
import logging
import os
import threading
import time
import timeit
import traceback

import wandb
import omegaconf
import nest
import torch

import libtorchbeast

from core import file_writer
from core import vtrace

from models import create_model
from models.baseline import NetHackNet

from torch import nn
from torch.nn import functional as F


logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    entropy_per_timestep = torch.sum(-policy * log_policy, dim=-1)
    return -torch.sum(entropy_per_timestep)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    policy_gradient_loss_per_timestep = cross_entropy * advantages.detach()
    return torch.sum(policy_gradient_loss_per_timestep)


def inference(
    inference_batcher, model, flags, actor_device, lock=threading.Lock()
):  # noqa: B008
    with torch.no_grad():
        for batch in inference_batcher:
            batched_env_outputs, agent_state = batch.get_inputs()
            observation, reward, done, *_ = batched_env_outputs
            # Observation is a dict with keys 'features' and 'glyphs'.
            observation["done"] = done
            observation, agent_state = nest.map(
                lambda t: t.to(actor_device, non_blocking=True),
                (observation, agent_state),
            )
            with lock:
                outputs = model(observation, agent_state)
            core_outputs, agent_state = nest.map(lambda t: t.cpu(), outputs)
            # Restructuring the output in the way that is expected
            # by the functions in actorpool.
            outputs = (
                tuple(
                    (
                        core_outputs["action"],
                        core_outputs["policy_logits"],
                        core_outputs["baseline"],
                    )
                ),
                agent_state,
            )
            batch.set_outputs(outputs)


# TODO(heiner): Given that our nest implementation doesn't support
# namedtuples, using them here doesn't seem like a good fit. We
# probably want to nestify the environment server and deal with
# dictionaries?
EnvOutput = collections.namedtuple(
    "EnvOutput", "frame rewards done episode_step episode_return"
)
AgentOutput = NetHackNet.AgentOutput
Batch = collections.namedtuple("Batch", "env agent")


def learn(
    learner_queue,
    model,
    actor_model,
    optimizer,
    scheduler,
    stats,
    flags,
    plogger,
    learner_device,
    lock=threading.Lock(),  # noqa: B008
):
    for tensors in learner_queue:
        tensors = nest.map(lambda t: t.to(learner_device), tensors)

        batch, initial_agent_state = tensors
        env_outputs, actor_outputs = batch
        observation, reward, done, *_ = env_outputs
        observation["reward"] = reward
        observation["done"] = done

        lock.acquire()  # Only one thread learning at a time.

        output, _ = model(observation, initial_agent_state, learning=True)

        # Use last baseline value (from the value function) to bootstrap.
        learner_outputs = AgentOutput._make(
            (output["action"], output["policy_logits"], output["baseline"])
        )

        # At this point, the environment outputs at time step `t` are the inputs
        # that lead to the learner_outputs at time step `t`. After the following
        # shifting, the actions in `batch` and `learner_outputs` at time
        # step `t` is what leads to the environment outputs at time step `t`.
        batch = nest.map(lambda t: t[1:], batch)
        learner_outputs = nest.map(lambda t: t[:-1], learner_outputs)

        # Turn into namedtuples again.
        env_outputs, actor_outputs = batch
        # Note that the env_outputs.frame is now a dict with 'features' and 'glyphs'
        # instead of actually being the frame itself. This is currently not a problem
        # because we never use actor_outputs.frame in the rest of this function.
        env_outputs = EnvOutput._make(env_outputs)
        actor_outputs = AgentOutput._make(actor_outputs)
        learner_outputs = AgentOutput._make(learner_outputs)

        rewards = env_outputs.rewards
        if flags.normalize_reward:
            model.update_running_moments(rewards)
            rewards /= model.get_running_std()

        total_loss = 0

        # STANDARD EXTRINSIC LOSSES / REWARDS
        if flags.entropy_cost > 0:
            entropy_loss = flags.entropy_cost * compute_entropy_loss(
                learner_outputs.policy_logits
            )
            total_loss += entropy_loss

        discounts = (~env_outputs.done).float() * flags.discounting

        # This could be in C++. In TF, this is actually slower on the GPU.
        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=actor_outputs.policy_logits,
            target_policy_logits=learner_outputs.policy_logits,
            actions=actor_outputs.action,
            discounts=discounts,
            rewards=rewards,
            values=learner_outputs.baseline,
            bootstrap_value=learner_outputs.baseline[-1],
        )

        # Compute loss as a weighted sum of the baseline loss, the policy
        # gradient loss and an entropy regularization term.
        pg_loss = compute_policy_gradient_loss(
            learner_outputs.policy_logits,
            actor_outputs.action,
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs.baseline
        )
        total_loss += pg_loss + baseline_loss

        # BACKWARD STEP
        optimizer.zero_grad()
        total_loss.backward()
        if flags.grad_norm_clipping > 0:
            nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())

        # LOGGING
        episode_returns = env_outputs.episode_return[env_outputs.done]
        stats["step"] = stats.get("step", 0) + flags.unroll_length * flags.batch_size
        stats["mean_episode_return"] = torch.mean(episode_returns).item()
        stats["mean_episode_step"] = torch.mean(env_outputs.episode_step.float()).item()
        stats["total_loss"] = total_loss.item()
        stats["pg_loss"] = pg_loss.item()
        stats["baseline_loss"] = baseline_loss.item()
        if flags.entropy_cost > 0:
            stats["entropy_loss"] = entropy_loss.item()

        stats["learner_queue_size"] = learner_queue.size()

        if not len(episode_returns):
            # Hide the mean-of-empty-tuple NaN as it scares people.
            stats["mean_episode_return"] = None

        # Only logging if at least one episode was finished
        if len(episode_returns):
            # TODO: log also SPS
            plogger.log(stats)
            if flags.wandb:
                wandb.log(stats, step=stats["step"])

        lock.release()


def train(flags):
    logging.info("Logging results to %s", flags.savedir)
    if isinstance(flags, omegaconf.DictConfig):
        flag_dict = omegaconf.OmegaConf.to_container(flags)
    else:
        flag_dict = vars(flags)
    plogger = file_writer.FileWriter(xp_args=flag_dict, rootdir=flags.savedir)

    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        learner_device = torch.device(flags.learner_device)
        actor_device = torch.device(flags.actor_device)
    else:
        logging.info("Not using CUDA.")
        learner_device = torch.device("cpu")
        actor_device = torch.device("cpu")

    if flags.max_learner_queue_size is None:
        flags.max_learner_queue_size = flags.batch_size

    # The queue the learner threads will get their data from.
    # Setting `minimum_batch_size == maximum_batch_size`
    # makes the batch size static. We could make it dynamic, but that
    # requires a loss (and learning rate schedule) that's batch size
    # independent.
    learner_queue = libtorchbeast.BatchingQueue(
        batch_dim=1,
        minimum_batch_size=flags.batch_size,
        maximum_batch_size=flags.batch_size,
        check_inputs=True,
        maximum_queue_size=flags.max_learner_queue_size,
    )

    # The "batcher", a queue for the inference call. Will yield
    # "batch" objects with `get_inputs` and `set_outputs` methods.
    # The batch size of the tensors will be dynamic.
    inference_batcher = libtorchbeast.DynamicBatcher(
        batch_dim=1,
        minimum_batch_size=1,
        maximum_batch_size=512,
        timeout_ms=100,
        check_outputs=True,
    )

    addresses = []
    connections_per_server = 1
    pipe_id = 0
    while len(addresses) < flags.num_actors:
        for _ in range(connections_per_server):
            addresses.append(f"{flags.pipes_basename}.{pipe_id}")
            if len(addresses) == flags.num_actors:
                break
        pipe_id += 1

    logging.info("Using model %s", flags.model)

    model = create_model(flags, learner_device)

    plogger.metadata["model_numel"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    logging.info("Number of model parameters: %i", plogger.metadata["model_numel"])

    actor_model = create_model(flags, actor_device)

    # The ActorPool that will run `flags.num_actors` many loops.
    actors = libtorchbeast.ActorPool(
        unroll_length=flags.unroll_length,
        learner_queue=learner_queue,
        inference_batcher=inference_batcher,
        env_server_addresses=addresses,
        initial_agent_state=model.initial_state(),
    )

    def run():
        try:
            actors.run()
        except Exception as e:
            logging.error("Exception in actorpool thread!")
            traceback.print_exc()
            print()
            raise e

    actorpool_thread = threading.Thread(target=run, name="actorpool-thread")

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return (
            1
            - min(epoch * flags.unroll_length * flags.batch_size, flags.total_steps)
            / flags.total_steps
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    stats = {}

    if flags.checkpoint and os.path.exists(flags.checkpoint):
        logging.info("Loading checkpoint: %s" % flags.checkpoint)
        checkpoint_states = torch.load(
            flags.checkpoint, map_location=flags.learner_device
        )
        model.load_state_dict(checkpoint_states["model_state_dict"])
        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_states["scheduler_state_dict"])
        stats = checkpoint_states["stats"]
        logging.info(f"Resuming preempted job, current stats:\n{stats}")

    # Initialize actor model like learner model.
    actor_model.load_state_dict(model.state_dict())

    learner_threads = [
        threading.Thread(
            target=learn,
            name="learner-thread-%i" % i,
            args=(
                learner_queue,
                model,
                actor_model,
                optimizer,
                scheduler,
                stats,
                flags,
                plogger,
                learner_device,
            ),
        )
        for i in range(flags.num_learner_threads)
    ]
    inference_threads = [
        threading.Thread(
            target=inference,
            name="inference-thread-%i" % i,
            args=(inference_batcher, actor_model, flags, actor_device),
        )
        for i in range(flags.num_inference_threads)
    ]

    actorpool_thread.start()
    for t in learner_threads + inference_threads:
        t.start()

    def checkpoint(checkpoint_path=None):
        if flags.checkpoint:
            if checkpoint_path is None:
                checkpoint_path = flags.checkpoint
            logging.info("Saving checkpoint to %s", checkpoint_path)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "stats": stats,
                    "flags": vars(flags),
                },
                checkpoint_path,
            )

    def format_value(x):
        return f"{x:1.5}" if isinstance(x, float) else str(x)

    try:
        train_start_time = timeit.default_timer()
        train_time_offset = stats.get("train_seconds", 0)  # used for resuming training
        last_checkpoint_time = timeit.default_timer()

        dev_checkpoint_intervals = [0, 0.25, 0.5, 0.75]

        loop_start_time = timeit.default_timer()
        loop_start_step = stats.get("step", 0)
        while True:
            if loop_start_step >= flags.total_steps:
                break
            time.sleep(5)
            loop_end_time = timeit.default_timer()
            loop_end_step = stats.get("step", 0)

            stats["train_seconds"] = round(
                loop_end_time - train_start_time + train_time_offset, 1
            )

            if loop_end_time - last_checkpoint_time > 10 * 60:
                # Save every 10 min.
                checkpoint()
                last_checkpoint_time = loop_end_time

            if len(dev_checkpoint_intervals) > 0:
                step_percentage = loop_end_step / flags.total_steps
                i = dev_checkpoint_intervals[0]
                if step_percentage > i:
                    checkpoint(flags.checkpoint[:-4] + "_" + str(i) + ".tar")
                    dev_checkpoint_intervals = dev_checkpoint_intervals[1:]

            logging.info(
                "Step %i @ %.1f SPS. Inference batcher size: %i."
                " Learner queue size: %i."
                " Other stats: (%s)",
                loop_end_step,
                (loop_end_step - loop_start_step) / (loop_end_time - loop_start_time),
                inference_batcher.size(),
                learner_queue.size(),
                ", ".join(
                    f"{key} = {format_value(value)}" for key, value in stats.items()
                ),
            )
            loop_start_time = loop_end_time
            loop_start_step = loop_end_step
    except KeyboardInterrupt:
        pass  # Close properly.
    else:
        logging.info("Learning finished after %i steps.", stats["step"])

    checkpoint()

    # Done with learning. Let's stop all the ongoing work.
    inference_batcher.close()
    learner_queue.close()

    actorpool_thread.join()

    for t in learner_threads + inference_threads:
        t.join()


def test(flags):
    test_checkpoint = os.path.join(flags.savedir, "test_checkpoint.tar")
    checkpoint = os.path.join(flags.load_dir, "checkpoint.tar")
    if not os.path.exists(os.path.dirname(test_checkpoint)):
        os.makedirs(os.path.dirname(test_checkpoint))

    logging.info("Creating test copy of checkpoint '%s'", checkpoint)

    checkpoint = torch.load(checkpoint)
    for d in checkpoint["optimizer_state_dict"]["param_groups"]:
        d["lr"] = 0.0
        d["initial_lr"] = 0.0

    checkpoint["scheduler_state_dict"]["last_epoch"] = 0
    checkpoint["scheduler_state_dict"]["_step_count"] = 0
    checkpoint["scheduler_state_dict"]["base_lrs"] = [0.0]
    checkpoint["stats"]["step"] = 0
    checkpoint["stats"]["_tick"] = 0

    flags.checkpoint = test_checkpoint
    flags.learning_rate = 0.0

    logging.info("Saving test checkpoint to %s", test_checkpoint)
    torch.save(checkpoint, test_checkpoint)

    train(flags)


def main(flags):
    if flags.wandb:
        wandb.init(
            project=flags.project,
            config=vars(flags),
            group=flags.group,
            entity=flags.entity,
        )
    if flags.mode == "train":
        if flags.write_profiler_trace:
            logging.info("Running with profiler.")
            with torch.autograd.profiler.profile() as prof:
                train(flags)
            filename = "chrome-%s.trace" % time.strftime("%Y%m%d-%H%M%S")
            logging.info("Writing profiler trace to '%s.gz'", filename)
            prof.export_chrome_trace(filename)
            os.system("gzip %s" % filename)
        else:
            train(flags)
    elif flags.mode.startswith("test"):
        test(flags)
