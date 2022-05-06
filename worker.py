import argparse
import os
from typing import Dict, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from utils.transformers_adaptor import BertOutput

from loader.data import Data
from loader.task.bert.mlm_task import MLMTask
from loader.task.pretrain_task import PretrainTask
from model.auto_bert import AutoBert
from utils.config_initializer import init_config
from utils.gpu import GPU
from utils.random_seed import seeding
from utils.smart_printer import SmartPrinter, printer, Color
from utils.logger import Logger


class Worker:
    def __init__(self, project_args, project_exp, cuda=None):
        self.args = project_args
        self.exp = project_exp
        self.print = printer[('MAIN', '·', Color.CYAN)]

        self.logging = Logger(self.args.store.log_path)
        SmartPrinter.logger = self.logging

        self.device = self.get_device(cuda)

        self.data = Data(
            project_args=self.args,
            project_exp=self.exp,
            device=self.device,
        )

        self.print(self.data.depots['train'][0])

        self.auto_model = AutoBert(
            device=self.device,
            model_init=self.data.bert_init,
            pretrain_depot=self.data.pretrain_depot,
        )

        self.auto_model.to(self.device)
        self.print(self.auto_model.model.config)
        self.save_model = self.auto_model

        self.static_modes = ['export', 'dev', 'test']

        if self.exp.mode in self.static_modes:
            self.m_optimizer = self.m_scheduler = None
        else:
            self.m_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, self.auto_model.parameters()),
                lr=self.exp.policy.lr
            )
            self.m_scheduler = get_linear_schedule_with_warmup(
                self.m_optimizer,
                num_warmup_steps=self.exp.policy.n_warmup,
                num_training_steps=len(self.data.train_set) // self.exp.policy.batch_size * self.exp.policy.epoch,
            )

            self.print('training params')
            total_memory = 0
            for name, p in self.auto_model.named_parameters():  # type: str, torch.Tensor
                total_memory += p.element_size() * p.nelement()
                if p.requires_grad and not name.startswith('bert.'):
                    self.print(name, p.data.shape, p.data.get_device())
            self.print('total memory usage:', total_memory / 1024 / 8)

        if not self.exp.load.super_load:
            self.attempt_loading()

    @staticmethod
    def get_device(cuda):
        if cuda == -1:
            return 'cpu'
        if not cuda:
            return GPU.auto_choose(torch_format=True)
        return "cuda:{}".format(cuda)

    def _attempt_loading(self, path):
        load_path = os.path.join(self.args.store.save_dir, path)
        self.print("load model from exp {}".format(load_path))
        state_dict = torch.load(load_path, map_location=self.device)

        model_ckpt = state_dict['model']

        self.save_model.load_state_dict(model_ckpt, strict=not self.exp.load.relax_load)
        load_status = False
        if self.exp.mode not in self.static_modes and not self.exp.load.load_model_only:
            load_status = True
            self.m_optimizer.load_state_dict(state_dict['optimizer'])
            self.m_scheduler.load_state_dict(state_dict['scheduler'])
        self.print('Load optimizer and scheduler:', load_status)

    def attempt_loading(self):
        if self.exp.load.load_ckpt is not None:
            self._attempt_loading(self.exp.load.load_ckpt)

    def log_interval(self, epoch, step, task: PretrainTask, loss):
        self.print(
            "epoch {}, step {}, "
            "task {}, "
            "loss {:.4f}".format(
                epoch,
                step + 1,
                task.name,
                loss.item()
            ))

    def train(self, *tasks: PretrainTask):
        self.print('Start Training')

        train_steps = len(self.data.train_set) // self.exp.policy.batch_size
        accumulate_step = 0
        assert self.exp.policy.accumulate_batch >= 1

        # t_loader = self.data.get_t_loader(task)
        self.m_optimizer.zero_grad()
        for epoch in range(self.exp.policy.epoch_start, self.exp.policy.epoch + self.exp.policy.epoch_start):
            for task in tasks:
                task.train()
                task.start_epoch(epoch - self.exp.policy.epoch_start, self.exp.policy.epoch)
            t_loader = self.data.get_loader(self.data.TRAIN, *tasks)
            self.auto_model.train()
            for step, batch in enumerate(tqdm(t_loader)):
                task = batch['task']
                task_output = self.auto_model(
                    batch=batch,
                    task=task,
                )

                loss = task.calculate_loss(batch, task_output, model=self.auto_model)
                loss.backward()

                accumulate_step += 1
                if accumulate_step == self.exp.policy.accumulate_batch:
                    self.m_optimizer.step()
                    self.m_scheduler.step()
                    self.m_optimizer.zero_grad()
                    accumulate_step = 0

                if self.exp.policy.check_interval:
                    if self.exp.policy.check_interval < 0:  # step part
                        if (step + 1) % max(train_steps // (-self.exp.policy.check_interval), 1) == 0:
                            self.log_interval(epoch, step, task, loss.loss)
                    else:
                        if (step + 1) % self.exp.policy.check_interval == 0:
                            self.log_interval(epoch, step, task, loss.loss)

            self.print('end epoch')
            avg_loss = self.dev(task=task)
            self.print("epoch {} finished, "
                  "task {}, "
                  "loss {:.4f}".format(epoch, task.name, avg_loss))

            if (epoch + 1) % self.exp.policy.store_interval == 0:
                epoch_path = os.path.join(self.args.store.ckpt_path, 'epoch_{}.bin'.format(epoch))
                state_dict = dict(
                    model=self.auto_model.state_dict(),
                    optimizer=self.m_optimizer.state_dict(),
                    scheduler=self.m_scheduler.state_dict(),
                )
                torch.save(state_dict, epoch_path)
        self.print('Training Ended')

    def dev(self, task: PretrainTask, steps=None, d_loader=None):
        avg_loss = torch.tensor(.0).to(self.device)
        self.auto_model.eval()
        task.eval()
        d_loader = d_loader or self.data.get_loader(self.data.DEV, task)
        for step, batch in enumerate(tqdm(d_loader)):
            with torch.no_grad():
                task_output = self.auto_model(
                    batch=batch,
                    task=task,
                )

                loss = task.calculate_loss(batch, task_output, model=self.auto_model)
                avg_loss += loss.loss

            if steps and step >= steps:
                break

        avg_loss /= len(self.data.dev_set) / self.exp.policy.batch_size

        return avg_loss.item()

    def test(self, task: PretrainTask):
        assert isinstance(task, MLMTask)
        self.auto_model.eval()
        task.test()
        loader = self.data.get_loader(self.data.TEST, task)

        overlap_rate_dict = dict()  # type: Dict[str, Union[list, float]]
        hit_rate_dict = dict()  # type: Dict[str, Union[list, float]]
        for hit_rate in self.exp.policy.hit_rates:
            overlap_rate_dict[str(hit_rate)] = []
            hit_rate_dict[str(hit_rate)] = []

        for step, batch in enumerate(tqdm(loader)):
            with torch.no_grad():
                output = self.auto_model(
                    batch=batch,
                    task=task,
                )[task.pred_items]
                mask_labels_col = batch['mask_labels_col']
                indexes = batch['append_info']['index']

                col_mask = mask_labels_col[task.pred_items]

                for i_batch in range(len(indexes)):
                    ground_truth = set(task.depot[indexes[i_batch]]['pred_items'])

                    for hit_rate in self.exp.policy.hit_rates:
                        candidates_per_channel = max(hit_rate // len(ground_truth), 1)

                        candidates = set()
                        for i_tok in range(task.dataset.max_sequence):
                            if col_mask[i_batch][i_tok]:
                                candidates.update(
                                    torch.argsort(
                                        output[i_batch][i_tok], descending=True).cpu().tolist()[:candidates_per_channel])

                        overlap_rate_dict[str(hit_rate)].append(len(candidates) / (candidates_per_channel * len(ground_truth)))
                        hit_rate_dict[str(hit_rate)].append(int(bool(candidates.intersection(ground_truth))))

        for hit_rate in self.exp.policy.hit_rates:
            for d in [overlap_rate_dict, hit_rate_dict]:
                d[str(hit_rate)] = torch.tensor(d[str(hit_rate)], dtype=torch.float).mean().item()
            self.print('HR@%4d: %.4f' % (hit_rate, hit_rate_dict[str(hit_rate)]))
            self.print('OR@%4d: %.4f' % (hit_rate, overlap_rate_dict[str(hit_rate)]))

    def export(self):
        # bert_aggregator = BertAggregator(
        #     layers=self.exp.save.layers,
        #     layer_strategy=self.exp.save.layer_strategy,
        #     union_strategy=self.exp.save.union_strategy,
        # )
        features = torch.zeros(
            self.data.depot.sample_size,
            self.args.model_config.hidden_size,
            dtype=torch.float
        ).to(self.device)

        for loader in [self.data.get_loader(self.data.TRAIN, self.data.non_task),
                       self.data.get_loader(self.data.DEV, self.data.non_task)]:
            for batch in tqdm(loader):
                with torch.no_grad():
                    task_output = self.auto_model(batch=batch, task=self.data.non_task)  # type: BertOutput
                    task_output = task_output.last_hidden_state.detach()  # type: torch.Tensor  # [B, S, D]
                    attention_sum = batch['attention_mask'].to(self.device).sum(-1).unsqueeze(-1).repeat(1, 1, self.args.model_config.hidden_size)
                    attention_mask = batch['attention_mask'].to(self.device).unsqueeze(-1).repeat(1, 1, self.args.model_config.hidden_size)
                    features[batch['append_info'][self.exp.save.key]] = (task_output * attention_mask).sum(1) / attention_sum

        save_path = os.path.join(self.args.store.ckpt_path, self.exp.save.feature_path)
        np.save(save_path, features.cpu().numpy(), allow_pickle=False)

    def run(self):
        # tasks = [self.data.pretrain_depot[task.name] for task in self.exp.tasks]
        tasks = self.data.tasks

        if self.exp.mode == 'train':
            self.train(*tasks)
        elif self.exp.mode == 'dev':
            dev_results = dict()
            for task in tasks:
                dev_results.update(self.dev(task, steps=100))
            display_string = []
            display_value = []
            for k in dev_results:
                display_string.append('%s {:.4f}' % k)
                display_value.append(dev_results[k])
            display_string = ', '.join(display_string)
            display_value = tuple(display_value)
            display_string = display_string.format(*display_value)
            self.print(display_string)
        elif self.exp.mode == 'export':
            self.export()
        elif self.exp.mode == 'test':
            if not self.exp.load.super_load:
                for task in tasks:
                    if task.name == 'non':
                        continue
                    self.test(task)
            else:
                epochs = eval(self.exp.load.epochs)
                for epoch in epochs:
                    ckpt_base_path = self.exp.load.ckpt_base_path
                    self._attempt_loading(os.path.join(ckpt_base_path, f'epoch_{epoch}.bin'))

                    for task in tasks:
                        if task.name == 'non':
                            continue
                        self.test(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--cuda', type=int, default=None)

    args = parser.parse_args()

    config, exp = init_config(args.config, args.exp)

    seeding(2021)

    worker = Worker(project_args=config, project_exp=exp, cuda=args.cuda)
    worker.run()
