import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions as BertOutput

from loader.data import Data
from loader.task_depot.mlm_task import MLMTask
from loader.task_depot.pretrain_task import PretrainTask
from model.auto_bert import AutoBert
from utils.config_initializer import init_config
from utils.gpu import GPU
from utils.random_seed import seeding
from utils.time_printer import printer as print
from utils.logger import Logger


class Worker:
    def __init__(self, project_args, project_exp, cuda=None):
        self.args = project_args
        self.exp = project_exp

        self.logging = Logger(self.args.store.log_path, print)
        print.logger = self.logging

        self.device = self.get_device(cuda)

        self.data = Data(
            project_args=self.args,
            project_exp=self.exp,
            device=self.device,
        )

        print(self.data.depots['train'][0])

        self.model = AutoBert(
            device=self.device,
            bert_init=self.data.bert_init,
            pretrain_depot=self.data.pretrain_depot,
        )

        self.model.to(self.device)
        print(self.model.bert.config)
        self.save_model = self.model

        self.static_modes = ['export', 'dev', 'test']

        if self.exp.mode in self.static_modes:
            self.m_optimizer = self.m_scheduler = None
        else:
            self.m_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.exp.policy.lr
            )
            self.m_scheduler = get_linear_schedule_with_warmup(
                self.m_optimizer,
                num_warmup_steps=self.exp.policy.n_warmup,
                num_training_steps=len(self.data.train_set) // self.exp.policy.batch_size * self.exp.policy.epoch,
            )

            print('training params')
            total_memory = 0
            for name, p in self.model.named_parameters():  # type: str, torch.Tensor
                total_memory += p.element_size() * p.nelement()
                if p.requires_grad and not name.startswith('bert.'):
                    print(name, p.data.shape)
            print('total memory usage:', total_memory / 1024 / 8)

        self.attempt_loading()

    @staticmethod
    def get_device(cuda):
        if cuda == -1:
            return 'cpu'
        if not cuda:
            return GPU.auto_choose(torch_format=True)
        return "cuda:{}".format(cuda)

    def attempt_loading(self):
        if self.exp.load.load_ckpt is not None:
            load_path = os.path.join(self.args.store.save_dir, self.exp.load.load_ckpt)
            print("load model from exp {}".format(load_path))
            state_dict = torch.load(load_path, map_location=self.device)

            model_ckpt = state_dict['model']
            # if self.exp.mode == 'test':
            #     del model_ckpt['embedding_tables.track_id.weight']
            #     del model_ckpt['extra_modules.sim-mlm.track_id.embedding_table']

            self.save_model.load_state_dict(model_ckpt, strict=not self.exp.load.relax_load)
            load_status = False
            if self.exp.mode not in self.static_modes and not self.exp.load.load_model_only:
                load_status = True
                self.m_optimizer.load_state_dict(state_dict['optimizer'])
                self.m_scheduler.load_state_dict(state_dict['scheduler'])
            print('Load optimizer and scheduler:', load_status)

    def log_interval(self, epoch, step, task: PretrainTask, loss):
        print(
            "epoch {}, step {}, "
            "task {}, "
            "loss {:.4f}".format(
                epoch,
                step + 1,
                task.name,
                loss.item()
            ))

    def train(self, *tasks: PretrainTask):
        print('Start Training')

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
            self.model.train()
            for step, batch in enumerate(tqdm(t_loader)):
                task = batch['task']
                task_output = self.model(
                    batch=batch,
                    task=task,
                )

                loss = task.calculate_loss(batch, task_output, model=self.model)
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

            print('end epoch')
            avg_loss = self.dev(task=task)
            print("epoch {} finished, "
                  "task {}, "
                  "loss {:.4f}".format(epoch, task.name, avg_loss))

            if (epoch + 1) % self.exp.policy.store_interval == 0:
                epoch_path = os.path.join(self.args.store.ckpt_path, 'epoch_{}.bin'.format(epoch))
                state_dict = dict(
                    model=self.model.state_dict(),
                    optimizer=self.m_optimizer.state_dict(),
                    scheduler=self.m_scheduler.state_dict(),
                )
                torch.save(state_dict, epoch_path)
        print('Training Ended')

    def dev(self, task: PretrainTask, steps=None, d_loader=None):
        avg_loss = torch.tensor(.0).to(self.device)
        self.model.eval()
        task.eval()
        d_loader = d_loader or self.data.get_loader(self.data.DEV, task)
        for step, batch in enumerate(tqdm(d_loader)):
            with torch.no_grad():
                task_output = self.model(
                    batch=batch,
                    task=task,
                )

                loss = task.calculate_loss(batch, task_output, model=self.model)
                avg_loss += loss.loss

            if steps and step >= steps:
                break

        avg_loss /= len(self.data.dev_set) / self.exp.policy.batch_size

        return avg_loss.item()

    def test(self, task: PretrainTask):
        avg_loss = self.dev(task=task, d_loader=self.data.get_loader(self.data.TEST, task))

        print("task {}, "
              "loss {:.4f}".format(task.name, avg_loss))

    def export(self):
        # bert_aggregator = BertAggregator(
        #     layers=self.exp.save.layers,
        #     layer_strategy=self.exp.save.layer_strategy,
        #     union_strategy=self.exp.save.union_strategy,
        # )
        features = torch.zeros(
            self.data.depot.sample_size,
            self.args.bert_config.hidden_size,
            dtype=torch.float
        ).to(self.device)

        for loader in [self.data.get_loader(self.data.TRAIN, self.data.non_task),
                       self.data.get_loader(self.data.DEV, self.data.non_task)]:
            for batch in tqdm(loader):
                with torch.no_grad():
                    task_output = self.model(batch=batch, task=self.data.non_task)  # type: BertOutput
                    task_output = task_output.last_hidden_state.detach()  # type: torch.Tensor  # [B, S, D]
                    attention_sum = batch['attention_mask'].to(self.device).sum(-1).unsqueeze(-1).repeat(1, 1, self.args.bert_config.hidden_size)
                    attention_mask = batch['attention_mask'].to(self.device).unsqueeze(-1).repeat(1, 1, self.args.bert_config.hidden_size)
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
            print(display_string)
        elif self.exp.mode == 'export':
            self.export()
        elif self.exp.mode == 'test':
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
