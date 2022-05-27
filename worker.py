import argparse
import copy
import os

import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from loader.data import Data
from loader.task.base_task import BaseTask
from loader.task.bert.bert4rec_task import Bert4RecTask
from loader.task.utils.base_curriculum_mlm_task import BaseCurriculumMLMTask
from utils import metric
from utils.config_initializer import init_config
from utils.dictifier import Dictifier
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

        self.auto_model = self.data.model(
            device=self.device,
            model_init=self.data.model_init,
            task_initializer=self.data.task_initializer,
        )

        self.auto_model.to(self.device)
        self.print(self.auto_model.model.config)
        self.save_model = self.auto_model
        self.disable_tqdm = bool(self.exp.display.disable_tqdm)

        self.static_modes = ['export', 'dev', 'test']
        self.in_static_modes = self.exp.mode in self.static_modes or self.exp.mode.startswith('test')

        if self.in_static_modes:
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
        if not self.in_static_modes and not self.exp.load.load_model_only:
            load_status = True
            self.m_optimizer.load_state_dict(state_dict['optimizer'])
            self.m_scheduler.load_state_dict(state_dict['scheduler'])
        self.print('Load optimizer and scheduler:', load_status)

    def attempt_loading(self):
        if self.exp.load.load_ckpt:
            self._attempt_loading(self.exp.load.load_ckpt)

    def log_interval(self, epoch, step, task: BaseTask, loss):
        self.print[f'epoch {epoch}'](f"step {step}, task {task.name}, loss {loss.item():.4f}")

    def train(self, *tasks: BaseTask):
        self.print('Start Training')

        train_steps = len(self.data.train_set) // self.exp.policy.batch_size
        accumulate_step = 0
        assert self.exp.policy.accumulate_batch >= 1

        # t_loader = self.data.get_t_loader(task)
        self.m_optimizer.zero_grad()
        for epoch in range(self.exp.policy.epoch_start, self.exp.policy.epoch + self.exp.policy.epoch_start):
            loader = self.data.get_loader(self.data.TRAIN, *tasks).train()
            loader.start_epoch(epoch - self.exp.policy.epoch_start, self.exp.policy.epoch)
            self.auto_model.train()

            for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
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

            for task in tasks:
                avg_loss = self.dev(task=task)
                self.print[f'epoch {epoch}'](f"task {task.name}, loss {avg_loss:.4f}")

            if (epoch + 1) % self.exp.policy.store_interval == 0:
                epoch_path = os.path.join(self.args.store.ckpt_path, 'epoch_{}.bin'.format(epoch))
                state_dict = dict(
                    model=self.auto_model.state_dict(),
                    optimizer=self.m_optimizer.state_dict(),
                    scheduler=self.m_scheduler.state_dict(),
                )
                torch.save(state_dict, epoch_path)

        self.print('Training Ended')

    def dev(self, task: BaseTask, steps=None):
        avg_loss = torch.tensor(.0).to(self.device)
        self.auto_model.eval()
        loader = self.data.get_loader(self.data.DEV, task).eval()

        for step, batch in enumerate(tqdm(loader, disable=self.disable_tqdm)):
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

    def test__curriculum(self, task: BaseTask, metric_pool: metric.MetricPool):
        assert isinstance(task, BaseCurriculumMLMTask)

        loader = self.data.get_loader(self.data.TEST, task).test()

        for batch in tqdm(loader, disable=self.disable_tqdm):
            with torch.no_grad():
                output = self.auto_model(
                    batch=batch,
                    task=task,
                )
                task.test__curriculum(batch, output, metric_pool)

    def test__left2right(self, task: BaseTask, metric_pool: metric.MetricPool):
        assert isinstance(task, Bert4RecTask)

        test_depot = self.data.sets[self.data.TEST].depot
        dictifier = Dictifier(aggregator=torch.stack)

        with torch.no_grad():
            for sample in tqdm(test_depot, disable=self.disable_tqdm):
                sample = copy.deepcopy(sample)
                task.test__left2right(sample, self.auto_model, metric_pool, dictifier=dictifier)

    def test_center(self, handler, task: BaseTask):
        metric_pool = metric.MetricPool()
        metric_pool.add(metric.OverlapRate())
        metric_pool.add(metric.HitRate(), ns=self.exp.policy.n_metrics)
        metric_pool.add(metric.Recall(), ns=self.exp.policy.n_metrics)
        metric_pool.init()

        self.auto_model.eval()
        task.test()

        handler(task, metric_pool)

        metric_pool.export()
        for metric_name, n in metric_pool.values:
            if n:
                self.print(f'{metric_name}@{n:4d}: {metric_pool.values[(metric_name, n)]:.4f}')
            else:
                self.print(f'{metric_name}     : {metric_pool.values[(metric_name, n)]:.4f}')

    # def export(self):
    #     # bert_aggregator = BertAggregator(
    #     #     layers=self.exp.save.layers,
    #     #     layer_strategy=self.exp.save.layer_strategy,
    #     #     union_strategy=self.exp.save.union_strategy,
    #     # )
    #     features = torch.zeros(
    #         self.data.depot.sample_size,
    #         self.args.model_config.hidden_size,
    #         dtype=torch.float
    #     ).to(self.device)
    #
    #     for loader in [self.data.get_loader(self.data.TRAIN, self.data.non_task),
    #                    self.data.get_loader(self.data.DEV, self.data.non_task)]:
    #         for batch in tqdm(loader):
    #             with torch.no_grad():
    #                 task_output = self.auto_model(batch=batch, task=self.data.non_task)  # type: BertOutput
    #                 task_output = task_output.last_hidden_state.detach()  # type: torch.Tensor  # [B, S, D]
    #                 attention_sum = batch['attention_mask'].to(self.device).sum(-1).unsqueeze(-1).repeat(1, 1, self.args.model_config.hidden_size)
    #                 attention_mask = batch['attention_mask'].to(self.device).unsqueeze(-1).repeat(1, 1, self.args.model_config.hidden_size)
    #                 features[batch['append_info'][self.exp.save.key]] = (task_output * attention_mask).sum(1) / attention_sum
    #
    #     save_path = os.path.join(self.args.store.ckpt_path, self.exp.save.feature_path)
    #     np.save(save_path, features.cpu().numpy(), allow_pickle=False)

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
        # elif self.exp.mode == 'export':
        #     self.export()
        elif self.exp.mode.startswith('test'):
            handler = object.__getattribute__(self, self.exp.mode)

            if not self.exp.load.super_load:
                for task in tasks:
                    if task.name == 'non':
                        continue
                    self.test_center(handler, task)
            else:
                epochs = eval(self.exp.load.epochs)
                for epoch in epochs:
                    ckpt_base_path = self.exp.load.ckpt_base_path
                    self._attempt_loading(os.path.join(ckpt_base_path, f'epoch_{epoch}.bin'))

                    for task in tasks:
                        if task.name == 'non':
                            continue
                        self.test_center(handler, task)


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
