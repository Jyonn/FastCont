from loader.task.pretrain_task import PretrainTask


class NonTask(PretrainTask):
    def _init_extra_module(self):
        return None

    def init_parallel(self):
        pass

    def produce_output(self, model_output: any, **kwargs):
        return model_output

    def rebuild_batch(self, batch):
        return batch

    def calculate_loss(self, batch, output, **kwargs):
        pass

    def __init__(self):
        super(NonTask, self).__init__(name='non')
