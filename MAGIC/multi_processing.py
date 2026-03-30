import time
from utils import *
import torch
import torch.multiprocessing as mp

class MultiProcessWorker(mp.Process):
    def __init__(self, id, trainer_maker, comm, seed, *args, **kwargs):
        self.id = id
        self.seed = seed
        self.trainer_maker = trainer_maker
        self.trainer = None
        super(MultiProcessWorker, self).__init__()
        self.comm = comm

    def _build_trainer(self):
        if self.trainer is not None:
            return
        try:
            self.trainer = self.trainer_maker(self.id + 1)
        except TypeError:
            self.trainer = self.trainer_maker()

    def run(self):
        torch.manual_seed(self.seed + self.id + 1)
        np.random.seed(self.seed + self.id + 1)
        self._build_trainer()

        while True:
            task = self.comm.recv()
            if type(task) == list:
                task, epoch = task

            if task == 'quit':
                return
            elif task == 'run_batch':
                batch, stat = self.trainer.run_batch(epoch)
                self.trainer.optimizer.zero_grad()
                s = self.trainer.compute_grad(batch)
                merge_stat(s, stat)
                self.comm.send(stat)
            elif task == 'send_grads':
                grads = []
                for p in self.trainer.params:
                    if p._grad is not None:
                        grads.append(p._grad.data)

                self.comm.send(grads)


class MultiProcessTrainer(object):
    def __init__(self, args, trainer_maker):
        self.comms = []
        # itself will do the same job as workers
        self.nworkers = args.nprocesses - 1
        for i in range(self.nworkers):
            comm, comm_remote = mp.Pipe()
            self.comms.append(comm)
            worker = MultiProcessWorker(i, trainer_maker, comm_remote, seed=args.seed)
            worker.start()
        try:
            self.trainer = trainer_maker(0)
        except TypeError:
            self.trainer = trainer_maker()
        self.is_random = args.random

    def quit(self):
        for comm in self.comms:
            comm.send('quit')

    def train_batch(self, epoch):
        # run workers in parallel
        for comm in self.comms:
            comm.send(['run_batch', epoch])

        # run its own trainer
        batch, stat = self.trainer.run_batch(epoch)
        self.trainer.optimizer.zero_grad()
        s = self.trainer.compute_grad(batch)
        merge_stat(s, stat)

        # check if workers are finished
        for comm in self.comms:
            s = comm.recv()
            merge_stat(s, stat)

        # Pull the current worker gradients every update instead of reusing
        # cached tensor references that may become stale on newer PyTorch.
        worker_grads = []
        for comm in self.comms:
            comm.send('send_grads')
            worker_grads.append(comm.recv())

        local_grads = []
        for p in self.trainer.params:
            if p._grad is not None:
                local_grads.append(p._grad.data)

        for i in range(len(local_grads)):
            for g in worker_grads:
                local_grads[i] += g[i]
            local_grads[i] /= stat['num_steps']

        self.trainer.optimizer.step()
        return stat

    def state_dict(self):
        return self.trainer.state_dict()

    def load_state_dict(self, state):
        self.trainer.load_state_dict(state)
