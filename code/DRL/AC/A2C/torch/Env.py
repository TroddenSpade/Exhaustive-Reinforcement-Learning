import torch.multiprocessing as mp

class Env(mp.Process):
    def __init__(self, id, create_env, pipe_end) -> None:
        super().__init__()
        self.id = id
        self.env = create_env()
        self.pipe_end = pipe_end

    def run(self):
        while True:
            cmd, kwargs = self.pipe_end.recv()
            if cmd == 'reset':
                self.pipe_end.send(self.env.reset(**kwargs))
            elif cmd == 'step':
                self.pipe_end.send(self.env.step(**kwargs))
            else:
                self.env.close()
                del self.env
                self.pipe_end.close()
                break