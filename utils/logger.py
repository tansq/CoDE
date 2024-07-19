import tensorboardX

class Logger(object):
    def __init__(self, model_name, header):
        self.header = header
        self.writer = tensorboardX.SummaryWriter(model_name)

    def __del(self):
        self.writer.close() 

    def log(self, phase, values):
        episode = values['episode']

        for col in self.header[1:]:
            self.writer.add_scalar(phase + "/" + col, float(values[col]), int(episode))


if __name__ == '__main__':
    train_logger = Logger(model_name='log', header=['epoch', 'RevLoss', 'lr'])

    train_logger.log(phase="train", values={
        'epoch': 1,
        'RevLoss': format(2.4, '.4f'),
        'lr': 0.001
    })