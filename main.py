from train import Trainer, parse_args

if __name__ == '__main__':
    args = parse_args('fast_scnn', 'citys', 1024, 512, 'train')
    trainer = Trainer(args)
    trainer.train()
