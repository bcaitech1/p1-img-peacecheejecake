from stage.exec import Trainer
from stage.config import load_config_from_yaml
from stage.utils import empty_logs

import argparse
from importlib import import_module
from pathlib import Path



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train", action='store_true')
    argparser.add_argument("--valid", action='store_true')
    argparser.add_argument("--eval", action='store_true')
    argparser.add_argument("--config", type=str, default="")
    argparser.add_argument("--state_path", type=str, default="")
    argparser.add_argument("--empty_logs", action='sotre_true')

    args = argparser.parse_args()

    if args.mpty_logs:
        empty_logs()
    
    config = load_config_from_yaml(args.config)
    if args.state_path:
        config.model.state_path = args.state_path

    trainer = Trainer(config)

    if args.train:
        trainer.train_and_save()
    if args.valid:
        trainer.valid()
    if args.eval:
        trainer.infer_and_save_csv()
