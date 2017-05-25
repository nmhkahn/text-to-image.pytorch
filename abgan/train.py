import argparse
import trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.0002)
    parser.add_argument("--beta1",
                        type=float,
                        default=0.5)
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999)
    parser.add_argument("--batch_size",
                        type=int,
                        default=64)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=50)
    parser.add_argument("--cuda",
                        action="store_true")
    parser.add_argument("--file_path",
                        type=str,
                        default="dataset/images.csv")

    return parser.parse_args()


def main(config):
    t = trainer.Trainer(config)
    # t.fit()

    t.load("model")

    import numpy as np
    z = np.random.randn(64, 128).astype(np.float32) * 2 - 1

    t.generate("fake", z)


if __name__ == "__main__":
    config = parse_args()
    main(config)
