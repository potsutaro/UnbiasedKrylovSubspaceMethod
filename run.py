import argparse
import yaml

from analysis import data_bootstrap
from analysis import data_binnedbootstrap
from analysis import sweep_subtraction
from analysis import tgevp_EnergyVariance
from analysis import tgevp_Spectroscopy
from analysis import tgevp_EigenvalueExt

from analysis import gen_mock_Ct
from analysis import tgevp_EigenvalueExtNoiseless

from config import Config


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument("--task", required=True)

    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    if args.task == "data":
        # data_bootstrap.main(cfg)
        data_binnedbootstrap.main(cfg)
        sweep_subtraction.main(cfg)

    if args.task == "RankCheck":
        sweep_subtraction.main(cfg)
        tgevp_EnergyVariance.main(cfg)


    if args.task == "Spectroscopy":
        tgevp_Spectroscopy.main(cfg)


    if args.task == "EvExt":
        tgevp_EigenvalueExt.main(cfg)

    if args.task == "mock":
        gen_mock_Ct.main(cfg)
        # data_bootstrap.main(cfg)
        data_binnedbootstrap.main(cfg)
        sweep_subtraction.main(cfg)

    if args.task == "EvExtNoiseless":
        tgevp_EigenvalueExtNoiseless.main(cfg)


if __name__ == "__main__":
    main()
