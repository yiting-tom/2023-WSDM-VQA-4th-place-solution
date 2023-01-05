import logging

from configs import paths
from wsdm_data import utils


L: logging.Logger = logging.getLogger(logging.basicConfig(level=logging.INFO))

def main():
    cmd = (
        f"cd {paths.ROOT}/scripts"
        " && "
        "sh ./evaluate_vqa.sh"
    )
    utils.run_and_wait(cmd)

if __name__ == "__main__":
    main()
