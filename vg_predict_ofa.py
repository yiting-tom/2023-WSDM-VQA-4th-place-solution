from configs import paths, consts
from wsdm_data import utils

def main():
    cmd = (
        f"cd {paths.ROOT}/scripts"
        " && "
        "sh ./evaluate_vg.sh"
    )
    utils.run_and_wait(cmd)
    

if __name__ == "__main__":
    main()