import argparse
import json
import os
import subprocess

if __name__ == "__main__":
    """
    script to submit a job that starts a wandb sweep agent to the GPULab wall.

    make sure to copy the setup-keypoint-detector.sh and run-sweep-on-wall.sh files to the /project mount and to make them executable using chmod + x
    before using this script.

    For more information on job submission, see https://doc.ilabt.imec.be/ilabt/gpulab/cli.html
    """

    ## check if gpulab CLI was installed and .pem file is availabe
    try:
        subprocess.run(["gpulab-cli", "--cert", "gpulab.pem"], check=True)
    except Exception as e:
        print(e)
        raise ValueError(
            "Gpulab CLI tool not installed or Unencrypted Cert not available in same folder, see https://doc.ilabt.imec.be/ilabt/gpulab/cli.html#submitting-a-gpulab-job"
        )

    ## get params for the job.

    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_ID")
    parser.add_argument("--wandb_key", help="wand auth key string")
    parser.add_argument("--project", help="Gpulab project to mount and to submit job against")

    parser.add_argument("--memory", default=12, help="cpu memory [GB] to request")
    parser.add_argument("--cpus", default=4)

    args = vars(parser.parse_args())

    # create job request and submit
    job_request = {
        "name": "sweep",
        "request": {
            "docker": {
                "image": "gitlab+deploy-token-8:YHsqg1MHW-C4GJpr_B-D@gitlab.ilabt.imec.be:4567/tlips/gpulab/jupyterlab-pytorch:latest",
                "storage": [{"hostPath": "/project", "containerPath": "/project"}],
                "command": f"""/project/run-sweep-on-wall.sh {args["wandb_key"]} {args['sweep_ID']}""",
                "user": "root",
            },
            "resources": {"cpus": args["cpus"], "gpus": 1, "cpuMemoryGb": args["memory"], "clusterId": 4},
        },
        "description": "Wandb sweep ",
    }

    # workaround as cli tool seems to expect file.
    with open("job.json", "w") as file:
        file.write(json.dumps(job_request))

    os.system(f"""gpulab-cli --cert gpulab.pem submit --project {args["project"]} < job.json""")
