import os
import sys
import glob
import json
import asyncio
import logging
import argparse
import datetime
from typing import Optional

from radical.asyncflow.logging import init_default_logger
from radical.asyncflow import RadicalExecutionBackend, WorkflowEngine

WFMS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = f'{WFMS_DIR}/../src'
TASKS_SUBMISSION_BATCH = 100

os.environ['RADICAL_CONFIG_USER_DIR'] = WFMS_DIR
# for debug purposes
os.environ['RADICAL_LOG_LVL'] = 'DEBUG'
os.environ['RADICAL_REPORT'] = 'TRUE'

logger = logging.getLogger(__name__)

def filter_input_images(images_dir: str, base_channel: str) -> list:
    output = []
    for image_path in glob.glob(f'{images_dir}/*'):
        f = os.path.basename(image_path).lower()
        if base_channel in f and f.endswith(('.png', '.tif', '.tiff')):
            output.append(image_path)
    return output


def datetime_now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class Pipeline:

    def __init__(self,
                 name: str,
                 config: dict,
                 flow: WorkflowEngine,
                 image_path: Optional[str] = None,
                 input_dir: Optional[str] = None,
                 output_dir: Optional[str] = None):

        self.flow = flow
        self.images = []
        self.name = name
        self.cfg = config['pipeline_cfg']

        if image_path:
            self.images.append(image_path)
        elif input_dir and os.path.isdir(input_dir):
            self.images.extend(filter_input_images(
                images_dir=input_dir, base_channel=self.cfg['base_channel']))

        self.output_dir = output_dir or \
                          self.cfg['output_dir'] or \
                          os.path.dirname(self.images[0])
        os.makedirs(self.output_dir, exist_ok=True)

        self.stage_id = 0

    
    def register_pipeline_tasks(self):
        """Register all pipeline tasks"""

        s1_task_desc = {"ranks": 1, "named_env": "rp",
                        "gpus_per_rank": 1,
                        "pre_exec": self.cfg['task_pre_exec'] or [],
                        "environment":self.cfg['task_environment'] or {}}

        s2_task_desc = {"ranks": 1, "named_env": "rp",
                        "pre_exec": self.cfg['task_pre_exec'] or [],
                        "environment":self.cfg['task_environment'] or {}}


        @self.flow.executable_task
        async def stage_1(image_path: str, task_description: dict = s1_task_desc):  # noqa: B006
            """Segmentation."""
            run_script = 'cell_segmentation_complete.py'
            arguments = ['--base_channel', self.cfg['base_channel'],
                        '--target_channel', self.cfg['target_channel'],
                        '--bbox_threshold', self.cfg['bbox_threshold'],
                        '--output_dir', self.cfg['output_dir'],
                        '--model_path', self.cfg['model_path']
            if self.cfg['save_bbox']:
                arguments += ['--save_bbox']

            cmd = f"python '$RP_PILOT_SANDBOX/{run_script} --image_path' {image_path}"
            cmd += ' '.join(arguments)

            return cmd

        @self.flow.executable_task
        def stage_2(task_description=s2_task_desc):   # noqa: B006
            """Analysis."""

            run_script = 'segmentation_analysis.py'
            arguments = ['--images_dir', self.output_dir,
                        '--base_channel', self.cfg['base_channel'],
                        '--target_channel', self.cfg['target_channel'],
                        '--week_name', self.cfg['week_name'],
                        '--plate_name', self.cfg['plate_name'],
                        '--plate_config', self.cfg['plate_config'],
                        '--feature_type', self.cfg['feature_type']]

            cmd = f"python $RP_PILOT_SANDBOX/{run_script}"
            cmd += ' '.join(arguments)


    async def run(self):
        """Main execution logic"""

        self.logger.pipeline_log(f"Pipeline {self.name} started")

        stage_1_submitted_tasks = []

        # Submit N tasks == N images per single pipeline
        for idx, image_path in enumerate(self.images):
            stage_1_submitted_tasks.append(self.stage_1(image_path))

        await asyncio.gather(*stage_1_submitted_tasks)

        stage_2_future = self.stage_2()

        results = await stage_2_future

        print(results)



# ------------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description='RADICAL-Pilot application for the Cell Painting Pipeline',
        usage='cell.rp.py [options]')
    parser.add_argument(
        '-w', '--work_dir',
        dest='work_dir',
        type=str,
        required=False,
        help='work space for RADICAL-Pilot session sandboxes')
    parser.add_argument(
        '-i', '--input_dir',
        dest='input_dir',
        type=str,
        required=False,
        help='directory path of input images')
    parser.add_argument(
        '-o', '--output_dir',
        dest='output_dir',
        type=str,
        required=False,
        help='directory path for output images')
    parser.add_argument(
        '-c', '--config_file',
        dest='config_file',
        type=str,
        required=True,
        help='configuration file with the run description')
    parser.add_argument(
        '-t', '--runtime',
        dest='runtime',
        type=int,
        help='requested runtime (min) for application to run',
        required=False)
    return parser.parse_args(sys.argv[1:])


async def main():

    args = get_args()

    init_default_logger(logging.INFO)

    config_file = args.config_file
    if '/' not in config_file:
        config_file = os.path.join(WFMS_DIR, config_file)

    with open(config_file, 'r') as f:
        config = json.load(f)

    if not len(config):
        raise ValueError(f'Config is empty (file: {config_file})')

    # adjust runtime
    if args.runtime:
        config['run_description']['runtime'] = int(args.runtime)

    input_dir = args.input_dir or config['pipeline_cfg']['input_dir']
    # NOTE: create a pipeline per directory with images, thus stage2 will be
    #       applied to all images generated from the same directory.
    input_dirs = [d for d in glob.glob(f'{input_dir}/*') if os.path.isdir(d)]
    input_dirs = input_dirs or [input_dir]

    pipes = []

    print(f'{datetime_now()} Processing {p.name} has started with ')

    backend = await RadicalExecutionBackend({'nodes': 1, 'resources': 'local.localhost'})
    flow = await WorkflowEngine.create(backend=backend)

    for pipe_id, input_dir in enumerate(input_dirs):
        p = Pipeline(name=f'pipe-{pipe_id}',
                     config=config,
                     flow=flow,
                     input_dir=input_dir,
                     output_dir=args.output_dir)

        p_task = p.run()
        pipes.append(p_task)

    results = await asyncio.gather(*pipes)

    print(f'All pipeline results: {results}')