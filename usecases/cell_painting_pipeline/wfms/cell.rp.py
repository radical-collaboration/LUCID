#!/usr/bin/env python3

import argparse
import datetime
import glob
import os
import queue
import sys

from collections import defaultdict
from typing import Optional

import radical.pilot as rp
import radical.utils as ru

WFMS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = f'{WFMS_DIR}/../src'
TASKS_SUBMISSION_BATCH = 100

os.environ['RADICAL_CONFIG_USER_DIR'] = WFMS_DIR
# for debug purposes
os.environ['RADICAL_LOG_LVL'] = 'DEBUG'
os.environ['RADICAL_REPORT'] = 'TRUE'

# in case of automation to prepare resource configuration
#   os.system(f'mkdir -p {WFMS_DIR}/.radical/pilot/configs && '
#             f'cd {WFMS_DIR} && cp resource_bnl.json .radical/pilot/configs/')


def filter_images(images_dir: str) -> list:
    output = []
    for image_path in glob.glob(f'{images_dir}/*'):
        f = os.path.basename(image_path).lower()
        if 'ch2' in f and f.endswith(('.png', '.tif', '.tiff')):
            output.append(image_path)
    return output


def datetime_now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class ExecManager:

    def __init__(self, config: ru.TypedDict, work_dir: Optional[str] = None):

        self.cfg = config.exec_cfg
        self.tasks_finished_queue = queue.Queue()

        # RADICAL-Pilot management components
        self._session = rp.Session()
        self._pmgr    = rp.PilotManager(self._session)
        self._tmgr    = rp.TaskManager(self._session)

        self._tmgr.register_callback(self.task_state_cb)

        resource_description = ru.as_dict(config.run_description)
        resource_description.update(
            input_staging=glob.glob(f'{SRC_DIR}/*'),
            # contains "radical.pilot.sandbox" with agent sandboxes per session
            sandbox=os.path.abspath(work_dir or WFMS_DIR))
        self._pilot = self._pmgr.submit_pilots(
            rp.PilotDescription(resource_description))

        self._tmgr.add_pilots(self._pilot)
        self._pilot.wait(rp.PMGR_ACTIVE)

    def close(self):
        self._session.close(download=True)

    def submit_tasks(self, *args, **kwargs):
        return self._tmgr.submit_tasks(*args, **kwargs)

    def get_finished_task(self):
        output = None
        try:
            # task prefix (== pipeline name), task state
            output = self.tasks_finished_queue.get_nowait()
        except queue.Empty:
            pass
        return output

    def task_state_cb(self, task, state):
        if state not in rp.FINAL:
            # ignore all non-finished state transitions
            return
        prefix = task.uid.split('.', 1)[0]
        self.tasks_finished_queue.put([prefix, task.state])

    def generate_pipe_uid(self):
        return ru.generate_id('p%(item_counter)06d',
                              ru.ID_CUSTOM, ns=self._session.uid)

    def generate_task_uid(self, prefix, stage_id):
        prefix = prefix.replace('.', '_')
        return ru.generate_id(f'{prefix}.{stage_id}.%(item_counter)06d',
                              ru.ID_CUSTOM, ns=self._session.uid)


class Pipeline:

    def __init__(self,
                 emgr: ExecManager,
                 image_path: Optional[str] = None,
                 images_dir: Optional[str] = None,
                 output_dir: Optional[str] = None):

        self.emgr = emgr  # exec manager TODO: should it be isolated?
        self.name = self.emgr.generate_pipe_uid()

        self.images = []
        if image_path:
            self.images.append(image_path)
        if images_dir and os.path.isdir(images_dir):
            self.images.extend(filter_images(images_dir))

        self.output_dir = output_dir or os.path.dirname(self.images[0])
        os.makedirs(self.output_dir, exist_ok=True)

        self.stage_id = 0

    def submit_next(self):
        next_stage_id = self.stage_id + 1

        if next_stage_id == 2:
            pass  # extra operations before going to the next stage

        try:
            submit_stage_func = getattr(self, f'stage_{next_stage_id}')
        except AttributeError:
            return 0

        self.stage_id = next_stage_id
        # submit tasks from the next stage
        return submit_stage_func()

    def stage_1(self):
        """Segmentation."""

        # segmentation_scripts = ['cell_nucleus_segmentation.py',
        #                         'cell_segmentation.py']
        segmentation_scripts = ['cell_segmentation_complete.py']

        arguments = []
        if self.output_dir:
            arguments = ['--output_dir', self.output_dir]

        submitted_tasks = 0
        tds = []
        idx_final = len(self.images) - 1
        for idx, image_path in enumerate(self.images):
            for run_script in segmentation_scripts:
                tds.append(rp.TaskDescription({
                    'uid': self.emgr.generate_task_uid(prefix=self.name,
                                                       stage_id=1),
                    'executable': 'python',
                    'arguments': [f'$RP_PILOT_SANDBOX/{run_script}',
                                  '--image_path', image_path] + arguments,
                    'pre_exec': self.emgr.cfg.task_pre_exec or [],
                    'named_env': 'rp',
                    'ranks': 1,
                    'gpus_per_rank': 1
                }))
            if len(tds) >= TASKS_SUBMISSION_BATCH or idx == idx_final:
                submitted_tasks += len(self.emgr.submit_tasks(tds))
                del tds[:]

        return submitted_tasks

    def stage_2(self):
        """Analysis."""
        return 0


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
        '-i', '--images_dir',
        dest='images_dir',
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


# ------------------------------------------------------------------------------
def main():

    args = get_args()

    config_file = args.config_file
    if '/' not in config_file:
        config_file = os.path.join(WFMS_DIR, config_file)
    config = ru.TypedDict(ru.read_json(config_file))
    if not len(config):
        raise ValueError(f'Config is empty (file: {config_file})')

    # adjust runtime
    if args.runtime:
        config.run_description.runtime = int(args.runtime)

    exec_mgr = ExecManager(config=config, work_dir=args.work_dir)

    images_dir = args.images_dir or config.pipeline_cfg.images_dir
    output_dir = args.output_dir or config.pipeline_cfg.output_dir
    # NOTE: create a pipeline per directory with images, thus stage2 will be
    #       applied to all images generated from the same directory.
    images_dirs = [d for d in glob.glob(f'{images_dir}/*') if os.path.isdir(d)]
    images_dirs = images_dirs or [images_dir]
    pipes = {}
    for images_dir in images_dirs:
        p = Pipeline(emgr=exec_mgr,
                     images_dir=images_dir,
                     output_dir=output_dir)
        pipes[p.name] = p

    # start executing pipelines (submit stages 1)
    tasks_active = defaultdict(int)
    for pipe_name, pipe in pipes.items():
        # start each pipeline
        tasks_active[pipe_name] += pipe.submit_next()  # num submitted tasks
        if tasks_active[pipe_name]:
            print(f'{datetime_now()} Pipeline {pipe_name} has started with '
                  f'{len(pipe.images)} images')
            print(f'{datetime_now()} Pipeline {pipe_name} | '
                  f'stage {pipe.stage_id}')

    # loop to track the status of the executed tasks and to submit next stages
    if sum(tasks_active.values()):
        while True:
            task_labels = exec_mgr.get_finished_task()
            if task_labels is None:
                # no finished tasks
                continue

            pipe_name, task_state = task_labels
            tasks_active[pipe_name] -= 1
            if tasks_active[pipe_name]:
                # if there were submitted a group of tasks within a stage,
                # and some of that tasks are still running
                continue

            tasks_active[pipe_name] += pipes[pipe_name].submit_next()
            if tasks_active[pipe_name]:
                print(f'{datetime_now()} Pipeline {pipe_name} | '
                      f'stage {pipes[pipe_name].stage_id}')
            else:
                print(f'{datetime_now()} Pipeline {pipe_name} has finished')

            # if there are no active tasks, then all pipelines finished
            if not sum(tasks_active.values()):
                break

    exec_mgr.close()


if __name__ == '__main__':
    main()

