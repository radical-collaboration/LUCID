#!/usr/bin/env python3

import argparse
import glob
import os
import queue
import sys

from collections import defaultdict

import radical.pilot as rp
import radical.utils as ru

WFMS_DIR   = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = f'{WFMS_DIR}/../src'
IMAGES_DIR = f'{WFMS_DIR}/../sample_imgs'  # TODO: to be updated

os.environ['RADICAL_CONFIG_USER_DIR'] = WFMS_DIR
# for debug purposes
os.environ['RADICAL_LOG_LVL'] = 'DEBUG'
os.environ['RADICAL_REPORT']  = 'TRUE'

# in case of automation to prepare resource configuration
#   os.system(f'mkdir -p {WFMS_DIR}/.radical/pilot/configs && '
#             f'cd {WFMS_DIR} && cp resource_bnl.json .radical/pilot/configs/')


class ExecManager:

    def __init__(self, config_file, work_dir=None):

        if '/' not in config_file:
            config_file = os.path.join(WFMS_DIR, config_file)
        self.cfg = ru.TypedDict(ru.read_json(config_file))
        if not self.cfg:
            raise FileNotFoundError(f'Config file not found: {config_file}')

        self.tasks_finished_queue = queue.Queue()

        # RADICAL-Pilot management components
        self._session = rp.Session()
        self._pmgr    = rp.PilotManager(self._session)
        self._tmgr    = rp.TaskManager(self._session)

        self._tmgr.register_callback(self.task_state_cb)

        resource_description = ru.as_dict(self.cfg.run_description)
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

    def __init__(self, emgr, image_path=None, image_dir=None):
        self.emgr = emgr  # exec manager TODO: should it be isolated?
        self.name = self.emgr.generate_pipe_uid()

        self.images = []
        if image_path:
            self.images.append(image_path)
        if image_dir and os.path.isdir(image_dir):
            self.images.extend(glob.glob(os.path.join(image_dir, '*.png')))

        self.stage_id = 0

    def submit_next(self):
        next_stage_id = self.stage_id + 1

        if next_stage_id == 2:
            pass  # extra operations before going to the next stage

        try:
            submit_stage_func = getattr(self, f'stage_{next_stage_id}')
        except AttributeError:
            print(f'Pipeline {self.name} has finished')
            return 0

        self.stage_id = next_stage_id
        # submit tasks from the next stage
        return submit_stage_func()

    def stage_1(self):
        """Segmentation."""

        segmentation_scripts = ['cell_nucleus_segmentation.py',
                                'cell_segmentation.py']

        tds = []
        for run_script in segmentation_scripts:
            for image_path in self.images:
                tds.append(rp.TaskDescription({
                    'uid'           : self.emgr.generate_task_uid(
                                          prefix=self.name, stage_id=1),
                    'executable'    : 'python',
                    'arguments'     : [f'$RP_PILOT_SANDBOX/{run_script}',
                                       '--image_path', image_path],
                    'pre_exec'      : self.emgr.cfg.task_pre_exec or [],
                    'named_env'     : 'rp',
                    'ranks'         : 1,
                    'gpus_per_rank' : 1
                }))

        return len(self.emgr.submit_tasks(tds))

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
        '-c', '--config_file',
        dest='config_file',
        type=str,
        required=True,
        help='configuration file with the run description')
    return parser.parse_args(sys.argv[1:])


# ------------------------------------------------------------------------------
def main():

    args = get_args()
    exec_mgr = ExecManager(config_file=args.config_file,
                           work_dir=args.work_dir)

    images_dir = args.images_dir or IMAGES_DIR

    # NOTE: for this test example, we create a pipeline per image, while
    #       the other option is to create a pipeline per directory with images.
    pipes = {}
    for image_path in glob.glob(f'{images_dir}/*'):
        p = Pipeline(emgr=exec_mgr, image_path=image_path)
        pipes[p.name] = p

    # start executing pipelines (submit stages 1)
    tasks_active = defaultdict(int)
    for pipe_name, pipe in pipes.items():
        # start each pipeline
        tasks_active[pipe_name] += pipe.submit_next()  # num submitted tasks

    # loop to track the status of the executed tasks and to submit next stages
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

        # if there are no active tasks, then all pipelines finished
        if not sum(tasks_active.values()):
            break

    exec_mgr.close()


if __name__ == '__main__':
    main()

