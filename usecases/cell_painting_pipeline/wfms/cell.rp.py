#!/usr/bin/env python3

import argparse
import glob
import os
import queue
import sys

from collections import defaultdict

import radical.pilot as rp
import radical.utils as ru

# for debug purposes
os.environ['RADICAL_LOG_LVL'] = 'DEBUG'
os.environ['RADICAL_REPORT']  = 'TRUE'

WFMS_DIR   = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = f'{WFMS_DIR}/../src'
IMAGES_DIR = f'{WFMS_DIR}/../sample_imgs'  # TODO: to be updated

CONDA_ENV = 've.cellsam'

RESOURCE_DESCRIPTION = {
    'resource'     : 'anl.polaris',
    'project'      : 'NNNNN',
    # https://docs.alcf.anl.gov/polaris/running-jobs/#queues
    'queue'        : 'debug',
    'nodes'        : 1,
    'runtime'      : 60,  # in minutes (== job-walltime)
    'input_staging': [f'{SRC_DIR}/*']
}

TASK_PRE_EXEC_ENV = [
    'module use /soft/modulefiles; module load conda',
    'eval "$(conda shell.posix hook)"',
    f'conda activate {CONDA_ENV}'
]


class ExecManager:

    def __init__(self, resource_description, work_dir=None):
        self.tasks_finished_queue = queue.Queue()

        # RADICAL-Pilot management components
        self._session = rp.Session()
        self._pmgr    = rp.PilotManager(self._session)
        self._tmgr    = rp.TaskManager(self._session)

        self._tmgr.register_callback(self.task_state_cb)

        # contains "radical.pilot.sandbox" with agent sandboxes per session
        resource_description['sandbox'] = os.path.abspath(work_dir or WFMS_DIR)
        self._pilot = self._pmgr.submit_pilots(
            rp.PilotDescription(resource_description))

        self._tmgr.add_pilots(self._pilot)
        self._pilot.wait(rp.PMGR_ACTIVE)

    def close(self):
        self._session.close(download=True)

    def submit_tasks(self, *args, **kwargs):
        self._tmgr.submit_tasks(*args, **kwargs)

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
        tds = []
        for run_script in ['cell_nucleus_segmentation.py',
                           'cell_segmentation.py']:
            for image_path in self.images:
                tds.append(rp.TaskDescription({
                    'uid'       : self.emgr.generate_task_uid(prefix=self.name,
                                                              stage_id=1),
                    'executable': 'python',
                    'arguments' : [f'$RP_PILOT_SANDBOX/{run_script}',
                                   '--image_path', image_path],
                    'pre_exec'  : TASK_PRE_EXEC_ENV
                    # TODO: resource requirements? 1 GPU per rank?
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
    return parser.parse_args(sys.argv[1:])


# ------------------------------------------------------------------------------
def main():

    args = get_args()
    exec_mgr = ExecManager(resource_description=RESOURCE_DESCRIPTION,
                           work_dir=args.work_dir)

    # NOTE: for this test example we create a pipeline per image, while
    #       the other option is to create a pipeline per directory with images.
    pipes = {}
    for image_path in glob.glob(f'{args.images_dir}/*'):
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

        # if there is no active tasks, then all pipelines finished
        if not sum(tasks_active.values()):
            break

    exec_mgr.close()


if __name__ == '__main__':
    main()

