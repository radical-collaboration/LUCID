#!/usr/bin/env python3

import os
import time
import random
import signal
import threading as mt
import argparse, sys
import json

import radical.pilot as rp
import radical.utils as ru


# # LUCID_Novel-lncRNA

# ## 1. Novel LncRNA Detection Pipeline
# ### 1.1. Top100.py
# *Example command prompt
# python Top100.py -i ./Data/LoRA_BERT/d0W1R1.csv -f ./Data/LoRA_BERT/d0W1R1.fasta -o d0W1R1_top100.fasta

# ### 1.2. BLAST_Target.py
# *Example command prompt
# python BLAST_Target.py -i ./Data/d0W1R1_BLAST.txt -o d0W1R1_novel.csv


# ## 2. Reverse: Calculating TPM/Counts from Target Sequence

# ### 2.1. TargetSequence_location.py
# *Example command prompt
# python TargetSequence_location.py -i ./Data/novel_lncRNA -o gene_location.csv


# ### 2.2. Location_StringTie.py
# *Example command prompt
# python Location_StringTie.py -i gene_location.csv -s ./Data/Strintie_output


# ------------------------------------------------------------------------------
#
class LUCID_RNA(object):

    # define task types (used as prefix on task-uid)
    TASK_TOP100 = 'top100'  # Task for Top100.py
    TASK_BLAST_TARGET = 'blast_target'  # Task for BLAST_Target.py
    TASK_TARGET_SEQ_LOCATION = 'target_seq_location'  # Task for TargetSequence_location.py
    TASK_LOCATION_STRINGTIE = 'location_stringtie'  # Task for Location_StringTie.py

    TASK_TYPES = [TASK_TOP100, TASK_BLAST_TARGET, 
                  TASK_TARGET_SEQ_LOCATION, TASK_LOCATION_STRINGTIE]  # List of task types for the workflow

    # --------------------------------------------------------------------------
    #
    def __init__(self):

        self.set_argparse()
 
        # control flow table
        self._protocol = {self.TASK_TOP100: self._control_top100,
                          self.TASK_BLAST_TARGET: self._control_blast_target,
                          self.TASK_TARGET_SEQ_LOCATION: self._control_target_seq_location,
                          self.TASK_LOCATION_STRINGTIE: self._control_location_stringtie}

        self._glyphs = {self.TASK_TOP100: 'T',
                        self.TASK_BLAST_TARGET: 'B',
                        self.TASK_TARGET_SEQ_LOCATION: 'S',
                        self.TASK_LOCATION_STRINGTIE: 'L'}

        # bookkeeping
        self._top100_tasks = 0
        self._blast_target_tasks = 0
        self._target_seq_location_tasks = 0
        self._location_stringtie_tasks = 0
        
        self._top100_tasks_max = self.args.num_top100
        self._blast_target_tasks_max = self.args.num_blast
        self._target_seq_location_tasks_max = self.args.num_target_seq
        self._location_stringtie_tasks_max = self.args.num_location_stringtie

        self._cores = self.args.num_cpus * self.args.num_nodes  # available resources
        self._cores_used = 0

        self._gpus = self.args.num_gpus * self.args.num_nodes  # available GPU resources
        self._gpus_used = 0

        self._lock = mt.RLock()
        self._tasks = {ttype: dict() for ttype in self.TASK_TYPES}
        self._final_tasks = list()

        # silence RP reporter, use own
        os.environ['RADICAL_REPORT'] = 'false'
        self._rep = ru.Reporter('lucid_lncrna')
        self._rep.title('LUCID_lncRNA')

        # RP setup
        self._session = rp.Session()
        self._pmgr = rp.PilotManager(session=self._session)
        self._tmgr = rp.TaskManager(session=self._session)

        pdesc = rp.PilotDescription({
            'resource': self.args.resource,
            'queue': self.args.queue,
            'runtime': self.args.runtime,
            'cores': self.args.num_cpus * self.args.num_nodes,
            'gpus': self.args.num_gpus * self.args.num_nodes,
            'project': self.args.project_id})
        print(pdesc)

        self._pilot = self._pmgr.submit_pilots(pdesc)

        self._tmgr.add_pilots(self._pilot)
        self._tmgr.register_callback(self._checked_state_cb)

    def set_argparse(self):
        parser = argparse.ArgumentParser(description="LUCID Novel lncRNA Detection Pipeline")

        # Add arguments for the script
        parser.add_argument('--resource', default='polaris',
                        help='the resource to use (e.g., polaris, theta, etc.)')
        parser.add_argument('--runtime', default='00:30:00',
                        help='the runtime for the pilot (e.g., 00:30:00)')
        parser.add_argument('--num_cpus', type=int, default=16,
                        help='number of CPU cores per node (default: 16)')
        parser.add_argument('--num_gpus', type=int, default=0,
                        help='number of GPUs per node (default: 0). Note: Ensure the resource supports GPU')
        parser.add_argument('--num_nodes', type=int, default=1,
                        help='number of nodes to use for the job (default: 1).')
        parser.add_argument('--num_top100', type=int, default=1,
                        help='number of Top100.py tasks to run (default: 1).')
        parser.add_argument('--num_blast', type=int, default=1,
                        help='number of BLAST_Target.py tasks to run (default: 1).')
        parser.add_argument('--num_target_seq', type=int, default=1,
                        help='number of TargetSequence_location.py tasks to run (default: 1)')
        parser.add_argument('--num_location_stringtie', type=int, default=1,
                        help='number of Location_StringTie.py tasks to run (default: 1)')
        parser.add_argument('--project_id', default='CSC249ADCD08',
                        help='Project ID for the resource allocation (default: CSC249ADCD08).')
        parser.add_argument('--queue', default='debug',
                        help='queue to use for the pilot (default: debug).')
        parser.add_argument('--work_dir', type=str, default=os.getcwd(),
                        help='working directory for the job (default: current working directory).')
        parser.add_argument('--input_csv', type=str, required=True,
                        help='input CSV file path for Top100.py')
        parser.add_argument('--input_fasta', type=str, required=True,
                        help='input FASTA file path for Top100.py')
        parser.add_argument('--blast_input', type=str, required=True,
                        help='input file for BLAST_Target.py')
        parser.add_argument('--novel_lncrna_dir', type=str, required=True,
                        help='directory with novel lncRNA data for TargetSequence_location.py')
        parser.add_argument('--stringtie_dir', type=str, required=True,
                        help='StringTie output directory for Location_StringTie.py')
    
        args = parser.parse_args()
        self.args = args

    # --------------------------------------------------------------------------
    #
    def __del__(self):
        self.close()

    # --------------------------------------------------------------------------
    #
    def close(self):
        if self._session is not None:
            self._session.close()
            self._session = None

    # --------------------------------------------------------------------------
    #
    def dump(self, task=None, msg=''):
        '''
        dump a representation of current task set to stdout
        '''

        # this assumes one core per task
        self._rep.plain('<<|')

        idle = self._cores

        n = len(self._tasks[self.TASK_TOP100])
        idle -= n
        self._rep.ok('%s' % self._glyphs[self.TASK_TOP100] * n)

        n = len(self._tasks[self.TASK_BLAST_TARGET])
        idle -= n
        self._rep.ok('%s' % self._glyphs[self.TASK_BLAST_TARGET] * n)
        
        n = len(self._tasks[self.TASK_TARGET_SEQ_LOCATION])
        idle -= n
        self._rep.ok('%s' % self._glyphs[self.TASK_TARGET_SEQ_LOCATION] * n)
        
        n = len(self._tasks[self.TASK_LOCATION_STRINGTIE])
        idle -= n
        self._rep.ok('%s' % self._glyphs[self.TASK_LOCATION_STRINGTIE] * n)

        self._rep.plain('%s' % '-' * idle +
                        '| %4d [%4d]' % (self._cores_used, self._cores))

        if task and msg:
            self._rep.plain(' %-15s: %s\n' % (task.uid, msg))
        else:
            if task:
                msg = task
            self._rep.plain(' %-15s: %s\n' % (' ', msg))

    # --------------------------------------------------------------------------
    #
    def start(self):
        '''
        submit initial set of tasks
        '''

        self.dump('Submit initial tasks')  

        # reset bookkeeping
        self._top100_tasks = 0
        self._blast_target_tasks = 0
        self._target_seq_location_tasks = 0
        self._location_stringtie_tasks = 0
        
        self._cores_used = 0
        self._gpus_used = 0
        self._tasks = {ttype: dict() for ttype in self.TASK_TYPES}
        
        # The workflow follows this sequence:
        # 1. Run Top100.py
        # 2. When Top100 is done, run BLAST_Target.py
        # 3. When BLAST_Target is done, run TargetSequence_location.py
        # 4. When TargetSequence_location is done, run Location_StringTie.py
        
        # Start with Top100.py
        self.run_top100(n=self.args.num_top100)

    # --------------------------------------------------------------------------
    #
    def stop(self):
        os.kill(os.getpid(), signal.SIGKILL)
        os.kill(os.getpid(), signal.SIGTERM)

    # --------------------------------------------------------------------------
    #
    def _get_ttype(self, uid):
        '''
        get task type from task uid
        '''
        ttype = uid.split('.')[0]
        assert ttype in self.TASK_TYPES, 'unknown task type: %s' % uid
        return ttype

    def _submit_task(self, ttype, args=None, n=1, cpu=1, gpu=0, argvals=''):
        '''
        submit 'n' new tasks of specified type
        '''
        assert ttype

        # NOTE: ttype can be a task description (or a list of those), or it can
        #       be a string.  In the first case, we submit the given
        #       description(s).  In the second case, we construct the task
        #       description from the remaining arguments and the ttype string.
        if isinstance(ttype, list) and isinstance(ttype[0], rp.TaskDescription):
            tds = ttype
        elif isinstance(ttype, rp.TaskDescription):
            tds = [ttype]
        else:
            raise TypeError('invalid task type %s' % type(ttype))

        with self._lock:
            tasks = self._tmgr.submit_tasks(tds)
            for task in tasks:
                self._register_task(task)

    # --------------------------------------------------------------------------
    #
    def _cancel_tasks(self, uids):
        '''
        cancel tasks with the given uids, and unregister them
        '''
        uids = ru.as_list(uids)

        # FIXME: does not work
        self._tmgr.cancel_tasks(uids)

        for uid in uids:
            ttype = self._get_ttype(uid)
            task = self._tasks[ttype][uid]
            self.dump(task, 'cancel [%s]' % task.state)
            self._unregister_task(task)

    # --------------------------------------------------------------------------
    #
    def _register_task(self, task):
        '''
        add task to bookkeeping
        '''
        with self._lock:
            ttype = self._get_ttype(task.uid)
            self._tasks[ttype][task.uid] = task

            cores = task.description['cpu_processes'] \
                  * task.description['cpu_threads']
            self._cores_used += cores
            # Update the GPU usage if applicable
            gpus = task.description.get('gpu_processes', 0)
            self._gpus_used += gpus

            # Update counters for task types
            if ttype == self.TASK_TOP100:
                self._top100_tasks += 1
            elif ttype == self.TASK_BLAST_TARGET:
                self._blast_target_tasks += 1
            elif ttype == self.TASK_TARGET_SEQ_LOCATION:
                self._target_seq_location_tasks += 1
            elif ttype == self.TASK_LOCATION_STRINGTIE:
                self._location_stringtie_tasks += 1

    # --------------------------------------------------------------------------
    #
    def _unregister_task(self, task):
        '''
        remove completed task from bookkeeping
        '''
        with self._lock:
            ttype = self._get_ttype(task.uid)

            if task.uid not in self._tasks[ttype]:
                return

            # removed tasks dont consume cores
            cores = task.description['cpu_processes'] \
                  * task.description['cpu_threads']
            self._cores_used -= cores
            # Update the GPU usage if applicable
            gpus = task.description.get('gpu_processes', 0)
            self._gpus_used -= gpus

            # remove task from bookkeeping
            self._final_tasks.append(task.uid)
            del self._tasks[ttype][task.uid]

    # --------------------------------------------------------------------------
    #
    def _state_cb(self, task, state):
        '''
        act on task state changes according to our protocol
        '''
        try:
            return self._checked_state_cb(task, state)
        except Exception as e:
            self._rep.error('\n\n---------\nexception caught: %s\n\n' % repr(e))
            try:
                self.close()  # Clean up resources before exit
            finally:
                self.stop()

    # --------------------------------------------------------------------------
    #
    def _checked_state_cb(self, task, state):
        # ignore all non-final state transitions
        if state not in rp.FINAL:
            return

        # ignore tasks which were already processed
        if task.uid in self._final_tasks:
            return

        # lock bookkeeping
        with self._lock:
            # raise alarm on failing tasks (but continue anyway)
            if state == rp.FAILED:
                self._rep.error('task %s failed: %s' % (task.uid, task.stderr))
                self.stop()

            # control flow depends on ttype
            ttype = self._get_ttype(task.uid)
            action = self._protocol[ttype]
            if not action:
                self._rep.exit('no action found for task %s' % task.uid)
            action(task)

            # remove final task from bookkeeping
            self._unregister_task(task)

    # --------------------------------------------------------------------------
    #
    def _control_top100(self, task):
        '''
        react on completed Top100 task
        '''
        self._top100_tasks -= 1
        
        # Generate output file name based on input file
        input_file = self.args.input_csv
        # Extract base name without extension
        base_name = os.path.basename(input_file).split('.')[0]
        top100_output = f"{base_name}_top100.fasta"
        
        if self._top100_tasks == 0:
            self.dump(task, 'completed, ALL Top100 tasks')
            # When all Top100 tasks are completed, start BLAST_Target tasks
            self.run_blast_target(n=self.args.num_blast, top100_output=top100_output)
        else:
            self.dump(task, 'completed, Top100 tasks continue')

    # --------------------------------------------------------------------------
    #
    def _control_blast_target(self, task):
        '''
        react on completed BLAST_Target task
        '''
        self._blast_target_tasks -= 1
        
        # Generate output file name based on input file
        input_file = self.args.blast_input
        # Extract base name without extension
        base_name = os.path.basename(input_file).split('.')[0]
        blast_output = f"{base_name}_novel.csv"
        
        if self._blast_target_tasks == 0:
            self.dump(task, 'completed, ALL BLAST_Target tasks')
            # When all BLAST_Target tasks are completed, start TargetSequence_location tasks
            self.run_target_seq_location(n=self.args.num_target_seq)
        else:
            self.dump(task, 'completed, BLAST_Target tasks continue')

    # --------------------------------------------------------------------------
    #
    def _control_target_seq_location(self, task):
        '''
        react on completed TargetSequence_location task
        '''
        self._target_seq_location_tasks -= 1
        
        if self._target_seq_location_tasks == 0:
            self.dump(task, 'completed, ALL TargetSequence_location tasks')
            # When all TargetSequence_location tasks are completed, start Location_StringTie tasks
            self.run_location_stringtie(n=self.args.num_location_stringtie)
        else:
            self.dump(task, 'completed, TargetSequence_location tasks continue')

    # --------------------------------------------------------------------------
    #
    def _control_location_stringtie(self, task):
        '''
        react on completed Location_StringTie task
        '''
        self._location_stringtie_tasks -= 1
        
        if self._location_stringtie_tasks == 0:
            self.dump(task, 'completed, ALL Location_StringTie tasks - Workflow complete!')
        else:
            self.dump(task, 'completed, Location_StringTie tasks continue')

    # --------------------------------------------------------------------------
    #
    def run_top100(self, n=1):
        '''
        Run Top100.py tasks
        '''
        with self._lock:
            tds = list()
            # Example command: python Top100.py -i ./Data/LoRA_BERT/d0W1R1.csv -f ./Data/LoRA_BERT/d0W1R1.fasta -o d0W1R1_top100.fasta
            
            # Generate output file name based on input file
            input_file = self.args.input_csv
            # Extract base name without extension
            base_name = os.path.basename(input_file).split('.')[0]
            output_file = f"{base_name}_top100.fasta"
            
            for _ in range(n):
                tds.append(rp.TaskDescription({
                         'pre_exec'     : ["module load conda", "conda activate lncrna_env"],
                         'uid'          : ru.generate_id(self.TASK_TOP100),
                         'cpu_processes': self.args.num_cpus,
                         'cpu_process_type' : None,
                         'cpu_threads'      : 1,
                         'cpu_thread_type'  : rp.OpenMP,
                         'gpu_processes'     : 0,
                         'gpu_process_type'  : None,
                         'executable'   : 'python',
                         'arguments'    : [f"{self.args.work_dir}/Top100.py",
                                          '-i', self.args.input_csv,
                                          '-f', self.args.input_fasta,
                                          '-o', output_file]}))

            self._submit_task(tds)

    # --------------------------------------------------------------------------
    #
    def run_blast_target(self, n=1, top100_output=None):
        '''
        Run BLAST_Target.py tasks
        '''
        with self._lock:
            tds = list()
            # Example command: python BLAST_Target.py -i ./Data/d0W1R1_BLAST.txt -o d0W1R1_novel.csv
            
            # Generate output file name based on input file
            input_file = self.args.blast_input
            # Extract base name without extension
            base_name = os.path.basename(input_file).split('.')[0]
            output_file = f"{base_name}_novel.csv"
            
            for _ in range(n):
                tds.append(rp.TaskDescription({
                         'pre_exec'     : ["module load conda", "conda activate lncrna_env"],
                         'uid'          : ru.generate_id(self.TASK_BLAST_TARGET),
                         'cpu_processes': self.args.num_cpus,
                         'cpu_process_type' : None,
                         'cpu_threads'      : 1,
                         'cpu_thread_type'  : rp.OpenMP,
                         'gpu_processes'     : 0,
                         'gpu_process_type'  : None,
                         'executable'   : 'python',
                         'arguments'    : [f"{self.args.work_dir}/BLAST_Target.py",
                                          '-i', self.args.blast_input,
                                          '-o', output_file]}))

            self._submit_task(tds)

    # --------------------------------------------------------------------------
    #
    def run_target_seq_location(self, n=1):
        '''
        Run TargetSequence_location.py tasks
        '''
        with self._lock:
            tds = list()
            # Example command: python TargetSequence_location.py -i ./Data/novel_lncRNA -o gene_location.csv
            
            for _ in range(n):
                tds.append(rp.TaskDescription({
                         'pre_exec'     : ["module load conda", "conda activate lncrna_env"],
                         'uid'          : ru.generate_id(self.TASK_TARGET_SEQ_LOCATION),
                         'cpu_processes': self.args.num_cpus,
                         'cpu_process_type' : None,
                         'cpu_threads'      : 1,
                         'cpu_thread_type'  : rp.OpenMP,
                         'gpu_processes'     : 0,
                         'gpu_process_type'  : None,
                         'executable'   : 'python',
                         'arguments'    : [f"{self.args.work_dir}/TargetSequence_location.py",
                                          '-i', self.args.novel_lncrna_dir,
                                          '-o', 'gene_location.csv']}))

            self._submit_task(tds)

    # --------------------------------------------------------------------------
    #
    def run_location_stringtie(self, n=1):
        '''
        Run Location_StringTie.py tasks
        '''
        with self._lock:
            tds = list()
            # Example command: python Location_StringTie.py -i gene_location.csv -s ./Data/Strintie_output
            
            for _ in range(n):
                tds.append(rp.TaskDescription({
                         'pre_exec'     : ["module load conda", "conda activate lncrna_env"],
                         'uid'          : ru.generate_id(self.TASK_LOCATION_STRINGTIE),
                         'cpu_processes': self.args.num_cpus,
                         'cpu_process_type' : None,
                         'cpu_threads'      : 1,
                         'cpu_thread_type'  : rp.OpenMP,
                         'gpu_processes'     : 0,
                         'gpu_process_type'  : None,
                         'executable'   : 'python',
                         'arguments'    : [f"{self.args.work_dir}/Location_StringTie.py",
                                          '-i', 'gene_location.csv',
                                          '-s', self.args.stringtie_dir]}))

            self._submit_task(tds)

# ------------------------------------------------------------------------------
#
if __name__ == '__main__':

    lucid_rna = LUCID_RNA()  # Create an instance of the LUCID_RNA class

    try:
        lucid_rna.start()

        while True:
            time.sleep(1)

    finally:
        lucid_rna.close()

# ------------------------------------------------------------------------------