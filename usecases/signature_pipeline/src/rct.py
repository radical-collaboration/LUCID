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


# ------------------------------------------------------------------------------
#
class LUCID_SIG(object):

    # define task types (used as prefix on task-uid)
    TASK_SIG_PROF_EXTRACT   = 'sig_prof_extract'  # Task for extracting signature profiles
    TASK_ANNOTATION_PREP   = 'annotation_prep'   # Task for downloading and preprocessing annotations
    TASK_GENOME_DOWNLOAD = 'genome_download'  # Task for downloading genome data
    TASK_MUTATION_PREP = 'mutation_prep'  # Task for preparing mutation data
    TASK_MUTATION_ANNOT = 'mutation_annot'  # Task for annotating mutation data

    TASK_TYPES     = [TASK_SIG_PROF_EXTRACT, TASK_ANNOTATION_PREP, 
                      TASK_GENOME_DOWNLOAD, TASK_MUTATION_PREP, TASK_MUTATION_ANNOT]  # List of task types for the workflow



    # --------------------------------------------------------------------------
    #
    def __init__(self):


        self.set_argparse()
 

        # control flow table
        self._protocol = {self.TASK_SIG_PROF_EXTRACT: self._control_sig_prof_extract,  # Control function for signature profile extraction 
                          self.TASK_ANNOTATION_PREP: self._control_annotation_prep,  # Control function for annotation prep
                          self.TASK_GENOME_DOWNLOAD: self._control_genome_download,  # Control function for genome download
                          self.TASK_MUTATION_PREP: self._control_mutation_prep,  # Control function for mutation prep
                          self.TASK_MUTATION_ANNOT: self._control_mutation_annot}  # Control function for mutation annotation


        self._glyphs   = {self.TASK_SIG_PROF_EXTRACT: 'S',  # Glyph for signature profile extraction tasks
                          self.TASK_ANNOTATION_PREP: 'A',  # Glyph for annotation preparation tasks
                          self.TASK_GENOME_DOWNLOAD: 'G',  # Glyph for genome download tasks
                          self.TASK_MUTATION_PREP: 'P',  # Glyph for mutation preparation tasks
                          self.TASK_MUTATION_ANNOT: 'M'}  # Glyph for mutation annotation tasks

        # bookkeeping
        self._sig_prof_extract = 0  # Counter for signature profile extraction tasks
        self._annotation_prep = 0  # Counter for annotation preparation tasks
        self._genome_download = 0  # Counter for genome download tasks
        self._mutation_prep = 0  # Counter for mutation preparation tasks
        self._mutation_annot = 0  # Counter for mutation annotation tasks
        
        self._sig_prof_extract_max = self.args.num_extact  # Max number of signature profile extraction tasks
        self._annotation_prep_max = self.args.num_annot_prep  # Max number of annotation preparation tasks
        self._genome_download_max = self.args.num_genome  # Max number of genome download tasks
        self._mutation_prep_max = self.args.num_mut_prep  # Max number of mutation preparation tasks
        self._mutation_annot_max = self.args.num_mut_annot  # Max number of mutation annotation tasks

        self._cores          = self.args.num_cpus * self.args.num_nodes  # available resources
        self._cores_used     =  0

        self._gpus           = self.args.num_gpus * self.args.num_nodes  #available Gpu resources
        self._gpus_used      = 0

        self._mutation_prep_started = False  # Flag to indicate if mutation preparation has started
        self._genome_download_started = False
        self._annot_prep_started = False
        self._sig_extract_started = False
        self._mutation_annot_started = False

        self._lock           = mt.RLock()
        self._tasks          = {ttype: dict() for ttype in self.TASK_TYPES}
        self._final_tasks    = list()

        # silence RP reporter, use own
        os.environ['RADICAL_REPORT'] = 'false'
        self._rep = ru.Reporter('lucid_sig')  # Name of the reporter)
        self._rep.title('LUCID_SIG')  # Title for the reporter

        # RP setup
        self._session = rp.Session()
        self._pmgr    = rp.PilotManager(session=self._session)
        self._tmgr    = rp.TaskManager(session=self._session)

        pdesc = rp.PilotDescription({
            'resource': self.args.resource,  # Resource to use (e.g., 'polaris')
            'queue'   : self.args.queue,
            'runtime' : self.args.runtime,  # Runtime for the pilot (e.g., '00:30:00')
            'cores'   : self.args.num_cpus * self.args.num_nodes,
            'gpus'    : self.args.num_gpus * self.args.num_nodes,
            'project' : self.args.project_id})
        print(pdesc)

        self._pilot = self._pmgr.submit_pilots(pdesc)

        self._tmgr.add_pilots(self._pilot)
        self._tmgr.register_callback(self._checked_state_cb)


    def set_argparse(self):
        parser = argparse.ArgumentParser(description="LUCID Signature Detection Pilot Script")

        # Add arguments for the script
        parser.add_argument('--resource', default='anl.polaris',
                        help='the resource to use (e.g., anl.polaris, ornl.frontier, etc.)')
        parser.add_argument('--runtime', type=int, default=60,
                        help='the runtime for the pilot (e.g., 60)')
        parser.add_argument('--num_cpus', type=int, default=32,
                        help='number of CPU cores per node (default: 32)')
        parser.add_argument('--num_gpus', type=int, default=0,
                        help='number of GPUs per node (default: 0). Note: Ensure the resource supports GPU')
        parser.add_argument('--num_nodes', type=int, default=1,
                        help='number of nodes to use for the job (default: 1). This should be set according to your resource limits')
        parser.add_argument('--num_annot', type=int, default=1,
                        help='number of sequence information annotation tasks to run (default: 1). This controls how many tasks will be submitted for sequence information annotation')
        parser.add_argument('--num_extact', type=int, default=1,
                        help='number of signature profile extraction tasks to run (default: 1). This controls how many tasks will be submitted for signature profile extraction. Adjust this based on your workload and resource availability')
        parser.add_argument('--num_annot_prep', type=int, default=1,
                        help='number of annotation preparation tasks to run (default: 1)')
        parser.add_argument('--num_genome', type=int, default=1,
                        help='number of genome download tasks to run (default: 1)')
        parser.add_argument('--num_mut_prep', type=int, default=1,
                        help='number of mutation preparation tasks to run (default: 1)')
        parser.add_argument('--num_mut_annot', type=int, default=1,
                        help='number of mutation annotation tasks to run (default: 1)')
        parser.add_argument('--project_id', default='hep-cce',
                        help='Project ID for the resource allocation (default: hep-cce). This should be set to your actual project ID for resource allocation purposes')
        parser.add_argument('--queue', default='debug',
                        help='queue to use for the pilot (default: debug). This should be set according to the queue policies of your resource. For example, Polaris uses "batch"')
        parser.add_argument('--work_dir', type=str, default=os.getcwd(),
                        help='working directory for the job (default: current working directory). This should be set to the directory where your scripts and input files are located')
        parser.add_argument('--genome_offline_path', type=str, default='{}/reference_genomes'.format(os.getcwd()),
                        help='Path to offline genome file for SigProfilerExtractor (default: empty, not used)')

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

        n     = len(self._tasks[self.TASK_SIG_PROF_EXTRACT])  # Number of signature profile extraction tasks
        idle -= n
        self._rep.ok('%s' % self._glyphs[self.TASK_SIG_PROF_EXTRACT] * n)
        
        n     = len(self._tasks[self.TASK_ANNOTATION_PREP])  # Number of annotation preparation tasks
        idle -= n
        self._rep.ok('%s' % self._glyphs[self.TASK_ANNOTATION_PREP] * n)
        
        n     = len(self._tasks[self.TASK_GENOME_DOWNLOAD])  # Number of genome download tasks
        idle -= n
        self._rep.ok('%s' % self._glyphs[self.TASK_GENOME_DOWNLOAD] * n)
        
        n     = len(self._tasks[self.TASK_MUTATION_PREP])  # Number of mutation preparation tasks
        idle -= n
        self._rep.ok('%s' % self._glyphs[self.TASK_MUTATION_PREP] * n)
        
        n     = len(self._tasks[self.TASK_MUTATION_ANNOT])  # Number of mutation annotation tasks
        idle -= n
        self._rep.ok('%s' % self._glyphs[self.TASK_MUTATION_ANNOT] * n)

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
        self._sig_prof_extract = 0  # Counter for signature profile extraction tasks
        self._annotation_prep = 0  # Counter for annotation preparation tasks
        self._genome_download = 0  # Counter for genome download tasks
        self._mutation_prep = 0  # Counter for mutation preparation tasks
        self._mutation_annot = 0  # Counter for mutation annotation tasks
        
        self._cores_used = 0
        self._gpus_used = 0
        self._tasks = {ttype: dict() for ttype in self.TASK_TYPES}  # Use TASK_TYPES directly
        
        #FIXME:
        # First download necesseary files
        # when they are done run Sig Profiler Extractor
        # when they are done run mutation prep and annotation prep
        # when they are done run mutation annotation

        # Start genome download first  Step 0
        self.run_genome_download(self.TASK_GENOME_DOWNLOAD, n=self.args.num_genome)
        # # TODO download the VCF file
        
        #Following is just to show what are the next steps:
        # # Then run annotation preparation Step 1
        #ex: self.run_annotation_prep(self.TASK_ANNOTATION_PREP, n=self.args.num_annot_prep)
        
        # # Run signature profile extraction  Step 1
        #ex: self.run_extraction(self.TASK_SIG_PROF_EXTRACT, n=self.args.num_extact)
        
        # # Run mutation preparation Step 2
        #ex: self.run_mutation_prep(self.TASK_MUTATION_PREP, n=self.args.num_mut_prep)
        
        # # Run mutation annotation Step 3
        #ex: self.run_mutation_annot(self.TASK_MUTATION_ANNOT, n=self.args.num_mut_annot)


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
                print("TASK has been submitted is \n" , task)
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
            task  = self._tasks[ttype][uid]
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
            gpus = task.description.get('gpu_processes', 0)  # Get GPU processes 
            self._gpus_used += gpus  # Update the GPU usage

            # Update counters for task types
            if ttype == self.TASK_SIG_PROF_EXTRACT:
                self._sig_prof_extract += 1
            elif ttype == self.TASK_ANNOTATION_PREP:
                self._annotation_prep += 1
            elif ttype == self.TASK_GENOME_DOWNLOAD:
                self._genome_download += 1
            elif ttype == self.TASK_MUTATION_PREP:
                self._mutation_prep += 1
            elif ttype == self.TASK_MUTATION_ANNOT:
                self._mutation_annot += 1


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
            gpus = task.description.get('gpu_processes', 0)  # Get GPU processes
            self._gpus_used -= gpus  # Update the GPU usage

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

        # this cb will react on task state changes.  Specifically it will watch
        # out for task completion notification and react on them, depending on
        # the task type.

      # if state in [rp.TMGR_SCHEDULING] + rp.FINAL:
      #     self.dump(task, ' -> %s' % task.state)

        # ignore all non-final state transitions
        if state not in rp.FINAL:
            return

        # ignore tasks which were already
        if task.uid in self._final_tasks:
            return

        # lock bookkeeping
        with self._lock:

            # raise alarm on failing tasks (but continue anyway)
            if state == rp.FAILED:
                self._rep.error('task %s failed: %s' % (task.uid, task.stderr))
                self.stop()

            # control flow depends on ttype
            ttype  = self._get_ttype(task.uid)
            action = self._protocol[ttype]
            if not action:
                self._rep.exit('no action found for task %s' % task.uid)
            action(task)

            # remove final task from bookkeeping
            self._unregister_task(task)


    # --------------------------------------------------------------------------
    #
    def _control_sig_prof_extract(self, task):
        '''
        react on completed signature profile extraction task
        '''
        self._sig_prof_extract -= 1  # Decrement the counter for signature profile extraction tasks
        
        if self._sig_prof_extract == 0 and self._annotation_prep == 0 and not self._mutation_prep_started:
            self._mutation_prep_started = True  # Set the flag to indicate mutation preparation has started
            self.run_mutation_prep(self.TASK_MUTATION_PREP, n=self.args.num_mut_prep)
            self.dump(task, 'completed, ALL Signature Profile Extraction and ALL Annotation Preparation')
        else:
            self.dump(task, 'completed, Signature Profile Extraction low')  # Log the status of signature profile extraction tasks


    # --------------------------------------------------------------------------
    #
    def _control_annotation_prep(self, task):
        '''
        react on completed annotation preparation task
        '''
        self._annotation_prep -= 1  # Decrement the counter for annotation preparation tasks
        
        if self._annotation_prep == 0 and self._sig_prof_extract == 0 and not self._mutation_prep_started:
            self.run_mutation_prep(self.TASK_MUTATION_PREP, n=self.args.num_mut_prep)
            self.dump(task, 'completed, ALL Annotation Preparation and ALL Signature Profile Extraction')
        else:
            self.dump(task, 'completed, Annotation Preparation low')

    # --------------------------------------------------------------------------
    #
    def _control_genome_download(self, task):
        '''
        react on completed genome download task
        '''
        self._genome_download -= 1  # Decrement the counter for genome download tasks
        
        if self._genome_download == 0:
            self.run_annotation_prep(self.TASK_ANNOTATION_PREP, n=self.args.num_annot_prep)
            self.run_extraction(self.TASK_SIG_PROF_EXTRACT, n=self.args.num_extact)
            self.dump(task, 'completed, ALL Genome Download')
        else:
            self.dump(task, 'completed, Genome Download low')

    # --------------------------------------------------------------------------
    #
    def _control_mutation_prep(self, task):
        '''
        react on completed mutation preparation task
        '''
        self._mutation_prep -= 1  # Decrement the counter for mutation preparation tasks
        
        if self._mutation_prep == 0:
            self.run_mutation_annot(self.TASK_MUTATION_ANNOT, n=self.args.num_mut_annot)
            self.dump(task, 'completed, ALL Mutation Preparation')
        else:
            self.dump(task, 'completed, Mutation Preparation low')

    # --------------------------------------------------------------------------
    #
    def _control_mutation_annot(self, task):
        '''
        react on completed mutation annotation task
        '''
        self._mutation_annot -= 1  # Decrement the counter for mutation annotation tasks
        
        if self._mutation_annot == 0:
            time.sleep(120)  # Wait for 120 seconds before closing the session
            self.close()  # Close the session when all mutation annotation tasks are done
            self.dump(task, 'completed, ALL Mutation Annotation')
        else:
            self.dump(task, 'completed, Mutation Annotation low')



    def run_extraction(self, ttype, n=1):
        self._sig_extract_started = True  # Set the flag to indicate signature profile extraction has started
        with self._lock:
            tds   = list()
            for _ in range(n):
                tds.append(rp.TaskDescription({
                         'pre_exec'     : ["module load python", "source /eagle/LUCID/okilic/SigDetect/source.me"],
                         'uid'          : ru.generate_id(ttype),
                         'cpu_processes': self.args.num_cpus,  # Number of CPU processes to use for the task
                         'cpu_process_type' : None,
                         'cpu_threads'      : 1,  # Number of CPU threads to use for the task
                         'cpu_thread_type'  : rp.OpenMP,
                         'gpu_processes'     : 0,
                         'gpu_process_type'  : None,
                         'executable'   : 'python',
                         'arguments'    : [ '{}/sigprofiler.py'.format(self.args.work_dir),
                                             '--input', './radiation_analysis_results/filtered_vcfs/',
                                             '--output', './radiation_analysis_results/filtered_vcfs/output/',
                                             '--project', 'test',  # Project name for SigProfilerExtractor, can be customized,
                                             '--reference', 'GRCh38'
                                             '--offline_path', '{}'.format(self.args.genome_offline_path)
                                            ]}))

            self._submit_task(tds)

    def run_annotation_prep(self, ttype, n=1):
        '''
        Run annotation preparation tasks 
        '''

        # ''' First check if annotation preparation has already run, if so, do not run again'''
        # if '{}/annotations'.format(self.args.work_dir) in os.listdir(self.args.work_dir):
        #     self._rep.info('Annotation preparation already completed, skipping...')
        #     lucid_sig._annot_prep_started = 1 
        #     lucid_sig._annotation_prep = 1
        #     return

        
        self._annot_prep_started = True  # Set the flag to indicate annotation preparation has started
        with self._lock:
            tds = list()
            for _ in range(n):
                tds.append(rp.TaskDescription({
                         'pre_exec'     : ["module load python", "source /eagle/LUCID/okilic/SigDetect/source.me"],
                         'uid'          : ru.generate_id(ttype),
                         'cpu_processes': self.args.num_cpus,
                         'cpu_process_type' : None,
                         'cpu_threads'      : 1,
                         'cpu_thread_type'  : rp.OpenMP,
                         'gpu_processes'     : 0,
                         'gpu_process_type'  : None,
                        #  'executable'   : 'python',
                         'executable'   : 'sleep',
                         'arguments'    : ['10']}))
                        #  'arguments'    : ['{}/annotation_preprocessing.py'.format(self.args.work_dir),
                        #                   '--build', 'hg38',
                        #                   '--annotation-dir', '{}/annotations'.format(self.args.work_dir)]}))

            self._submit_task(tds)

    def run_genome_download(self, ttype, n=1):
        '''
        Run genome download tasks
        '''
        self._genome_download_started = True  # Set the flag to indicate genome download has started
        with self._lock:
            tds = list()
            for _ in range(n):
                tds.append(rp.TaskDescription({
                         'pre_exec'     : ["module load python", "source /eagle/LUCID/okilic/SigDetect/source.me"],
                         'uid'          : ru.generate_id(ttype),
                         'cpu_processes': self.args.num_cpus,
                         'cpu_process_type' : None,
                         'cpu_threads'      : 1,
                         'cpu_thread_type'  : rp.OpenMP,
                         'gpu_processes'     : 0,
                         'gpu_process_type'  : None,
                        #  'executable'   : 'python',
                         'executable'   : 'sleep',
                        #  'arguments'    : ['{}/genome_download.py'.format(self.args.work_dir)]}))
                         'arguments'    : ['10']}))

            self._submit_task(tds)

    def run_mutation_prep(self, ttype, n=24):  # Set default n to 24 for 24 chromosomes
        '''
        Run mutation preparation tasks, one task per chromosome
        '''
        self._mutation_prep_started = True  # Set the flag to indicate mutation preparation has started
        with self._lock:
            tds = list()
            # Run for each chromosome (1-22, X, Y) individually
            chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y']
            
            # Create one task per chromosome
            for chrom in chromosomes:
                tds.append(rp.TaskDescription({
                         'pre_exec'     : ["module load python", "source /eagle/LUCID/okilic/SigDetect/source.me"],
                         'uid'          : ru.generate_id(ttype),
                         'cpu_processes': self.args.num_cpus,
                         'cpu_process_type' : None,
                         'cpu_threads'      : 1,
                         'cpu_thread_type'  : rp.OpenMP,
                         'gpu_processes'     : 0,
                         'gpu_process_type'  : None,
                         'executable'   : 'python',
                         'arguments'    : ['{}/mutation_preprocessing.py'.format(self.args.work_dir),
                                          '--chromosome', chrom,  # Just one chromosome per task
                                          '--output', '{}/processed_data'.format(self.args.work_dir),
                                          '--summary', '{}/summary_data'.format(self.args.work_dir)]}))

            # Only submit as many tasks as requested by the parameter (but default is now 24)
            self._submit_task(tds[:n])

    def run_mutation_annot(self, ttype, n=24):  # Set default n to 24 for 24 chromosomes
        '''
        Run mutation annotation tasks, one task per chromosome
        '''
        self._mutation_annot_started = True  # Set the flag to indicate mutation annotation has started
        with self._lock:
            tds = list()
            # Run for each chromosome (1-22, X, Y) individually
            chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y']
            
            # Create one task per chromosome
            for chrom in chromosomes:
                tds.append(rp.TaskDescription({
                         'pre_exec'     : ["module load python", "source /eagle/LUCID/okilic/SigDetect/source.me"],
                         'uid'          : ru.generate_id(ttype),
                         'cpu_processes': self.args.num_cpus,
                         'cpu_process_type' : None,
                         'cpu_threads'      : 1,
                         'cpu_thread_type'  : rp.OpenMP,
                         'gpu_processes'     : 0,
                         'gpu_process_type'  : None,
                         'executable'   : 'python',
                         'arguments'    : ['{}/mutation_annotation.py'.format(self.args.work_dir),
                                          '--chromosome', chrom,  # Just one chromosome per task
                                          '--input-dir', '{}/processed_data'.format(self.args.work_dir),
                                          '--output-dir', '{}/annotated_data'.format(self.args.work_dir),
                                          '--annotation-dir', '{}/annotations'.format(self.args.work_dir),
                                          '--build', 'hg38']}))

            # Only submit as many tasks as requested by the parameter (but default is now 24)
            self._submit_task(tds[:n])


# ------------------------------------------------------------------------------
#
if __name__ == '__main__':

    lucid_sig = LUCID_SIG()  # Create an instance of the LUCID_SIG class

    try:
        lucid_sig.start()

        while True:  
            # Only consider workflow complete if each stage started AND finished
            genome_done = lucid_sig._genome_download_started and lucid_sig._genome_download == 0
            annot_done = lucid_sig._annot_prep_started and lucid_sig._annotation_prep == 0
            extract_done = lucid_sig._sig_extract_started and lucid_sig._sig_prof_extract == 0
            mut_prep_done = lucid_sig._mutation_prep_started and lucid_sig._mutation_prep == 0
            mut_annot_done = lucid_sig._mutation_annot_started and lucid_sig._mutation_annot == 0
            
            # Check if we've reached the end of the workflow
            if genome_done and annot_done and extract_done and mut_prep_done and mut_annot_done:
                lucid_sig.dump("All workflows completed. Exiting...")
                break
                
            time.sleep(1)
        #   # ddmd.dump()
        #     time.sleep(1)

    finally:
        lucid_sig.close()


# ------------------------------------------------------------------------------
