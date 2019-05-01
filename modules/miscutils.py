import os, sys, json, copy, random, shutil, subprocess, time
import pandas as pd
import numpy as np
import fastai

''' misc utils '''

def fastai_version(min_version=None, allow_dev=True, print_minor_v=False):
    ''' check fastai version we're using; default return True if min_version met'''
    try:
        v = fastai.__version__
    except:
        try:
            import fastai
            v = fastai.__version__
        except:
            print('failed to extract fastai.__version__')
            return False

    major_v, middle_v, minor_v, *misc_v = v.split('.')

    if print_minor_v:
        print(major_v, middle_v, minor_v, *misc_v)
        return

    if not(allow_dev):
        if 'dev' in misc_v[0]:
            return False

    if int(minor_v) >= min_version:
        return True

    return False

def is_local(name='DESKTOP-5VTC260'):
    sys_vars = os.environ
    return ( sys_vars.get('NAME', '') == name
             or sys_vars.get('COMPUTERNAME', '') == name)


def move_file_obj(src, dest='new-models', patterns=['.pth']
                  ,max_files=10, dry_run=False, log=True):
    
    ''' src, dest are paths/path-strs
        patterns - list of str: omit `*`, every pattern, p, 
                   is implicitly `*p*`, all of individuals patterns
                   are eval'd as an AND operation on filename
        max_files - reduces mega cp mistake; set to larger num if needed
        dry_run - bool; don't actually cp; just print which files will cp
        log - bool; False to suppress results message
     '''
    
    def filter_func(p):
        for _pattern in patterns:
            if not(_pattern in str(p)):
                return False
        return True
    
    files = [e for e in os.listdir(src) if filter_func(e)]

    if len(files) > max_files:
        print('aborting - max_files is %i but files-found is %i' 
              % (max_files, len(files)) )
        print('try dry_run=True to see which [extra] files are being found')
        return

    if dry_run:
        print('attempting to copy: ', *files, sep='\n')
        return
    
    successes, fails = [], []
    for _i, _f in enumerate(files):
        fsrc = os.path.join(src, _f)
        try:
            shutil.copy(fsrc, dest)
            successes.append(str(fsrc))
        except:
            fails.append(str(fsrc))

    if not(log): return

    print('succesfully copied %i files, failed on %i files' % (len(successes), len(fails)))
    print('from %s to %s' % (str(src), str(dest)) )
    print('successes:', *successes, sep=' | ')
    print('fails:', *fails, sep=' | ')

    return


def scp_from_gcloud(src, dest):
    ''' src  - relative to [server] custom2 dir
        dest - relative to [local] custom2 dir
        Must be run from wsl kernel to access gcloud cmd
        no progress bar; can ls -l <copied_file> to see its size grow
    '''

    cmd = '''gcloud compute scp --zone us-west2-b
    jupyter@fastai-basic-gpu:tutorials/fastai/course-v3/nbs/custom2/%s
    %s ''' % (str(src), str(dest))
     
    cmd = cmd.replace('\n', ' ')
    
    # put in a temp bash file, to run the cmd:
    # https://stackoverflow.com/questions/51316937/cannot-use-gcloud-compute-ssh-command-in-python-subprocess/51354235
    fn = 'gcloud_cmd_tmp.sh'
    with open(fn, 'w') as f:
        f.write(cmd)
    
    try:
        dir0 = os.listdir(dest if dest != '' else '.')
    except Exception as e:
        print('error - dest is not found locally: ', *e.args)
        return

    print('running popen and waiting...')
    try:
        p = subprocess.Popen(cmd, 
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, 
                            shell=True
                            )
        
    except KeyboardInterrupt:
        print('keyboard interrupt - killing gcloud subproc')
        p.terminate()        
    except:
        print('unknown exception - exiting subproc early')

    ret_code = p.wait()

    if ret_code != 0:
        print('error - return code from subproc is %i' % ret_code)

    dir1 = os.listdir(dest if dest != '' else '.')

    new_dir = [e for e in dir1 if e not in dir0]

    if len(new_dir) == 0:
        print('error - no new files found in dest: %s' % dest)
    else:
        print('new files in dest (%s)' % dest)
        print(*new_dir, sep='\n')
    
    return

    
    