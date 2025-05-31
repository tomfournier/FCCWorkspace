import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--build",   help="Build the environment",          action='store_true')
parser.add_argument("--shell",   help="Make a shell script",            action='store_true')
parser.add_argument("--fcc",     help="Set up FCCAnalyses",             action='store_true')
parser.add_argument("--combine", help="Set up Combine",                 action='store_true')
parser.add_argument("--local",   help="Set up local environment",       action='store_true')
parser.add_argument("--venv",    help="Install local environment",      action='store_true')
args = parser.parse_args()

#__________________________________________________________
def set_fcc():
    cmt_source = '# Source Key4hep and setup script\n'
    source = 'source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2024-03-10\n'
    source += 'cd FCCAnalyses\n'
    source += 'source ./setup.sh\n'
    if not args.build:
        source += 'cd ..\n\n'
    os.system(source)

    cmt_export = '# Set LOCAL_DIR to the directory of this script\n'
    export = 'export LOCAL_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)\n'
    os.system(export)

    cmt_build = '# Check if the build directory exists'
    build = 'cd FCCAnalyses\n'
    build += 'fccanalysis build -j 8\n'
    build += 'cd ..\n'
    if not os.path.exist('FCCAnalyses/build') or args.build:
        os.system(build)
    else:
        print('FCCAnalyses is already built')
        print('If you want to rebuild it, rerun setup.py with --build argument')

    shell_build = 'if [ ! -d "build" ]; then\n\t'
    shell_build += 'fccanalysis build -j 8\n'
    shell_build += 'else\n\t'
    shell_build += 'echo "FCCAnalyses is already built."\n\t'
    shell_build += 'echo "If you want to rebuild it, please run:"\n\t'
    shell_build += 'echo "cd FCCAnalyses"'
    shell_build += 'echo "source ./setup.sh"\n\t'
    shell_build += 'echo "fccanalysis build -j 8"\n\t'
    shell_build += 'echo "cd .."\n'
    shell_build += 'fi\n\n'
    shell_build += 'cd ..\n\n'
    if args.shell:
        with open('set_FCC.sh', 'w') as f:
            f.write('#!/bin/bash\n\n')
            f.write(cmt_source)
            f.write(source+"\n")
            f.write(cmt_build)
            f.write(shell_build)
            f.write(cmt_export)
            f.write(export)
            f.close()

#__________________________________________________________
def set_combine():
    source = 'cd HiggsAnalysis/CombinedLimit\n'
    source += '. env_standalone.sh\n'
    source += 'cd ../../\n'

    os.system(source)
    if args.shell:
        with open('setup_Combine.sh', 'w') as f:
            f.write(source)
            f.close()

#__________________________________________________________
def set_local():
    local = 'eval $(pdm venv activate)\n'
    os.system(local)
    if args.shell:
        with open('setup_local.sh', 'w') as f:
            f.write(local)
            f.close()

#__________________________________________________________
def install_venv():
    export = 'export ROOTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"\n'
    os.system(export)

    python = input('Do you use python or python3 as command? (answer python or python3):')
    latest_pip = input('Do you have the latest pip? (answer yes or no like wrotten):')
    if str(latest_pip)=='no':
        print('> getting latest pip')
        pip = f'{python} -m pip install --upgrade pip\n'
        os.system(pip)

    PDM = input('Do you have the pdm module? (answer yes or no):')
    if str(PDM)=='no':
        print('Installing pdm')
        install = 'pip install pdm\n'
        os.system(install)

    print('> initiating environment')
    print('> choose the environment that start with /cvmfs')
    print('> please choose venv as environment name')
    init = 'pdm init\n'
    os.system(init)

    print('> activating environment')
    acti = 'eval $(pdm venv activate)'
    acti += 'pdm config --local venv.with_pip true'
    os.sytem(acti)

    package = input('Do you have packages that you want to add to the environment?')
    if package!='':
        os.sytem(f'pdm add {package}')
        os.system('pdm export --format=requirements --without-hashes > requirements.txt')
        print('> required package are displayed in requirement.txt')
    
    inst = 'pdm install'
    os.system(inst)

    shell = '#!/bin/bash\n'
    shell += 'export ROOTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"\n\n'