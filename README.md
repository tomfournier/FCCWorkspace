# FCCWorkspace

This repository is made to study the ZH cross-section measurement at FCC-ee. It is greatly inspired by [FCCWorkplace](https://github.com/Ang-Li-93/FCCWorkplace) by Ang Li and [FCCPhysics](https://github.com/jeyserma/FCCPhysics) by Jan Eysermans and use the [FCCAnalyses](https://github.com/HEP-FCC/FCCAnalyses/tree/pre-edm4hep1) and [HiggsAnalysis-CombinedLimit](https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit) frameworks to do the events selection and the cross-section fit.

This repository is made to be installed in `lxplus` from CERN. Be careful to put it in the `/eos` storage as `/afs` has a small storage.

## Cloning the repository

To clone the repository, use the following command:

```shell
git clone --recursive https://github.com/tomfournier/FCCWorkspace.git
cd FCCWorkspace
git submodule update --remote
```

## FCCAnalyses

Then you will have to build `FCCAnalyses` to use its framework. First be sure that you are in the branch `pre-edm4hep1`, if you are not there will be problems when doing the analysis. To verify this you just have to use the command

```shell
cd FCCAnalyses
git branch
```

and verify that you see something like this:

```terminal
master
* pre-edm4hep1
```

If you are not in the good branch, you just have to execute this command:

```shell
git checkout pre-edm4hep1
```

When you are sure to be in the good branch, you can start building `FCCAnalyses`. To do this you have to source `setup/FCCAnalyses.sh` by using the following command (be sure to be in `FCCWorkspace` folder):

```shell
source setup/FCCAnalyses.sh
```

When you execute the command, the compiling will start and take a few minutes to setup `FCCAnalyses`. When it's done you can start doing the events selection. The next time you login, you also have to source `setup/FCCAnalyses.sh` but this time it won't recompile `FCCAnalyses` so it won't take this much time.

## Combined-Limit

To build `Combined-Limit` you will not have to verify in which branch you are as the master branch's building is working. To build you just have to run the following command and wait patiently for the building to be done, it will also take a few minutes.

```shell
source setup/CombinedLimit.sh
```

After each login you'll just have to source `setup/CombinedLimit.sh` by using the previous command but this time it won't recompile `Combined-Limit` so it won't take much time.

## Recompiling FCCAnalyses and Combined-Limit

If you applied modifications to either `FCCAnalyses` or `Combined-Limit`, you will have to recompile the repositories. To do this you can either execute:

```shell
source setup/FCCAnalyses.sh build
source setup/CombinedLimit.sh build
```

if you just modified a file. But if you added or removed a file, you have to recompile from scratch by executing the following:

```shell
source setup/FCCAnalyses.sh rebuild
source setup/CombinedLimit.sh rebuild
```

This will remove the `build/` (and `install/` for `FCCAnalyses`) folder to compile from scratch.

Normally you won't have to modify `Combined-Limit` but you may encounter this situation for `FCCAnalyses`. IF you are in a situation where you have to compile the repository of either `FCCAnalyses` or `Combined-Limit`, the scripts to set them up make alias to not have to redo the command each time.

## Careful

Be careful to use a different terminal to use `CombinedLimit` and `FCCAnalyses` as they are not compatible.

## Local environment

It is possible but not mandatory to make a local environment by sourcing `setup/install_env.sh` by using the following command:

```shell
source setup/install_env.sh <env_name>
```

with `<env_name>` either `fccanalysis`, `combined-limit`, `default` or nothing. If you don't put any argument, `default` will be taken. These 3 arguments are possible in case you need different environments for `FCCAnalyses` and `Combined-Limit`, the default one is there just in case. If you need more environments, you can modify `setup/install_env.sh` to add a new argument, you'll just have to follow the syntax of the script.

Since these environment use `pip` to install the modules, `ROOT` won't directly available which will cause conflict if you use these environment on a `Jupyter` notebook in VSCode for example as VSCode won't detect `ROOT`. It is thus recommended to use these environment if you already setup `ROOT` before starting VSCode or if you use this environment by executing your scripts on the terminal.

## VSCode setup

If you want to use VSCode to use this repository, it is recommended that you sourced `ROOT` before executing VSCode as it won't detect `ROOT` otherwise. `.vscode` folder were put in `FCCWorkspace/` and `analysis/ZH/xsec/` for `C/C++` and `python` IntelliSense.

Workspace specific settings were written in `fccanalysis.code-workspace` and `combined-limit.code-workspace` for `FCCAnalyses` or `Combined-Limit` specific IntelliSense. You can choose to put these settings in `settings.json` if you only need only one specific IntelliSense.

If you find a way to have workspace specific settings in `settings.json` for `python`. Contact me as I would be very interested in it.

### Remote-SSH with VSCode

As this repository was made to be used on `lxplus`, you will probably need to use `Remote-SSH` connection if you want to use it with VSCode. For this to work, it is recommended to have a `~/.ssh/config` file with these parameters:

```yaml
Host *
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
  
  ServerAliveInterval 300

  ControlMaster auto
  ControlPersist 20m
  ConnectTimeout 15
  ControlPath ~/.ssh/%r@%h:%p
  
  ForwardAgent yes
  ForwardX11Trusted yes
  
  TCPKeepAlive yes
  XAuthLocation /opt/X11/bin/xauth

  GSSAPIAuthentication yes
  GSSAPIDelegateCredentials yes
```

and to have different `Host` for connections to work with `FCCAnalyses` or `Combined-Limit`. For example, you can use this:

```yaml
Host fccanalysis
  HostName lxplus.cern.ch
  User <username>

Host combined-limit
  HostName lxplus.cern.ch
  User <username>
```

so that on the `Remote-SSH` tab, the workspace specific configurations are well separated. You can also add workspace specific arguments to the two `Host` more easily.

As `/afs` has a small storage limit, it is recommended to do a symlink to `/eos` storage beforehand on a terminal SSH connection and to use these parameters in `Remote-SSH`:

```json
"remote.SSH.enableAgentForwarding": true,
"remote.SSH.lockfilesInTmp": true,
"remote.SSH.serverInstallPath": {
    "<Host1>": "/path/to/symlink/to/eos/storage/to/path/install",
    "<Host2>": "/another/path/to/symlink/to/eos/storage/to/path/install",
}
```

The most important setting being the install path. This parameter permit you to choose the default installation path of `.vscode-server` with the extensions that VSCode use. This folder can be quite heavy and thus put constrain to `/afs`. That's why the symlink is heavily recommended in this case.

A more practical thing in having different `Host` is to be able to choose different installation paths for different workspace in case you use incompatible extensions.

### VSCode extensions

Another important point to confortably run this repository is the extensions needed. There are not much constraints in this case as this repository only need a few extensions for the experience in VSCode to be pleasant.

Here is a list of the extensions needed:

- `ms-python.python` (VSCode's python extension)
- `ms-python.flake8` (Python linting)
- `ms-python.autopep8` (Python formatter)
- `ms-python.vscode-pylance` (Python langage support)
- `albertopdrf.root-file-viewer` (To directly browse `.root` file)
- `clangd` (C/C++ linting and formatting)

Flake8 and autopep8 are not necessary to run the repository but they are good to keep the code clean and structured. A lot of flake8 errors and warnings are ignored in the settings, you can add or remove some if you want.

Python extension is obviously important to run the repository and Pylance is generally automatically installed with it. 

Clangd is used if you want to look at the `C/C++` file in `FCCAnalyses` and `Combined-Limit`. To have the IntelliSense, `setup/FCCAnalyses.sh` and `setup/CombinedLimit.sh` ask `CMake` to make a `compile_commands.json` during the compilation by using this argument `-DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE`. You can remove it if you don't want to use Clangd or won't use C/C++ files. 

Note that `functions/functions.h` will raise a lot of IntelliSense error as this file is not included in the build. If you know a method to remove the warning while keeping IntelliSense, I would be very interested in it.

Finally, probably the most important extension is `root-file-viewer` that permit you to read `.root` files directly on VSCode and see the histograms and TTree distributions which can be useful to verify that your code works well.

## Conclusion

Normally you should be able to setup and run this repository with the instructions given earlier. If you have any idea to improve the repository or the instructions given here, don't hesitate to contact me.

For more details on how to run the analysis, I refer you to the corresponding `README.md`. If you don't find them clear enough, don't hesitate to improve them or to contact me for suggestions or questions.