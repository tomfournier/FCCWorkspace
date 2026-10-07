# FCCWorkspace

## First words

It is strongly recommended to read this `README.md` before setting up the repository, as it addresses some problems you may encounter.

Enjoy the read!

## Introduction

This repository is made to study the ZH cross-section measurement and Higgs mass measurement at FCC-ee. It is greatly inspired by [FCCWorkplace](https://github.com/Ang-Li-93/FCCWorkplace) by Ang Li and [FCCPhysics](https://github.com/jeyserma/FCCPhysics) by Jan Eysermans and use the [FCCAnalyses](https://github.com/HEP-FCC/FCCAnalyses/tree/pre-edm4hep1) and [HiggsAnalysis-CombinedLimit](https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit) frameworks to do the events selection and the fit.

This repository is made to be installed in `lxplus` from CERN. Be careful to put it in the `/eos` storage as `/afs` has a small storage.

## Cloning the repository

To clone the repository, use the following command:

```shell
git clone --recursive https://github.com/tomfournier/FCCWorkspace.git
cd FCCWorkspace
```

If you cloned the repository without the `--recursive` argument, or if you are
working with an existing clone whose submodules are not initialized, run:

```shell
git submodule update --init --recursive
```

## FCCAnalyses

Then you will have to build `FCCAnalyses` to use its framework. To do this you have to source `setup/FCCAnalyses.sh` by using the following command (be sure to be in `FCCWorkspace` folder):

```shell
source setup/FCCAnalyses.sh
```

When you execute the command, compilation will start and take a few minutes to set up `FCCAnalyses`. When it is finished, you can start performing the event selection. The next time you log in, you also have to source `setup/FCCAnalyses.sh`, but it will not recompile `FCCAnalyses` and will therefore take less time.

## Combined-Limit

To build `Combined-Limit` you just have to run the following command and wait patiently for the building to be done, it will also take a few minutes.

```shell
source setup/CombinedLimit.sh
```

After each login, you only have to source `setup/CombinedLimit.sh`. It will not recompile `Combined-Limit`, so it will not take much time.

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

Normally you will not have to modify `Combined-Limit`, but you may encounter this situation with `FCCAnalyses`. If you have to compile either repository frequently, the setup scripts create aliases so that you do not have to repeat the full command each time.

The aliases are displayed when you execute the setup scripts, and you can also find them by using the `alias` command in the terminal.

## Careful

Use separate terminals for `CombinedLimit` and `FCCAnalyses`, as their environments are not compatible. You can set up `FCCAnalyses` first and then set up `CombinedLimit` to perform the fit, but you will no longer be able to run `FCCAnalyses`-related scripts in that shell. Open a new shell for them.

## VSCode setup

If you want to use VS Code with this repository, it is recommended that you source `ROOT` before starting VS Code, as it will not detect `ROOT` otherwise. `.vscode` folders were placed in `FCCWorkspace/` and the analysis folders for C/C++ and Python IntelliSense.

Workspace-specific settings are provided in `workspaces/fccanalysis.code-workspace` and `workspaces/combined-limit.code-workspace` for `FCCAnalyses` and `Combined-Limit`, respectively. You can put these settings in `settings.json` if you only need one of the two configurations.

If you find a way to configure workspace-specific Python settings in `settings.json`, please let me know.

### Remote-SSH with VS Code

As this repository is intended to be used on `lxplus`, you will probably need to use a `Remote-SSH` connection with VS Code. For this to work, it is recommended to have a `~/.ssh/config` file with these parameters:

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

  StrictHostKeyChecking accept-new
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

### Possible OS incompatibility

Some settings, such as `UseKeychain` and `XAuthLocation`, are macOS-specific and may cause errors on Linux or Windows. Test the connection in a terminal before using VS Code.

If you encounter a problem on Linux or Windows, check these parameters:

- UseKeychain
- XAuthLocation
- ControlMaster
- ControlPersist
- ControlPath

If you still encounter a problem after removing the macOS-specific parameters, contact me so I can update the documentation.

### `/afs/` and `/eos/`

As `/afs` has a small storage limit, it is recommended to create a symlink to `/eos` first using a terminal SSH connection, and to use these parameters in `Remote-SSH`:

```json
"remote.SSH.enableAgentForwarding": true,
"remote.SSH.lockfilesInTmp": true,
"remote.SSH.serverInstallPath": {
    "<Host1>": "/path/to/symlink/to/eos/storage/to/path/install",
    "<Host2>": "/another/path/to/symlink/to/eos/storage/to/path/install",
}
```

The most important setting is the installation path. This parameter lets you choose the default installation path for `.vscode-server` and its extensions. This directory can be large and can therefore put a strain on `/afs`, which is why the symlink is strongly recommended.

A practical benefit of having different `Host` entries is that you can choose different installation paths for different workspaces if you use incompatible extensions.

**It is strongly recommended to create the symlinks directly in a terminal through the SSH connection before using VS Code, to avoid problems later.**

### Connection problems with Remote-SSH and VS Code

Sometimes you may encounter connection problems with VS Code. Here are some examples and their solutions. If you find a better solution, please let me know so I can update the documentation.

#### Loss of connection when saving a file

If your connection stops before VS Code can save your file, you may enter a loop of failed reconnection attempts, which can take a lot of time.

It is advisable to close the remote connection and prevent VS Code from trying to save the file. The loop is generally caused by VS Code repeatedly trying to save it, so stopping the save attempts may break the loop.

To reduce the risk of losing changes, enable **Auto Save** in the `File` menu of VS Code and choose an appropriate `files.autoSave` setting.

#### Failure of connection after recently closed remote connection

Sometimes you may have trouble reconnecting to `lxplus` after closing the connection for some time. In that case, you can close the multiplexed SSH connection by running this command in your local terminal, not in the remote terminal:

```shell
ssh -O exit <ssh-session>
```

with `<ssh-session>` the name of your `Host` in the `.ssh/config`.

#### Connection failure after a VSCode update

Sometime after an update, you can have a connection failure that does not come from the previous cases.

It may be caused by the `.vscode-server` directory that VS Code installs for the remote connection. This directory contains the files needed to establish the connection and the extensions installed in the remote VS Code environment.

After closing the remote connection, you can remove this directory with:

```shell
rm -rf path/to/directory/.vscode-server
```

VS Code will install `.vscode-server` from scratch the next time you connect. This also removes its remote extensions, which will need to be installed again.

#### Other

This list does not cover every possible case; it contains the recurring problems I have encountered. If you find another useful case, please let me know so I can update the documentation.

### VSCode extensions

Another important point for using this repository comfortably is installing the required extensions. There are not many constraints, as this repository only needs a few extensions for a pleasant VS Code experience.

Here is a list of the extensions needed:

- `ms-python.python` (VSCode's python extension)
- `ms-python.flake8` (Python linting)
- `ms-python.autopep8` (Python formatter)
- `ms-python.vscode-pylance` (Python language support)
- `albertopdrf.root-file-viewer` (To browse `.root` files directly)
- `clangd` (C/C++ linting and formatting)
- `mhutchie.git-graph` (For repository browsing)

Flake8 and autopep8 are not necessary to run the repository, but they help keep the code clean and structured. Many Flake8 errors and warnings are ignored in the settings; you can add or remove exclusions if needed.

The Python extension is important for running the repository, and Pylance is generally installed automatically with it.

Clangd can be used to work with C/C++ files in `FCCAnalyses` and `Combined-Limit`. To provide IntelliSense, `setup/FCCAnalyses.sh` and `setup/CombinedLimit.sh` ask CMake to create `compile_commands.json` during compilation using `-DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE`. You can remove this option if you do not use Clangd or C/C++ files.

Probably the most important extension is `root-file-viewer`, which lets you read `.root` files directly in VS Code and view histograms and TTree distributions. This can be useful for verifying that your code works correctly.

Last but not least, Git Graph is an extension you can use to view the history of your repository and its submodules. It is useful when checking your logs or commits.

## Conclusion

Normally you should be able to setup and run this repository with the instructions given earlier. If you have any idea to improve the repository or the instructions given here, don't hesitate to contact me.

For more details on how to run the analysis, refer to the corresponding `README.md` files in the relevant folders. If you do not find them clear enough, feel free to improve them or contact me with suggestions or questions.

## Authors & References

This analysis is part of the FCC physics program for precision Higgs measurements. 
It was written by Tom Fournier with the help from Ang Li and Jan Eysermans (see [Introduction](#Introduction)). There was also a contribution from Amaury Lhoste for the improvement of the ZH cross-section analysis.

---

**Last updated:** October 2026