# FCCWorkspace

## First words

It is advised to read this entire `README.md` file before setting up the repository, as it documents some problems you may encounter.

Enjoy the read!

## Introduction

This repository is made to study the ZH cross-section measurement and Higgs mass measurement at FCC-ee. It is greatly inspired by [FCCWorkplace](https://github.com/Ang-Li-93/FCCWorkplace) by Ang Li and [FCCPhysics](https://github.com/jeyserma/FCCPhysics) by Jan Eysermans and use the [FCCAnalyses](https://github.com/HEP-FCC/FCCAnalyses/tree/pre-edm4hep1) and [HiggsAnalysis-CombinedLimit](https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit) frameworks to do the events selection and the fit.

This repository is made to be installed in `lxplus` from CERN. Be careful to put it in the `/eos` storage as `/afs` has a small storage.

## Cloning the repository

To clone the repository, use the following command:

```shell
git clone --recursive https://github.com/tomfournier/FCCWorkspace.git
cd FCCWorkspace
git submodule update --init --recursive
```

## FCCAnalyses

Then you will have to build `FCCAnalyses` to use its framework. To do this, source `setup/FCCAnalyses.sh` with the following command (be sure to be in the `FCCWorkspace` folder):

```shell
source setup/FCCAnalyses.sh
```

When you execute the command, compilation will start and may take a few minutes while `FCCAnalyses` is set up. When it is done, you can start performing the event selection. The next time you log in, you will still have to source `setup/FCCAnalyses.sh`, but it will not recompile `FCCAnalyses` and should take less time.

## Combined-Limit

To build `Combined-Limit`, run the following command. The build may take a few minutes.

```shell
source setup/CombinedLimit.sh
```

After each login, you will have to source `setup/CombinedLimit.sh` again, but it will not recompile `Combined-Limit` and should take less time.

## Recompiling FCCAnalyses and Combined-Limit

If you applied modifications to either `FCCAnalyses` or `Combined-Limit`, you will have to recompile the repositories. To do this you can either execute:

```shell
source setup/FCCAnalyses.sh build
source setup/CombinedLimit.sh build
```

if you have only modified a file. If you have added or removed a file, recompile from scratch by executing the following:

```shell
source setup/FCCAnalyses.sh rebuild
source setup/CombinedLimit.sh rebuild
```

This removes the `build/` folder and, for `FCCAnalyses`, the `install/` folder before compiling from scratch.

Normally, you will not have to modify `Combined-Limit`, but you may encounter this situation with `FCCAnalyses`. If you have to compile either repository frequently, the setup scripts create aliases so that you do not have to repeat the full commands each time.

The aliases are displayed when you execute the setup scripts, and you can also find them by running the `alias` command in the terminal.

## Careful

Use separate terminals for `CombinedLimit` and `FCCAnalyses`, as their environments are not compatible. You can set up `FCCAnalyses` first and then set up `CombinedLimit` in the same terminal to perform the fit, but you will no longer be able to run `FCCAnalyses`-related scripts there; use a new shell for them instead.

## VSCode setup

If you want to use VS Code with this repository, it is recommended that you source `ROOT` before launching VS Code, as it will not detect `ROOT` otherwise. `.vscode` folders are included in `FCCWorkspace/` and the analysis folders for C/C++ and Python IntelliSense.

Workspace-specific settings are provided in `workspaces/fccanalysis.code-workspace` and `workspaces/combined-limit.code-workspace` for `FCCAnalyses` and `Combined-Limit`, respectively. You can put these settings in `settings.json` if you only need one of the two configurations.

If you find a way to have workspace-specific Python settings in `settings.json`, please let me know; I would be very interested.

### Remote-SSH with VS Code

As this repository is intended to be used on `lxplus`, you will probably need the Remote-SSH extension if you want to use it with VS Code. For this to work, it is recommended to have a `~/.ssh/config` file with these parameters:

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

Some settings are macOS-specific and may cause errors on Linux or Windows. You can check them by testing the connection in a terminal before using VS Code.

If you run into problems on Linux or Windows, check these parameters first:

- UseKeychain
- XAuthLocation
- ControlMaster
- ControlPersist
- ControlPath

If you still have problems after removing the unsupported parameters, please contact me so I can update the documentation.

### `afs/` and `eos/`

As `/afs` has a small storage limit, it is recommended to do a symlink to `/eos` storage beforehand on a terminal SSH connection and to use these parameters in `Remote-SSH`:

```json
"remote.SSH.enableAgentForwarding": true,
"remote.SSH.lockfilesInTmp": true,
"remote.SSH.serverInstallPath": {
    "<Host1>": "/path/to/symlink/to/eos/storage/to/path/install",
    "<Host2>": "/another/path/to/symlink/to/eos/storage/to/path/install",
}
```

The most important setting is the installation path. This parameter lets you choose where VS Code installs `.vscode-server` and its extensions. This folder can be quite large and can put a strain on `/afs`, which is why using a symlink is strongly recommended.

A practical benefit of having different `Host` entries is that you can choose different installation paths for different workspaces if you use incompatible extensions.

**It is strongly advised to set up the symlinks directly in a terminal through the SSH connection before using VS Code, to avoid problems later.**

### Connection problems with Remote-SSH and VS Code

Sometimes you may encounter connection problems with VS Code. Here are some examples I have encountered and their solutions. If you find a better solution, please contact me so I can update the documentation.

#### Loss of connection when saving a file

If your connection stops before VS Code can save your file, you may enter a loop of failed reconnection attempts that can take a lot of time.

It is advisable to close the remote connection and prevent VS Code from trying to save the file. The loop is generally caused by VS Code repeatedly trying to save it, so stopping the save attempts may allow you to exit the loop.

To reduce the risk of losing changes, enable `Auto Save` from the `File` menu in VS Code. This saves changes periodically rather than only when you save manually.

#### Connection failure after recently closing a remote connection

Sometimes, you may have problems reconnecting to `lxplus` after closing the connection for some time (probably for lunch break). In these cases, you can force the multiplexed SSH connection to close by running this command in your local terminal, not in the remote terminal:

```shell
ssh -O exit <ssh-session>
```

where `<ssh-session>` is the name of your `Host` entry in `.ssh/config`.

#### Connection failure after a VSCode update

Sometimes, an update may cause a connection failure unrelated to the previous cases.

This may be caused by the `.vscode-server` directory that VS Code installs for the remote connection. It contains the server and your extensions for the remote VS Code session.

You can remove this directory with the following command while the remote connection is closed:

```shell
rm -rf path/to/directory/.vscode-server
```

VS Code will then install `.vscode-server` from scratch the next time you connect. This also removes the remote extensions, which will need to be installed again.

#### Other

This does not cover every possible case, only the recurring problems I have encountered. If you find another case that could be useful to document, please contact me.

### VSCode extensions

Another important point for comfortably using this repository is installing the recommended extensions. There are not many requirements, as the repository only needs a few extensions for a pleasant VS Code experience.

Here is a list of the extensions needed:

- `ms-python.python` (VSCode's python extension)
- `ms-python.flake8` (Python linting)
- `ms-python.autopep8` (Python formatter)
- `ms-python.vscode-pylance` (Python language support)
- `albertopdrf.root-file-viewer` (To browse `.root` files directly)
- `clangd` (C/C++ linting and formatting)
- `mhutchie.git-graph` (For repository browsing)

Flake8 and autopep8 are not necessary to run the repository, but they help keep the code clean and structured. Many Flake8 errors and warnings are ignored in the settings; you can add or remove exclusions if needed.

The Python extension is important for working with the repository, and Pylance is generally installed automatically with it.

Clangd is useful if you want to work with the C/C++ files in `FCCAnalyses` and `Combined-Limit`. To provide IntelliSense, `setup/FCCAnalyses.sh` and `setup/CombinedLimit.sh` ask CMake to generate a `compile_commands.json` file during compilation by using `-DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE`. You can remove this option if you do not use Clangd or C/C++ files.

Probably the most important extension is `root-file-viewer`, which lets you read `.root` files directly in VS Code and view histograms and TTree distributions that can help verify that your code works correctly.

Last but not least, `Git Graph` can be used to view the logs of your repository and its submodules. It is useful when checking your history or commits.

## Conclusion

You should normally be able to set up and run this repository using the instructions above. If you have any ideas for improving the repository or its documentation, please contact me.

For more details on how to run the analyses, see the corresponding `README.md` files in their folders. If you do not find them clear enough, please improve them or contact me with suggestions and questions.