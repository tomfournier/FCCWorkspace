# FCCWorkspace

This repository is made to study the ZH cross-section measurement at FCC-ee. It is greatly inspired by [FCCWorkplace](https://github.com/Ang-Li-93/FCCWorkplace) by Ang Li and [FCCPhysics](https://github.com/jeyserma/FCCPhysics) by Jan Eysermans and use the [FCCAnalyses](https://github.com/HEP-FCC/FCCAnalyses/tree/pre-edm4hep1) and [HiggsAnalysis-CombinedLimit](https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit) frameworks to do the events selection and the cross-section fit.

## Installation and set-up

This section will describe step-by-step how to install and set-up the repository.

### Cloning the repository

To clone the repository, use the following command:

```shell
git clone --recursive https://github.com/tomfournier/FCCWorkspace.git
cd FCCWorkspace
git submodule update --remote
```

### FCCAnalyses

Then you will have to build `FCCAnalyses` to use its framework. First be sure that you are in the branch `pre-edm4hep1`, if you are not the build will not work. To verify this you just have to use the command

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

When you are sure to be in the good branch, you can start building `FCCAnalyses`. To do this you have to source `setup_FCCAnalyses.sh` by using the following command (be sure to be in `FCCWorkspace` folder):

```shell
source setup_FCCAnalyses.sh
```

When you execute the command, the building will start and take a few minute to setup `FCCAnalyses`. When it's done you can start doing the events selection. The next time you login, you also have to source `setup_FCCAnalyses.sh` but this time it won't rebuild `FCCAnalyses` so it won't take this much time.

### HiggsAnalysis-CombinedLimit

To build `HiggsAnalysis-CombinedLimit` you will not have to verify in which branch you are as the master branch's building is working. To build you just have to run the following command and wait patiently for the building to be done, it will also take a few minutes.

```shell
source setup_CombinedLimit.sh
```

After each login you'll just have to source `setup_CombinedLimit.sh` by using the previous command but this time it won't rebuild `HiggsAnalysis/CombinedLimit` so it won't take much time.

### Local environment

It is possible but not mandatory to make a local environment by sourcing `install_venv.sh` by using the following command:

```shell
source install_venv.sh
```

When running the command, `pdm` will ask you which environment to take, you will have to choose the one which start by `/cvmfs`, end by `python` and have `2024-03-10` in its path.

## Careful

Be careful to use a different terminal to use `CombinedLimit` and `FCCAnalyses` as they are not compatible

