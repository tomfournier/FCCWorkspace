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

and verify that you see somethinglike this:

```terminal
master
* pre-edm4hep1
```

If you are not you just have to do:

```shell
git checkout pre-edm4hep1
```

When you are sure to be in the good branch, you can start building `FCCAnalyses`. To do this you have to source `setup_FCCAnalyses.sh` by using the following command (be sure to be in `FCCWorkspace`):

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

# ZH cross-section analysis

The ZH cross-section analysis is in the `analysis/ZH/xsec` folder and separated in 4 phases and are explicitely noted by a number at the beginning of each folder, the first one concerns the production of MVA inputs for the training of a BDT. The second one concerns the training of the BDT and its evaluation. The third one concerns the production of histograms for the events selection. The fourth and last one concerns the fit of the ZH cross-section and the verification of the model-independence of the measurement by a bias test. 

A `userConfig.py` file is also present and contains the location of the output files, the samples names and other variables that are used across the analysis and centralized here so as to have to modify a variable only one time instead of going to each concerned files to modify it by hand. A description of the information in it is made in the following section.

## userConfig file

The `userConfig.py` file is divised in different zone delimited by headers that look like:

```
##################
##### HEADER #####
##################
```

### Parameters

The first header is for global parameters that you can import with the `import_module` function from `importlib` module in the analysis files. These parameters are put there so as to not have to put it by hand in each file.

### Location of files

This section is very important as it determines the structure of your output directory. The output directory is where all your data, plots, histograms, etc. will be put so it has to be well structured so that you can easily find what you want.

If this configuration does not suit you, you change it by modifying the path of the different variables in this header.

The current structure of the `xsec` directory is displayed below. To be easier to look at, not all the folders are displayed and only their function are displayed:

- `ecm` is the folder where you put all your files with a given center of mass energy
    - By default it is set to `240 GeV`
- `selection` is the folder where you put all you files with given selection 
    - There are several selections available and will be further explained below
- `final_state` is the folder where you put all your files with a given final state
    - For the moment only `ee` and `mumu` are available

```
├── 1-MVAInputs
├── 2-BDT
├── 3-Combine
├── 4-Fit
└── output
    ├── data
    │   ├── combine
    │   │   └── ecm
    │   │       └── selection
    │   │            └── final_state
    │   │               ├── bias
    │   │               │   ├── datacard
    │   │               │   ├── log
    │   │               │   ├── results
    │   │               │   │   ├── bias
    │   │               │   │   └── fit
    │   │               │   └── WS
    │   │               └── nominal
    │   │                   ├── datacard
    │   │                   ├── log
    │   │                   ├── results
    │   │                   └── WS
    │   ├── histograms
    │   │   ├── MVAFinal
    │   │   │   └── ecm
    │   │   │       └── final_state
    │   │   ├── preprocessed
    │   │   │   └── ecm
    │   │   │       └── selection
    │   │   └── processed
    │   │       └── ecm
    │   │           └── selection
    │   └── MVA
    │       └── ecm
    │           └── final_state
    │               └── selection
    │                   ├── BDT
    │                   ├── MVAInputs
    │                   └── MVAProcessed
    └── plots
        └── ecm
            └── final_state
                ├── evaluation
                │   └── selection
                ├── measurement
                │   └── selection
                │       ├── cutflow
                │       ├── higgsDecays
                │       │   ├── high
                │       │   ├── low
                │       │   └── nominal
                │       ├── makePlot
                │       │   ├── high
                │       │   ├── low
                │       │   └── nominal
                │       └── significance
                └── MVAInputs
                    └── selection

```

The output directory is separated in two folders: `data` and `plots`. In data, all your results that go from your histograms to your BDT will be put there. In plots, as its name says, all your plots will be put there.

The data directory is separated in 3 folders: `MVA`, `histograms` and `combine`. The MVA part is where all that concern the BDT: the training samples and the BDT itself will be put. For the histograms, as its name says, this is where all your histograms that you will produce across the analysis will be put. The combine part is where all that concern the fit of the cross-section and the bias test are.

#### Add a path

If you want to add a folder in the analysis, you can put it in this header. The name of the variable of the path to the folder that you want must start by `loc.`, you can choose an explicit after to help you remember which path the variable contains.

If your path require one of the variables explained before (`ecm`, `final_state` or `selection`) you have to put exactly their name in you path, the reason why is explained below.

### Functions to extract the path

The paths in the previous header follow a rule of putting `selection`, `ecm` or `final_state` in place of the selection, center of mass energy or final state that you use to be more general and because there are two functions in this header that are made to convert the general path in the path with the selection, center of mass energy or final state that you want.

#### get_loc

`get_loc` is precisely the function that convert the general path to the specific that you want depending on the situation. Here is the function:

```python
def get_loc(path: str, cat: str , ecm: int, sel: str) -> str:
    path = path.replace('final_state', cat)
    path = path.replace('ecm', str(ecm))
    path = path.replace('selection', sel)
    return path
```

The function takes as argument the path (variable starting by `loc.`), the final state (`cat`), the center of mass energy (`ecm`) and the selection (`sel`)

It can be bothersome to put the selection by hand in each file when calling `get_loc`. To simplify this, we use the function `select` that returns the selection that you want.

```python
def select(recoil120: bool, miss: bool = False, bdt: bool = False) -> str:
    sel = 'Baseline'
    if recoil120: sel += '_120'
    elif miss: sel += '_miss'
    elif bdt: sel += '_missBDT'
    return sel
```

If you want to add a selection, you just have to modify this function by adding an argument on the function and on the file that you want.

#### Variables for BDT

This header contains the variables for the BDT in `train_vars` and their label that will be used when plotting the performance of the BDT in `latex_mapping`. If you want to add a variable, be sure to add it in both variables.

There is also a variable for the label of the processes, `Label`, if you add a process for the BDT training, be sure to update this variable

#### Other variables

This header contains all the other variables, for the moment there is only the $Z$ and Higgs decays that are considered in the analysis in the `z_decays` and `h_decays` respectively and `param`, the variable that will be used in the preselection to do the events selection only on a fraction of the total events or separate the resulting events in separate files.

If you want to add other variables, you can put them there.

## Running the analysis

There are tree types of file in the analysis: the one that use `FCCAnalyses`, the one that use `HiggsAnalysis-CombinedLimit` and the one that use neither.

### Running `FCCAnalyses` files

To run `FCCAnalyses` file, there are four possible command depending on the file that you run

```shell=1
fccanalysis run <preselection_file>.py
fccanalysis final <final_selection_file>.py
fccanalysis plots <plots_file>.py
fccanalysis combine <combine_file>.py
```

#### pre-selection

The first command is for files that do a pre-selection of the events. In this analysis, there are only two of those files: 

- `1-MVAInputs/pre-selection.py` 
- `3-Combine/selection.py`

Contrary to `1-MVAInputs/pre-selection.py`, `3-Combine/selection.py` directly make histograms and not `TTree` and thus does not need the next command.

#### final-selection

The second command is for files that make histograms out of pre-selection file that use `RDFanalysis` class. In this analysis, there is only one such file: 

- `1-MVAInputs/final-selection.py`

#### plots

The third command is for files that make plots out of histograms. In this analysis, there is only one such file:

- `1-MVAInputs/plots.py`

Be careful as `3-Combine/plots.py` does not use the `FCCAnalyses` framework and thus do not require the third command.

#### combine

The fourth and last command is for files that prepare histograms for the fit of the cross-section. In this analysis there is only one such file:

- `3-Combine/combine.py`

### Running `HiggsAnalysis-CombinedLimit` files

`FCCAnalyses` and `HiggsAnalysis-CombinedLimit` are not compatible and thus require to use different terminals when using both. In this analysis, they all are in the `4-Fit` directory:

- `4-Fit/make_pseudo.py`
- `4-Fit/fit.py`
- `4-Fit/bias_test.py`

To run them you just have to be in a separate terminal and source `HiggsAnalysis-CombinedLimit` as explained before and run them the same way than the other files in the section below.

### Running other files

All the otherr files not yet cited do not need either of the two framework and can be used in either of these set up. To run these files you just need to call them with the `python` or `python3` command and with the adequate argument.

### Argument of the files

To run the files in the different selections, center of mass energies or final states you will need to use argument when calling the file. As these arguments are not compatible with `FCCAnalyses` there are no argument in files that run on `FCCAnalyses` framework.

There is only one mandatory argument that can be required: the final state. But it is not mandatory in some files as they will run for each final state.

To know which file can be runned without mandatory argument, you can see in the code if there are lines that look like this:

```python
if arg.cat=='':
    print('\n----------------------------------------------------------------\n')
    print('Final state was not selected, please select one to run this code')
    print('\n----------------------------------------------------------------\n')
    exit(0)
```

or just run the code without adding argument and see if it works.

For files in `4-Fit`, you don't have to put the final state if you do the combined fit by adding the `--combine` argument but be sure to run the code in both channel individually before doing the combined fit as it uses the datacard from the individual fits to run.
