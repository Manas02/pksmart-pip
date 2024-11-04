# PKSmart

Drug exposure is a key contributor to the safety and efficacy of drugs. It can be defined using human pharmacokinetics (PK) parameters that affect the blood concentration profile of a drug, such as steady-state volume of distribution (VDss), total body clearance (CL), half-life (t½), fraction unbound in plasma (fu) and mean residence time (MRT). In this work, we used molecular structural fingerprints, physicochemical properties, and predicted animal PK data as features to model the human PK parameters VDss, CL, t½, fu and MRT for 1,283 unique compounds. First, we predicted animal PK parameters [VDss, CL, fu] for rats, dogs, and monkeys for 372 unique compounds using molecular structural fingerprints and physicochemical properties. Second, we used Morgan fingerprints, Mordred descriptors and predicted animal PK parameters in a hyperparameter-optimised Random Forest algorithm to predict human PK parameters. When validated using repeated nested cross-validation, human VDss was best predicted with an R2 of 0.55 and a Geometric Mean Fold Error (GMFE) of 2.09; CL with accuracies of R2=0.31 and GMFE=2.43, fu with R2=0.61 and GMFE=2.81, MRT with R2=0.28 and GMFE=2.49, and t½ with R2=0.31 and GMFE=2.46 for models combining Morgan fingerprints, Mordred descriptors and predicted animal PK parameters. We evaluated models with an external test set comprising 315 compounds for VDss (R2=0.33 and GMFE=2.58) and CL (R2=0.45 and GMFE=1.98). We compared our models with proprietary pharmacokinetic (PK) models from AstraZeneca and found that model predictions were similar with Pearson correlations ranging from 0.77-0.78 for human PK parameters of VDss and fu and 0.46-0.71 for animal (dog and rat) PK parameters of VDss, CL and fu. To the best of our knowledge, this is the first work that publicly releases PK models on par with industry-standard models. Early assessment and integration of predicted PK properties are crucial, such as in DMTA cycles, which is possible with models in this study based on the input of only chemical structures. We developed a webhosted application PKSmart (https://broad.io/PKSmart) which users can access using a web browser with all code also downloadable for local use.

## Install using `PyPI`

```sh
pip install pksmart
```

### Install from source

1. Clone this repo
```sh
git clone https://github.com/Manas02/pksmart-pip
```

2. Install the `PKSmart` Package
```sh
poetry install
poetry build
```

## Usage 

### Help
Simply run `pksmart` or `pksmart -h` or `pksmart --help` to get helper.

![](https://github.com/Manas02/pksmart-pip/raw/main/pksmart_help.png?raw=True)

### Running PKSmart as CLI
Run `pksmart -s` or `pksmart --smi` or `pksmart --smiles` to run inference on a single SMILES string.

![](https://github.com/Manas02/pksmart-pip/raw/main/pksmart_run_smiles.png?raw=True)

Alternatively, Run `pksmart -f` or `pksmart --file` to run inference using a file containing newline separated SMILES strings.

![](https://github.com/Manas02/pksmart-pip/raw/main/pksmart_run_file.png?raw=True)

### Running PKSmart as Library

```py
import pksmart


if __name__ == "__main__":
    smiles = "CCCCCO"
    out = pksmart.predict_pk_params(smiles)
    print(out)
```

## Cite

If you use PKSmart in your work, please cite:

> PKSmart: An Open-Source Computational Model to Predict in vivo Pharmacokinetics of Small Molecules
> Srijit Seal, Maria-Anna Trapotsi, Vigneshwari Subramanian, Ola Spjuth, Nigel Greene, Andreas Bender
> bioRxiv 2024.02.02.578658; doi: https://doi.org/10.1101/2024.02.02.578658