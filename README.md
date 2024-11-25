# PKSmart
Drug exposure is a critical determinant of drug safety and efficacy, defined through human phar-macokinetics (PK) parameters such as steady-state volume of distribution (VDss), total body clear-ance (CL), half-life (t½), fraction unbound in plasma (fu), and mean residence time (MRT), which influence a drug's blood concentration profile. In this study, we modelled these human PK parame-ters for 1,283 unique compounds using molecular structural fingerprints, physicochemical proper-ties, and predicted animal PK data. We first predicted animal PK parameters (VDss, CL, fu) for rats, dogs, and monkeys for 371 compounds using molecular structural fingerprints and physico-chemical properties. Next, we employed Morgan fingerprints, Mordred descriptors, and predicted animal PK parameters in a hyperparameter-optimized Random Forest algorithm to predict human PK parameters. Repeated nested cross-validation demonstrated predictive performance for VDss (R² = 0.53 and Geometric Mean Fold Error, GMFE= 2.13), CL (R² = 0.31, GMFE = 2.46), fu (R² = 0.63, GMFE = 2.71), MRT (R² = 0.27, GMFE = 2.50), and t½ (R² = 0.31, GMFE = 2.46). External validation on 315 compounds showed strong performance for VDss (R² = 0.39, GMFE = 2.46) and CL (R² = 0.46, GMFE = 1.95). Comparison with AstraZeneca PK models revealed similar predic-tive performance, with Pearson correlations ranging from 0.46 to 0.73 for animal PK parameters of VDss, CL, and fu. PKSmart models further demonstrated predictive performance for VDss (R² = 0.33, RMSE = 0.58), comparable to AstraZeneca's human VDss models (R² = 0.30, RMSE = 0.60) on an external test set of 51 compounds. To our knowledge, this is the first publicly available set of PK models with performance on par with industry standard models. These models enable early in-tegration of predicted PK properties into workflows such as Design-Make-Test-Analyze (DMTA) cycles using only chemical structures as input. We also developed a web-hosted application, PKSmart (https://broad.io/PKSmart), accessible via browser, with all associated code available for local use.

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
