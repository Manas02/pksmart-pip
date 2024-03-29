# PKSmart

## Install

```sh
pip install pksmart
```

### Install from source

1. Clone this repo
```sh
git clone https://github.com/Manas02/pksmart-pip
```

2. Pip install the `PKSmart` Package
```sh
pip install .
```
> Note: Use `pip install -e .` to make an editable installation.

## Usage 

### Help
Simply run `pksmart` or `pksmart -h` or `pksmart --help` to get helper.
![](./pksmart_help.png)

### Running PKSmart as CLI
Run `pksmart -s` or `pksmart --smi` or `pksmart --smiles` to run inference.
![](./pksmart_run.png)


### Running PKSmart as Library

```py
import pksmart


if __name__ == "__main__":
    out = pksmart.predict_pk_params("CCCCCO")
    print(out)
```

## Cite

If you use PKSmart in your work, please cite:

> PKSmart: An Open-Source Computational Model to Predict in vivo Pharmacokinetics of Small Molecules
> Srijit Seal, Maria-Anna Trapotsi, Vigneshwari Subramanian, Ola Spjuth, Nigel Greene, Andreas Bender
> bioRxiv 2024.02.02.578658; doi: https://doi.org/10.1101/2024.02.02.578658