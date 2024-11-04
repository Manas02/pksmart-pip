import pksmart

if __name__ == "__main__":
    out = pksmart.predict_pk_params("C/[NH+]=C1/C[NH+](O)C(c2ccccc2)=c2cc(Cl)ccc2=N1")
    print(out)
