
from urllib.parse import unquote, quote
import requests
import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize
import pickle
from mordred import Calculator, descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from sklearn.feature_selection import VarianceThreshold
from itertools import compress
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit import RDPaths
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import ColumnDataSource, figure, output_file, show
from bokeh.models import Range1d, Label
from bokeh.layouts import column, row

from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from bokeh.io import output_notebook
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Table
from sqlalchemy import Column, BIGINT, Index
from sqlalchemy.orm import mapper
from sqlalchemy.ext.declarative import declarative_base



def mol2svg(mol):
    d2d = rdMolDraw2D.MolDraw2DSVG(200, 100)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()

@st.cache

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def main():
    
    st.set_page_config(
    
    
    page_title="PKSmart PK Predictor",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
    )
    
    st.image("logo_front.png", width=500)
    #st.title("PKSmart PK Predictor")
    st.write(
    """
    [![Follow](https://img.shields.io/twitter/follow/srijitseal?style=social)](https://www.twitter.com/srijitseal)
    """
)
    
    smiles=st.text_input("Enter SMILES of the query compound:")

    #predict
    results = ''
    CL =''
    CL_fe =''
    Vd =''
    Vd_fe =''
    MRT =''
    MRT_fe =''
    thalf =''
    thalf_fe =''
    fup_upperbound=''

    fup =''
    fup_fe =''

    dog_VDss_L_kg =''
    dog_CL_mL_min_kg =''
    dog_fup =''
    monkey_VDss_L_kg=''
    monkey_CL_mL_min_kg=''
    monkey_fup =''
    rat_VDss_L_kg =''
    rat_CL_mL_min_kg =''
    rat_fup =''

    CL_Tc =''
    Vd_Tc =''
    MRT_Tc =''
    thalf_Tc =''
    fup_Tc =''

    pca_res= []
    pcv_1 =''
    pcv_2 =''

    comment=''

    smiles_r=''
    molsvgs_train = ''
    data_train=dict()

    preds_dict={}
    preds= pd.DataFrame(columns=['Endpoint Predicted', 'Predicted value','Predicted Fold error', 'Predicted Range'])
    
    if st.button('Predict PK parameters'):

        with st.spinner('Predicting PK parameters...'):
            
            
            if(smiles==''):
                st.write("No input provided")
                st.success("Fail!")
                st.stop()
            
            #st.snow()

            try:
                smiles_quoted =  quote(smiles)
                results = requests.get(f"http://127.0.0.1:8000/smiles/{smiles_quoted}")
                results = results.json()
            

                smiles_r = results.get("smiles_r")

                dog_VDss_L_kg =results.get("dog_VDss_L_kg")
                dog_CL_mL_min_kg =results.get("dog_CL_mL_min_kg")
                dog_fup =results.get("dog_fup")
                monkey_VDss_L_kg=results.get("monkey_VDss_L_kg")
                monkey_CL_mL_min_kg=results.get("monkey_CL_mL_min_kg")
                monkey_fup =results.get("monkey_fup")
                rat_VDss_L_kg =results.get("rat_VDss_L_kg")
                rat_CL_mL_min_kg =results.get("rat_CL_mL_min_kg")
                rat_fup =results.get("rat_fup")

                CL_fe = results.get("CL_fe")
                CL = results.get("CL")
                Vd_fe = results.get("Vd_fe")
                Vd = results.get("Vd")
                MRT_fe = results.get("MRT_fe")
                MRT = results.get("MRT")
                thalf_fe = results.get("thalf_fe")
                thalf = results.get("thalf")
                fup_fe = results.get("fup_fe")
                fup = results.get("fup")

                CL_Tc =results.get("CL_Tc")
                Vd_Tc =results.get("Vd_Tc")
                MRT_Tc =results.get("MRT_Tc")
                thalf_Tc =results.get("thalf_Tc")
                fup_Tc =results.get("fup_Tc")

                pcv_1 = results.get("pcv_1")
                pcv_2 = results.get("pcv_2")

                comment= results.get("comment")

                st.write(comment)

                CL_range =[np.round(CL/CL_fe,2), np.round(CL*CL_fe, 2)]
                Vd_range =[np.round(Vd/Vd_fe,2) , np.round(Vd*Vd_fe, 2)]
                #fup_range =[np.round(fup/fup_fe,2) , np.round(fup*fup_fe, 2)]
                MRT_range =[np.round(MRT/MRT_fe,2), np.round(MRT*MRT_fe, 2)]
                thalf_range =[np.round(thalf/thalf_fe,2), np.round(thalf*thalf_fe, 2)]

                if(np.round(fup*fup_fe, 2)<=1):
                    fup_upperbound= np.round(fup*fup_fe, 2)
                    fup_range =[np.round(fup/fup_fe,2) , np.round(fup*fup_fe, 2)]
                    
                else:
                    fup_upperbound= 1.00
                    fup_range =[np.round(fup/fup_fe,2), fup_upperbound]
                    



                molecule = Chem.MolFromSmiles(smiles_r)
                
                st.image(Draw.MolToImage(molecule), width=200)
                st.write(f"Standardised SMILES Query compound: {smiles_r}")

                preds_dict = {'Endpoint Predicted':['Human Clearance (CL)', 
                'Human Volume of distribution (VDss)', 
                "Human Fraction unbound in plasma (fup)", 
                "Human Mean Residence Time (MRT)", 
                "Human Half-life (thalf)",

                "Dog Volume of distribution (VDss)",
                "Dog Clearance (CL)",
                "Dog Fraction unbound in plasma (fup)",
                "Monkey Volume of distribution (VDss)",
                "Monkey Clearance (CL)",
                "Monkey Fraction unbound in plasma (fup)",
                "Rat Volume of distribution (VDss)",
                "Rat Clearance (CL)",
                "Rat Fraction unbound in plasma (fup)"
                ],
                

                'Predicted value':[f'{CL} mL/min/kg', f'{Vd} L/kg', f'{fup}', f'{MRT} h', f'{thalf} h',
                f'{dog_VDss_L_kg} L/kg',
                f'{dog_CL_mL_min_kg} mL/min/kg',
                f'{dog_fup}',
                f'{monkey_VDss_L_kg} L/kg',
                f'{monkey_CL_mL_min_kg} mL/min/kg',
                f'{monkey_fup}',
                f'{rat_VDss_L_kg} L/kg', 
                f'{rat_CL_mL_min_kg} mL/min/kg',
                f'{rat_fup}'
                ],
                
                'Estimated Fold error':[np.round(CL_fe,2), np.round(Vd_fe,2), np.round(fup_fe,2), np.round(MRT_fe,2), np.round(thalf_fe,2),
                
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                ],

                'Predicted Range':[ f'{np.round(CL/CL_fe,2)} to {np.round(CL*CL_fe, 2)} mL/min/kg', 
                f'{np.round(Vd/Vd_fe,2)} to {np.round(Vd*Vd_fe, 2)} L/kg', 
                f'{np.round(fup/fup_fe,2)} to {fup_upperbound}', 
                f'{np.round(MRT/MRT_fe,2)} to {np.round(MRT*MRT_fe, 2)} h', 
                f'{np.round(thalf/thalf_fe,2)} to {np.round(thalf*thalf_fe, 2)} h', 

                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                ]
                }

                preds_dict_download = {

                'smiles':[smiles],
                'smiles_r':[smiles_r],

                'Clearance_(CL)':[CL],
                'Clearance_(CL)_units': ["mL/min/kg"],
                'Clearance_(CL)_folderror': [CL_fe],
                'Clearance_(CL)_upperbound': np.max(CL_range),
                'Clearance_(CL)_lowerbound': np.min(CL_range),

                'Volume_of_distribution_(VDss)':[Vd],
                'Volume_of_distribution_(VDss)_units': ["L/kg"],
                'Volume_of_distribution_(VDss)_folderror': [Vd_fe],
                'Volume_of_distribution_(VDss)_upperbound': np.max(Vd_range),
                'Volume_of_distribution_(VDss)_lowerbound': np.min(Vd_range),

                'Fraction_unbound_in_plasma_(fup)':[fup],
                'Fraction_unbound_in_plasma_(fup)_units': ["dimensionless"],
                'Fraction_unbound_in_plasma_(fup)_folderror': [fup_fe],
                'Fraction_unbound_in_plasma_(fup)_upperbound': np.max(fup_range),
                'Fraction_unbound_in_plasma_(fup)_lowerbound': np.min(fup_range),

                'Mean_Residence_Time_(MRT)':[MRT],
                'Mean_Residence_Time_(MRT)_units': ["h"],
                'Mean_Residence_Time_(MRT)_folderror': [MRT_fe],
                'Mean_Residence_Time_(MRT)_upperbound': np.max(MRT_range),
                'Mean_Residence_Time_(MRT)_lowerbound': np.min(MRT_range),

                'Half_life_(thalf)':[thalf],
                'Half_life_(thalf)_units': ["h"],
                'Half_life_(thalf)_folderror': [thalf_fe],
                'Half_life_(thalf)_upperbound': np.max(thalf_range),
                'Half_life_(thalf)_lowerbound': np.min(thalf_range),

                "Dog_Volume_of_distribution_(VDss)": [dog_VDss_L_kg],
                'Dog_Volume_of_distribution_(VDss)_units': ["L/kg"],

                "Dog_Clearance_(CL)": [dog_CL_mL_min_kg],
                'Dog_Clearance_(CL)_units': ["mL/min/kg"],

                "Dog_Fraction_unbound_in_plasma_(fup)": [dog_fup],
                'Dog_Fraction_unbound_in_plasma_(fup)_units': ["dimensionless"],

                "Monkey_Volume_of_distribution_(VDss)": [monkey_VDss_L_kg],
                'Monkey_Volume_of_distribution_(VDss)_units': ["L/kg"],

                "Monkey_Clearance_(CL)": [monkey_CL_mL_min_kg],
                'Monkey_Clearance_(CL)_units': ["mL/min/kg"],

                "Monkey_Fraction_unbound_in_plasma (fup)": [monkey_fup],
                'Monkey_Fraction_unbound_in_plasma_(fup)_units': ["dimensionless"],

                "Rat_Volume_of_distribution_(VDss)": [rat_VDss_L_kg],
                'Rat_Volume_of_distribution_(VDss)_units': ["L/kg"],

                "Rat_Clearance_(CL)": [rat_CL_mL_min_kg],
                'Rat_Clearance_(CL)_units': ["mL/min/kg"],

                "Rat_Fraction_unbound_in_plasma_(fup)": [rat_fup],
                'Rat_Fraction_unbound_in_plasma_(fup)_units': ["dimensionless"],

                "comment": comment

                }
        
                preds = pd.DataFrame(preds_dict)
                st.markdown(  """ ### Human PK paramter predictions for query compound""")
                
                st.table(preds.iloc[:5, :].style.format({"Estimated Fold error": "{:.2f}"}))
                
                preds_dict_download = pd.DataFrame(preds_dict_download)
                preds_dict_download = convert_df(preds_dict_download)

                #Dowload Predictions

                st.download_button(
                label="Download predictions as CSV",
                data=preds_dict_download,
                file_name='Human_PK_prediction.csv',
                mime='text/csv',
                )

                #interactive plot
                st.markdown(  """ ###  Overlaying Query compound on physicochemical space of training data""")

                molsvgs_train = pickle.load(open("../molsvgs_train.sav", 'rb'))
                pca_res = results.get("pca_res")
                pca_res = np.array(pca_res)

                train_data_features= pd.read_csv("../Train_data_features.csv") 
                train_data_features["Data"] = "Train"

                human_VDss_L_kg=10**train_data_features[:]['human_VDss_L_kg']
                human_CL_mL_min_kg=10**train_data_features[:]['human_CL_mL_min_kg']
                human_fup=10**train_data_features[:]['human_fup']
                human_mrt=10**train_data_features[:]['human_mrt']
                human_thalf=10**train_data_features[:]['human_thalf']

                file = open("../features_mfp_mordred_columns_human.txt", "r")
                file_lines = file.read()
                features_mfp_mordred_columns = file_lines.split("\n")
                features_mfp_mordred_columns = features_mfp_mordred_columns[:-1]
            
                mols_test = Chem.MolFromSmiles(smiles_r)
                molsvgs_test = [mol2svg(mols_test)]

                data_train = dict(
                    x= pca_res[:-1][:,0],
                    y=pca_res[:-1][:,1],
                    img = molsvgs_train,
                    human_VDss_L_kg=human_VDss_L_kg,
                    human_CL_mL_min_kg=human_CL_mL_min_kg,
                    human_fup=human_fup,
                    human_mrt=human_mrt,
                    human_thalf=human_thalf
                    )

                data_test = dict(
                
                    x= pca_res[-1:][:,0],
                    y=pca_res[-1:][:,1],
                    img = molsvgs_test[-1:],
                    #human_CL_mL_min_kg=[np.round(CL,2)],
                )


                source_train = ColumnDataSource(data_train)
                source_test= ColumnDataSource(data_test)

                TOOLTIPS = """
                <div>
                human_VDss_L_kg: @human_VDss_L_kg<br>
                human_CL_mL_min_kg: @human_CL_mL_min_kg<br>
                human_fup: @human_fup<br>
                human_mrt: @human_mrt<br>
                human_thalf: @human_thalf<br>
                @img{safe}
                </div>
                """

                st.write("Projecting query compound in the structural-physicochemical space of the training data for Human PK parameters:")

                p = figure(plot_width=600, plot_height=600, tooltips=TOOLTIPS,
                        title=f"Principal Component Analysis using selected Mordred descriptors ({np.round(pcv_1+pcv_2, 2)}% variance explained)",
                        x_axis_label=f"Principal Component 1 ({np.round(pcv_1, 2)}% explained variance)",
                        y_axis_label=f"Principal Component 2 ({np.round(pcv_2, 2)}% explained variance)")
                
                p.circle('x', 'y', size=10, source=source_train, color="red", legend_label="Training data")
                p.circle('x', 'y', size=10, source=source_test, color="blue", legend_label="Query Compound")
                
                p.legend.location = "top_left"

                # change border and background of legend
                p.legend.border_line_width = 3
                p.legend.border_line_color = "black"
                

                st.bokeh_chart(p, use_container_width=True)

                st.markdown(  """ 1""")
                #Plot of Applicability Domains VDss
                st.markdown(  """ ###  Plot of Applicability Domain Analysis for each endpoint""")
                
                pen = pd.read_csv("../human_trainingdata_5nn_similarity_all_compounds_endpoint_wise.csv")
                plots = []
                Tcs = [Vd_Tc, CL_Tc, fup_Tc, MRT_Tc ,thalf_Tc]

                for counter, col in enumerate(pen.Endpoint.unique()[:3]):

                    
                
                    df = pen[(pen.Endpoint==col)]
                    st.write("The 5-nearest Neighbour Tanimoto similarity for the query compound for the endpoint of", col, "is", Tcs[counter])
                    
                    p = figure(width=500, height=500, toolbar_location="right",
                    title=col)

                    # Histogram
                    bins = np.linspace(0, 01.0, 50)
                    x = df[df["Within Applicability Domain?"]==True]["5-nearest Neighbour Tanimoto similarity"].values
                    hist, edges = np.histogram(x, density=False, bins=bins)
                    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                                            fill_color="skyblue", line_color="white")
                    x = df[df["Within Applicability Domain?"]==False]["5-nearest Neighbour Tanimoto similarity"].values
                    hist, edges = np.histogram(x, density=False, bins=bins)
                    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                                            fill_color="red", line_color="white")


                    p.circle(x=np.array([Tcs[counter]]), y=np.array([2]), size=20, color="navy", alpha=0.5)
                    p.step([0.20, 0.20], [140, 1], line_width=2, mode="center")
                    p.step([Tcs[counter], Tcs[counter]], [100, 1], line_width=2, mode="center", color="firebrick")

                    p.x_range = Range1d(0, 1)
                    p.y_range.start = 0
                    p.xaxis.axis_label = "5-nearest Neighbour Tanimoto similarity"
                    p.yaxis.axis_label = "Count"


                    mytext_2 = Label(x=Tcs[counter], y=100, text=f'Query Compound ({Tcs[counter]})')
                    mytext_1 = Label(x=0.20, y=140, text='Threshold for Applicability Doimain (0.20)')

                    p.add_layout(mytext_1)
                    p.add_layout(mytext_2)

                    plots.append(p)

                st.bokeh_chart(row(*plots))

                plots = []

                for counter,col in enumerate(pen.Endpoint.unique()[3:]):
    
                    st.write("The 5-nearest Neighbour Tanimoto similarity for the query compound for the endpoint of", col, "is", Tcs[counter+3])
                    df = pen[(pen.Endpoint==col)]

                    p = figure(width=500, height=500, toolbar_location="right",
                    title=col)

                    # Histogram
                    bins = np.linspace(0, 01.0, 50)
                    x = df[df["Within Applicability Domain?"]==True]["5-nearest Neighbour Tanimoto similarity"].values
                    hist, edges = np.histogram(x, density=False, bins=bins)
                    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                                            fill_color="skyblue", line_color="white")
                    x = df[df["Within Applicability Domain?"]==False]["5-nearest Neighbour Tanimoto similarity"].values
                    hist, edges = np.histogram(x, density=False, bins=bins)
                    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                                            fill_color="red", line_color="white")


                    p.circle(x=np.array([Tcs[counter+3]]), y=np.array([2]), size=20, color="navy", alpha=0.5)
                    p.step([0.20, 0.20], [140, 1], line_width=2, mode="center")
                    p.step([Tcs[counter+3], Tcs[counter+3]], [100, 1], line_width=2, mode="center", color="firebrick")

                    p.x_range = Range1d(0, 1)
                    p.y_range.start = 0
                    p.xaxis.axis_label = "5-nearest Neighbour Tanimoto similarity"
                    p.yaxis.axis_label = "Count"


                    mytext_2 = Label(x=Tcs[counter+3], y=100, text=f'Query Compound ({Tcs[counter+3]})')
                    mytext_1 = Label(x=0.20, y=140, text='Threshold for Applicability Doimain(0.20)')

                    p.add_layout(mytext_1)
                    p.add_layout(mytext_2)

                    plots.append(p)

                st.bokeh_chart(row(*plots))

                    

                st.markdown(  """ ### Predicted Animal PK paramters used by PKSmart Human PK predictor model """)             
                st.write("This model used the following predicted animal data:")
            
                
                
                expander_dog = st.expander("See predictions for Dog PK data")
                expander_dog.write("Predicted Dog PK Parameters ")
                expander_dog.table(preds.iloc[5:8, :2])

                expander_monkey = st.expander("See predictions for Monkey PK data")
                expander_monkey.write("Predicted Monkey PK Parameters")
                expander_monkey.table(preds.iloc[8:11, :2])
                
                expander_rat = st.expander("See predictions for Rat PK data")
                expander_rat.write("Predicted Rat PK Parameters")
                expander_rat.table(preds.iloc[11:, :2])

                #st.write("Predicted CL", np.round(CL,2), "ml/min/kg")
                #st.write("Expected Fold Error:", np.round(CL_fe,2))
                #st.write("Expected range of prediction:", np.round(CL/CL_fe,2), " to ", np.round(CL*CL_fe, 2), "ml/min/kg")

                #st.write("Predicted Vdss", np.round(Vd,2), "L/kg")
                #st.write("Expected Fold Error:", np.round(CL_fe,2))
                #st.write("Expected range of prediction:", np.round(Vd/Vd_fe,2), " to ", np.round(Vd*Vd_fe, 2), "L/kg")

                #st.write("Predicted fup", np.round(fup,2))
                #st.write("Expected Fold Error:", np.round(fup_fe,2))
                #st.write("Expected range of prediction:", np.round(fup/fup_fe,2), " to ", np.round(fup*fup_fe, 2), "")

                #st.write("Predicted MRT", np.round(MRT,2))
                #st.write("Expected Fold Error:", np.round(MRT_fe,2))
                #st.write("Expected range of prediction:", np.round(MRT/MRT,2), " to ", np.round(MRT*MRT_fe, 2), "")
                
                #st.write("Predicted thalf", np.round(thalf,2))
                #st.write("Expected Fold Error:", np.round(thalf_fe,2))
                #st.write("Expected range of prediction:", np.round(thalf/thalf_fe,2), " to ", np.round(thalf*thalf_fe, 2), "")


                

                st.success("Complete")

            except:
                
                if (smiles==' '):
                    st.write(f"Empty SMILES : Unsuccessful!")
                    
                else:
                    st.write(f"{smiles} Unsuccessful! Check SMILES")
                
                st.success("Complete")

if __name__ == '__main__': 
    main()   

