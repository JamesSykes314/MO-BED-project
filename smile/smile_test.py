from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd

# From GPT-4o
smiles = [
    "CC(C)NCC(O)COC1=CC=CC=C1",
    "CC(C)CC1=CC=CC=C1",
    "CCN(CC)CCOC1=CC=CC=C1",
    "CC1=CC=C(C=C1)C(C)C(=O)O",
    "CC(C)C(=O)OC1=CC=CC=C1C(=O)O",
    "CC1=C(C(=O)NC(=O)N1)N",
    "CC1=CC=C(C=C1)C2=CC=CC=C2",
    "C1=CC=CC=C1CC(C(=O)O)N",
    "CCC(CC)COC1=CC=CC=C1",
    "CC(C)COC1=CC=CC=C1",
    "CC(C)(C)C1=CC=C(C=C1)C2=CC=CC=C2",
    "CC1=CC=C(O1)C(=O)NCC2=CC=CC=C2",
    "CC(C)(C)C1=CC=CC=C1C2=CC=CC=C2",
    "CC(C)CC(C(=O)O)N",
    "CC(C)OC1=CC=CC=C1",
    "CCN1CCN(CC1)C2=CC=CC=C2",
    "CC1=CC=C(C=C1)C2=CC=CC=C2O",
    "CC(C)NCC(O)COC1=CC=CC=C1",
    "CCC(CC)N",
    "CCCCCC1=CC=CC=C1",
    "CCC(CC)COC1=CC=CC=C1",
    "CCCCCCCCC1=CC=CC=C1",
    "CC(C)CC(C(=O)O)N",
    "CC(C)OC1=CC=CC=C1",
    "CCN1CCN(CC1)C2=CC=CC=C2",
    "CC1=CC=C(C=C1)C2=CC=CC=C2O",
    "CC(C)CC(C(=O)O)NCC1=CC=CC=C1",
    "CC(C)OC1=CC=CC=C1C2=CC=CC=C2",
    "CCN1CCN(CC1)C2=CC=CC=C2O",
    "CC1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3",
    "CC(C)CC1=CC=CC=C1C2=CC=CC=C2",
    "CCC(CC)COC1=CC=CC=C1C2=CC=CC=C2",
    "CC1=CC=C(O1)C(=O)NCC2=CC=CC=C2C3=CC=CC=C3",
    "CC(C)(C)C1=CC=CC=C1C2=CC=CC=C2O",
    "CC(C)CC(C(=O)O)NCC1=CC=CC=C1C2=CC=CC=C2",
    "CC(C)OC1=CC=CC=C1C2=CC=CC=C2O",
    "CCN1CCN(CC1)C2=CC=CC=C2C3=CC=CC=C3",
    "CC1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3O",
    "CC(C)CC1=CC=CC=C1C2=CC=CC=C2C3=CC=CC=C3",
    "CCC(CC)COC1=CC=CC=C1C2=CC=CC=C2C3=CC=CC=C3",
    "CC1=CC=C(O1)C(=O)NCC2=CC=CC=C2C3=CC=CC=C3O",
    "CC(C)(C)C1=CC=CC=C1C2=CC=CC=C2C3=CC=CC=C3",
    "CC(C)CC(C(=O)O)NCC1=CC=CC=C1C2=CC=CC=C2C3=CC=CC=C3",
    "CC(C)OC1=CC=CC=C1C2=CC=CC=C2C3=CC=CC=C3",
    "CCN1CCN(CC1)C2=CC=CC=C2C3=CC=CC=C3O",
    "CC1=CC=C(C=C1)C2=CC=CC=C2C3=CC=CC=C3C4=CC=CC=C4",
    "CC(C)CC1=CC=CC=C1C2=CC=CC=C2C3=CC=CC=C3O",
    "CCC(CC)COC1=CC=CC=C1C2=CC=CC=C2C3=CC=CC=C3C4=CC=CC=C4",
    "CC1=CC=C(O1)C(=O)NCC2=CC=CC=C2C3=CC=CC=C3C4=CC=CC=C4",
    "CC(C)(C)C1=CC=CC=C1C2=CC=CC=C2C3=CC=CC=C3O",
    "CCOCC(CO)OC1=CC=CC=C1"
]

newset = set(smiles)
smiles = list(newset)
if len(smiles) == len(set(smiles)):
    print("All SMILES strings are unique.")
else:
    print("There are duplicate SMILES strings in the list.")

# Function to calculate cheminformatic descriptors
def generate_cheminformatic_variables(smiles_list):
    descriptors = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        desc = {
            "SMILES": smi,
            "Molecular Weight": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "Hydrogen Bond Donors": Descriptors.NumHDonors(mol),
            "Hydrogen Bond Acceptors": Descriptors.NumHAcceptors(mol),
            "Topological Polar Surface Area": Descriptors.TPSA(mol),
            "Molar Refractivity": Descriptors.MolMR(mol),
            "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
            "Aromatic Rings": Descriptors.NumAromaticRings(mol),
            "Heavy Atoms": Descriptors.HeavyAtomCount(mol)
        }
        descriptors.append(desc)
    return descriptors

# Generate descriptors for the SMILES strings
descriptors_list = generate_cheminformatic_variables(smiles)
variables_df = pd.DataFrame(descriptors_list)
print("done")