import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem import AllChem, MACCSkeys, FragmentCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Function to load SMILES data
def load_smiles_data(file_path):
    """Load SMILES data from CSV file"""
    df = pd.read_csv(file_path)
    
    return df

# Function to extract various molecular subgroups/fingerprints
def extract_molecular_features(smiles_list):
    """Extract various molecular subgroups and fingerprints from SMILES"""
    results = {}
    
    # Convert SMILES to molecules
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_mols = [m for m in mols if m is not None]
    
    # Store valid SMILES and their indices
    valid_indices = [i for i, m in enumerate(mols) if m is not None]
    results['valid_indices'] = valid_indices
    results['valid_mols'] = valid_mols
    
    # 1. Morgan Fingerprints (ECFP-like)
    morgan_fps = []
    bit_info_list = []  # To track which atoms contribute to each bit
    
    for mol in valid_mols:
        info = {}  # Dictionary to capture atom environments
        fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, bitInfo=info)
        morgan_fps.append(fp)
        bit_info_list.append(info)
    
    results['morgan_fps'] = morgan_fps
    results['bit_info'] = bit_info_list
    
    # 2. MACCS Keys (166 structural keys)
    maccs_fps = [MACCSkeys.GenMACCSKeys(m) for m in valid_mols]
    results['maccs_fps'] = maccs_fps
    
    # 3. Murcko Scaffolds (core structure)
    scaffolds = [MurckoScaffold.GetScaffoldForMol(m) for m in valid_mols]
    scaffold_smiles = [Chem.MolToSmiles(s) if s else "" for s in scaffolds]
    results['scaffolds'] = scaffold_smiles
    
    # 4. Functional Groups (using SMARTS patterns)
    # Define some common functional groups
    functional_groups = {
        'alcohol': '[#6]-[#8X2H]',
        'carboxylic_acid': '[CX3](=O)[OX2H1]',
        'amine': '[NX3;H2,H1,H0;!$(NC=O)]',
        'amide': '[NX3][CX3](=[OX1])',
        'ester': '[#6][CX3](=O)[OX2][#6]',
        'ketone': '[#6][CX3](=O)[#6]',
        'aldehyde': '[CX3H1](=O)[#6]',
        'ether': '[OD2]([#6])[#6]',
        'phenyl': 'c1ccccc1',
        'nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])]'
    }
    
    # Check for functional groups
    func_group_results = []
    for mol in valid_mols:
        features = {}
        for name, smarts in functional_groups.items():
            pattern = Chem.MolFromSmarts(smarts)
            if mol.HasSubstructMatch(pattern):
                features[name] = 1
            else:
                features[name] = 0
        func_group_results.append(features)
    
    results['functional_groups'] = pd.DataFrame(func_group_results)
    
    return results

def get_bit_smiles(molecule, bit_info, radius=2):
    """
    Extract SMILES notation for a specific bit in a molecule
    """
    atom_idx, env_radius = bit_info
    if env_radius > radius:
        env_radius = radius
        
    # Find atoms within the environment
    env = Chem.FindAtomEnvironmentOfRadiusN(molecule, env_radius, atom_idx)
    
    if not env:
        return None
    
    # Create an atom map for the environment
    amap = {}
    submol = Chem.PathToSubmol(molecule, env, atomMap=amap)
    
    if submol is None or submol.GetNumAtoms() == 0:
        return None
    
    # Convert to SMILES
    return Chem.MolToSmiles(submol)

# Function to analyze subgroups vs properties
def analyze_subgroups_vs_properties(features, properties):
    """Analyze correlations and extract SMILES for important bits"""
    # Create dataframe from fingerprints as before
    fp_df = pd.DataFrame([list(fp) for fp in features['morgan_fps']])
    fp_df.columns = [f'bit_{i}' for i in range(fp_df.shape[1])]
    
    # Add functional groups
    full_df = pd.concat([fp_df, features['functional_groups']], axis=1)
    
    # Add properties
    for prop_name, prop_values in properties.items():
        valid_props = [prop_values[i] for i in features['valid_indices']]
        full_df[prop_name] = valid_props
    
    # Calculate correlations
    corr_df = full_df.corr()[list(properties.keys())].iloc[:-len(properties)]
    
    # Find important bits for each property
    important_bits = {}
    for prop in properties.keys():
        # Get fingerprint bits only (first 1024 columns)
        fp_corrs = corr_df[prop].iloc[:1024]
        
        # Find top positive and negative correlations
        top_pos = fp_corrs.nlargest(5)
        top_neg = fp_corrs.nsmallest(5)
        
        # Store bit information
        important_bits[f"{prop}_positive"] = [(int(bit.split('_')[1]), corr) 
                                              for bit, corr in top_pos.items()]
        important_bits[f"{prop}_negative"] = [(int(bit.split('_')[1]), corr) 
                                              for bit, corr in top_neg.items()]
    
    # Extract SMILES for important bits
    bit_smiles = {}
    for category, bits in important_bits.items():
        bit_smiles[category] = {}
        for bit, corr in bits:
            # Find molecules that have this bit
            for mol_idx, bit_info in enumerate(features['bit_info']):
                if bit in bit_info:
                    mol = features['valid_mols'][mol_idx]
                    # Get the first atom info for this bit
                    atom_info = bit_info[bit][0]  # Take the first environment
                    smiles = get_bit_smiles(mol, atom_info)
                    if smiles:
                        bit_smiles[category][bit] = {
                            'correlation': corr,
                            'smiles': smiles
                        }
                        break  # Found a SMILES for this bit, move to next one
    
    return full_df, corr_df, important_bits, bit_smiles


def main(smiles_file, property_cols=None):
    """Main function to process SMILES and analyze subgroups"""
    # Load data
    data = load_smiles_data(smiles_file)
    smiles_col = 'SMILES'  # Adjust if your column name is different
    
    # Extract properties if provided
    properties = {}
    if property_cols:
        for col in property_cols:
            if col in data.columns:
                properties[col] = data[col].tolist()
    
    # Extract molecular features
    features = extract_molecular_features(data[smiles_col].tolist())
    
    # If properties provided, analyze correlations
    if properties:
        full_df, corr_df, important_bits, bit_smiles = analyze_subgroups_vs_properties(features, properties)
        
        # Print results in text format
        print("IMPORTANT SUBSTRUCTURES (SMILES FORMAT)")
        print("=======================================")
        
        for category, bits in bit_smiles.items():
            prop, effect = category.split('_')
            effect_label = "INCREASES" if effect == "positive" else "DECREASES"
            
            print(f"\nSubstructures that {effect_label} {prop}:")
            print("-" * 40)
            
            for bit, info in bits.items():
                print(f"Bit {bit} (correlation: {info['correlation']:.3f}):")
                print(f"SMILES: {info['smiles']}")
                print()
        
        return features, full_df, corr_df, important_bits, bit_smiles
    
    return features

def create_smiles_labeled_heatmap(corr_df, bit_smiles, property_cols, n_top=20):
    """
    Create a heatmap of correlations where bit labels are replaced with SMILES notations
    """
    all_smiles_labels = {}
    
    # For each property, gather smiles for top positive and negative correlations
    for prop in property_cols:
        # Get top positive correlations
        pos_category = f"{prop}_positive"
        neg_category = f"{prop}_negative"
        
        # Collect SMILES for top bits
        if pos_category in bit_smiles:
            for bit, info in bit_smiles[pos_category].items():
                bit_col = f"bit_{bit}"
                # Create a label that includes bit number, correlation and SMILES
                all_smiles_labels[bit_col] = f"{bit}: {info['smiles']}"
        
        if neg_category in bit_smiles:
            for bit, info in bit_smiles[neg_category].items():
                bit_col = f"bit_{bit}"
                all_smiles_labels[bit_col] = f"{bit}: {info['smiles']}"
    
    # Create a list of all important bit columns
    important_bits = list(all_smiles_labels.keys())
    
    # Find up to n_top features for each property
    top_features = set()
    for prop in property_cols:
        # Sort by absolute correlation to get most important features
        sorted_features = corr_df[prop].abs().sort_values(ascending=False)
        # Filter to only include bits we have SMILES for
        sorted_features = sorted_features[sorted_features.index.isin(important_bits)]
        # Take top n features
        top_features.update(sorted_features.head(n_top).index)
    
    # Convert to list and sort
    top_features = sorted(list(top_features))
    
    # Create a filtered correlation dataframe with only the important features
    plot_df = corr_df.loc[top_features, property_cols]
    
    # Create a mapping of bit columns to SMILES labels
    smiles_labels = {bit: all_smiles_labels.get(bit, bit) for bit in plot_df.index}
    
    # Create the heatmap
    plt.figure(figsize=(15, len(top_features)*0.3 + 3))
    ax = sns.heatmap(plot_df, annot=True, cmap='coolwarm', center=0, 
                     linewidths=.5, fmt=".2f", cbar_kws={"shrink": .8})
    
    # Replace y-axis labels with SMILES
    ax.set_yticklabels([smiles_labels[bit] for bit in plot_df.index], fontsize=9)
    
    plt.title("Molecular Substructures Correlation with Properties", fontsize=14)
    plt.tight_layout()
    
    return plt.gcf()

if __name__ == "__main__":
    # Replace with your file path
    file_path = '/Users/sambolton/Desktop/Cummins_Lab/LXRa vs LXRb Data/CDD_SMILES.csv'
    
    # Replace with your property column names
    property_cols = ['Data Summary: Average LXRa 100uM %','Data Summary: Average LXRb 100nM %']
    
    features, full_df, corr_df, important_bits, bit_smiles = main(file_path, property_cols)
    
    fig = create_smiles_labeled_heatmap(corr_df, bit_smiles, property_cols)
    plt.show()
   
    # full_df.to_csv("subgroup_features.csv", index=False)
    # corr_df.to_csv("subgroup_correlations.csv", index=True)
    # print("Analysis complete. Results saved to subgroup_correlations.png")