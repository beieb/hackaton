import pandas as pd
import ast

# Read the CSV file
df = pd.read_csv('data/features_metadata.csv')

# Afficher toutes les maladies uniques
print("="*80)
print("Liste de toutes les maladies:")
print("="*80)

# Parser la colonne 'pathology' (qui contient des listes en string)
all_diseases = set()
for pathologies in df['pathology'].dropna():
    try:
        diseases_list = ast.literal_eval(pathologies)
        all_diseases.update(diseases_list)
    except:
        pass

print(f"Nombre de maladies: {len(all_diseases)}")
for disease in sorted(all_diseases):
    print(f"  - {disease}")

# Vérifier les features sans maladies associées
print("\n" + "="*80)
print("Features sans maladie associée:")
print("="*80)
features_sans_maladie = df[df['pathology'].isnull() | (df['pathology'] == '[]')]
print(f"Nombre: {len(features_sans_maladie)}")
print(features_sans_maladie[['SAS', 'Component']])

# Compter les features par maladie
print("\n" + "="*80)
print("Nombre de features par maladie:")
print("="*80)
disease_counts = {}
for pathologies in df['pathology'].dropna():
    try:
        diseases_list = ast.literal_eval(pathologies)
        for disease in diseases_list:
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
    except:
        pass

for disease, count in sorted(disease_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {disease}: {count}")