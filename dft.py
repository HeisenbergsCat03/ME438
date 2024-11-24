from pyscf import gto, dft
elements = ["Li", "Na", "K", "Mg", "Ca"]

basis_set = 'def2-svp'  
functional = 'PBE'   
energies = []
def perform_dft(element, distance=2.5):
    
    print(f"Performing DFT calculation for Si-{element}...")

    mol = gto.Mole()
    if element == "Si":
        mol.atom = "Si 0.0 0.0 0.0"
    else:
        mol.atom = f"""
        Si 0.0 0.0 0.0
        {element} 0.0 0.0 {distance}
        """
    mol.basis = basis_set
    mol.charge = 0  
    mol.spin = 1 if element in ["Li", "Na", "K"] else 0  
    mol.build()


    mf = dft.RKS(mol)
    mf.xc = functional
    energy = mf.kernel()

    print(f"DFT Total Energy for Si-{element}: {energy:.6f} Hartree\n")
    return energy
def dft():
    energies = {}
    for elem in elements + ["Si"]: 
        energies[elem] = perform_dft(elem)

    print("Summary of DFT Calculations:")
    for elem, energy in energies.items():
        print(f"Si-{elem}: {energy:.6f} Hartree")
    return energies

def create_csv():
    energies = dft()
    import csv
    ### MAKE A NEW CSV FILE WITH  ELEMENTS AND THEIR ENERGIES
    with open('dft_energies.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Element', 'Energy (Hartree)'])
        for elem, energy in energies.items():
            writer.writerow([elem, energy])


