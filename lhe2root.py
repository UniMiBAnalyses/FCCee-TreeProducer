# Converts a LHE file into a ROOT file with a structure inspired by the NanoAOD format.
# Only the branches resembling the LHEPart collection commonly used in NanoAOD datasets.

import pylhe
import sys

import numpy as np
import awkward as ak
import uproot

# -------------------------
# GOBAL VARIABLES
# -------------------------
LHE_FILE = sys.argv[1]
ROOT_FILE = sys.argv[2]

EVENTS = pylhe.read_lhe_with_attributes(LHE_FILE)

# -------------------------
# FUNCTIONS
# -------------------------
def write_root(output_file, lhepart):
    with uproot.recreate(output_file) as f:
        tree = f.mktree("Events", {
            "LHEPart": lhepart
        })

def build_TTree_nanoAOD(arr, output_file):
    lhepart = ak.zip({
        "eta": ak.values_astype(arr.particles.vector.eta, np.float32),
        "firstMotherIdx": arr.particles.mother1,  
        "incomingpz": ak.where(arr.particles.status == -1,
                               arr.particles.vector.pz, -99),
        "lastMotherIdx": arr.particles.mother2,
        "mass": ak.values_astype(arr.particles.vector.M, np.float32),
        "pdgId": arr.particles.id,
        "phi": ak.values_astype(arr.particles.vector.phi, np.float32),
        "pt": ak.values_astype(arr.particles.vector.pt, np.float32),
        "spin": arr.particles.spin,
        "status": arr.particles.status,
    })

    write_root(output_file, lhepart)

def build_TTree_nanoAOD_reweighted(arr, output_file):
    lhepart = ak.zip({
        "eta": arr.particles.vector.eta,
        "firstMotherIdx": arr.particles.mother1,
        "incomingpz": ak.where(arr.particles.status == -1,
                               arr.particles.vector.pz, 0),
        "lastMotherIdx": arr.particles.mother2,
        "mass": arr.particles.vector.M,
        "pdgId": arr.particles.id,
        "phi": arr.particles.vector.phi,
        "pt": arr.particles.vector.pt,
        "spin": arr.particles.spin,
        "status": arr.particles.status,
    })
    weight = ak.zip({
        "weights": arr.weights.values,
    })

    write_root(output_file, {"LHEPart": lhepart, "Weight": weight})

# -------------------------
# MAIN
# -------------------------
def main():
    print(f"Reading LHE file: {LHE_FILE}")

    # --------------------------
    # LHE → AWKWARD
    # --------------------------
    # arr is an array where each entry corresponds to an event (each entry will be an array of arrays containing general info about the event, list of particles etc.)
    arr = pylhe.to_awkward(EVENTS)

    # --------------------------
    # CHECKS Cross section
    # --------------------------
    # I save the cross section so I can check that the weights match
    xs = pylhe.LHEFile.fromfile(LHE_FILE).init.procInfo[0].xSection
    xs = float(xs)

    # calculate the nominal weight for all events and check that the sum returns the cross section
    weight = arr.eventinfo.weight
    norm_weight = weight * xs * 1000 / ak.sum(weight)
    print("cross-section (pb):", xs)
    print("cross-section reconstructed (fb):", ak.sum(norm_weight))
    print("expected (fb):", xs * 1000)

    # --------------------------
    # LHE → ROOT (nanoAOD)
    # --------------------------
    if "weights" in ak.fields(arr):
        build_TTree_nanoAOD_reweighted(arr, ROOT_FILE)
    else:
        build_TTree_nanoAOD(arr, ROOT_FILE)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py input.lhe output.root")
        sys.exit(1)
        
    main()
