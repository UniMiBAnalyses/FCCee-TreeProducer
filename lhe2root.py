# Converts a LHE file into a ROOT file with a structure inspired by the NanoAOD format.
# Only the branches resembling the LHEPart collection commonly used in NanoAOD datasets.

from MG5_simulation.gridpack.grid_cms_lhc.genproductions_scripts.bin.MadGraph5_aMCatNLO.local.VBSee_100TeV.VBSee_100TeV_gridpack.work.MG5_aMC_v2_9_18.models.MSSM_SLHA2 import particles
import pylhe
import sys

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
def build_TTree_nanoAOD(arr, output_file):
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

    with uproot.recreate(output_file) as f:
        tree = f.mktree("Events", {
            "LHEPart": lhepart
        })

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
    weights = arr.eventinfo.weight
    norm_weights = weights * xs * 1000 / ak.sum(weights)
    print("cross-section (pb):", xs)
    print("cross-section reconstructed (fb):", ak.sum(norm_weights))
    print("expected (fb):", xs * 1000)

    # --------------------------
    # LHE → ROOT (nanoAOD)
    # --------------------------
    build_TTree_nanoAOD(arr, ROOT_FILE)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py input.lhe output.root")
        sys.exit(1)
        
    main()
