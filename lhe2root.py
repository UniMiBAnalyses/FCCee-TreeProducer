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

#EVENTS = pylhe.read_lhe_with_attributes(LHE_FILE)


# -------------------------
# FUNCTIONS
# -------------------------
# Write ROOT file: TTree with defined branches
def write_root(output_file, branches):
    with uproot.recreate(output_file) as f:

        tree = f.mktree(
            "Events",
            {
                key: ak.type(val)
                for key, val in branches.items()
            }
        )

        tree.extend(branches)

# Define branches for TTree: only LHEPart
def build_TTree_nanoAOD(arr, kind_event, output_file):
    lhepart = ak.zip({
        "eta": ak.values_astype(arr.particles.vector.eta, np.float32),
        "firstMotherIdx": arr.particles.mother1,  
        "incomingpz": ak.where(arr.particles.status == -1,
                               arr.particles.vector.pz, -999),
        "lastMotherIdx": arr.particles.mother2,
        "mass": ak.values_astype(arr.particles.vector.M, np.float32),
        "pdgId": arr.particles.id,
        "phi": ak.values_astype(arr.particles.vector.phi, np.float32),
        "pt": ak.values_astype(arr.particles.vector.pt, np.float32),
        "spin": arr.particles.spin,
        "status": arr.particles.status,
    })
    kind_event = ak.zip({
        "kind_event": kind_event,
    })

    write_root(output_file, {"LHEPart": lhepart, "kind_event": kind_event.kind_event})

# Define branches for TTree: LHEPart + weight
def build_TTree_nanoAOD_reweighted(arr, kind_event, output_file):
    lhepart = ak.zip({
        "eta": arr.particles.vector.eta,
        "firstMotherIdx": arr.particles.mother1,
        "incomingpz": ak.where(arr.particles.status == -1,
                               arr.particles.vector.pz, -999),
        "lastMotherIdx": arr.particles.mother2,
        "mass": arr.particles.vector.M,
        "pdgId": arr.particles.id,
        "phi": arr.particles.vector.phi,
        "pt": arr.particles.vector.pt,
        "spin": arr.particles.spin,
        "status": arr.particles.status,
    })
    weight = ak.zip({
        "weight": arr.eventinfo.weight,
        "weights": arr.weights.values,
    })
    kind_event = ak.zip({
        "kind_event": kind_event,
    })

    write_root(output_file, {"LHEPart": lhepart, "Weight": weight, "kind_event": kind_event})

# Select events with 6 finel particles (for the classification)
def select_events(arr):
    part = ak.zip({
        "pdgId": arr.particles.id,
        "status": arr.particles.status,
        "mother1": arr.particles.mother1,
    })

    lep = part[part.status == 1]
    lep = lep[ak.num(lep) == 6]
    return part, lep

# Classify VBS events
def classify_events(arr):
    part, lep = select_events(arr)
    mother_idx = lep.mother1 - 1
    mother = part[mother_idx]

    is_tag_lep = (
        ((abs(lep.pdgId) == 11) | (abs(lep.pdgId) == 12))
        &
        (mother.status == -1))

    n_tag = ak.sum(is_tag_lep, axis=1)

    is_from_boson = (
        ((abs(mother.pdgId) == 23) | (abs(mother.pdgId) == 24))
        &
        (mother.status == 2))

    n_boson_lep = ak.sum(is_from_boson, axis=1)

    is_Z = is_from_boson & (abs(mother.pdgId) == 23)
    is_W = is_from_boson & (abs(mother.pdgId) == 24)

    nZ = ak.sum(is_Z, axis=1)
    nW = ak.sum(is_W, axis=1)

    is_ZZ = (nZ == 4)
    is_ZW = (nZ == 2) & (nW == 2)
    is_WW = (nW == 4)

    is_VBS = (
        (n_tag == 2)
        &
        (n_boson_lep == 4)
        &
        (is_ZZ | is_ZW | is_WW))

    return is_ZZ, is_WW, is_ZW, is_VBS


# -------------------------
# MAIN
# -------------------------
def main():
    print(f"Reading LHE file: {LHE_FILE}")

    # LHE → AWKWARD
    # arr is an array where each entry corresponds to an event (each entry will be an array of arrays containing general info about the event, list of particles etc.)
    arr = pylhe.to_awkward(pylhe.read_lhe_with_attributes(LHE_FILE))

    # CHECKS Cross section
    # I save the cross section so I can check that the weights match
    xs = pylhe.LHEFile.fromfile(LHE_FILE).init.procInfo[0].xSection
    xs = float(xs)

    # calculate the nominal weight for all events and check that the sum returns the cross section
    weight = arr.eventinfo.weight
    norm_weight = weight * xs * 1000 / ak.sum(weight)
    print("cross-section (pb):", xs)
    print("cross-section reconstructed (fb):", ak.sum(norm_weight))
    print("expected (fb):", xs * 1000)

    # CLASSIFY EVENTS: negative: signal; positive: background; zero: something wrong appened
    is_ZZ, is_WW, is_ZW, is_VBS = classify_events(arr)
    kind_event = ak.where(~is_VBS, +1, ak.where(is_WW, -1, ak.where(is_ZW, -2, ak.where(is_ZZ, -3, 0))))

    # LHE → ROOT (nanoAOD)
    if "weights" in ak.fields(arr):
        build_TTree_nanoAOD_reweighted(arr, kind_event, ROOT_FILE)
    else:
        build_TTree_nanoAOD(arr, kind_event, ROOT_FILE)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py input.lhe output.root")
        sys.exit(1)
        
    main()
