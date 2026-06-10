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
    weight = ak.zip({
        "weight": arr.eventinfo.weight,
    })
    kind_event = ak.zip({
        "kind_event": kind_event,
    })

    write_root(output_file, {"LHEPart": lhepart, "Weight": weight,"kind_event": kind_event.kind_event})

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

    There_is_higgs = ak.any(abs(part.pdgId) == 25, axis=1)

    if len(lep) == 0 or len(part) == 0:
            empty = ak.Array(np.zeros(len(arr), dtype=bool))
            return empty, empty, empty, empty, There_is_higgs

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
    is_from_higgs = (
        (abs(mother.pdgId) == 25)
        &
        (mother.status == 2))

    granmother_idx = mother.mother1 - 1
    granmother = part[granmother_idx]
    boson_from_incoming = is_from_boson & (granmother.status == -1)

    n_boson_lep = ak.sum(is_from_boson, axis=1)

    is_Z = is_from_boson & (abs(mother.pdgId) == 23) & boson_from_incoming
    is_W = is_from_boson & (abs(mother.pdgId) == 24) & boson_from_incoming

    is_H = is_from_higgs & (granmother.status == -1)

    grangranmother_idx = granmother.mother1 - 1
    grangranmother = part[grangranmother_idx]

    is_Z_from_H = is_from_boson & (abs(mother.pdgId) == 23) & (granmother.status == 2) & (abs(granmother.pdgId) == 25) & (grangranmother.status == -1)
    is_W_from_H = is_from_boson & (abs(mother.pdgId) == 24) & (granmother.status == 2) & (abs(granmother.pdgId) == 25) & (grangranmother.status == -1)

    nZ = ak.sum(is_Z, axis=1)
    nW = ak.sum(is_W, axis=1)
    nH = ak.sum(is_H, axis=1)
    nZ_from_H = ak.sum(is_Z_from_H, axis=1)
    nW_from_H = ak.sum(is_W_from_H, axis=1)

    is_ZZ = (nZ == 4)                     & (n_tag == 2)
    is_ZW = (nZ == 2) & (nW == 2)         & (n_tag == 2)
    is_WW = (nW == 4)                     & (n_tag == 2)
    is_HZ  = (nH == 2) & (nZ_from_H == 2) & (n_tag == 2)
    is_HW  = (nH == 2) & (nW_from_H == 2) & (n_tag == 2)


    is_VBS = (
        (n_tag == 2)
        &
        (n_boson_lep == 4)
        &
        (is_ZZ | is_ZW | is_WW)
    )

    return is_ZZ, is_WW, is_ZW, is_HZ, is_HW, is_VBS, There_is_higgs


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
    norm_weight = weight * xs * 1000
    print("cross-section (pb):", xs)
    print("cross-section reconstructed (fb):", ak.sum(norm_weight / ak.sum(weight)))
    print("expected (fb):", xs * 1000)

    # CLASSIFY EVENTS: negative: signal; positive: background; 0: something deeply wrong appened (like code that compiles on the first try)
    # WW: -1 ; WZ:-2 ; ZZ:-3 ; ZZ with Higgs: -4 ; WW with Higgs: -5
    # Bkg generic: +1, bkg with Higgs: +2
    is_ZZ, is_WW, is_ZW, is_HZ, is_HW, is_VBS, theres_H = classify_events(arr)

    kind_event = ak.where(~is_VBS, ak.where(is_HZ, -4, ak.where(is_HW, -5, ak.where(theres_H, +2, +1))), ak.where(is_WW, -1, ak.where(is_ZW, -2, ak.where(is_ZZ, -3, 0))))

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
