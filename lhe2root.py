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

# kind_event codes (negative: signal; positive: background)
# The set of codes is chosen event by event from the final state:
#   purely leptonic final state -> CODES_LEP (same as the full-leptonic analysis)
#   final state with quarks (semileptonic / hadronic) -> CODES_HAD (shifted by 10)
CODES_LEP = dict(WW=-1,  WZ=-2,  ZZ=-3,  ZZ_H=-4,  WW_H=-5,  BKG=+1,  BKG_H=+2)
CODES_HAD = dict(WW=-11, WZ=-12, ZZ=-13, ZZ_H=-14, WW_H=-15, BKG=+11, BKG_H=+12)

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
        "weight": (arr.eventinfo.weight), #/ len(arr.eventinfo.weight)),
    })
    branches = {"LHEPart": lhepart, "Weight": weight}
    # kind_event is None when the events could not be classified (see main): branch not written
    if kind_event is not None:
        kind_event = ak.zip({
            "kind_event": kind_event,
        })
        branches["kind_event"] = kind_event.kind_event

    write_root(output_file, branches)

# Define branches for TTree: LHEPart + reweight
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
    branches = {"LHEPart": lhepart, "Weight": weight}
    # kind_event is None when the events could not be classified (see main): branch not written
    if kind_event is not None:
        kind_event = ak.zip({
            "kind_event": kind_event,
        })
        branches["kind_event"] = kind_event

    write_root(output_file, branches)

# Select events with 6 finel particles (for the classification)
# NB: here "lep" = all final-state particles (status 1), leptons OR quarks
def select_events(arr):
    part = ak.zip({
        "pdgId": arr.particles.id,
        "status": arr.particles.status,
        "mother1": arr.particles.mother1,
    })

    lep = part[part.status == 1]
    lep = lep[ak.num(lep) == 6]
    return part, lep

# Classify VBS events -> need 6 final particles
def classify_events(arr):
    part, lep = select_events(arr)

    There_is_higgs = ak.any(abs(part.pdgId) == 25, axis=1)

    if len(lep) == 0 or len(part) == 0:
            empty = ak.Array(np.zeros(len(arr), dtype=bool))
            # same 7 values as the normal return (is_ZZ, is_WW, is_ZW, is_HZ, is_HW, is_VBS, There_is_higgs)
            return empty, empty, empty, empty, empty, empty, There_is_higgs

    mother_idx = lep.mother1 - 1
    mother = part[mother_idx]
    
    # tag particles: e/nu_e whose mother is an incoming e+/e-
    # (a lepton coming from a V decay has the V as mother, so it is NOT a tag)
    is_tag_lep = (
        ((abs(lep.pdgId) == 11) | (abs(lep.pdgId) == 12))
        &
        (mother.status == -1))

    n_tag = ak.sum(is_tag_lep, axis=1)

    # final-state particles (leptons or quarks) coming from a W/Z or from an H in intermediate state
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

    n_boson_lep = ak.sum(is_from_boson, axis=1)   # number of final particles coming from a V (4 for VBS)

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

# Final state with quarks (or gluons)? True -> semileptonic/hadronic, False -> purely leptonic
def is_hadronic_final_state(arr):
    pid = arr.particles.id
    final = arr.particles.status == 1
    coloured = (abs(pid) <= 6) | (pid == 21)
    return ak.any(final & coloured, axis=1)

# Map the classification flags to the kind_event code, using a given set of codes
def assign_codes(flags, codes):
    is_ZZ, is_WW, is_ZW, is_HZ, is_HW, is_VBS, theres_H = flags
    return ak.where(
        ~is_VBS,
        ak.where(is_HZ, codes["ZZ_H"], ak.where(is_HW, codes["WW_H"], ak.where(theres_H, codes["BKG_H"], codes["BKG"]))),
        ak.where(is_WW, codes["WW"], ak.where(is_ZW, codes["WZ"], ak.where(is_ZZ, codes["ZZ"], 0))),
    )

# Assign the kind_event code to every event
# CLASSIFY EVENTS: negative: signal; positive: background; 0: something deeply wrong appened (like code that compiles on the first try)
# Purely leptonic: WW -1 ; WZ -2 ; ZZ -3 ; ZZ with Higgs -4 ; WW with Higgs -5 ; bkg +1 ; bkg with Higgs +2
# Semileptonic:    WW -11; WZ -12; ZZ -13; ZZ with Higgs -14; WW with Higgs -15; bkg +11; bkg with Higgs +12
def compute_kind_event(arr):
    flags = classify_events(arr)
    kind_lep = assign_codes(flags, CODES_LEP)
    kind_had = assign_codes(flags, CODES_HAD)
    return ak.where(is_hadronic_final_state(arr), kind_had, kind_lep)


# -------------------------
# MAIN
# -------------------------
def main():
    print(f"Reading LHE file: {LHE_FILE}")

    # LHE → AWKWARD
    # arr is an array where each entry corresponds to an event (each entry will be an array of arrays containing general info about the event, list of particles etc.)
    arr = pylhe.to_awkward(pylhe.read_lhe_with_attributes(LHE_FILE))

    # CHECK: the classification works only on events with exactly 6 final-state particles
    # (select_events would drop the others and kind_event would no longer be aligned with the events).
    # If some event does not satisfy this, the classification is skipped and the output TTree
    # is written without the kind_event branch.
    n_final = ak.sum(arr.particles.status == 1, axis=1)
    n_bad = int(ak.sum(n_final != 6))
    can_classify = (n_bad == 0)
    if not can_classify:
        print(f"WARNING: {n_bad} / {len(arr)} events do not have exactly 6 final-state particles "
              f"(counts found: {sorted(set(ak.to_list(n_final)))}).")
        print("WARNING: there is no function to classify these events: the output TTree will be "
              "written WITHOUT the kind_event branch.")

    # CHECKS Cross section
    # Save the cross section to check that the weights match
    xs = pylhe.LHEFile.fromfile(LHE_FILE).init.procInfo[0].xSection
    xs = float(xs)

    # calculate the nominal weight for all events and check that the sum returns the cross section
    weight = arr.eventinfo.weight
    norm_weight = weight * xs * 1000
    print("cross-section (pb):", xs)
    print("cross-section reconstructed (fb):", ak.sum(norm_weight / ak.sum(weight)))
    print("expected (fb):", xs * 1000)

    if can_classify:
        kind_event = compute_kind_event(arr)

        # quick summary of the classification
        print(f"events with quarks/gluons in the final state: {int(ak.sum(is_hadronic_final_state(arr)))} / {len(arr)}")
        for code in sorted(set(ak.to_list(kind_event))):
            print(f"kind_event {code:+d}: {int(ak.sum(kind_event == code))} events")
    else:
        kind_event = None

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