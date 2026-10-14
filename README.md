# FCCee-TreeProducer

## Description
This script converts an LHE file into a ROOT file with a structure inspired by NanoAOD, storing only branches analogous to the `LHEPart` collection.

On top of that, it classifies each event of a Vector Boson Scattering (VBS) sample at an e+e- collider and stores the result in a `kind_event` branch. The classification works for purely leptonic as well as semileptonic/hadronic final states.

## Requirements
- pylhe
- awkward
- uproot
- numpy

## Usage
Run the script by providing an input LHE file and a name for the output ROOT file:
```bash
python3 lhe_to_root_had.py input.lhe output.root
```

While running, the script prints a check of the cross section (the sum of the normalised event weights must reproduce the cross section stored in the LHE file) and a summary of the number of events per `kind_event` code.

## Output
The output file contains a TTree named `Events` with the following branches:

| Branch | Content |
|---|---|
| `LHEPart` | `eta`, `firstMotherIdx`, `lastMotherIdx`, `incomingpz`, `mass`, `pdgId`, `phi`, `pt`, `spin`, `status` for every particle of the event |
| `Weight` | `weight` (event weight); if the LHE file contains reweighting weights, also `weights` |
| `kind_event` | event classification code (see below) |

## Event classification (`kind_event`)
The classification only uses the PDG ids and the status of the **mothers** of the final-state particles, never the type of the final-state particles themselves. It therefore works unchanged when the daughters of the vector bosons are leptons or quarks.

An event is classified as VBS if it has exactly 6 final-state particles (`status == 1`) such that:
- 2 of them are electrons or electron neutrinos whose mother is an incoming e+/e- (the *tag* particles);
- the other 4 are daughters of two vector bosons (W/Z) in intermediate state (`status == 2`), whose mother is in turn an incoming particle: 2 daughters per boson.

The events are then assigned to WW, WZ or ZZ according to the bosons. Events with a Higgs in intermediate state, with the tags and a Higgs decaying via ZZ\* or WW\*, are classified as ZZ+H or WW+H. All the other events are background.

The set of codes is chosen event by event from the final state:
- **purely leptonic** final state (no quarks or gluons): first column;
- **semileptonic / hadronic** final state (at least one quark or gluon): second column.

| Process | Leptonic | Semileptonic / hadronic |
|---|---|---|
| WW | -1 | -11 |
| WZ | -2 | -12 |
| ZZ | -3 | -13 |
| ZZ with Higgs | -4 | -14 |
| WW with Higgs | -5 | -15 |
| Background | +1 | +11 |
| Background with Higgs | +2 | +12 |

Negative codes are signal, positive codes are background. The code `0` flags a VBS-like event that matches none of the categories (it should not occur).

The codes are defined at the top of the script (`CODES_LEP`, `CODES_HAD`) and can be changed there.

## Notes and limitations
- **W, Z and H must appear as intermediate particles (`status == 2`) in the LHE file.** This is the case when the decays are done with MadSpin or with the MadGraph decay-chain syntax (e.g. `, (w+ > j j)`). If the final state is generated directly, without intermediate resonances, the VBS events cannot be recognised and end up in the background category.
- **The classification needs exactly 6 final-state particles in every event.** If at least one event has a different number (for example an extra gluon), the script prints a warning, does not classify any event of the file, and writes the TTree **without** the `kind_event` branch.
- The tag particles are identified only as electrons or electron neutrinos coming from the incoming particles. The script does not check that the sample contains only fusion diagrams: this depends on how the process is defined in MadGraph.