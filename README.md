# FCCee-TreeProducer

## Description
This script converts an LHE file into a ROOT file with a structure inspired by NanoAOD, storing only branches analogous to the `LHEPart` collection.

## Requirements
- pylhe  
- awkward  
- uproot  

## Usage
Run the script by providing an input LHE file and a name for output ROOT file:
```bash
python3 script.py input.lhe output.root
```
