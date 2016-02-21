import yaml
import ROOT
import itertools

with open('../hhdb/hhdb/samples/hadhad/2015/backgrounds.yml') as infile:
    backgrounds = yaml.load(infile)

ntuple = ROOT.TFile('ntuples/v8/hhskim/hhskim.root', 'r')

# IN YAML BUT NOT NTUPLES
yaml_no_ntup = []
for c in backgrounds.keys():
    for s in backgrounds[c]['samples']:
        if not ntuple.GetListOfKeys().Contains(s):
            yaml_no_ntup.append((c, s))


# IN NTUPLES BUT NOT YAML
ntup_no_yaml = []
for a in ntuple.GetListOfKeys():
    if '_daod' in a.GetName() or 'data' in a.GetName():# or 'gg' in a.GetName() or 'VBF' in a.GetName():
        continue
    for c in backgrounds.keys():
        if a.GetName() in backgrounds[c]['samples']:
            break
    else:
        # DETERMINE CATEGORY
        if "tautau" in a.GetName():
            c = 'pythia_ztautau'
        elif "Ztt" in a.GetName():
            c = 'mc_ztautau'
        else:
            c = 'others'
        ntup_no_yaml.append((c, a.GetName()))

ntup_no_yaml_no_sys = []
for cat, proc in ntup_no_yaml:
    if '_up' in proc or '_down' in proc or 'Up' in proc or 'Down' in proc or 'MET' in proc:
        continue
    else:
        print cat, proc
        ntup_no_yaml_no_sys.append((cat,proc))

print "IN YAML BUT NOT NTUP", yaml_no_ntup
print "IN NTUP BUT NOT YAML", ntup_no_yaml_no_sys
print "DELETING STUFF FROM YAML"

for c, s in yaml_no_ntup:
    print "delete ", s, " from ", c
    backgrounds[c]['samples'].remove(s)

print "APPENDING STUFF TO YAML"
for c, s in ntup_no_yaml_no_sys:
    print "append ", s, " from ", c
    backgrounds[c]['samples'].append(s)

with open('../hhdb/hhdb/samples/hadhad/2015/backgrounds.yml', 'w') as outfile:
    outfile.write( yaml.dump(backgrounds) )
