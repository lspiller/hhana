#!/usr/bin/env bash

#perl -i.orig -pe "s/((?<=HistoFileLow\=\")|(?<=HistoFileHigh\=\"))workspaces\/[a-zA-Z_]*\///g" workspaces/hh_nos_nonisol_mcz_cuts/hh_15_combination_125/*.xml
#perl -i.orig -pe "s/((?<=HistoFileLow\=\")|(?<=HistoFileHigh\=\"))workspaces\/[a-zA-Z_]*\///g" workspaces/hh_nos_nonisol_mcz_cuts/*/*.xml
#perl -i.orig -pe "s/((?<=HistoFileLow\=\")|(?<=HistoFileHigh\=\"))workspaces\/[a-zA-Z_]*\///g" workspaces/hh_nos_nonisol_mcz_mva/*/*.xml

find ./workspaces/ -mindepth 0 -maxdepth 5 -type f | grep xml | grep -v fixed | xargs -n1 -I {} perl -i.orig -pe "s/((?<=HistoFileLow\=\")|(?<=HistoFileHigh\=\"))workspaces\/[a-zA-Z_]*\///g" {}
