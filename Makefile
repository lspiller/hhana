
HHSTUDENT ?= hhskim
HHNTUP ?= ntuples/prod/hhskim
HHNTUP_RUNNING ?= ntuples/running/hhskim

.PHONY: dump

default: clean

clean-root:
	rm -f $(HHNTUP)/$(HHSTUDENT).root

clean-h5:
	rm -f $(HHNTUP)/$(HHSTUDENT).h5

clean-ntup: clean-root clean-h5

check-files:
	./checkfile $(HHNTUP_RUNNING)/$(HHSTUDENT)*.root

check-ntup:
	./checkfile $(HHNTUP)/$(HHSTUDENT).root

browse:
	rootpy browse $(HHNTUP)/$(HHSTUDENT).root

roosh:
	roosh $(HHNTUP)/$(HHSTUDENT).root

$(HHNTUP_RUNNING)/$(HHSTUDENT).data12-JetTauEtmiss.root:
	if [ -f $(HHNTUP_RUNNING)/$(HHSTUDENT).data12-JetTauEtmiss_1.root ]; then \
		test -d $(HHNTUP_RUNNING)/data || mkdir $(HHNTUP_RUNNING)/data; \
		mv $(HHNTUP_RUNNING)/$(HHSTUDENT).data12-JetTauEtmiss_*.root $(HHNTUP_RUNNING)/data; \
		hadd $(HHNTUP_RUNNING)/$(HHSTUDENT).data12-JetTauEtmiss.root $(HHNTUP_RUNNING)/data/$(HHSTUDENT).data12-JetTauEtmiss_*.root; \
		test -d $(HHNTUP_RUNNING)/data_log || mkdir $(HHNTUP_RUNNING)/data_log; \
		mv $(HHNTUP_RUNNING)/$(HHSTUDENT).data12_*.e[0-9]* $(HHNTUP_RUNNING)/data_log/; \
		mv $(HHNTUP_RUNNING)/$(HHSTUDENT).data12_*.o[0-9]* $(HHNTUP_RUNNING)/data_log/; \
		mv $(HHNTUP_RUNNING)/supervisor-$(HHSTUDENT)-$(HHSTUDENT).data12-JetTauEtmiss_*.log $(HHNTUP_RUNNING)/data_log/; \
	fi

init-data-12: $(HHNTUP_RUNNING)/$(HHSTUDENT).data12-JetTauEtmiss.root

$(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_TES_EOP_UP.root:
	test -d $(HHNTUP_RUNNING)/embed_tes || mkdir $(HHNTUP_RUNNING)/embed_tes
	
	for TES_TERM in TES TES_TRUE TES_FAKE TES_EOP TES_CTB TES_Bias TES_EM TES_LCW TES_PU TES_OTHERS; do \
		if [ -f $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_$${TES_TERM}_UP_1.root ]; then \
			mv $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_$${TES_TERM}_*.root $(HHNTUP_RUNNING)/embed_tes; \
		fi; \
		if [ -f $(HHNTUP_RUNNING)/embed_tes/$(HHSTUDENT).embed12-HH-IM_$${TES_TERM}_UP_1.root ]; then \
			hadd $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_$${TES_TERM}_UP.root $(HHNTUP_RUNNING)/embed_tes/$(HHSTUDENT).embed12-HH-IM_$${TES_TERM}_UP_*.root; \
			hadd $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_$${TES_TERM}_DOWN.root $(HHNTUP_RUNNING)/embed_tes/$(HHSTUDENT).embed12-HH-IM_$${TES_TERM}_DOWN_*.root; \
		fi; \
	done

$(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM.root:
	test -d $(HHNTUP_RUNNING)/embed || mkdir $(HHNTUP_RUNNING)/embed
	
	if [ -f $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_1.root ]; then \
		mv $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_[0-9]*.root $(HHNTUP_RUNNING)/embed; \
	fi
	if [ -f $(HHNTUP_RUNNING)/embed/$(HHSTUDENT).embed12-HH-IM_1.root ]; then \
		hadd $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM.root $(HHNTUP_RUNNING)/embed/$(HHSTUDENT).embed12-HH-IM_*.root; \
	fi
	
	if [ -f $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-UP_1.root ]; then \
		mv $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-UP_*.root $(HHNTUP_RUNNING)/embed; \
	fi
	if [ -f $(HHNTUP_RUNNING)/embed/$(HHSTUDENT).embed12-HH-UP_1.root ]; then \
		hadd $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-UP.root $(HHNTUP_RUNNING)/embed/$(HHSTUDENT).embed12-HH-UP_*.root; \
	fi
	
	if [ -f $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-DN_1.root ]; then \
		mv $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-DN_*.root $(HHNTUP_RUNNING)/embed; \
	fi
	if [ -f $(HHNTUP_RUNNING)/embed/$(HHSTUDENT).embed12-HH-DN_1.root ]; then \
		hadd $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-DN.root $(HHNTUP_RUNNING)/embed/$(HHSTUDENT).embed12-HH-DN_*.root; \
	fi
	
.PHONY: embed-12-log
embed-12-log:
	test -d $(HHNTUP_RUNNING)/embed_log || mkdir $(HHNTUP_RUNNING)/embed_log
	-mv $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-*.e[0-9]* $(HHNTUP_RUNNING)/embed_log/
	-mv $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-*.o[0-9]* $(HHNTUP_RUNNING)/embed_log/
	-mv $(HHNTUP_RUNNING)/supervisor-$(HHSTUDENT)-$(HHSTUDENT).embed12-HH-*.log $(HHNTUP_RUNNING)/embed_log/

init-embed-12-sys: $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM_TES_EOP_UP.root

init-embed-12-nominal: $(HHNTUP_RUNNING)/$(HHSTUDENT).embed12-HH-IM.root

init-embed-12: init-embed-12-nominal init-embed-12-sys embed-12-log

init-mc-12:
	test -d $(HHNTUP_RUNNING)/mc_log || mkdir $(HHNTUP_RUNNING)/mc_log
	-mv $(HHNTUP_RUNNING)/$(HHSTUDENT).*mc12*.e[1-9]* $(HHNTUP_RUNNING)/mc_log/
	-mv $(HHNTUP_RUNNING)/$(HHSTUDENT).*mc12*.o[1-9]* $(HHNTUP_RUNNING)/mc_log/
	-mv $(HHNTUP_RUNNING)/supervisor-$(HHSTUDENT)-$(HHSTUDENT).*mc12*.log $(HHNTUP_RUNNING)/mc_log/

init-ntup: init-data-12 init-embed-12 init-mc-12

$(HHNTUP)/$(HHSTUDENT).root:
	./merge-ntup -s $(HHSTUDENT) -o $(HHNTUP)/$(HHSTUDENT).root $(HHNTUP)/$(HHSTUDENT).*.root

$(HHNTUP)/$(HHSTUDENT).h5: $(HHNTUP)/$(HHSTUDENT).root
	root2hdf5 --complib lzo --complevel 0 --quiet $^

ntup: $(HHNTUP)/$(HHSTUDENT).h5

.PHONY: ntup-update
ntup-update:
	./merge-ntup -s $(HHSTUDENT) -o $(HHNTUP)/$(HHSTUDENT).root $(HHNTUP_RUNNING)/$(HHSTUDENT).*.root
	root2hdf5 --update --complib lzo --complevel 0 --quiet $(HHNTUP)/$(HHSTUDENT).root

$(HHNTUP)/merged_grl.xml:
	ls $(HHNTUP)/data/*.root | sed 's/$$/:\/lumi/g' | xargs grl or > $@

$(HHNTUP)/observed_grl.xml: $(HHNTUP)/merged_grl.xml ../higgstautau-dev/grl/2012/current.xml 
	grl and $^ > $@

~/observed_grl.xml: $(HHNTUP)/observed_grl.xml
	cp $^ $@

grl: ~/observed_grl.xml

clean-grl:
	rm -f $(HHNTUP)/observed_grl.xml
	rm -f ~/observed_grl.xml
	rm -f $(HHNTUP)/merged_grl.xml

clean-pyc:                                                                      
	find . -name "*.pyc" -exec rm {} \;

clean: clean-pyc


bundle:
	rm -f ~/higgstautau-mva-plots.tar.gz
	tar -vpczf ~/higgstautau-mva-plots.tar.gz plots/*.png plots/*.eps cache/classify/*/tree.pdf

test:
	nosetests -s -v mva

dump:
	@./dump -t higgstautauhh -s "taus_pass && (RunNumber==207528)" --select-file etc/embed_select_ac.txt -o RunNumber,EventNumber $(HHNTUP)/$(HHSTUDENT).embed12-HH-IM.root
	@./dump -t higgstautauhh -e 50 -s "taus_pass" -o EventNumber $(HHNTUP)/$(HHSTUDENT).AlpgenJimmy_AUET2CTEQ6L1_ZtautauNp4.mc12a.root
	@./dump -t higgstautauhh -e 50 -s "taus_pass" -o EventNumber $(HHNTUP)/$(HHSTUDENT).AlpgenJimmy_AUET2CTEQ6L1_ZtautauNp0.mc12a.root

.PHONY: plots
plots:
	nohup ./ana plot --output-formats eps png > var_plots.log &
	nohup ./ana train evaluate --output-formats eps png > bdt_plots.log & 
	nohup ./ana train evaluate --output-formats eps png --categories mva_deta_controls --unblind > deta_control_plots.log & 

.PHONY: workspace
workspace:
	nohup ./ana workspace > workspace.log &
	nohup ./ana workspace --unblind --mu 123 --workspace-suffix unblinded_random_mu > workspace_unblind_random.log &
