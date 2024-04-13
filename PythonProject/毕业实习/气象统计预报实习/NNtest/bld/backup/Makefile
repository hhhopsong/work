FC     = ifort 
FFLAGS = -r8 -i4 -C 
OBJS   = main.o \
         statistic_subroutine.o ran_mod.o \
         pra_data_mod.o NNbase.o types_mod.o \
         preliminary.o 
#----------------------------------------
ROOT_DIR = /home/FCtest/NNtest
EXEC     = $(ROOT_DIR)/test
VPATH    = $(ROOT_DIR)/code
#----------------------------------------
.PHONY  : all
all     : $(EXEC)
	@echo "EXEC have compiled"
$(EXEC) : $(OBJS)
	$(FC) $(FFLAGS) $^ -o $@  
main.o  : main.f90  NNbase.o pra_data_mod.o preliminary.o
	$(FC) -c $(FFLAGS) $< -o $@ 
NNbase.o: NNbase.f90 pra_data_mod.o ran_mod.o types_mod.o statistic_subroutine.o
	$(FC) -c $(FFLAGS) $< -o $@
pra_data_mod.o: pra_data_mod.f90 
	$(FC) -c $(FFLAGS) $< -o $@
preliminary.o:preliminary.f90 pra_data_mod.o ran_mod.o
	$(FC) -c $(FFLAGS) $< -o $@
types_mod.o: types_mod.f90 ran_mod.o
	$(FC) -c $(FFLAGS) $< -o $@
statistic_subroutine.o: statistic_subroutine.f90
	$(FC) -c $(FFLAGS) $< -o $@
ran_mod.o: ran_mod.f90 
	$(FC) -c $(FFLAGS) $< -o $@
#----------------------------------------
.PHONY  : clean
clean   :
	rm -fr *.mod *.o $(EXEC)
#----------------------------------------
.PHONY  : debug
debug   :
	@echo "VPATH :"$(VPATH)
	@echo "FFLAGS:"$(FFLAGS)
