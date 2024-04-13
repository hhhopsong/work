!====================================================================
!  Simple Neural Network (NN) Model.  
!  <NNbasetrain> only one training process. 
!  <NNlootrain>  Firstly, selectting good seeds for NNstate base on LOO. 
!                And then, NN model was trained again with the good seeds.
!  Auther: Xiangjun Shi(史湘军)  
!  Email: shixj@nuist.edu.cn  
!====================================================================    
    module NNbase
       use pra_data_mod, only: Np,N0,Nobs,Niters,LearningRate,LossThreshold,&
                               SeedNNstate,SeedNNsamples,SeedNNbatch,Nhid,N1,N2,Xobs,Yobs, &
                               Samplesdir,Ntrain,Nbatch,Nvalidate,ORTestSet,Ntest,Nallseeds,Ngoodseeds        
       use ran_mod, only: RandomSorting,SetSeed
       use statistic_subroutine, only: LossCorrelation,sorting
       use types_mod, only:NNstate2,NNstate1, &
                           NNstate1_init,NNstate2_init,NNstate1_dealloc,NNstate2_dealloc
       implicit none
       private
       save       

       public :: NNsamples
       public :: NNbaseinit
       public :: NNbasetrain
       public :: NNlootrain
       
       real,parameter :: beta1=0.9, beta2=0.999    !!! Adam LearningRate <<<<< DO NOT CHANGE>>>>>
       
       type(NNstate2) :: state2
       type(NNstate1) :: state1      
       
       real Xtrain(N0,Ntrain),Xtest(N0,Ntest),Xvalidate(N0,Nvalidate)
       real Ytrain(Ntrain),Ytest(Ntest),Yvalidate(Nvalidate)
       
    contains
    
    subroutine NNsamples 
       implicit none
       character(len=6) cSeed
       integer iSort(Nobs) 
       integer i,count
       Xtrain=0.
       Xtest=0.
       Xvalidate=0.
       Ytrain=0.
       Ytest=0.
       Yvalidate=0.
       if (N0.gt.NP) then
          write(*,*) "N0 must <= Np", N0,Np
          stop 2324
       endif    
       call SetSeed(SeedNNsamples)  !!!!!!!!!
       write(cSeed(1:6),'(i6)') SeedNNsamples+100000
       call RandomSorting(Nobs,iSort)
       open(22,file=Samplesdir//cSeed//"_TrainSamples.txt",form="formatted")
       open(24,file=Samplesdir//cSeed//"_ValidateSamples.txt",form="formatted")
       count=0
       do i=1,Ntrain
          count=count+1 
          Xtrain(1:N0,i)=Xobs(1:N0,iSort(count))
          Ytrain(i)     =Yobs(iSort(count))
          write(22,('(i4,2x,<N0>f8.4,2x,f8.4)')) i,Xtrain(1:N0,i),Ytrain(i)
       enddo  
       do i=1,Nvalidate
          count=count+1 
          Xvalidate(1:N0,i)=Xobs(1:N0,iSort(count))
          Yvalidate(i)     =Yobs(iSort(count))
          write(24,('(i4,2x,<N0>f8.4,2x,f8.4)')) i,Xvalidate(1:N0,i),Yvalidate(i)
       enddo 
       if (ORTestSet) then !ORTestSet
          open(23,file=Samplesdir//cSeed//"_TestSamples.txt",form="formatted")
          do i=1,Ntest
             count=count+1 
             Xtest(1:N0,i)=Xobs(1:N0,iSort(count))
             Ytest(i)     =Yobs(iSort(count))
             write(23,('(i4,2x,<N0>f8.4,2x,f8.4)')) i,Xtest(1:N0,i),Ytest(i)
          enddo 
          close(23)
       else
          write(*,*) "No test set"
       endif
       if (count.ne.Nobs) then
          write (*,*) " NNbase.f90 count.ne.Nobs"
          stop 3424238
       endif   
       close(22)
       close(24)   
    endsubroutine NNsamples 

    subroutine NNbaseinit
       call SetSeed(SeedNNstate)   !!!!!!!!!
       if (Nhid.eq.1) then
          call NNstate1_init(state1,SeedNNstate,N0,N1)
       elseif (Nhid.eq.2) then
          call NNstate2_init(state2,SeedNNstate,N0,N1,N2)
       else
          write(*,*) "Nhid must be 1 or 2d",Nhid
          stop 92420
       endif
    endsubroutine NNbaseinit
    
    subroutine NNbasetrain
       implicit none
       call SetSeed(SeedNNbatch)   !!!!!!!!!
       if (Nhid.eq.1) call NNbasetrain1
       if (Nhid.eq.2) call NNbasetrain2
    endsubroutine NNbasetrain
    
    subroutine NNlootrain
       implicit none
       if (Nhid.eq.1) call NNlootrain1
       if (Nhid.eq.2) call NNlootrain2
    endsubroutine NNlootrain

    subroutine NNbasetrain1 
       implicit none
       real YtrainML(Ntrain),YtestML(Ntest),YvalidateML(Nvalidate)
       real Xbatch(N0,Nbatch),Ybatch(Nbatch)
       real lossTrain,corrTrain,lossValidate,corrValidate
       real lossTest,corrTest,lossTestMin,lossTestPre
       integer iSort(Ntrain),Nextract
       integer i,ii,ip,CountUP
       Nextract=int(Ntrain/Nbatch)
       lossTestMin=999.9
       lossTestPre=999.9
       CountUP=0
       do while (state1%nstep.le.Niters)
          call RandomSorting(Ntrain,iSort)
          ip=0
          do ii=1,Nextract
             do i=1,Nbatch
                ip=ip+1
                Xbatch(1:N0,i)=Xtrain(1:N0,iSort(ip))
                Ybatch(i)=Ytrain(iSort(ip))
             enddo  
             call NNupdate1(state1,Nbatch,Xbatch,Ybatch)  !!
          enddo  
          state1%nstep=state1%nstep+1
          if (mod(state1%nstep,min(5,Niters/100)).eq.0) then         !!!
             call GetY1(state1,Ntrain,Xtrain,YtrainML)
             call LossCorrelation(Ytrain,YtrainML,Ntrain,lossTrain,corrTrain)
             if (ORTestSet) then
                call GetY1(state1,Ntest,Xtest,YtestML)
                call LossCorrelation(Ytest,YtestML,Ntest,lossTest,corrTest)
                write(*,'(i4,2x,a10,f8.5,3x,a10,f7.4,2x,a10,f8.5,3x,a10,f7.4)') state1%nstep, &
                                                "lossTrain",lossTrain,"corrTrain",corrTrain, &
                                                "lossTest",lossTest,"corrTest",corrTest
                if ((lossTest+lossTestPre)/2.0.lt.lossTestMin) then
                   CountUP=0 
                else
                   CountUP=CountUP+1
                   !write(*,*) CountUP
                endif 
                if (((lossTest+lossTestPre)/2.0.gt.lossTestMin*1.01).or.(CountUP.ge.2)) then
                   !write(*,*) "min lossTest:",lossTestMin
                   exit
                endif   
                lossTestMin=lossTestMin*0.9+0.1*(lossTest+lossTestPre)/2.0
                lossTestMin=min((lossTest+lossTestPre)/2.0,lossTestMin)
                lossTestPre=lossTest
             else
                write(*,'(i4,2x,a10,f8.5,3x,a10,f7.4)') state1%nstep,"lossTrain",lossTrain,"corrTrain",corrTrain 
                if (lossTrain.le.LossThreshold) then
                   state1%loss=lossTrain
                   state1%corr=corrTrain
                   exit
                endif 
             endif    
          endif    
       enddo
       call GetY1(state1,Nvalidate,Xvalidate,YvalidateML)
       call LossCorrelation(Yvalidate,YvalidateML,Nvalidate,lossValidate,corrValidate)
       write(*,'(a4,i6,2x,a10,f8.5,3x,a10,f7.4)') "seed",state1%seed,"lossTrain",state1%loss,"corrTrain",state1%corr
       write(*,'(i4,2x,a13,f8.5,3x,a13,f7.4)') state1%nstep,"lossValidate",lossValidate,"corrValidate",corrValidate
       call NNstate1_dealloc(state1)
    endsubroutine NNbasetrain1
    
    subroutine NNbasetrain2 
       implicit none
       real YtrainML(Ntrain),YtestML(Ntest),YvalidateML(Nvalidate)
       real Xbatch(N0,Nbatch),Ybatch(Nbatch)
       real lossTrain,corrTrain,lossValidate,corrValidate
       real lossTest,corrTest,lossTestMin,lossTestPre
       integer iSort(Ntrain),Nextract
       integer i,ii,ip,CountUP
       Nextract=int(Ntrain/Nbatch)
       lossTestMin=999.9
       lossTestPre=999.9
       CountUP=0
       do while (state2%nstep.le.Niters)
          call RandomSorting(Ntrain,iSort)
          ip=0
          do ii=1,Nextract
             do i=1,Nbatch
                ip=ip+1
                Xbatch(1:N0,i)=Xtrain(1:N0,iSort(ip))
                Ybatch(i)=Ytrain(iSort(ip))
             enddo  
             call NNupdate2(state2,Nbatch,Xbatch,Ybatch)  !!
          enddo  
          state2%nstep=state2%nstep+1
          if (mod(state2%nstep,min(5,Niters/100)).eq.0) then         !!!
             call GetY2(state2,Ntrain,Xtrain,YtrainML)
             call LossCorrelation(Ytrain,YtrainML,Ntrain,lossTrain,corrTrain)
             if (ORTestSet) then
                call GetY2(state2,Ntest,Xtest,YtestML)
                call LossCorrelation(Ytest,YtestML,Ntest,lossTest,corrTest)
                write(*,'(i4,2x,a10,f8.5,3x,a10,f7.4,2x,a10,f8.5,3x,a10,f7.4)') state2%nstep, &
                                                "lossTrain",lossTrain,"corrTrain",corrTrain, &
                                                "lossTest",lossTest,"corrTest",corrTest
                if ((lossTest+lossTestPre)/2.0.lt.lossTestMin) then
                   CountUP=0 
                else
                   CountUP=CountUP+1
                   !write(*,*) CountUP
                endif 
                if (((lossTest+lossTestPre)/2.0.gt.lossTestMin*1.01).or.(CountUP.ge.2)) then
                   !write(*,*) "min lossTest:",lossTestMin
                   exit
                endif   
                lossTestMin=lossTestMin*0.9+0.1*(lossTest+lossTestPre)/2.0
                lossTestMin=min((lossTest+lossTestPre)/2.0,lossTestMin)
                lossTestPre=lossTest
             else
                write(*,'(i4,2x,a10,f8.5,3x,a10,f7.4)') state2%nstep,"lossTrain",lossTrain,"corrTrain",corrTrain 
                if (lossTrain.le.LossThreshold) then
                   state2%loss=lossTrain
                   state2%corr=corrTrain
                   exit
                endif     
             endif    
          endif    
       enddo
       call GetY2(state2,Nvalidate,Xvalidate,YvalidateML)
       call LossCorrelation(Yvalidate,YvalidateML,Nvalidate,lossValidate,corrValidate)
       write(*,'(a4,i6,2x,a10,f8.5,3x,a10,f7.4)') "seed",state2%seed,"lossTrain",state2%loss,"corrTrain",state2%corr
       write(*,'(i4,2x,a13,f8.5,3x,a13,f7.4)') state2%nstep,"lossValidate",lossValidate,"corrValidate",corrValidate
       !++test
       !write(*,'(a8,<N2>f8.4)') "W2:",state2%W2(1:N2)
       !write(*,*) "YvalidateML(1:5):",YvalidateML(1:5)
       !--test
       call NNstate2_dealloc(state2)
    endsubroutine NNbasetrain2
    
    subroutine NNtrain1(seed,Nin,Xin,Yin,stateout1) 
       implicit none
       integer,intent(in) :: seed,Nin 
       real,intent(in) ::Xin(N0,Nin)
       real,intent(in) ::Yin(Nin)
       type(NNstate1), intent(out) :: stateout1
       real Xbatch(N0,Nbatch),Ybatch(Nbatch)
       real YML(Nin),loss,corr
       integer iSort(Nin),Nextract
       integer i,ii,ip
       Nextract=int(Nin/Nbatch)
       call SetSeed(Seed)
       call NNstate1_init(stateout1,Seed,N0,N1)
       !call SetSeed(SeedNNbatch)   !!!!!!!!!
       do while (stateout1%nstep.le.Niters)
          call RandomSorting(Nin,iSort)
          ip=0
          do ii=1,Nextract
             do i=1,Nbatch
                ip=ip+1
                Xbatch(1:N0,i)=Xin(1:N0,iSort(ip))
                Ybatch(i)=Yin(iSort(ip))
             enddo  
             call NNupdate1(stateout1,Nbatch,Xbatch,Ybatch)  !!
          enddo  
          stateout1%nstep=stateout1%nstep+1
          call GetY1(stateout1,Nin,Xin,YML)
          call LossCorrelation(Yin,YML,Nin,loss,corr)
          if (loss.le.LossThreshold) then
             !write(*,'(i3,2x,i6,2x,a10,f8.5,3x,a10,f7.4)') stateout1%nstep,stateout1%seed,"lossTrain",loss,"corrTrain",corr 
             stateout1%loss=loss
             stateout1%corr=corr
             exit
          endif   
       enddo 
       if (stateout1%nstep.ge.Niters) then
          write(*,*) "stateout1%nstep.ge.Niters",seed,Niters
       endif
    endsubroutine NNtrain1
    
    subroutine NNtrain2(seed,Nin,Xin,Yin,stateout2) 
       implicit none
       integer,intent(in) :: seed,Nin 
       real,intent(in) ::Xin(N0,Nin)
       real,intent(in) ::Yin(Nin)
       type(NNstate2), intent(out) :: stateout2
       real Xbatch(N0,Nbatch),Ybatch(Nbatch)
       real YML(Nin),loss,corr
       integer iSort(Nin),Nextract
       integer i,ii,ip
       Nextract=int(Nin/Nbatch)
       call SetSeed(Seed)
       call NNstate2_init(stateout2,Seed,N0,N1,N2)
       !call SetSeed(SeedNNbatch)   !!!!!!!!!
       do while (stateout2%nstep.le.Niters)
          call RandomSorting(Nin,iSort)
          ip=0
          do ii=1,Nextract
             do i=1,Nbatch
                ip=ip+1
                Xbatch(1:N0,i)=Xin(1:N0,iSort(ip))
                Ybatch(i)=Yin(iSort(ip))
             enddo  
             call NNupdate2(stateout2,Nbatch,Xbatch,Ybatch)  !!
          enddo  
          stateout2%nstep=stateout2%nstep+1
          call GetY2(stateout2,Nin,Xin,YML)
          call LossCorrelation(Yin,YML,Nin,loss,corr)
          if (loss.le.LossThreshold) then
             !write(*,'(i3,2x,i8,2x,a10,f8.5,3x,a10,f7.4)') stateout2%nstep,stateout2%seed,"lossTrain",loss,"corrTrain",corr 
             stateout2%loss=loss
             stateout2%corr=corr
             exit
          endif   
       enddo 
       if (stateout2%nstep.ge.Niters) then
          write(*,*) "stateout2%nstep.ge.Niters",seed,Niters
       endif
    endsubroutine NNtrain2

    subroutine NNlootrain1 
       implicit none
       type(NNstate1) :: goodstate1(Ngoodseeds)
       type(NNstate1) :: statetemp1
       real XtrainLOO(N0,Ntrain-1),YtrainLOO(Ntrain-1)
       real YtrainML(Ntrain,Nallseeds),YvalidateMLseed(Nvalidate,Ngoodseeds)
       real YvalidateMLave(Nvalidate)
       real lossTrain(Nallseeds),corrTrain(Nallseeds),lossValidate,corrValidate
       integer AllSeeds(Nallseeds),GoodSeeds(Ngoodseeds),seedsort(Nallseeds)
       integer i,iRe,ip,iseed
       do iseed=1,Nallseeds
          AllSeeds(iseed)=SeedNNstate*29+iseed*197
          !AllSeeds(iseed)=SeedNNstate+100000+iseed
       enddo    
       do iRe=1,Ntrain
          !write(*,*) "Remove:",iRe 
          ip=0
          do i=1,Ntrain
          if (i.ne.iRe) then   
             ip=ip+1
             XtrainLOO(1:N0,ip)=Xtrain(1:N0,i)
             YtrainLOO(ip)=Ytrain(i)
          endif   
          enddo 
          do iseed=1,Nallseeds
             call NNtrain1(AllSeeds(iseed),Ntrain-1,XtrainLOO,YtrainLOO,statetemp1) 
             call GetY1(statetemp1,1,Xtrain(1:N0,iRe),YtrainML(iRe,iseed)) 
             call NNstate1_dealloc(statetemp1)
             !write(*,'(i8,2x,a10,f8.5,3x,a10,f7.4)') AllSeeds(iseed),"lossTrain",statetemp1%loss,"CorrTrain",statetemp1%corr
          enddo 
       enddo
       do iseed=1,Nallseeds
          call LossCorrelation(Ytrain(1:Ntrain),YtrainML(1:Ntrain,iseed),Ntrain,lossTrain(iseed),corrTrain(iseed))
          write(*,'(i3,2x,i8,2x,a13,f8.5,3x,a13,f7.4)') &
               iseed,AllSeeds(iseed),"lossTrainLOO",lossTrain(iseed),"corrTrainLOO",corrTrain(iseed)
       enddo
       call sorting(corrTrain,Nallseeds,seedsort)
       YvalidateMLave(:)=0.
       YvalidateMLseed(:,:)=0.
       do iseed=1,Ngoodseeds
          GoodSeeds(iseed)=AllSeeds(seedsort(iseed)) 
          !write(*,'(i3,2x,i8,2x,a13,f8.5,3x,a13,f7.4)') &
          !       iseed,GoodSeeds(iseed),"lossTrainLOO",lossTrain(seedsort(iseed)),"corrTrainLOO",corrTrain(seedsort(iseed)) 
          call NNtrain1(GoodSeeds(iseed),Ntrain,Xtrain,Ytrain,goodstate1(iseed))
          write(*,'(i3,2x,i8,2x,a10,f8.5,3x,a10,f7.4)') iseed,GoodSeeds(iseed),"lossTrain",goodstate1(iseed)%loss,"CorrTrain",goodstate1(iseed)%corr
          call GetY1(goodstate1(iseed),Nvalidate,Xvalidate(1:N0,1:Nvalidate),YvalidateMLseed(1:Nvalidate,iseed))
          call LossCorrelation(Yvalidate(1:Nvalidate),YvalidateMLseed(1:Nvalidate,iseed),Nvalidate,lossValidate,corrValidate)
          write(*,'(a10,f8.5,3x,a10,f7.4,2x,a10,f8.5,3x,a10,f7.4)') &
                     "lossTrain",goodstate1(iseed)%loss,"corrTrain",goodstate1(iseed)%corr, &
                     "lossValidate",lossValidate,       "corrValidate",corrValidate 
          YvalidateMLave(:)=YvalidateMLave(:)+YvalidateMLseed(:,iseed)      
       enddo
       YvalidateMLave(:)=YvalidateMLave(:)/Ngoodseeds
       call LossCorrelation(Yvalidate(1:Nvalidate),YvalidateMLave(1:Nvalidate),Nvalidate,lossValidate,corrValidate)
       write(*,'(a15,2x,a13,f8.5,3x,a13,f7.4)') &
                     "GoodSeedsAve--","lossValidate",lossValidate,"corrValidate",corrValidate 
    endsubroutine NNlootrain1
 
    subroutine NNlootrain2 
       implicit none
       !real Xtrain(N0,Ntrain),Xvalidate(N0,Nvalidate)
       !real Ytrain(Ntrain),Yvalidate(Nvalidate)
       type(NNstate2) :: goodstate2(Ngoodseeds)
       type(NNstate2) :: statetemp2
       real XtrainLOO(N0,Ntrain-1),YtrainLOO(Ntrain-1)
       real YtrainML(Ntrain,Nallseeds),YvalidateMLseed(Nvalidate,Ngoodseeds)
       real YvalidateMLave(Nvalidate)
       real lossTrain(Nallseeds),corrTrain(Nallseeds),lossValidate,corrValidate
       integer AllSeeds(Nallseeds),GoodSeeds(Ngoodseeds),seedsort(Nallseeds)
       integer i,iRe,ip,iseed
       do iseed=1,Nallseeds
          AllSeeds(iseed)=SeedNNstate*59+iseed*97
          !AllSeeds(iseed)=SeedNNstate+100000+iseed
       enddo    
       do iRe=1,Ntrain
          !write(*,*) "Remove:",iRe 
          ip=0
          do i=1,Ntrain
          if (i.ne.iRe) then   
             ip=ip+1
             XtrainLOO(1:N0,ip)=Xtrain(1:N0,i)
             YtrainLOO(ip)=Ytrain(i)
          endif   
          enddo 
          do iseed=1,Nallseeds
             call NNtrain2(AllSeeds(iseed),Ntrain-1,XtrainLOO,YtrainLOO,statetemp2) 
             call GetY2(statetemp2,1,Xtrain(1:N0,iRe),YtrainML(iRe,iseed)) 
             call NNstate2_dealloc(statetemp2)
             !write(*,'(i6,2x,a10,f8.5,3x,a10,f7.4)') AllSeeds(iseed),"lossTrain",statetemp2%loss,"CorrTrain",statetemp2%corr
          enddo 
       enddo
       do iseed=1,Nallseeds
          call LossCorrelation(Ytrain(1:Ntrain),YtrainML(1:Ntrain,iseed),Ntrain,lossTrain(iseed),corrTrain(iseed))
          write(*,'(i3,2x,i6,2x,a13,f8.5,3x,a13,f7.4)') &
               iseed,AllSeeds(iseed),"lossTrainLOO",lossTrain(iseed),"corrTrainLOO",corrTrain(iseed)
       enddo
       call sorting(corrTrain,Nallseeds,seedsort)
       YvalidateMLave(:)=0.
       YvalidateMLseed(:,:)=0.
       do iseed=1,Ngoodseeds
          GoodSeeds(iseed)=AllSeeds(seedsort(iseed)) 
          !write(*,'(i3,2x,i6,2x,a13,f8.5,3x,a13,f7.4)') &
          !       iseed,GoodSeeds(iseed),"lossTrainLOO",lossTrain(seedsort(iseed)),"corrTrainLOO",corrTrain(seedsort(iseed)) 
          call NNtrain2(GoodSeeds(iseed),Ntrain,Xtrain,Ytrain,goodstate2(iseed))
          write(*,'(i3,2x,i6,2x,a10,f8.5,3x,a10,f7.4)') iseed,GoodSeeds(iseed),"lossTrain",goodstate2(iseed)%loss,"CorrTrain",goodstate2(iseed)%corr
          call GetY2(goodstate2(iseed),Nvalidate,Xvalidate(1:N0,1:Nvalidate),YvalidateMLseed(1:Nvalidate,iseed))
          call LossCorrelation(Yvalidate(1:Nvalidate),YvalidateMLseed(1:Nvalidate,iseed),Nvalidate,lossValidate,corrValidate)
          write(*,'(a10,f8.5,3x,a10,f7.4,2x,a10,f8.5,3x,a10,f7.4)') &
                     "lossTrain",goodstate2(iseed)%loss,"corrTrain",goodstate2(iseed)%corr, &
                     "lossValidate",lossValidate,       "corrValidate",corrValidate 
          YvalidateMLave(:)=YvalidateMLave(:)+YvalidateMLseed(:,iseed)      
       enddo
       YvalidateMLave(:)=YvalidateMLave(:)/Ngoodseeds
       call LossCorrelation(Yvalidate(1:Nvalidate),YvalidateMLave(1:Nvalidate),Nvalidate,lossValidate,corrValidate)
       write(*,'(a15,2x,a13,f8.5,3x,a13,f7.4)') &
                     "GoodSeedsAve--","lossValidate",lossValidate,"corrValidate",corrValidate 
    endsubroutine NNlootrain2
 
    
    subroutine GetY1(state1,NN,Xin,Yout) 
       implicit none
       type(NNstate1), intent(in) :: state1
       integer, intent(in) :: NN
       real,intent(in) :: Xin(N0,NN)
       real,intent(out) :: Yout(NN)
       real X_1(N1,NN),X1(N1,NN)
       integer i0,i1,ii
       do ii=1,NN  
          do i1=1,N1
             X_1(i1,ii)=state1%B0(i1)
             do i0=1,N0
                X_1(i1,ii)=X_1(i1,ii)+Xin(i0,ii)*state1%W0(i0,i1)
             enddo 
             X1(i1,ii)=2.0/(1.0+exp(-2*X_1(i1,ii)))-1   !!!  tanh(x)=2sigmoid(2x)-1 ; sigmoid(x)=1/(1+exp(-x))
          enddo  
          Yout(ii)=state1%B1 
          do i1=1,N1
             Yout(ii)=Yout(ii)+X1(i1,ii)*state1%W1(i1)  !!!
          enddo    
       enddo  
    endsubroutine GetY1
    
    subroutine GetY2(state2,NN,Xin,Yout) 
       implicit none
       type(NNstate2), intent(in) :: state2
       integer, intent(in) :: NN
       real,intent(in) :: Xin(N0,NN)
       real,intent(out) :: Yout(NN)
       real X_1(N1,NN),X1(N1,NN)
       real X_2(N2,NN),X2(N2,NN)
       integer i0,i1,i2,ii
       do ii=1,NN  
          do i1=1,N1
             X_1(i1,ii)=state2%B0(i1)
             do i0=1,N0
                X_1(i1,ii)=X_1(i1,ii)+Xin(i0,ii)*state2%W0(i0,i1)
             enddo 
             X1(i1,ii)=2.0/(1.0+exp(-2*X_1(i1,ii)))-1   !!!  tanh(x)=2sigmoid(2x)-1 ; sigmoid(x)=1/(1+exp(-x))
          enddo    
          do i2=1,N2
             X_2(i2,ii)=state2%B1(i2)
             do i1=1,N1
                X_2(i2,ii)=X_2(i2,ii)+X1(i1,ii)*state2%W1(i1,i2)
             enddo 
             X2(i2,ii)=2.0/(1.0+exp(-2*X_2(i2,ii)))-1   !!!  tanh 
          enddo    
          Yout(ii)=state2%B2 
          do i2=1,N2
             Yout(ii)=Yout(ii)+X2(i2,ii)*state2%W2(i2)  !!!
          enddo    
       enddo  
    endsubroutine GetY2

    subroutine NNupdate1(state1,NN,Xin,Yin) 
       implicit none
       type(NNstate1), intent(inout) :: state1
       integer, intent(in) :: NN
       real,intent(in) :: Xin(N0,NN)
       real,intent(in) :: Yin(NN)
       real X_1(N1,NN),X1(N1,NN)
       real Yout(NN)
       real dYout_dLoss(NN)
       real dB1_dYout(NN),dB1_dLoss(NN)
       real dW1_dYout(N1,NN),dW1_dLoss(N1,NN)
       real dX1_dYout(N1,NN),dX_1_dX1(N1,NN),dX_1_dLoss(N1,NN)
       real dB0_dX_1(N1,NN),dB0_dLoss(N1,NN)
       real dW0_dX_1(N0,N1,NN),dW0_dLoss(N0,N1,NN)
       real LR
       integer i0,i1,ii,istep
       !---forward 
       do ii=1,NN  
          do i1=1,N1
             X_1(i1,ii)=state1%B0(i1)
             do i0=1,N0
                X_1(i1,ii)=X_1(i1,ii)+Xin(i0,ii)*state1%W0(i0,i1)
             enddo 
             X1(i1,ii)=2.0/(1.0+exp(-2*X_1(i1,ii)))-1   !!!  tanh(x)=2sigmoid(2x)-1 ; sigmoid(x)=1/(1+exp(-x))
          enddo  
          Yout(ii)=state1%B1 
          do i1=1,N1
             Yout(ii)=Yout(ii)+X1(i1,ii)*state1%W1(i1)  !!!
          enddo    
       enddo 
       !---backward
       dYout_dLoss(1:NN)=2.0*(Yout(1:NN)-Yin(1:NN))    !!!Here, loss=(Yout(1)-Yin(1))**2 + (Yout(2)-Yin(2))**2 + .......
       dB1_dYout(1:NN)=1.
       dB1_dLoss(1:NN)=dYout_dLoss(1:NN)
       dW1_dYout(1:N1,1:NN)=X1(1:N1,1:NN)
       do ii=1,NN
          dW1_dLoss(1:N1,ii)=dW1_dYout(1:N1,ii)*dYout_dLoss(ii)
       enddo   
       do ii=1,NN
          dX1_dYout(1:N1,ii)= state1%W1(1:N1) 
       enddo   
       do ii=1,NN
          do i1=1,N1    
             dX_1_dX1(i1,ii)=1- X1(i1,ii)**2           !!!dtanh(x)/dx=1-tanh(x)**2
             dX_1_dLoss(i1,ii)=dX_1_dX1(i1,ii)*dX1_dYout(i1,ii)*dYout_dLoss(ii)
          enddo
       enddo  
       dB0_dX_1(1:N1,1:NN)=1.
       dB0_dLoss(1:N1,1:NN)=dX_1_dLoss(1:N1,1:NN)
       do ii=1,NN
          do i1=1,N1 
             do i0=1,N0
                dW0_dX_1(i0,i1,ii)=Xin(i0,ii)
                dW0_dLoss(i0,i1,ii)=dW0_dX_1(i0,i1,ii)*dX_1_dLoss(i1,ii)
             enddo   
          enddo
       enddo    
       state1%dB0(:)=0.
       state1%dB1=0.
       state1%dW0(:,:)=0.
       state1%dW1(:)=0.
       do ii=1,NN
          state1%dB0(1:N1)=state1%dB0(1:N1)+dB0_dLoss(1:N1,ii)
          state1%dB1      =state1%dB1      +dB1_dLoss(ii)
          state1%dW0(1:N0,1:N1)= state1%dW0(1:N0,1:N1)+dW0_dLoss(1:N0,1:N1,ii)
          state1%dW1(1:N1)     = state1%dW1(1:N1)     +dW1_dLoss(1:N1,ii)
       enddo  
       state1%dB0=state1%dB0/NN
       state1%dB1=state1%dB1/NN 
       state1%dW0=state1%dW0/NN
       state1%dW1=state1%dW1/NN
       !---update B and W  using Adam
       istep=state1%nstep+1
       LR = LearningRate * sqrt(1.0 - beta2**istep)/(1.0 - beta1**istep) 
       state1%mB0 = beta1 * state1%mB0 + (1 - beta1) *  state1%dB0 
       state1%vB0 = beta2 * state1%vB0 + (1 - beta2) * (state1%dB0**2) 
       state1%B0  = state1%B0 - LR * state1%mB0/sqrt(state1%vB0)  !  - LR*...
       state1%mB1 = beta1 * state1%mB1 + (1 - beta1) *  state1%dB1 
       state1%vB1 = beta2 * state1%vB1 + (1 - beta2) * (state1%dB1**2) 
       state1%B1  = state1%B1 - LR * state1%mB1/sqrt(state1%vB1)  
       state1%mW0 = beta1 * state1%mW0 + (1 - beta1) *  state1%dW0 
       state1%vW0 = beta2 * state1%vW0 + (1 - beta2) * (state1%dW0**2) 
       state1%W0  = state1%W0 - LR * state1%mW0/sqrt(state1%vW0)  
       state1%mW1 = beta1 * state1%mW1 + (1 - beta1) *  state1%dW1 
       state1%vW1 = beta2 * state1%vW1 + (1 - beta2) * (state1%dW1**2) 
       state1%W1  = state1%W1 - LR * state1%mW1/sqrt(state1%vW1)  
    endsubroutine NNupdate1
    
    subroutine NNupdate2(state2,NN,Xin,Yin) 
       implicit none
       type(NNstate2), intent(inout) :: state2
       integer, intent(in) :: NN
       real,intent(in) :: Xin(N0,NN)
       real,intent(in) :: Yin(NN)
       real X_1(N1,NN),X1(N1,NN)
       real X_2(N2,NN),X2(N2,NN)
       real Yout(NN)
       real dYout_dLoss(NN)
       real dB2_dYout(NN),dB2_dLoss(NN)
       real dW2_dYout(N2,NN),dW2_dLoss(N2,NN)
       real dX2_dYout(N2,NN),dX_2_dX2(N2,NN),dX_2_dLoss(N2,NN)
       real dB1_dX_2(N2,NN),dB1_dLoss(N2,NN)
       real dW1_dX_2(N1,N2,NN),dW1_dLoss(N1,N2,NN)
       real dX1_dX_2(N1,N2,NN),dX1_dLoss(N1,NN),dX_1_dX1(N1,NN),dX_1_dLoss(N1,NN)
       real dB0_dX_1(N1,NN),dB0_dLoss(N1,NN)
       real dW0_dX_1(N0,N1,NN),dW0_dLoss(N0,N1,NN)
       real LR
       integer i0,i1,i2,ii,istep
       !---forward 
       do ii=1,NN 
          do i1=1,N1
             X_1(i1,ii)=state2%B0(i1)
             do i0=1,N0
                X_1(i1,ii)=X_1(i1,ii)+Xin(i0,ii)*state2%W0(i0,i1)
             enddo 
             X1(i1,ii)=2.0/(1.0+exp(-2*X_1(i1,ii)))-1   !!!  tanh 
          enddo    
          do i2=1,N2
             X_2(i2,ii)=state2%B1(i2)
             do i1=1,N1
                X_2(i2,ii)=X_2(i2,ii)+X1(i1,ii)*state2%W1(i1,i2)
             enddo 
             X2(i2,ii)=2.0/(1.0+exp(-2*X_2(i2,ii)))-1   !!!  tanh 
          enddo    
          Yout(ii)=state2%B2 
          do i2=1,N2
             Yout(ii)=Yout(ii)+X2(i2,ii)*state2%W2(i2)  !!!
          enddo  
       enddo
       !---backward
       dYout_dLoss(1:NN)=2.0*(Yout(1:NN)-Yin(1:NN))  !!!Here, loss=(Yout(1)-Yin(1))**2 + (Yout(2)-Yin(2))**2 + .......
       dB2_dYout(1:NN)=1.
       dB2_dLoss(1:NN)=dYout_dLoss(1:NN)
       dW2_dYout(1:N2,1:NN)=X2(1:N2,1:NN)
       do ii=1,NN
          dW2_dLoss(1:N2,ii)=dW2_dYout(1:N2,ii)*dYout_dLoss(ii)
       enddo    
       do ii=1,NN
          dX2_dYout(1:N2,ii)= state2%W2(1:N2) 
       enddo    
       do ii=1,NN
          do i2=1,N2    
             dX_2_dX2(i2,ii)=1- X2(i2,ii)**2  !!!dtanh(x)/dx=1-tanh(x)**2
             dX_2_dLoss(i2,ii)=dX_2_dX2(i2,ii)*dX2_dYout(i2,ii)*dYout_dLoss(ii)
          enddo
       enddo  
       dB1_dX_2(1:N2,1:NN)=1.
       dB1_dLoss(1:N2,1:NN)=dX_2_dLoss(1:N2,1:NN)
       do ii=1,NN
          do i2=1,N2 
             do i1=1,N1
                dW1_dX_2(i1,i2,ii)=X1(i1,ii)
                dW1_dLoss(i1,i2,ii)=dW1_dX_2(i1,i2,ii)*dX_2_dLoss(i2,ii)
             enddo   
          enddo
       enddo   
       do ii=1,NN
          dX1_dX_2(1:N1,1:N2,ii)= state2%W1(1:N1,1:N2) 
          do i1=1,N1
             dX1_dLoss(i1,ii)=0.
             do i2=1,N2
                dX1_dLoss(i1,ii)=dX1_dLoss(i1,ii)+dX1_dX_2(i1,i2,ii)*dX_2_dLoss(i2,ii)
             enddo 
             dX_1_dX1(i1,ii)=1- X1(i1,ii)**2   !!!dtanh(x)/dx
             dX_1_dLoss(i1,ii)=dX_1_dX1(i1,ii)*dX1_dLoss(i1,ii)
          enddo    
       enddo  
       dB0_dX_1(1:N1,1:NN)=1.
       dB0_dLoss(1:N1,1:NN)=dX_1_dLoss(1:N1,1:NN)
       do ii=1,NN
          do i1=1,N1 
             do i0=1,N0
                dW0_dX_1(i0,i1,ii)=Xin(i0,ii)
                dW0_dLoss(i0,i1,ii)=dW0_dX_1(i0,i1,ii)*dX_1_dLoss(i1,ii)
             enddo   
          enddo
       enddo
       state2%dB0(:)=0.
       state2%dB1(:)=0. 
       state2%dB2=0.
       state2%dW0(:,:)=0.
       state2%dW1(:,:)=0.
       state2%dW2(:)=0.
       do ii=1,NN
          state2%dB0(1:N1)=state2%dB0(1:N1)+dB0_dLoss(1:N1,ii)
          state2%dB1(1:N2)=state2%dB1(1:N2)+dB1_dLoss(1:N2,ii)
          state2%dB2      =state2%dB2      +dB2_dLoss(ii)
          state2%dW0(1:N0,1:N1)= state2%dW0(1:N0,1:N1)+dW0_dLoss(1:N0,1:N1,ii)
          state2%dW1(1:N1,1:N2)= state2%dW1(1:N1,1:N2)+dW1_dLoss(1:N1,1:N2,ii)
          state2%dW2(1:N2)     = state2%dW2(1:N2)     +dW2_dLoss(1:N2,ii)
       enddo  
       state2%dB0=state2%dB0/NN
       state2%dB1=state2%dB1/NN 
       state2%dB2=state2%dB2/NN
       state2%dW0=state2%dW0/NN
       state2%dW1=state2%dW1/NN
       state2%dW2=state2%dW2/NN
       !---update B and W  using Adam
       istep=state2%nstep+1
       LR = LearningRate * sqrt(1.0 - beta2**istep)/(1.0 - beta1**istep) 
       state2%mB0 = beta1 * state2%mB0 + (1 - beta1) *  state2%dB0 
       state2%vB0 = beta2 * state2%vB0 + (1 - beta2) * (state2%dB0**2) 
       state2%B0  = state2%B0 - LR * state2%mB0/sqrt(state2%vB0)  !  - LR*...
       state2%mB1 = beta1 * state2%mB1 + (1 - beta1) *  state2%dB1 
       state2%vB1 = beta2 * state2%vB1 + (1 - beta2) * (state2%dB1**2) 
       state2%B1  = state2%B1 - LR * state2%mB1/sqrt(state2%vB1)  
       state2%mB2 = beta1 * state2%mB2 + (1 - beta1) *  state2%dB2 
       state2%vB2 = beta2 * state2%vB2 + (1 - beta2) * (state2%dB2**2) 
       state2%B2  = state2%B2 - LR * state2%mB2/sqrt(state2%vB2)  
       state2%mW0 = beta1 * state2%mW0 + (1 - beta1) *  state2%dW0 
       state2%vW0 = beta2 * state2%vW0 + (1 - beta2) * (state2%dW0**2) 
       state2%W0  = state2%W0 - LR * state2%mW0/sqrt(state2%vW0)  
       state2%mW1 = beta1 * state2%mW1 + (1 - beta1) *  state2%dW1 
       state2%vW1 = beta2 * state2%vW1 + (1 - beta2) * (state2%dW1**2) 
       state2%W1  = state2%W1 - LR * state2%mW1/sqrt(state2%vW1) 
       state2%mW2 = beta1 * state2%mW2 + (1 - beta1) *  state2%dW2 
       state2%vW2 = beta2 * state2%vW2 + (1 - beta2) * (state2%dW2**2) 
       state2%W2  = state2%W2 - LR * state2%mW2/sqrt(state2%vW2+1e-6)  
       !write(*,'(a13,2x,<N2>f8.3)') "state2%dW2",state2%dW2(1:N2)
       !write(*,'(a13,2x,<N2>f8.3)') "state2%mW2",state2%mW2(1:N2)
       !write(*,'(a13,2x,<N2>f8.3)') "state2%vW2",state2%vW2(1:N2)
       !write(*,'(a13,2x,<N2>f8.3)') "state2%W2" ,state2%W2(1:N2)
    endsubroutine NNupdate2


    endmodule NNbase
    
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       !++test  Adam 学习率自适应梯度下降法(http://arxiv.org/abs/1412.6980v8)
       !real,parameter :: learning_rate = 0.1 
       !real,parameter :: beta1=0.9    !0.9
       !real,parameter :: beta2=0.999  !0.999
       !real X,Y,dX,dY,r,LR
       !real Xv,Xm,Yv,Ym
       !integer i,istep
       !X=2.0 
       !Y=4.0
       !istep=0
       !Xv=0.
       !Xm=0.
       !Yv=0.
       !Ym=0.
       !do while (istep.le.2000) 
       !   istep = istep + 1  
       !   r=sqrt(X**2+Y**2)
       !   dX=-X*10/(r+0.1)
       !   dY=-Y*10/(r+0.1) 
       !   LR = learning_rate * sqrt(1.0 - beta2**istep)/(1.0 - beta1**istep) 
       !   Xm = beta1 * Xm + (1 - beta1) * dX 
       !   Xv = beta2 * Xv + (1 - beta2) * (dX**2) 
       !   X= X + LR * Xm/sqrt(Xv) 
       !   Ym = beta1 * Ym + (1 - beta1) * dY 
       !   Yv = beta2 * Yv + (1 - beta2) * (dY**2) 
       !   Y= Y + LR * Ym/sqrt(Yv) 
       !   if ((mod(istep,20).eq.0).or.(istep.le.100)) then
       !      write(*,'(i6,2x,f6.4,2x,f6.2,2x,2f6.2,2x,2f7.3,2x,2f7.2,2x,2f7.1)') istep,LR,r,X,Y,dX,dY,Xm,Ym,Xv,Yv
       !   endif  
       !enddo
       !--test
       

