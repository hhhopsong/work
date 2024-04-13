!====================================================================
!  Simple Neural Network (NN) Model     
!  Auther: Xiangjun Shi(Ê·Ïæ¾ü)  
!  Email: shixj@nuist.edu.cn  
!====================================================================   
    program main
       use pra_data_mod, only: Nhid,NNtraintype
       use preliminary , only: LoadData
       use NNbase, only: NNsamples,NNbaseinit,NNbasetrain,NNlootrain
       implicit none
       
       if (Nhid.eq.1) write(*,*) "One Hidden Layer"
       if (Nhid.eq.2) write(*,*) "Two Hidden Layers"
       
       call LoadData 
       
       call NNsamples
       if (NNtraintype.eq.0) then
          call NNbaseinit
          call NNbasetrain
       elseif (NNtraintype.eq.1) then
          write(*,*) "Leave-One-Out Test" 
          call NNlootrain
       else
          write(*,*) "NNtraintype must be 0 or 1",NNtraintype
          stop 232432
       endif      


    end program main

