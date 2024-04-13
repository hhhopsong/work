!====================================================================
!  Training Set,Test Set, and Validation Set used for the NN Model     
!  Auther: Xiangjun Shi(Ê·Ïæ¾ü)  
!  Email: shixj@nuist.edu.cn  
!====================================================================  
    module preliminary  
       use pra_data_mod,only: Np,Nobs,ORProduceData,Xobs,Yobs,ObsDatadir                   
       implicit none
       private
       save

       public :: LoadData

    contains
    
    subroutine ProduceObsData
       implicit none
       real Rrandom
       integer ip,iobs
       call random_seed()     !!!!!
       do iobs=1,Nobs
          Yobs(iobs)=0. 
          do ip=1,Np
             call random_number(Rrandom) 
             Xobs(ip,iobs)=Rrandom*2.0-1.0
             Yobs(iobs)=Yobs(iobs)+Xobs(ip,iobs)**(mod(iobs,3)+1)
          enddo
          Yobs(iobs)=sin(Yobs(iobs)+Rrandom*0.1)
       enddo    
       open(22,file=ObsDatadir//"OBS_samples.txt",form="formatted")
       do iobs=1,Nobs
          write(22,('(i4,2x,<Np>f8.4,2x,f8.4)')) iobs,Xobs(1:Np,iobs),Yobs(iobs)
       enddo
       close(22)
    endsubroutine ProduceObsData
    
    subroutine LoadData
       implicit none
       real Xtemp(Np),Ytemp
       integer itemp,iobs
       if (ORProduceData) call ProduceObsData
       open(22,file=ObsDatadir//"OBS_samples.txt",form="formatted")
       do iobs=1,Nobs
          read(22,('(i4,2x,<Np>f8.4,2x,f8.4)')) itemp,Xtemp(1:Np),Ytemp
          Xobs(1:Np,iobs)=Xtemp(1:Np)
          Yobs(iobs)=Ytemp
          !write(*,('(i4,2x,<Np>f8.4,2x,f8.4)')) iobs,Xobs(1:Np,iobs),Yobs(iobs)
       enddo
       close(22) 
    endsubroutine LoadData
      
  
    end module preliminary


