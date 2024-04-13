!====================================================================
!  Random Variables used for the NN Model    
!  Auther: Xiangjun Shi(Ê·Ïæ¾ü)  
!  Email: shixj@nuist.edu.cn  
!====================================================================   
    module ran_mod
       implicit none
       private
       save
       integer , save :: Reverseflag = 0
       
       public :: Rnormal             !  return a normal distribution normal(mean,sigma)
       public :: RandomSorting       !  integer, intent(out) :: Sout(N)
       public :: SetSeed             !  SeedValue<=0,random_seed is set based on system_clock
                                     !  SeedValue>0, random_seed is set based on SeedValue
    contains

    subroutine SetSeed(SeedValue)
      implicit none
      integer, intent(in) :: SeedValue
      integer,dimension(:),allocatable :: Rseed
      integer n,clock
      !write(*,*) "<<<<random_seed>>>>"
      call random_seed(SIZE=n)
      !write(*,*) "SIZE=n",n
      allocate(Rseed(n))
      if (SeedValue.le.0) then
         call system_clock(COUNT=clock)
         !write(*,*) "clock",clock
         Rseed=clock
      else
         Rseed=SeedValue+1000000000
      endif    
      !write(*,*) "PUT=SeedValue+1809648271(or 1000000000)",Rseed  
      call random_seed(PUT=Rseed) 
      call random_seed(GET=Rseed)
      !write(*,*) "GET=Rseed",Rseed 
      Reverseflag = 0  !!!! Reverseflag=0 after seting seedNumber
    endsubroutine SetSeed
        
    function Rnormal(mean,sigma)
       implicit none
       real, parameter :: pi = 3.1415926
       real :: Rnormal
       real :: mean, sigma
       real :: u1, u2, y1, y2
       call random_number(u1)     !returns random number between 0 - 1 
       call random_number(u2)
       if (Reverseflag.eq.0) then
          y1 = sqrt(-2.0*log(u1))*cos(2.0*pi*u2)
          Rnormal = mean + sigma*y1
          Reverseflag = 1
       else
          y2 = sqrt(-2.0*log(u1))*sin(2.0*pi*u2)
          Rnormal = mean + sigma*y2
          Reverseflag = 0
       endif 
    end function Rnormal

    subroutine RandomSorting(N,Sout)
      implicit none
      integer, intent(in) :: N
      integer, intent(out) :: Sout(N) 
      real rantemp(N),Rrandom
      integer i,ii,itemp
      !call random_seed()
      do i=1,N
         call random_number(Rrandom)
         rantemp(i)=Rrandom
         Sout(i)=i
      enddo
      do i=1,N
         do ii=i,N
            if (rantemp(i).lt.rantemp(ii)) then
               Rrandom=rantemp(i)
               rantemp(i)=rantemp(ii)
               rantemp(ii)=Rrandom
               itemp=Sout(i)
               Sout(i)=Sout(ii)
               Sout(ii)=itemp
            endif    
         enddo    
      enddo   
    endsubroutine RandomSorting

    endmodule ran_mod

    
    !!!!!!!!!!!!!!test!!!!!!!!!!!!!!!!!!!!
       !integer,parameter :: Nbin=20
       !integer,parameter :: Nran=1000
       !integer,parameter :: SeedValue= 2875   ! <=0 random_seed is system_clock; >0 random_seed is SeedValue
       !real,parameter :: BinUpper= 3.0
       !real,parameter :: Binlower=-3.0
       !real Bin(0:Nbin),mean,sigma
       !real Rrandom,dbin
       !integer RanSort(Nran),CountBin(Nbin)
       !integer i,n
       !dbin=(BinUpper-Binlower)/Nbin
       !Bin(0)=Binlower
       !do i=0,Nbin
       !   Bin(i)=i*dbin+Binlower
       !   !write(*,'(a8,i4,f8.2)') "Bin:",i,Bin(i)
       !enddo 
       !call SetSeed(SeedValue) 
       !call RandomSorting(Nran,RanSort)
       !write(*,'(10i4)'), RanSort(1:10)
       !mean=0.
       !sigma=1.0
       !CountBin(:)=0
       !do n=1,Nran
       !   Rrandom=Rnormal(mean,sigma) 
       !   if (n.ge.Nran-20) write(*,'(i4,f12.8)') n,Rrandom
       !   do i=1,Nbin
       !      if ((Rrandom.ge.Bin(i-1)).and.(Rrandom.lt.Bin(i))) then
       !          CountBin(i)=CountBin(i)+1
       !      endif
       !   enddo   
       !enddo 
       !do i=1,Nbin
       !  write(*,'(i4,f5.1,a2,f5.1,2x,i5)'), i,Bin(i-1),"-",Bin(i),CountBin(i)
       !enddo 
   !!!!!!!!!!!!!!test!!!!!!!!!!!!!!!!!!!!