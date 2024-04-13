!====================================================================
!  Statistic Subroutines used for the NN Model    
!  Auther: Xiangjun Shi(Ê·Ïæ¾ü)  
!  Email: shixj@nuist.edu.cn  
!====================================================================    
    module statistic_subroutine                   
       implicit none
       private
       save
       
       public :: LossCorrelation
       public :: sorting
        
    contains
    
   subroutine sorting(Var,N,sort) 
      implicit none
      integer,intent(in) :: N
      real,intent(in) :: Var(1:N)
      integer,intent(out) :: sort(1:N)
      integer i,ii,sorttemp(1:N),inttemp
      real realtemp,Vartemp(1:N)
      do ii=1,N
         sorttemp(ii)=ii
         Vartemp(ii)=Var(ii)
      enddo
      do i=1,N
         do ii=1,N-i
            if (Vartemp(ii).gt.Vartemp(ii+1)) then
               realtemp=Vartemp(ii)
               Vartemp(ii)=Vartemp(ii+1)
               Vartemp(ii+1)=realtemp
               inttemp=sorttemp(ii)
               sorttemp(ii)=sorttemp(ii+1)
               sorttemp(ii+1)=inttemp
            endif
         enddo
      enddo
      do ii=1,N
         sort(ii)=sorttemp(N+1-ii)  ! "low=>high" was changed to "high=>low"
      enddo
   endsubroutine sorting


   subroutine LossCorrelation(X1,X2,N,loss,corr)
      implicit none
      integer N
      real X1(N),X2(N)
      real Xnormalization1(N),Xnormalization2(N)
      real loss,corr    
      integer i
      call normalization(X1,N,Xnormalization1)
      call normalization(X2,N,Xnormalization2)
      corr=0.
      loss=0.
      do i=1,N
         corr=corr+Xnormalization1(i)*Xnormalization2(i)
         loss=loss+(X1(i)-X2(i))**2
      enddo
      corr=corr/N
      loss=sqrt(loss/N)
   endsubroutine LossCorrelation
   
  
   subroutine normalization(Xin,N,Xout)
      implicit none
      integer N
      real Xin(N),Xout(N)
      real sum
      integer i
      sum=0.
      do i=1,N
         sum=sum+Xin(i)
      enddo
      sum=sum/N
      do i=1,N
         Xout(i)=Xin(i)-sum
      enddo
      sum=0.
      do i=1,N
         sum=sum+Xout(i)**2
      enddo
      if (sum.eq.0.) return
      sum=sum/N
      sum=sqrt(sum)
      do i=1,N
         Xout(i)=Xout(i)/sum
      enddo
   endsubroutine normalization
   
   

   
 end module statistic_subroutine 
 
