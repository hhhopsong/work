    program readData
       implicit none 
       integer,parameter :: Nall=506, Nvar=13
       real :: XX(Nall,Nvar),X(Nall,Nvar)  ! 社会中各种影响房价的因素
       real :: YY(Nall),Y(Nall)            ! 房价
       real :: XYcorr(Nvar)
       integer sort(1:Nvar)
       integer :: i,iv,ii

    
       open(22,file='XX.txt')
       do i=1,Nall
        read(22,*) (XX(i,iv),iv=1,Nvar)
        !write(*,'(<Nvar>(f10.5))') XX(i,:)
       enddo
       close(22)    
    
       open(22,file='YY.txt')
       do i=1,Nall
        read(22,*) YY(i)
        !write(*,'((f7.2))') YY(i)
       enddo
       close(22)    
       
       call  normalization(YY(1:Nall),Nall,Y(1:Nall))
       
       do iv=1,Nvar
          call  normalization(XX(1:Nall,iv),Nall,X(1:Nall,iv))
          call correlation(X(1:Nall,iv),Y(1:Nall),Nall,XYcorr(iv))
          XYcorr(iv)=XYcorr(iv)*XYcorr(iv)
       enddo
       
       call sorting(XYcorr,Nvar,sort)
       do iv=1,Nvar
          ii=sort(iv) 
          write(*,'(i2,2x,i2,2x,f6.3)') iv,ii,XYcorr(ii)
          do i=1,Nall
             XX(i,iv)=X(i,ii)
          enddo    
       enddo

       
       
       open(22,file="OBS_samples.txt",form="formatted")
       do i=1,Nall
          write(22,('(i4,2x,<Nvar>f8.4,2x,f8.4)')) i,XX(i,1:Nvar),Y(i)
          write(*,('(i4,2x,<Nvar>f8.4,2x,f8.4)')) i,XX(i,1:Nvar),Y(i)
       enddo
       close(22) 

    end program readData
    
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
         sort(ii)=sorttemp(N+1-ii)
      enddo
    endsubroutine sorting
    

   subroutine correlation(X1,X2,N,out)
      implicit none
      integer N
      real X1(N),X2(N)
      real Xnormalization1(N),Xnormalization2(N)
      real out    ! correlation between X1 and X2
      real confidence_level,temp,sum
      integer i
      call normalization(X1,N,Xnormalization1)
      call normalization(X2,N,Xnormalization2)
      out=0.
      do i=1,N
         out=out+Xnormalization1(i)*Xnormalization2(i)
      enddo
      out=out/N
    endsubroutine correlation
