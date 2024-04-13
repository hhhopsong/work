!====================================================================
!  The weights and bias in the NN Model    
!  Auther: Xiangjun Shi(Ê·Ïæ¾ü)  
!  Email: shixj@nuist.edu.cn  
!====================================================================    
    module types_mod
       use ran_mod, only: Rnormal
       implicit none
       private          ! Make default type private to the module
       
       public NNstate1
       public NNstate2 
       public NNstate1_init
       public NNstate2_init
       public NNstate1_dealloc
       public NNstate2_dealloc
       
       !---------------------------------------------------
       type NNstate1  !One Hidden Layer
          integer :: seed         ! random_seed     
          integer :: nstep        ! number of iterations
          real :: loss,corr       ! loss&correlation
          real :: B1,dB1,mB1,vB1  ! just One output variable
          real, dimension(:), allocatable :: W1,dW1,mW1,vW1
          real, dimension(:), allocatable :: B0,dB0,mB0,vB0
          real, dimension(:,:), allocatable :: W0,dW0,mW0,vW0
          !W0(N0,N1),W1(N1),B0(N1),B1                                                     
       end type NNstate1
       
       type NNstate2  !Two Hidden Layers   
          integer :: seed        ! random_seed 
          integer :: nstep       ! number of iterations
          real :: loss,corr      ! loss&correlation 
          real :: B2,dB2,mB2,vB2 ! just One output variable
          real, dimension(:), allocatable :: W2,dW2,mW2,vW2
          real, dimension(:), allocatable :: B0,dB0,mB0,vB0
          real, dimension(:), allocatable :: B1,dB1,mB1,vB1
          real, dimension(:,:), allocatable :: W0,dW0,mW0,vW0
          real, dimension(:,:), allocatable :: W1,dW1,mW1,vW1
          !W0(N0,N1),W1(N1,N2),W2(N2),B0(N1),B1(N2),B2                                                       
       end type NNstate2

    contains !--------------------------------------------

       subroutine NNstate1_init(state,seed,N0,N1)
          implicit none
          type(NNstate1), intent(inout) :: state
          integer, intent(in) :: seed,N0,N1
          real mean,sigma(0:1)
          integer i,ii
          allocate(state%W0(N0,N1))
          allocate(state%dW0(N0,N1))
          allocate(state%mW0(N0,N1))
          allocate(state%vW0(N0,N1))
          allocate(state%W1(N1))
          allocate(state%dW1(N1))
          allocate(state%mW1(N1))
          allocate(state%vW1(N1))
          allocate(state%B0(N1))
          allocate(state%dB0(N1)) 
          allocate(state%mB0(N1))
          allocate(state%vB0(N1))
          state%seed=seed
          state%nstep=0
          state%loss=0.
          state%corr=0.
          state%B0(:)=0. !
          state%dB0(:)=0.
          state%mB0(:)=0.
          state%vB0(:)=0.
          state%B1=0.    !
          state%dB1=0.
          state%mB1=0.
          state%vB1=0.
          state%dW0(:,:)=0.
          state%mW0(:,:)=0.
          state%vW0(:,:)=0.
          state%dW1(:)=0.
          state%mW1(:)=0.
          state%vW1(:)=0.
          !write(*,*) "state%W0(N0,N1)",N0,N1
          mean=0.
          sigma(0)=1.0/sqrt(N0*1.0)
          sigma(1)=1.0/sqrt(N1*1.0)
          do i=1,N1
             do ii=1,N0
                state%W0(ii,i)=Rnormal(mean,sigma(0))
             enddo   
             !write(*,'(<N0>f8.4)') state%W0(:,i)
          enddo
          !write(*,*) "state%W1(N1)",N1
          do i=1,N1
             state%W1(i)=Rnormal(mean,sigma(1))   
          enddo
          !write(*,'(<N1>f8.4)') state%W1(:)
       end subroutine NNstate1_init
    
       subroutine NNstate2_init(state,seed,N0,N1,N2)
          implicit none
          type(NNstate2), intent(inout) :: state
          integer, intent(in) :: seed,N0,N1,N2
          real mean,sigma(0:2)
          integer i,ii
          allocate(state%W0(N0,N1))
          allocate(state%dW0(N0,N1))
          allocate(state%mW0(N0,N1))
          allocate(state%vW0(N0,N1))
          allocate(state%W1(N1,N2))
          allocate(state%dW1(N1,N2))
          allocate(state%mW1(N1,N2))
          allocate(state%vW1(N1,N2))
          allocate(state%W2(N2))
          allocate(state%dW2(N2)) 
          allocate(state%mW2(N2))
          allocate(state%vW2(N2)) 
          allocate(state%B0(N1))
          allocate(state%dB0(N1)) 
          allocate(state%mB0(N1))
          allocate(state%vB0(N1))
          allocate(state%B1(N2))
          allocate(state%dB1(N2)) 
          allocate(state%mB1(N2)) 
          allocate(state%vB1(N2)) 
          state%seed=seed
          state%nstep=0
          state%loss=0.
          state%corr=0.
          state%B0(:)=0.    !
          state%B1(:)=0.    !
          state%B2=0.       !
          state%dB0(:)=0.
          state%dB1(:)=0.
          state%dB2=0.
          state%mB0(:)=0.
          state%mB1(:)=0.
          state%mB2=0.
          state%vB0(:)=0.
          state%vB1(:)=0.
          state%vB2=0.
          state%dW0(:,:)=0.
          state%dW1(:,:)=0.
          state%dW2(:)=0.
          state%mW0(:,:)=0.
          state%mW1(:,:)=0.
          state%mW2(:)=0.
          state%vW0(:,:)=0.
          state%vW1(:,:)=0.
          state%vW2(:)=0.
          !write(*,*) "state%W0(N0,N1)",N0,N1
          mean=0.
          sigma(0)=1.0/sqrt(N0*1.0)
          sigma(1)=1.0/sqrt(N1*1.0)
          sigma(2)=1.0/sqrt(N2*1.0) 
          do i=1,N1
             do ii=1,N0
                state%W0(ii,i)=Rnormal(mean,sigma(0))
             enddo    
             !write(*,'(<N0>f8.4)') state%W0(:,i)
          enddo
          !write(*,*) "state%W1(N1,N2)",N0,N1
          do i=1,N2
             do ii=1,N1
                state%W1(ii,i)=Rnormal(mean,sigma(1))
             enddo  
             !write(*,'(<N1>f8.4)') state%W1(:,i)
          enddo
          !write(*,*) "state%W2(N2)",N2
          do i=1,N2
             state%W2(i)=Rnormal(mean,sigma(2))  
          enddo
          !write(*,'(<N2>f8.4)') state%W2(:)
       end subroutine NNstate2_init
       
       subroutine NNstate1_dealloc(state)
          implicit none
          type(NNstate1), intent(inout) :: state
          deallocate(state%W0)
          deallocate(state%dW0)
          deallocate(state%mW0)
          deallocate(state%vW0)
          deallocate(state%W1)
          deallocate(state%dW1)
          deallocate(state%mW1)
          deallocate(state%vW1)
          deallocate(state%B0)
          deallocate(state%dB0) 
          deallocate(state%mB0) 
          deallocate(state%vB0) 
       end subroutine NNstate1_dealloc
       
       subroutine NNstate2_dealloc(state)
          implicit none
          type(NNstate2), intent(inout) :: state
          deallocate(state%W0)
          deallocate(state%dW0)
          deallocate(state%mW0)
          deallocate(state%vW0)
          deallocate(state%W1)
          deallocate(state%dW1)
          deallocate(state%mW1)
          deallocate(state%vW1)
          deallocate(state%W2)
          deallocate(state%dW2) 
          deallocate(state%mW2) 
          deallocate(state%vW2) 
          deallocate(state%B0)
          deallocate(state%dB0)
          deallocate(state%mB0)
          deallocate(state%vB0)
          deallocate(state%B1)
          deallocate(state%dB1)
          deallocate(state%mB1)
          deallocate(state%vB1)
       end subroutine NNstate2_dealloc
       
    end module types_mod
    
   