!====================================================================
!  Simple Neural Network (NN) Model.  
!  One or Two hidden layers.  
!  The Output of NN model is One variable.     
!  Auther: Xiangjun Shi(Ê·Ïæ¾ü)  
!  Email: shixj@nuist.edu.cn  
!====================================================================   
    module pra_data_mod
      implicit none
      
      integer,parameter :: Nobs=66                                ! the size of obs samples 
      integer,parameter :: Np=30                                  ! the size of predictors, > N0
      integer,parameter :: Ntrain=40                              ! Training Set
      integer,parameter :: Nbatch=20                              ! batch size  
      integer,parameter :: Nvalidate=26                           ! Validation Set  
      !logical, parameter :: ORTestSet = .True.                     ! False, Training is stopped based on Niters and LossThreshold.
      integer,parameter :: Ntest=100                               ! Test Set. Checking over-fitting during tranning. 
      logical, parameter :: ORTestSet = .False.                     ! False, Training is stopped based on Niters and LossThreshold.
      real,parameter :: LossThreshold = 0.45                       ! Stop Training <LossThreshold.  At ORTestSet = .False.
      integer,parameter :: Niters = 500                            ! max number of iterations, must>=100 Suggest 100-500.
      real,parameter :: LearningRate  = 0.01                       ! LearningRate should be turned based on performance. Suggest 0.01~0.1
      integer,parameter ::  SeedNNstate=43149                          ! used for the NNstate_init
      integer,parameter ::  SeedNNsamples=45                      ! used for the NNsamples
      integer,parameter ::  SeedNNbatch=0                          ! used for selecting the batch for trainning
                                                                   ! Seed<=0,random_seed is set based on system_clock
      !!!!<<<<<<<<Two Hidden Layers>>>>>>>>>
      !integer,parameter :: Nhid=2            
      !integer,parameter :: N0=6,N1=4,N2=3    !N0-input layer;N1-first hidden layer; N2-second hidden layer 
      !!!!<<<<<<<<One Hidden Layer>>>>>>>>>
      integer,parameter :: Nhid=1            
      integer,parameter :: N0=6,N1=4,N2=0   ! Note,the N2 must be defined 0.
      
      !<<<<<Type of Trainning>>>>>>
      !integer,parameter :: NNtraintype=0     ! 0-basetrain;  
      integer,parameter :: NNtraintype=1    ! 1-lootrain 
      integer,parameter :: Nallseeds=200      ! lootrain, seeds number begin from SeedNNstate 
      integer,parameter :: Ngoodseeds=10      ! lootrain, how mang good seeds 
      
      !character(len = *), parameter :: rootdir="/work/shixj/NNtest/"    ! root dir include code directory
      character(len = *), parameter :: rootdir="/home/FCtest/NNtest/"    ! root dir include code directory
      character(len = *), parameter :: ObsDatadir=trim(rootdir)//"ObsData/"   ! 
      character(len = *), parameter :: Samplesdir=trim(rootdir)//"samples/"
      
      logical, parameter :: ORProduceData = .False.    ! False,Read Obs Data or Produced Data. <house data Nobs=506,Np=13> <SST_WPSH, Nobs=66,Np=30>
      !logical, parameter :: ORProduceData = .True.    ! Ture,Producing Obs Samples based on a given equation

      real Xobs(Np,Nobs),Yobs(Nobs)

    endmodule pra_data_mod
    








