**********************************************************************
      SUBROUTINE SPW2G     !! spectral -> grid transform
     M         ( GDATA ,
     I           WDATA ,
     C           PNM   , NMO   , TRIGS , IFAX ,
     F           HGRAD , HFUNC ,
     D           IMAX  , JMAX  , KMAX  , IDIM  , JDIM  ,
     D           LMAX  , MMAX  , NMAX  , MINT  , NMDIM , JMXHF, 
     W           ZDATA , WORK                                   )
*
*   [PARAM] 
      INTEGER    IMAX
      INTEGER    JMAX
      INTEGER    KMAX
      INTEGER    IDIM
      INTEGER    JDIM
      INTEGER    LMAX
      INTEGER    MMAX
      INTEGER    NMAX
      INTEGER    MINT
      INTEGER    NMDIM
      INTEGER    JMXHF
*
*   [MODIFY]       
      REAL*8     GDATA ( IDIM*JDIM, KMAX )    !! grid point data
*
*   [INPUT]
      REAL*8     WDATA ( NMDIM, KMAX     )    !! spectral data
*
      REAL*8     PNM   ( NMDIM, JMXHF )       !! Legendre function
      INTEGER    NMO   ( 2, 0:MMAX , 0:LMAX ) !! order of spect. suffix
      REAL*8     TRIGS ( * )                  !! triangle function table
      INTEGER    IFAX  ( * )                  !! factorziation of IMAX
*
      CHARACTER  HGRAD*4                      !! flag of differential
      CHARACTER  HFUNC*4                      !! flag of sign
*
*   [WORK] 
      REAL*8     ZDATA ( IDIM*JDIM, KMAX )    !! zonal spectral
      REAL*8     WORK  ( IDIM*JDIM, KMAX )    !! work
*
*   [INTERNAL WORK] 
      LOGICAL    LDPNM                        !! y differentail flag
      LOGICAL    LOFFS                        !! offset flag
      INTEGER    KMAXD
      PARAMETER (KMAXD=100)
      REAL*8     DOFFS ( KMAXD )              !! offset value
      INTEGER    IJ, K, I
      REAL*8     WORKZ
*
      IF ( IMAX .EQ. 1 .OR. JMAX .EQ. 1 ) THEN
         WRITE (6,*) ' ### SPW2G: THIS ROUTINE IS FOR 3 DIM.'
         CALL XABORT( 2 )
         RETURN
      ENDIF
*
*          < 1. LOFFS, LDPNM : flag >
*
      IF ( HFUNC(4:4) .EQ. 'O' ) THEN
         LOFFS = .TRUE.
         IF ( KMAXD .LT. KMAX ) THEN
            WRITE (6,*) ' ### SPW2G: WORK AREA(KMAXD) TOO SMALL < ',
     &                  KMAX
            CALL XABORT( 1 )
            RETURN
         ENDIF
      ELSE
         LOFFS = .FALSE.
      ENDIF
*
      IF ( HGRAD(1:1) .EQ. 'Y' ) THEN
         LDPNM = .TRUE.
         LOFFS = .FALSE.
      ELSE
         LDPNM = .FALSE.
      ENDIF
*
*          < 2. spectral -> zonal wave >
*
      IF ( LOFFS ) THEN
         DO 2000 K = 1, KMAX
            DOFFS( K ) = WDATA( NMO(1,0,0), K )
            WDATA( NMO(1,0,0), K ) = 0.  
 2000    CONTINUE
      ENDIF
*
      CALL SPW2Z
     O         ( ZDATA ,
     I           WDATA ,
     C           PNM   , NMO   ,
     F           LDPNM ,
     D           JMAX  , KMAX  , IDIM  , JDIM  ,
     D           LMAX  , MMAX  , NMAX  , MINT  , NMDIM , JMXHF ,
     W           WORK                                           )
*
      IF ( LOFFS ) THEN
         DO 2100 K = 1, KMAX
            WDATA( NMO(1,0,0), K ) = DOFFS( K )
 2100    CONTINUE
      ENDIF
*
      IF ( HGRAD(1:1) .EQ. 'X' ) THEN
*
*          < 3. x deriv. >
*
         CALL GRADX
     M         ( ZDATA,
     D           IDIM , JDIM , KMAX  , MMAX , MINT ,
     W           WORK                                 )
         LOFFS = .FALSE.
      ENDIF
*
*          < 4. zonal wvae -> grid >
*
      CALL FFT99X
     O         ( WORK  ,
     M           ZDATA ,
     C           TRIGS , IFAX  ,
     C           1     , IDIM  , IMAX  , JDIM*KMAX , 1    )
*

      DO 4000 K = 1, KMAX
         WORKZ = WORK( 1,K )
         DO 4010 IJ = IDIM*JMAX+1, IDIM*JDIM
            WORK( IJ,K ) = WORKZ
 4010    CONTINUE 
         DO 4020 I = IMAX+1, IDIM
            DO 4020 IJ = 1, IDIM*JDIM, IDIM
               WORK( IJ+I-1,K ) = WORKZ
 4020    CONTINUE 
 4000 CONTINUE 
*
      IF ( LOFFS ) THEN
         DO 4100 K = 1, KMAX
#ifdef SYS_SX3
*vdir noloopchg
#endif
            DO 4100 IJ = 1, IDIM*JDIM
               WORK ( IJ,K ) = WORK ( IJ,K ) + DOFFS( K )
 4100    CONTINUE
      ENDIF
*
*          < 5. output data >
*
      IF      ( HFUNC(1:1) .EQ. 'A' ) THEN
*
*                                ( add )
        DO 5000 K = 1, KMAX
           DO 5000 IJ = 1, IDIM*JDIM
              GDATA( IJ,K ) = GDATA( IJ,K ) + WORK( IJ,K )
 5000   CONTINUE
*
      ELSE IF ( HFUNC(1:1) .EQ. 'S' ) THEN
*
*                                ( sub )
        DO 5100 K = 1, KMAX
           DO 5100 IJ = 1, IDIM*JDIM
              GDATA( IJ,K ) = GDATA( IJ,K ) - WORK( IJ,K )
 5100   CONTINUE
*
      ELSE IF ( HFUNC(1:1) .EQ. 'N' ) THEN
*
*                                ( negative )
        DO 5200 K = 1, KMAX
           DO 5200 IJ = 1, IDIM*JDIM
              GDATA( IJ,K ) = - WORK( IJ,K )
 5200   CONTINUE
*
      ELSE
*                                ( positive )
        DO 5300 K = 1, KMAX
           DO 5300 IJ = 1, IDIM*JDIM
              GDATA( IJ,K ) = WORK( IJ,K )
 5300   CONTINUE
      ENDIF
*
      RETURN
      END
**********************************************************************
