**********************************************************************
      SUBROUTINE SPG2W     !! grid -> spectral
     M         ( WDATA ,
     I           GDATA ,
     C           PNM   , NMO   , TRIGS , IFAX , GW   ,
     F           HGRAD , HFUNC ,
     D           IMAX  , JMAX  , KMAX  , IDIM  , JDIM  ,
     D           LMAX  , MMAX  , NMAX  , MINT  , NMDIM , JMXHF ,
     W           ZDATA , WORK                                    )
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
      REAL*8     WDATA ( NMDIM, KMAX     )    !! spectral data
*
*   [INPUT]
      REAL*8     GDATA ( IDIM*JDIM, KMAX )    !! grid point data
*
      REAL*8     PNM   ( NMDIM, JMXHF )       !! Legendre function
      INTEGER    NMO   ( 2, 0:MMAX , 0:LMAX ) !! order of spect. suffix
      REAL*8     GW    ( JDIM )               !! Gaussian weight
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
      INTEGER    IJ, K
*
      IF ( IMAX .EQ. 1 .OR. JMAX .EQ. 1 ) THEN
         WRITE (6,*) ' ### SPG2W: THIS ROUTINE IS FOR 3 DIM.'
         CALL XABORT( 2 )
         RETURN
      ENDIF
*
*          < 1. LOFFS, LDPNM : flag >
*
      IF ( HFUNC(4:4) .EQ. 'O' ) THEN
         LOFFS = .TRUE.
         IF ( KMAXD .LT. KMAX ) THEN
            WRITE (6,*) ' ### SPG2W: WORK AREA(KMAXD) TOO SMALL < ',
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
*          < 2. duplicate input >
*
      IF ( LOFFS ) THEN
         DO 2000 K = 1, KMAX
            DOFFS ( K ) = GDATA( 1,K )
#ifdef SYS_SX3
*vdir noloopchg
#endif
            DO 2010 IJ = 1, IDIM*JDIM
               WORK ( IJ,K ) = GDATA( IJ,K ) - DOFFS( K )
 2010       CONTINUE
 2000    CONTINUE
      ELSE
         CALL COPY  ( WORK , GDATA , IDIM*JDIM*KMAX )
      ENDIF
*
*          < 3. grid -> zonal wave >
*
      CALL FFT99X
     M         ( WORK  ,
     O           ZDATA ,
     C           TRIGS , IFAX  ,
     C           1     , IDIM  , IMAX  , JDIM*KMAX , 0    )
*
      IF ( HGRAD(1:1) .EQ. 'X' ) THEN
*
*                  < 4. x deriv. >
*
         CALL GRADX
     M         ( ZDATA,
     D           IDIM , JDIM , KMAX , MMAX , MINT  ,
     W           WORK )
*
         LOFFS = .FALSE.
      ENDIF
*
*          < 5. zonal wave -> spectral >
*
      CALL SPZ2W
     M         ( WDATA ,
     I           ZDATA ,
     C           PNM   , NMO   , GW    ,
     F           LDPNM , HFUNC ,
     D           JMAX  , KMAX  , IDIM  , JDIM  ,
     D           LMAX  , MMAX  , NMAX  , MINT  , NMDIM , JMXHF ,
     W           WORK                                           )
*
      IF ( LOFFS ) THEN
         IF (      ( HFUNC(1:1) .EQ. 'N' )
     &        .OR. ( HFUNC(1:1) .EQ. 'S' )    ) THEN
            DO 5100 K = 1, KMAX
               WDATA( NMO(1,0,0),K ) = WDATA( NMO(1,0,0),K )
     &                               - DOFFS( K )
 5100       CONTINUE
         ELSE
            DO 5200 K = 1, KMAX
               WDATA( NMO(1,0,0),K ) = WDATA( NMO(1,0,0),K )
     &                               + DOFFS( K )
 5200       CONTINUE
         ENDIF
      ENDIF
      DO 5300 K = 1, KMAX
         WDATA( NMO(2,0,0), K ) = 0.D0
 5300 CONTINUE
*
      RETURN
      END
**********************************************************************
