      SUBROUTINE W2G       !! spherical trans.(spect.->grid)
     M         ( GDATA ,
     I           WDATA ,
     I           HGRAD , HFUNC , KMAXD )
*
*   [PARAM]
#include        "zcdim.F"                /* # of grid point & wave */
#include        "zddim.F"                /* NMDIM                  */
#include        "zccom.F"                /* stand. physical const. */
#include        "zcord.F"                /* coordinate             */
*
      INTEGER    KMAXD
*
*   [MODIFY]
      REAL*8     GDATA ( IDIM*JDIM, KMAXD )  !! grid point data
*
*   [INPUT]
      REAL*8     WDATA ( NMDIM, KMAXD    )   !! spectral data
      CHARACTER  HGRAD*4                     !! flag of differential
      CHARACTER  HFUNC*4                     !! flag of sign
*
*   [INTERNAL WORK]
      REAL*8     ZDATA ( IDIM*JDIM, KMAX )   !! zonal spectral
      REAL*8     WORK  ( IDIM*JDIM, KMAX )   !! work
*
      REAL * 8   QSINLA( JDIM )              !! sin(lat.):double
      REAL * 8   QGW   ( JDIM )              !! Gaussian weight:double
      REAL * 8   QPNM  ( 0:NMAX+1, 0:MMAX )  !! Pnm Legendre
      REAL * 8   QDPNM ( 0:NMAX+1, 0:MMAX )  !! mu differential of Pnm
*
      INTEGER    J
*
*   [INTERNAL SAVE]
      REAL*8     PNM   ( JMXHF*NMDIM )       !! Pnm Legendre
      REAL*8     DPNM  ( JMXHF*NMDIM )       !! mu differential of Pnm
      REAL*8     TRIGS ( IDIM*2 )            !! triangle function table
      INTEGER    IFAX  ( 10 )                !! factorziation of IMAX
      INTEGER    NMO   ( 2, 0:MMAX, 0:LMAX ) !! order of spect. suffix
      REAL*8     GWX   ( JDIM )              !! Gaussian weight
      REAL*8     GWDEL ( JDIM )              !! Gaussian weight for diff
      SAVE       PNM, DPNM, TRIGS, IFAX, NMO, GWX, GWDEL
      LOGICAL    OSET                        !! flag of setting const.
      LOGICAL    OFIRST
      DATA       OSET   / .FALSE. /
      DATA       OFIRST / .TRUE. /
*
      IF ( OFIRST ) THEN
         WRITE (6,*) ' @@@ DSPHE: SPHERICAL TRANSFORM INTFC. 93/12/07'
         OFIRST= .FALSE.
      ENDIF
*
      IF ( .NOT. OSET ) THEN
         WRITE (6,*) ' ### W2G: SPSTUP MUST BE CALLED BEFORE'
         CALL XABORT( 1 )
         RETURN
      ENDIF
*
      IF ( HGRAD(1:1) .EQ. 'Y' ) THEN
         CALL    SPW2G
     M         ( GDATA ,
     I           WDATA ,
     C           DPNM  , NMO   , TRIGS , IFAX ,
     F           HGRAD , HFUNC ,
     D           IMAX  , JMAX  , KMAXD , IDIM  , JDIM  ,
     D           LMAX  , MMAX  , NMAX  , MINT  , NMDIM , JMXHF ,
     W           ZDATA , WORK                                   )
      ELSE
         CALL    SPW2G
     M         ( GDATA ,
     I           WDATA ,
     C           PNM   , NMO   , TRIGS , IFAX ,
     F           HGRAD , HFUNC ,
     D           IMAX  , JMAX  , KMAXD , IDIM  , JDIM  ,
     D           LMAX  , MMAX  , NMAX  , MINT  , NMDIM , JMXHF ,
     W           ZDATA , WORK                                   )
      ENDIF
*
      RETURN
*======================================================================