*======================================================================
      ENTRY      G2W       !!  spherical trans.(grid->spect.)
     M         ( WDATA ,
     I           GDATA ,
     I           HGRAD , HFUNC , KMAXD )
*
      IF ( .NOT. OSET ) THEN
         WRITE (6,*) ' ### G2W: SPSTUP MUST BE CALLED BEFORE'
         CALL XABORT( 1 )
         RETURN
      ENDIF
*
      IF      ( HGRAD(1:1) .EQ. 'Y' ) THEN
         CALL    SPG2W
     M         ( WDATA ,
     I           GDATA ,
     C           DPNM  , NMO   , TRIGS , IFAX  , GWDEL ,
     F           HGRAD , HFUNC ,
     D           IMAX  , JMAX  , KMAXD , IDIM  , JDIM  ,
     D           LMAX  , MMAX  , NMAX  , MINT  , NMDIM , JMXHF ,
     W           ZDATA , WORK                                   )
      ELSE IF ( HGRAD(1:1) .EQ. 'X' ) THEN
         CALL    SPG2W
     M         ( WDATA ,
     I           GDATA ,
     C           PNM   , NMO   , TRIGS , IFAX  , GWDEL ,
     F           HGRAD , HFUNC ,
     D           IMAX  , JMAX  , KMAXD , IDIM  , JDIM  ,
     D           LMAX  , MMAX  , NMAX  , MINT  , NMDIM , JMXHF ,
     W           ZDATA , WORK                                   )
      ELSE
         CALL    SPG2W
     M         ( WDATA ,
     I           GDATA ,
     C           PNM   , NMO   , TRIGS , IFAX  , GWX   ,
     F           HGRAD , HFUNC ,
     D           IMAX  , JMAX  , KMAXD , IDIM  , JDIM  ,
     D           LMAX  , MMAX  , NMAX  , MINT  , NMDIM , JMXHF ,
     W           ZDATA , WORK                                   )
      ENDIF
*
      RETURN
*=====================================================================
