"""
      INTEGER     IMAX           # No.of longitudinal grid
      INTEGER     JMAX           # number of latitude grid
      INTEGER     NMAX           # maximum total wave number
      INTEGER     MINT           # interval of zonal wave number
      INTEGER     MMAX           # maximum zonal wave number
      INTEGER     LMAX           # maximum meridional wave number
      INTEGER     IDIM           # size of longitude dimension
      INTEGER     JDIM           # size of latitude dimension
      INTEGER     IJDIM          # IDIM*JDIM
      INTEGER     IJSDIM         # size of physical process
      INTEGER     KDIM           # vertical dimension size
      INTEGER     IJKDIM         # total size of matrix
"""


IMAX = 128  # No.of longitudinal grid
JMAX = 64  # number of latitude grid
NMAX = 42  # maximum total wave number
KMAX = 20
MINT = 1  # interval of zonal wave number
NVAR = 4
MAXH = 2
MAXV=3
NOMIT=NMAX + 2
MSIZR=1



MMAX = NMAX  # maximum zonal wave number
LMAX = NMAX  # maximum meridional wave number
IDIM = IMAX + 1  # size of longitude dimension
JDIM = JMAX  # size of latitude dimension
IJDIM = IDIM * JDIM  # IDIM*JDIM
IJSDIM = IJDIM  # size of physical process
KDIM = KMAX  # vertical dimension size
IJKDIM = IJDIM * KDIM  # total size of matrix
MMXMI = MMAX // MINT
NTR = MMAX
NMDIM = (MMXMI + 1) * (2 * (NMAX + 1) - MMXMI * MINT) - (NMAX - LMAX) // MINT * (NMAX - LMAX + 1)  # of horizontal wave
JMXHF = JMAX // 2 + 1  # JMAX/2+1
MSIZ=(NMDIM - NOMIT) * (KMAX * NVAR + 1)
MSIZ2=(NMDIM - NOMIT) * KMAX