      PROGRAM  BarotropicalVorcityEquationModel
      !implicit none
      !chs   1999 9 21 by chs

      parameter(m=20,n=16,l=24,detat=3600.0)
      dimension h(m,n),mm(m,n),ff(m,n),av(m,n),af(m,n), &
                zt(m,n),z(m,n,0:l)
      real      mm

      write(*,*)'-----------------------------------------'
      write(*,*)' barotropical vorcity equation  model	'
      write(*,*)'    finished by chs 1999'
      write(*,*)'-----------------------------------------'

      write(*,*)'1 step read initial height fields.........'
      call read(h,m,n)
      write(*,*)'2 step calculate m and f......'
      open(11,file='chs0.dat')
      call mf(mm,ff,m,n)
      write(*,*)
      write(*,*) '!!!!!! begin ..........!!!!!!!!!!!!'

      write(*,*)'3 step initialize p-height array z......'
      do i=1,m
         do j=1,n
            z(i,j,0)=h(i,j)
         enddo
      enddo

      do kk=1,l
        write(*,*)'!!!!! time step=',kk
        call abv(av,af,h,mm,ff,m,n)
        call pzpt(zt,af,m,n)
        do i=3,m-2
           do j=3,n-2
              if(kk.eq.1) then
                  z(i,j,kk)=z(i,j,kk-1)+zt(i,j)*detat
              else
                  z(i,j,kk)=z(i,j,kk-2)+2*zt(i,j)*detat
              endif
           enddo
        enddo

         do i=3,m-2
            do j=3,n-2
                h(i,j)=z(i,j,kk)
            enddo
         enddo

         do i=1,m
            z(i,1,kk)=h(i,1)
            z(i,2,kk)=h(i,2)
            z(i,15,kk)=h(i,15)
            z(i,16,kk)=h(i,16)
         enddo
         do j=1,n
            z(1,j,kk)=h(1,j)
            z(2,j,kk)=h(2,j)
            z(19,j,kk)=h(19,j)
            z(20,j,kk)=h(20,j)
         enddo
      enddo

       write(*,*)'Please write out the result into a file'
       !请将结果输出至文件中以便画图分析

       stop
       end

!chs     the subroutine of reading original 500hpa height field data

        subroutine read(hh,m0,n0)
        dimension hh(m0,n0)
        open(1,file='dat.txt')
        read(1,*)((hh(i,j),i=1,m0),j=n0,1,-1)

        return
        end

!chs     the subroutine of calculating mm and ff
!chs     mm:the amplification factor of map projection
!chs     ff:coriolis parameter

        subroutine mf(mm,ff,m0,n0)
        parameter(m=20,n=16,se=11423.37,sk=0.7156,d=300.0,&
                  omega=7.292e-5,r=6371.0)
        dimension s(m,n),a(m,n),mm(m0,n0),ff(m0,n0)
        real mm

        return
        end

!chs     the subroutine of calculating absolute vorticity

        subroutine abv(av,af,h,mm,ff,m0,n0)
        parameter  (aff=1.0e-4,d=3.0e5,g=9.8)
        dimension  av(m0,n0),af(m0,n0),h(m0,n0),mm(m0,n0),ff(m0,n0)
        real mm

        return
        end

!chs     the subroutine of calculating potential voyticity tendency

        subroutine pzpt(zt,af,m0,n0)
        parameter(m=20,n=16,alph=1.6,ee=1.0e-5)
        dimension zt(m0,n0),af(m0,n0),r(m,n)
        real rmax

        return
        end
