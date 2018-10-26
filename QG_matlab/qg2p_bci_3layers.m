%%%
clear all


f0=8.64;
beta=1.728e-3;
r=0.05;
rek=0;
L=1000;
W=1000;
Ltop=25;
Htop=0;

dt=1/128;
%dt=1/16;
%dt=1/32;
tpl=1/dt;
%tmax=6*365.25;
%tmax=365.25;
tmax=100000;
%tmax=500;
%tmax=30;
%tmax=5;

del=0.2;


U1=20;
U2=0;
U3=0;
%bci2
%beta*Ltop^2/U2

kappa=0;
%kappa=1;

%kappa*dt/dx/dx*4

nx=64;ny=64;
%nx=256;ny=256;
dx=L/nx;
dy=W/ny;

k0x=2*pi/L;
k0y=2*pi/W;
[k,l]=meshgrid([0:nx/2,-nx/2+1:-1]*k0x,[0:ny/2,-ny/2+1:-1]*k0y);


%% defining new stratification params.

H1=2000/1e3; H2=2000/1e3; H3=2000/1e3; % in km
grav0=9.8/1e3*(0.86e5)^2; % in km/days^2 
grav1=grav0*1e-3;
grav2=grav0*1e-6;

F11=f0^2/H1/grav1;
F21=f0^2/H2/grav1;
F22=f0^2/H2/grav2;
F32=f0^2/H3/grav2;

%%

beta1=beta+F11*(U1-U2);
beta2=beta+F21*(U2-U1)+F22*(U2-U3);
beta3=beta+F32*(U3-U2);

%% defining PV inversion coeffs.

a=(k.*k+l.*l);
wv2=a;
b=F11;
c=F21;
d=F22;
e=F32;

detM=-a.*(a.^2+a*(b+c+d+e)+b*(d+e)+c*e);

a11=(a.^2+c*a+d*a+e*a+c*e)./detM; 

a12=(a*b+e*b)./detM;

a13=(b*d)./detM;

a21=(a*c+e*c)./detM;

a22=(a.^2+b*a+e*a+b*e)./detM;

a23=(a*d+b*d)./detM;

a31=(c*e)./detM;

a32=(a*e+b*e)./detM;

a33=(a.*2+b*a+c*a+d*a+b*d)./detM;

a11(1,1)=0;
a12(1,1)=0;
a13(1,1)=0;
a21(1,1)=0;
a22(1,1)=0;
a23(1,1)=0;
a31(1,1)=0;
a32(1,1)=0;
a33(1,1)=0;
%%





%%
[x,y]=meshgrid([1/2:1:nx]/nx*L-L/2,[1/2:1:ny]/ny*W-W/2);


top=topog(x,y,L,W,Ltop,f0*Htop);

k0=2*pi/nx/dx;
if ~exist('q1')
  amp=0.01;
  q1=0;
  q2=0;
  q3=0;
  for ik=[1,2,3,5,8]
    for il=[-3,-1,0,1,3]
      q1 =q1 + amp*rand(1,1)*cos(ik*k0*x+il*k0*y+2*pi*rand(1,1));
      q2 =q2 + amp*rand(1,1)*cos(ik*k0*x+il*k0*y+2*pi*rand(1,1));
      q3 =q3 + amp*rand(1,1)*cos(ik*k0*x+il*k0*y+2*pi*rand(1,1));
    end
  end
end;
