%%%
f0=8.64;
beta=1.728e-3;

r=0;
rek=0.1;
L=4000;
W=L;
Ltop=25;
Htop=0;

dt=1/64;
%dt=1/16;
%dt=1/32;
tpl=1/dt;
%tmax=6*365.25;
%tmax=365.25;
tmax=100000;
%tmax=500;
%tmax=30;
%tmax=5;

Rd=40;
%Rd=10;
%Rd=2;
del=0.2;


U1=20; U2=0;
%bci2

%beta*Ltop^2/U2

kappa=0;
kappa=1;

%kappa*dt/dx/dx*4

nx=256;ny=256;
dx=L/nx;
dy=W/ny;

k0x=2*pi/L;
k0y=2*pi/W;
[k,l]=meshgrid([0:nx/2,-nx/2+1:-1]*k0x,[0:ny/2,-ny/2+1:-1]*k0y);

F1=1/Rd^2/(1+del);
F2=del*F1;

beta1=beta+F1*(U1-U2);
beta2=beta-F2*(U1-U2);

wv2=(k.*k+l.*l);
det=wv2.*(wv2+F1+F2);
a11=-(wv2+F2)./det;
a12=-F1./det;
a21=-F2./det;
a22=-(wv2+F1)./det;

a11(1,1)=0;
a12(1,1)=0;
a21(1,1)=0;
a22(1,1)=0;

[x,y]=meshgrid([1/2:1:nx]/nx*L-L/2,[1/2:1:ny]/ny*W-W/2);


top=topog(x,y,L,W,Ltop,f0*Htop);

k0=2*pi/nx/dx;
if ~exist('q1')
  amp=0.1;
  q1=0;
  q2=0;
  for ik=[1,2,5,8]
    for il=[-3,-1,0,1,3]
      q1 =q1 + amp*rand(1,1)*cos(ik*k0*x+il*k0*y+2*pi*rand(1,1));
      q2 =q2 + amp*rand(1,1)*cos(ik*k0*x+il*k0*y+2*pi*rand(1,1));
    end
  end
end;
