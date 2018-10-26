%%%
%load initial.mat;
 clear all; close all
 qg2p_bci_beta

frc=exp(-kappa*dt*wv2-r*dt);

switch(nx)
    case 128
      cphi = 0.69*pi;
    case 256 
      cphi = 0.715*pi;
    case 512
      cphi = 0.735*pi;
    otherwise
      cphi = 0.65*pi;
end

wvx=sqrt((k*dx).^2+(l*dy).^2);
filtr=exp(-18*(wvx-cphi).^7).*(wvx>cphi)+(wvx<=cphi);
filtr(isnan(filtr))=1;

kmax2=((nx/2-1)*k0x).^2;

trunc=(wv2<kmax2);


%%%%%%%%%%%%%%%%%%%%%%%%%

t=0;
tc=0;

psimax=[];
ts=[];
stat=[];
FLUX=[];FLUX_q=[];
clear psi1_ts  psi2_ts v1_ts v2_ts u1_ts q1_ts q2_ts  


qh1=fft2(q1);
qh2=fft2(q2);

dqh1dt_p=0;
dqh2dt_p=0;
dt0=dt;dt1=0;

U1_mean=[];
    fig=figure;
    set(fig,'position',[50 50 1000 500]);
tic
while t<=tmax+dt/2
  
  q1=real(ifft2(qh1.*trunc));
  q2=real(ifft2(qh2.*trunc));
  [ph1,ph2]=invert(qh1,qh2,a11,a12,a21,a22);
  [u1,v1]=caluv(ph1,k,l,trunc);
  [u2,v2]=caluv(ph2,k,l,trunc);
  zeta1=real(ifft2(ph1.*(k.^2+l.^2)));
  dqh1dt=-advect(q1,u1+U1,v1,k,l)-beta1*1i*k.*ph1;
  dqh2dt=-advect(q2+top,u2+U2,v2,k,l)-beta2*1i*k.*ph2+rek*wv2.*ph2;
  if(rem(tc,tpl)==0)
    ts=[ts,t];
    psi1=real(ifft2(ph1));    psi2=real(ifft2(ph2));

    stat=[stat,[0.5/(1+del)*mean((del*(u1(:).^2+v1(:).^2)+(u2(:).^2+v2(:).^2))); mean(mean(psi1.*v2))]];
    FLUX= [FLUX,mean(mean(psi1.*v2))];
    FLUX_q= [FLUX_q,mean(mean(q1.*v1))];
    QGPV=rs(q1);
    U1_mean=[U1_mean, mean(u1,2)];
    
    figure(fig); clf(1)
    
    axes('position',[0.1 0.1 0.4 0.8])
    imagesc(0:dx:L,0:dy:W,zeta1); %colormap('gray')
    title(sprintf('Input: QGPV @ Time=%g',t)); caxis([-1 1]*10)
    xlabel('km'); ylabel('km')
    
    axes('position',[0.85 0.1 0.1 0.8]);
    plot(ts,rs(FLUX));  if max(ts)~=0 xlim([0 max(ts)]); end
    xlabel('Time, days')
    
    axes('position',[0.55 0.1 0.25 0.8]);
    imagesc(U1_mean); %caxis([-50 50]);
    drawnow();
    toc; tic
  end

  qh1=frc.*filtr.*(qh1+dt0*dqh1dt+dt1*dqh1dt_p);
  qh2=frc.*filtr.*(qh2+dt0*dqh2dt+dt1*dqh2dt_p);

  dqh1dt_p=frc.*dqh1dt;
  dqh2dt_p=frc.*dqh2dt;

  if tc==0
    dt0=1.5*dt;dt1=-0.5*dt;
  end
  tc=tc+1;
  t=tc*dt;

  if sum(isnan(qh1(:)))>0
    return;
  end
  
end


