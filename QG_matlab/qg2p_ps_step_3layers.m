%%%
qg2p_bci_3layers

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

qh1=fft2(q1);
qh2=fft2(q2);
qh3=fft2(q3);

dqh1dt_p=0;
dqh2dt_p=0;
dqh3dt_p=0;

dt0=dt;dt1=0;

while t<=tmax+dt/2

  q1=real(ifft2(qh1.*trunc));
  q2=real(ifft2(qh2.*trunc));
  q3=real(ifft2(qh3.*trunc));

  
  [ph1,ph2,ph3]=invert_3layers(qh1,qh2,qh3, a11,a12, a13, a21,a22,a23, a31,a32,a33);
  [u1,v1]=caluv(ph1,k,l,trunc);
  [u2,v2]=caluv(ph2,k,l,trunc);
  [u3,v3]=caluv(ph3,k,l,trunc);

 
  dqh1dt=-advect(q1,u1+U1,v1,k,l)-beta1*i*k.*ph1;
  dqh2dt=-advect(q2,u2+U2,v2,k,l)-beta2*i*k.*ph2;
  dqh3dt=-advect(q3+top,u3+U3,v3,k,l)-beta3*i*k.*ph3+rek*wv2.*ph3; %wv2=l^2+k^2

  if(rem(tc,tpl)==0)
%    w=calw(dqh1dt,dqh2dt,u1+U1,v1,u2+U2,v2,a11,a12,a21,a22,trunc);
    ts=[ts,t];
%    stat=[stat,[max(max(w));min(min(w))]];
%    stat=[stat,[mean(mean(q1.^2))/2;mean(mean(q2.^2))/2]];
%    stat=[stat,max(max(abs(q1-q1_0)))];
%    df(t,psi,min(min(psi)),max(max(psi)),'/tmp/stp',2);
%    df(t,psi,min(min(psi)),max(max(psi)),2);
%    df(t,q,min(min(q)),max(max(q)),2);
    psi1=real(ifft2(ph1));
    stat=[stat,[0.5/(1+del)*mean((del*(u1(:).^2+v1(:).^2)+(u2(:).^2+v2(:).^2)));
    mean(mean(psi1.*v2))]];
    ff=[rs(q1),rs(q2),rs(q3)];
%    ff=[rs(p1),rs(p2)];
%    ff=rs(w);
%    ff=[rs(q1),rs(q2),rs(w)];
   figure(1);
   imagesc(ff);axis('xy','equal');title(sprintf('q : t=%g',t));
   drawnow;
     figure(2);
     plot(ts,stat(2,:));
     drawnow();
%    df(t,x,y,q);
%    save(["bcijetpulse-",num2str(t),".mat"]);
  end

  qh1=frc.*filtr.*(qh1+dt0*dqh1dt+dt1*dqh1dt_p);
  qh2=frc.*filtr.*(qh2+dt0*dqh2dt+dt1*dqh2dt_p);
  qh3=frc.*filtr.*(qh3+dt0*dqh3dt+dt1*dqh3dt_p);

  dqh1dt_p=frc.*dqh1dt;
  dqh2dt_p=frc.*dqh2dt;
  dqh3dt_p=frc.*dqh3dt;

  if tc==0
    dt0=1.5*dt;dt1=-0.5*dt;
  end
  tc=tc+1;
  t=tc*dt;

  if sum(isnan(qh1(:)))>0
    return;
  end
end

%pause;
%df(-1);

%Sf=1/(1+2*kappa*tmax/el^2)*exp(-1/(2*(el^2+2*kappa*tmax))*((x-1).^2+y.^2));
%plot(y(:,85),S(:,85),y(:,85),S0(:,85),y(:,85),Sf(:,85))
%plot(y(:,43),S(:,43),y(:,43),S0(:,43),y(:,43),Sf(:,43))
plot(ts,stat);
