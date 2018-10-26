%%%
%load initial.mat;
clear all

%save_dir='data3/'; 
rng('shuffle')
save_dir = input(' type save_dir = ', 's');

mkdir(save_dir);

qg2p_bci

%Variables
Nsamples=20000; % total
jump_days=1; %days between samples saved
savefreq=1000; %saves and overwrites previous 
%%%%%%%%



tpl=fix(jump_days/dt);

tmax=Nsamples*jump_days;

input=zeros(Nsamples,nx,ny);
output=zeros(Nsamples,2);

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
FLUX=[];
clear psi1_ts  psi2_ts v1_ts v2_ts u1_ts q1_ts q2_ts  


qh1=fft2(q1);
qh2=fft2(q2);

dqh1dt_p=0;
dqh2dt_p=0;
dt0=dt;dt1=0;


%     fig=figure;
%     set(fig,'position',[50 50 1000 500]);
tic
    sample_indx=1;
while t<=tmax+dt/2

  q1=real(ifft2(qh1.*trunc));
  q2=real(ifft2(qh2.*trunc));
  [ph1,ph2]=invert(qh1,qh2,a11,a12,a21,a22);
  [u1,v1]=caluv(ph1,k,l,trunc);
  [u2,v2]=caluv(ph2,k,l,trunc);

  dqh1dt=-advect(q1,u1+U1,v1,k,l)-beta1*i*k.*ph1;
  dqh2dt=-advect(q2+top,u2+U2,v2,k,l)-beta2*i*k.*ph2+rek*wv2.*ph2;

  if(rem(tc,tpl)==0)
    ts=[ts,t];
    psi1=real(ifft2(ph1));    psi2=real(ifft2(ph2));

    stat=[stat,[0.5/(1+del)*mean((del*(u1(:).^2+v1(:).^2)+(u2(:).^2+v2(:).^2))); mean(mean(psi1.*v2))]];
    FLUX= [FLUX,mean(mean(psi1.*v2))];
    QGPV=rs(q1+beta1*y);
    
    output=stat';
    input(sample_indx,:,:)=QGPV;
    
    if rem(sample_indx,savefreq)==0
    toc;
    in=input(1:sample_indx,:,:); save([save_dir 'input.mat'],'in');
    out=output(1:sample_indx,:); save([save_dir 'output.mat'],'out');
    display(sample_indx);
    end
    
    sample_indx=sample_indx+1;
  
    % figure(fig);
    % axes('position',[0.1 0.1 0.45 0.8])
    % imagesc(0:dx:W,0:dy:L,QGPV); %colormap('gray')
    % title(sprintf('Input: QGPV @ Time=%g',t)); caxis([0 1])
    %    xlabel('km'); ylabel('km')
    %  axes('position',[0.65 0.1 0.3 0.8]);
    % plot(ts,FLUX);
    % hold on; 
    % xlabel('Time, days')
    % plot(ts(end),FLUX(end),'ro');title('Output: FLUX');
    % %xlim([0 tmax]); ylim([-0.1 1]*1e4)
    % drawnow(); clf(fig)
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

