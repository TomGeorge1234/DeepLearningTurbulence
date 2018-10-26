function top=topog(x,y,L,W,Ltop,amp)
  top=amp*exp(-0.5/Ltop^2*((x+L/4).^2+y.^2));
