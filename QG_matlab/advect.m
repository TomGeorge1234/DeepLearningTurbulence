function qdot=advect(q,u,v,k,l)
  qdot=i*k.*fft2(u.*q)+i*l.*fft2(v.*q);
