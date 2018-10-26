function [u,v]=caluv(ph,k,l,trunc)
  u=-real(ifft2(i*l.*ph.*trunc));
  v=real(ifft2(i*k.*ph.*trunc));
