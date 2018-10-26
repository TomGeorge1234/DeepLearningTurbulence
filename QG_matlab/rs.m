function qp=rs(q)
  del=max(max(q))-min(min(q));
  if(del==0)
    qp=0*q;qp(1,1)=1;
    return;
  end
  qp=(q-min(min(q)))/del;

end
